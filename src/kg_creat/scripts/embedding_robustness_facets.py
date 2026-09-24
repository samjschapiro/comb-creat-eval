"""Is the leaderboard, and the "association originality predicts every other facet" finding, an artifact
of the sentence encoder? Surprise and originality (base and emergent) are the embedding-derived dimensions.
This recomputes all of them under several encoders, in memory, with the CURRENT definitions (base
originality for blending = rho over the item's generic spaces; emergent originality over the invention's
elements), rebuilds the stationary composite exactly as compute_composite.py does, and reports:

  1. leaderboard agreement across encoders (overall and per task; Spearman, Kendall, top-k overlap, largest
     rank move), plus a sanity check that the canonical encoder reproduces composite.json;
  2. facet stability: for every ordered encoder pair (X, Y), the correlation across models between
     association originality scored under X and every other facet scored under Y, using the UNGATED facet
     means of analyze_facet_correlations.py; and, per pair, whether association originality is still the
     facet with the highest mean cross-task correlation.

Judge fields are never touched and no canonical file is written.
    .venv_mlx/bin/python -m src.kg_creat.scripts.embedding_robustness_facets
"""
import glob
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

from src.kg_creat import scoring
from src.kg_creat.embed import get_embedder
from src.kg_creat.parse import EmittedPath
from src.kg_creat.scripts.score import _norm

RESP = "data/kg_creat/kombine_test30/responses"
SCORES = "data/kg_creat/kombine_test30/scores"
OUT = Path("data/kg_creat/kombine_test30/analysis/embedding_robustness_facets.json")
ENCODERS = ["mlx-community/all-MiniLM-L6-v2-4bit",       # canonical: everything in the paper is scored with it
            "mlx-community/bge-small-en-v1.5-4bit",
            "mlx-community/multilingual-e5-small-mlx"]
TASKS = {"baseline": "association", "analogy": "analogy", "blending": "blending"}
DIMS = {"association": ["utility", "surprise", "originality"],
        "analogy": ["utility", "surprise", "originality", "em_originality", "em_utility", "em_integration"],
        "blending": ["utility", "surprise", "originality", "em_originality", "em_utility", "em_integration"]}
K = 5


def _elements(triple_lists, u, v):
    """score.py::_artifact_elements: relations and non-input concepts, tagged so shared strings stay distinct."""
    au, av = _norm(u), _norm(v)
    els = set()
    for ts in triple_lists:
        for t in ts:
            if len(t) != 3:
                continue
            els.add(("r", _norm(t[1])))
            for x in (t[0], t[2]):
                if _norm(x) not in (au, av):
                    els.add(("c", _norm(x)))
    return els


def artifacts():
    """One record per scored artifact head: what the embedding dimensions need plus the judge verdicts."""
    heads = {}
    for f in sorted(glob.glob(f"{SCORES}/*/path_scores.json")):
        m = f.split("/")[-2]
        for r in json.load(open(f)):
            key = (m, r["mode"], r["prompt_id"], r.get("path_idx"))
            if r["mode"] == "analogy":
                if "pair_sat" in r:
                    heads[key] = r
            elif r.get("triples"):
                heads[key] = r
    out = []
    for f in sorted(glob.glob(f"{RESP}/*/responses.json")):
        m = f.split("/")[-2]
        for r in json.load(open(f)):
            mode = r.get("mode")
            if mode not in TASKS or not r.get("items"):
                continue
            u, v = r.get("u_label"), r.get("v_label")
            item = (u, v)
            for pi, path in enumerate(r.get("paths") or []):
                key = (m, mode, r["prompt_id"], pi)
                if key not in heads:
                    continue
                h = heads[key]
                p = EmittedPath(path)
                it = (r["items"] or [{}])[min(pi // (2 if mode == "analogy" else 1), len(r["items"]) - 1)]
                rec = {"model": m, "task": TASKS[mode], "item": item, "sat": bool(h.get("pair_sat" if mode == "analogy" else "sat") is True),
                       "entities": list(p.entities), "emg": set()}
                if mode == "blending":
                    rec["g"] = (it.get("generic_space") or "").strip()
                    tags = it.get("tags") or []
                    tr = (it.get("paths") or [[]])[0]
                    rec["base"] = {("c", _norm(rec["g"]))} if rec["g"] else set()          # O_bl(g) := rho(g)
                    rec["emg"] = _elements([[t for i, t in enumerate(tr) if i < len(tags) and tags[i] == "emergent"]], u, v)
                    rec["em_utility"] = None if h.get("blend_utility") is None else float(bool(h.get("blend_utility")))
                    sc = h.get("blend_integration")
                    rec["em_integration"] = None if not sc else (float(sc) - 1) / 2
                elif mode == "analogy":
                    if pi % 2 or pi + 1 >= len(r["paths"]):
                        continue
                    rec["pair"] = (list(p.entities), list(EmittedPath(r["paths"][pi + 1]).entities))
                    rec["base"] = _elements([path, r["paths"][pi + 1]], u, v)
                    imgs = [pr["image"] for pr in (it.get("projection") or []) if isinstance(pr, dict) and len(pr.get("image", [])) == 3]
                    rec["emg"] = _elements([imgs], u, v)
                    if it.get("invention"):
                        rec["emg"].add(("c", _norm(it["invention"])))
                    rec["em_utility"] = None if h.get("invention_utility") is None else float(bool(h.get("invention_utility")))
                    rec["em_integration"] = None if h.get("invention_integration") is None else float(bool(h.get("invention_integration")))
                else:
                    rec["base"] = _elements([path], u, v)
                out.append(rec)
    return out


def score_with(recs, embed):
    """Surprise, base originality and emergent originality for every artifact under one encoder."""
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    cache = {}

    def V(s):
        key = s if isinstance(s, str) else tuple(s)
        if key not in cache:
            cache[key] = un(np.asarray(embed(str(s if isinstance(s, str) else s[1])), float))
        return cache[key]

    for r in recs:                                          # surprise, per the task's definition
        if r["task"] == "blending":
            u, v = r["item"]
            r["R"] = float((scoring.cosine_distance(V(u), V(r["g"])) + scoring.cosine_distance(V(v), V(r["g"]))) / 2) if r.get("g") else None
        elif r["task"] == "analogy":
            ea, eb = r["pair"]
            dd = [scoring.cosine_distance(V(ea[i]), V(eb[i])) for i in range(min(len(ea), len(eb)))]
            r["R"] = (sum(dd) / len(dd)) if dd else None
        else:
            e = r["entities"]
            dd = [scoring.cosine_distance(V(e[i]), V(e[i + 1])) for i in range(len(e) - 1)]
            r["R"] = (sum(dd) / len(dd)) if dd else None
    for field, out_key in (("base", "originality"), ("emg", "em_originality")):
        pools = defaultdict(set)                            # one pool per (task, item), base and emergent apart
        for r in recs:
            pools[(r["task"], r["item"])].update(tuple(e) for e in r.get(field, ()))
        pool_vecs = {key: {s: V(s) for s in ss} for key, ss in pools.items()}
        for r in recs:
            if r["task"] == "association" and field == "emg":
                continue
            pool = pool_vecs[(r["task"], r["item"])]
            rhos = []
            for s in {tuple(e) for e in r.get(field, ())}:
                d = sorted(scoring.cosine_distance(pool[s], o) for t, o in pool.items() if t != s)
                if d:
                    rhos.append(sum(d[:min(K, len(d))]) / min(K, len(d)))
            r[out_key] = (sum(rhos) / len(rhos)) if rhos else None
    return recs


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def composite(recs, models):
    """compute_composite.py's stationary percent-of-max score, gated by utility."""
    per_task, overall = {}, {}
    for m in models:
        per_task[m] = {}
        for task, dims in DIMS.items():
            rs = [r for r in recs if r["model"] == m and r["task"] == task]
            if not rs:
                per_task[m][task] = float("nan"); continue
            vals = []
            for dim in dims:
                if dim == "utility":
                    vals.append(_mean([1.0 if r["sat"] else 0.0 for r in rs]))
                else:
                    key = {"surprise": "R"}.get(dim, dim)
                    vals.append(_mean([(min(1.0, max(0.0, r[key])) if r["sat"] else 0.0) for r in rs if r.get(key) is not None]))
            per_task[m][task] = 100.0 * _mean(vals)
        overall[m] = _mean(list(per_task[m].values()))
    return overall, per_task


def facets(recs, models):
    """analyze_facet_correlations.py's UNGATED facet means: {(task, dim): {model: value}}."""
    out = defaultdict(dict)
    for m in models:
        for task, dims in DIMS.items():
            rs = [r for r in recs if r["model"] == m and r["task"] == task]
            for dim in dims:
                if dim == "utility":
                    out[(task, dim)][m] = _mean([1.0 if r["sat"] else 0.0 for r in rs])
                else:
                    key = {"surprise": "R"}.get(dim, dim)
                    out[(task, dim)][m] = _mean([r[key] for r in rs if r.get(key) is not None])
    return out


def _ranks(vals, models):
    order = sorted(models, key=lambda m: -vals[m])
    return {m: i + 1 for i, m in enumerate(order)}


def main():
    recs_base = artifacts()
    models = sorted({r["model"] for r in recs_base})
    short = lambda n: n.split("/")[-1].replace("-4bit", "").replace("-mlx", "")
    print(f"{len(recs_base)} artifacts, {len(models)} models, {len(ENCODERS)} encoders\n")
    comp, fac = {}, {}
    for name in ENCODERS:
        recs = deepcopy(recs_base)
        score_with(recs, get_embedder(name))
        comp[name] = composite(recs, models)
        fac[name] = facets(recs, models)
        print(f"  scored under {short(name)}")

    # ---- sanity: the canonical encoder should reproduce composite.json
    canon = json.load(open(f"{SCORES}/composite.json"))["per_model"]
    ov, pt = comp[ENCODERS[0]]
    diffs = [abs(ov[m] - canon[m]["overall"]) for m in models if m in canon]
    rho_c = spearmanr([ov[m] for m in models], [canon[m]["overall"] for m in models]).statistic
    print(f"\nSANITY vs composite.json (canonical MiniLM): overall Spearman {rho_c:.4f}, max |diff| {max(diffs):.2f} points")

    # ---- 1. leaderboard agreement
    print("\nLEADERBOARD AGREEMENT (each encoder vs canonical MiniLM, and every pair)")
    agree = {}
    for i, a in enumerate(ENCODERS):
        for b in ENCODERS[i + 1:]:
            row = {}
            for scope in ["overall", "association", "analogy", "blending"]:
                va = comp[a][0] if scope == "overall" else {m: comp[a][1][m][scope] for m in models}
                vb = comp[b][0] if scope == "overall" else {m: comp[b][1][m][scope] for m in models}
                xa, xb = [va[m] for m in models], [vb[m] for m in models]
                ra, rb = _ranks(va, models), _ranks(vb, models)
                row[scope] = {"spearman": round(float(spearmanr(xa, xb).statistic), 3),
                              "kendall": round(float(kendalltau(xa, xb).statistic), 3),
                              "top5_overlap": len({m for m in models if ra[m] <= 5} & {m for m in models if rb[m] <= 5}),
                              "top10_overlap": len({m for m in models if ra[m] <= 10} & {m for m in models if rb[m] <= 10}),
                              "max_rank_move": max(abs(ra[m] - rb[m]) for m in models),
                              "mean_abs_diff_points": round(float(np.mean([abs(va[m] - vb[m]) for m in models])), 2)}
            agree[f"{short(a)} ~ {short(b)}"] = row
            for scope, s in row.items():
                print(f"  {short(a):22s} ~ {short(b):22s} {scope:12s} rho {s['spearman']:+.3f}  tau {s['kendall']:+.3f}  "
                      f"top5 {s['top5_overlap']}/5  top10 {s['top10_overlap']}/10  max move {s['max_rank_move']}  mean |dpts| {s['mean_abs_diff_points']}")
    print("\nTOP 10 OVERALL under each encoder")
    for name in ENCODERS:
        ov = comp[name][0]
        print(f"  {short(name):22s} " + ", ".join(m.split("_", 1)[1] for m in sorted(models, key=lambda m: -ov[m])[:10]))

    # ---- 2. facet stability across encoders
    keys = [(t, d) for t, dims in DIMS.items() for d in dims]
    A = ("association", "originality")
    cross = lambda f: [k for k in keys if k[0] != f[0]]                     # facets of the other two tasks
    base_cross = lambda f: [k for k in cross(f) if not k[1].startswith("em_")]  # the six base facets the paper's r-bar uses

    def corr(fx, fy):
        xs = np.array([fx[m] for m in models]); ys = np.array([fy[m] for m in models])
        ok = ~(np.isnan(xs) | np.isnan(ys))
        r, p = pearsonr(xs[ok], ys[ok])
        return float(r), float(p)

    facet_out = {}
    print("\nFACET STABILITY: association originality under X vs each facet under Y")
    for X in ENCODERS:
        for Y in ENCODERS:
            row = {}
            for k in keys:
                if k == A:
                    continue
                r, p = corr(fac[X][A], fac[Y][k])
                row[f"{k[0]}.{k[1]}"] = {"r": round(r, 3), "p": round(p, 4)}
            rbar = float(np.mean([row[f"{k[0]}.{k[1]}"]["r"] for k in base_cross(A)]))
            # is association originality still the facet with the highest mean cross-task r, scoring every facet under X against Y?
            means = {}
            for f in keys:
                if f[1] in ("em_utility", "em_integration", "utility"):
                    pass
                means[f"{f[0]}.{f[1]}"] = float(np.mean([corr(fac[X][f], fac[Y][k])[0] for k in base_cross(f)]))
            rank = sorted(means, key=lambda k: -means[k])
            facet_out[f"{short(X)} -> {short(Y)}"] = {"association_originality_row": row, "r_bar_base_cross_task": round(rbar, 3),
                                                      "facet_mean_cross_task_r": {k: round(v, 3) for k, v in means.items()},
                                                      "best_facet": rank[0], "association_originality_rank": rank.index("association.originality") + 1}
            sig = [f"{k}={v['r']:+.2f}" for k, v in row.items() if v["p"] < 0.05 and not k.startswith("association.")]
            print(f"  X={short(X):20s} Y={short(Y):20s} r-bar {rbar:+.3f}  best facet {rank[0]:24s} (assoc.orig rank {rank.index('association.originality') + 1})  "
                  f"sig cross-task: {', '.join(sig)}")

    OUT.write_text(json.dumps({"encoders": ENCODERS, "n_artifacts": len(recs_base), "models": models, "k": K,
                               "sanity_vs_composite_json": {"spearman": round(float(rho_c), 4), "max_abs_diff_points": round(max(diffs), 3)},
                               "leaderboard_agreement": agree,
                               "top10_overall": {short(n): [m for m in sorted(models, key=lambda m: -comp[n][0][m])[:10]] for n in ENCODERS},
                               "facet_stability": facet_out}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
