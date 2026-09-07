"""Does the leaderboard depend on which sentence encoder we scored with?

Surprise and originality are the two embedding-derived dimensions, and both feed the composite, so a
ranking built on one encoder could in principle be an artifact of that encoder's geometry. This
recomputes BOTH dimensions under several encoders, rebuilds the per-task and overall composites from
each, and reports how much the ranking moves. Judge fields are never touched and the canonical
`path_scores.json` files are never written -- everything happens in memory from the responses.

Also re-tests the item-level result the encoder question was raised about: distant anchor pairs give
less original blends (r = -0.47 under MiniLM). If that flips or vanishes under another encoder it was
geometry, not behaviour.

    .venv_mlx/bin/python -m src.kg_creat.scripts.embedding_robustness
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
OUT = Path("data/kg_creat/kombine_test30/analysis/embedding_robustness.json")
ENCODERS = ["mlx-community/all-MiniLM-L6-v2-4bit",       # the one everything is scored with
            "mlx-community/bge-small-en-v1.5-4bit",
            "mlx-community/multilingual-e5-small-mlx"]
TASKS = {"baseline": "association", "analogy": "analogy", "blending": "blending"}


def artifacts():
    """One record per scored artifact, carrying what the two embedding dimensions need plus the
    utility flag already decided by the judges (which no encoder can change)."""
    sat = {}
    for f in sorted(glob.glob(f"{SCORES}/*/path_scores.json")):
        m = f.split("/")[-2]
        for r in json.load(open(f)):
            key = (m, r["mode"], r["prompt_id"], r.get("path_idx"))
            if r["mode"] == "analogy":
                if "pair_sat" in r:
                    sat[key] = r["pair_sat"] is True
            elif r.get("triples"):
                sat[key] = r.get("sat") is True
    out = []
    for f in sorted(glob.glob(f"{RESP}/*/responses.json")):
        m = f.split("/")[-2]
        for r in json.load(open(f)):
            mode = r.get("mode")
            if mode not in TASKS or not r.get("items"):
                continue
            item = (r.get("u_label"), r.get("v_label"))
            for pi, path in enumerate(r.get("paths") or []):
                key = (m, mode, r["prompt_id"], pi)
                if key not in sat:
                    continue
                p = EmittedPath(path)
                it = (r["items"] or [{}])[min(pi // (2 if mode == "analogy" else 1), len(r["items"]) - 1)]
                rec = {"model": m, "task": TASKS[mode], "item": item, "sat": sat[key],
                       "entities": list(p.entities)}
                if mode == "blending":
                    rec["g"] = (it.get("generic_space") or "").strip()
                    tri = [(it.get("paths") or [[]])[0]]
                elif mode == "analogy":
                    if pi % 2 or pi + 1 >= len(r["paths"]):
                        continue
                    rec["pair"] = (list(p.entities), list(EmittedPath(r["paths"][pi + 1]).entities))
                    tri = [path, r["paths"][pi + 1]]
                else:
                    tri = [path]
                # same element rule as score.py::_artifact_elements -- relations count too, and a
                # concept and a relation sharing a surface string are separate elements
                rec["elements"] = sorted({e for ts in tri for tp in ts if len(tp) == 3
                                          for e in ([("r", _norm(tp[1]))] +
                                                    [("c", _norm(x)) for x in (tp[0], tp[2])
                                                     if _norm(x) not in (_norm(item[0]), _norm(item[1]))])})
                out.append(rec)
    return out


def score_with(recs, embed, k=5):
    """Surprise and originality for every artifact under one encoder."""
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    cache = {}

    def V(s):
        """`s` is either a plain string (surprise) or an (kind, surface) element pair."""
        key = s if isinstance(s, str) else tuple(s)
        if key not in cache:
            cache[key] = un(np.asarray(embed(str(s if isinstance(s, str) else s[1])), float))
        return cache[key]

    for r in recs:                                          # surprise, per the task's definition
        if r["task"] == "blending":
            u, v = r["item"]
            r["R"] = float((scoring.cosine_distance(V(u), V(r["g"])) +
                            scoring.cosine_distance(V(v), V(r["g"]))) / 2) if r.get("g") else None
        elif r["task"] == "analogy":
            ea, eb = r["pair"]
            m = min(len(ea), len(eb))
            dd = [scoring.cosine_distance(V(ea[i]), V(eb[i])) for i in range(m)]
            r["R"] = (sum(dd) / len(dd)) if dd else None
        else:
            e = r["entities"]
            dd = [scoring.cosine_distance(V(e[i]), V(e[i + 1])) for i in range(len(e) - 1)]
            r["R"] = (sum(dd) / len(dd)) if dd else None

    pools = defaultdict(set)                                # originality: pool per (task, item)
    for r in recs:
        pools[(r["task"], r["item"])].update(tuple(e) for e in r["elements"])
    pool_vecs = {key: {s: V(s) for s in ss} for key, ss in pools.items()}
    for r in recs:
        pool = pool_vecs[(r["task"], r["item"])]
        rhos = []
        for s in {tuple(e) for e in r["elements"]}:
            d = sorted(scoring.cosine_distance(pool[s], o) for t, o in pool.items() if t != s)
            if d:
                rhos.append(sum(d[:min(k, len(d))]) / min(k, len(d)))
        r["originality"] = (sum(rhos) / len(rhos)) if rhos else None
    return recs


def composite(recs, models):
    """Utility-gated, equal-weight z-composite -- the same shape compute_composite.py builds."""
    dims = defaultdict(dict)
    for task in ("association", "analogy", "blending"):
        for m in models:
            rs = [r for r in recs if r["model"] == m and r["task"] == task]
            if not rs:
                continue
            dims[(task, "utility")][m] = float(np.mean([r["sat"] for r in rs]))
            for dim in ("R", "originality"):
                vals = [(min(1.0, max(0.0, r[dim])) if r["sat"] else 0.0)
                        for r in rs if r.get(dim) is not None]
                dims[(task, dim)][m] = float(np.mean(vals)) if vals else np.nan
    z = {}
    for key, per_model in dims.items():
        x = np.array([per_model.get(m, np.nan) for m in models], float)
        z[key] = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    per_task = {t: np.nanmean([z[(t, d)] for d in ("utility", "R", "originality")], axis=0)
                for t in ("association", "analogy", "blending")}
    overall = np.nanmean(list(per_task.values()), axis=0)
    return overall, per_task


def main():
    recs_base = artifacts()
    models = sorted({r["model"] for r in recs_base})
    print(f"{len(recs_base)} artifacts, {len(models)} models, {len(ENCODERS)} encoders\n")

    rankings, per_task_all, item_effect = {}, {}, {}
    for name in ENCODERS:
        recs = deepcopy(recs_base)                          # fresh copy per encoder
        score_with(recs, get_embedder(name))
        overall, per_task = composite(recs, models)
        rankings[name] = overall
        per_task_all[name] = per_task

        embed = get_embedder(name)
        un = lambda x: x / (np.linalg.norm(x) + 1e-9)
        bl = [r for r in recs if r["task"] == "blending" and r.get("originality") is not None]
        items = sorted({r["item"] for r in bl})
        D = np.array([1 - float(un(np.asarray(embed(u), float)) @ un(np.asarray(embed(v), float)))
                      for u, v in items])
        O = np.array([np.mean([r["originality"] for r in bl if r["item"] == it]) for it in items])
        r, p = pearsonr(D, O)
        item_effect[name] = {"r": round(float(r), 3), "p": float(p), "n_items": len(items)}
        print(f"  {name.split('/')[-1]:32s} distance~blend originality  r = {r:+.2f} (p = {p:.4f})")

    print("\nVARIANCE SHARE (item vs model) per encoder -- does the decomposition survive?")
    var_share = defaultdict(dict)
    for name in ENCODERS:
        recs = deepcopy(recs_base)
        score_with(recs, get_embedder(name))
        for task in ("association", "analogy", "blending"):
            for dim in ("R", "originality"):
                cell = defaultdict(dict)
                for r in recs:
                    if r["task"] == task and r.get(dim) is not None:
                        cell[r["model"]].setdefault(r["item"], []).append(r[dim])
                items = sorted({i for m in cell for i in cell[m]})
                M = np.array([[np.mean(cell[m][i]) if i in cell[m] else np.nan for i in items]
                              for m in models], float)
                M = M[:, ~np.isnan(M).any(axis=0)]
                v_i, v_m = M.mean(axis=0).var(ddof=1), M.mean(axis=1).var(ddof=1)
                v_r = (M - M.mean(axis=0)[None, :] - M.mean(axis=1)[:, None] + M.mean()).var(ddof=1)
                tot = v_i + v_m + v_r
                var_share[f"{task}.{'surprise' if dim == 'R' else dim}"][name.split("/")[-1]] = \
                    {"item_pct": round(100*v_i/tot, 1), "model_pct": round(100*v_m/tot, 1)}
    print(f"  {'task.dimension':28s} " + "  ".join(f"{n.split('/')[-1][:14]:>16s}" for n in ENCODERS))
    for k, per in var_share.items():
        cells = "  ".join(f"{per[n.split('/')[-1]]['item_pct']:5.0f}/{per[n.split('/')[-1]]['model_pct']:<10.0f}"
                          for n in ENCODERS)
        print(f"  {k:28s} {cells}   (item%/model%)")

    print("\nRANK AGREEMENT on the overall composite")
    pairs = {}
    for i, a in enumerate(ENCODERS):
        for b in ENCODERS[i + 1:]:
            rho = spearmanr(rankings[a], rankings[b]).statistic
            tau = kendalltau(rankings[a], rankings[b]).statistic
            top5 = len(set(np.argsort(-rankings[a])[:5]) & set(np.argsort(-rankings[b])[:5]))
            pairs[f"{a.split('/')[-1]} ~ {b.split('/')[-1]}"] = {
                "spearman": round(float(rho), 3), "kendall": round(float(tau), 3), "top5_overlap": top5}
            print(f"  {a.split('/')[-1][:26]:26s} ~ {b.split('/')[-1][:26]:26s} rho = {rho:+.3f}  "
                  f"tau = {tau:+.3f}  top-5 overlap {top5}/5")

    mean_rank = np.mean([np.argsort(np.argsort(-rankings[e])) for e in ENCODERS], axis=0)
    spread = np.ptp([np.argsort(np.argsort(-rankings[e])) for e in ENCODERS], axis=0)
    print("\nPER-MODEL RANK (mean over encoders, and the worst disagreement)")
    for i in np.argsort(mean_rank):
        print(f"  {models[i]:36s} mean rank {mean_rank[i]+1:5.1f}   spread {int(spread[i])}")

    print("\nPER-TASK rank agreement")
    task_pairs = {}
    for t in ("association", "analogy", "blending"):
        rhos = [spearmanr(per_task_all[a][t], per_task_all[b][t]).statistic
                for i, a in enumerate(ENCODERS) for b in ENCODERS[i + 1:]]
        task_pairs[t] = round(float(np.mean(rhos)), 3)
        print(f"  {t:12s} mean rho across encoder pairs = {np.mean(rhos):+.3f}")

    OUT.write_text(json.dumps({"encoders": ENCODERS, "n_artifacts": len(recs_base),
                               "rank_agreement": pairs, "per_task_mean_rho": task_pairs,
                               "distance_vs_blend_originality": item_effect,
                               "variance_share_per_encoder": var_share,
                               "models": models, "mean_rank": mean_rank.tolist(),
                               "rank_spread": spread.tolist()}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
