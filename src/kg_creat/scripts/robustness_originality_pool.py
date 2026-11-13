"""Rebuttal 3a: how much does the leaderboard depend on WHO is in the originality reference pool?

Originality is pool-relative: an element's rho is its mean cosine distance to its k = 5 nearest neighbours
among the distinct elements everyone produced for the same item. The paper's pool (rescore_originality_combined.py
+ build_paper_scores.py) is every model + every human (kept and flagged) per item, with, for analogy / blending,
separate base and emergent pools. This script re-runs exactly that computation in memory under different pools,
overwrites only the fields the shared-pool rescorer writes (association path `originality`; analogy / blending
path-0 head `originality` + `em_originality`), and recomputes the paper composite with compute_composite's own
`artifact_dims` (utility gate, fraction-of-max dimensions, equal-weight task means, overall = mean of tasks).

Pools:
  full            the paper's pool -- a reproduction check against the stored paper_scores values
  lomo:<model>    leave one model out of the pool (its answers are not scored either; 34 models ranked)
  lopo:<provider> leave a whole provider out (e.g. all 10 Anthropic models)
  split<i>:A>B    a FIXED external pool: models in half A are the reference, models in half B are scored against
                  it (and vice versa); 10 random halvings. The scored model's own answers are NOT in its pool,
                  so an exact duplicate in the pool counts as distance 0 (in-pool variants keep the paper's rule
                  of skipping the identical surface).
  humans          a fixed external pool of human answers only (kept + flagged); worked-example items, which no
                  human saw, get no originality and drop out of the originality means.

Humans stay in every in-pool variant's pool, as in the paper. Rankings are over the 35 models; the human row is
not ranked here.

    .venv_mlx/bin/python -m src.kg_creat.scripts.robustness_originality_pool
Writes data/kg_creat/kombine_test30/analysis/originality_pool_robustness.json.
"""
import copy
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.compute_composite import TASK_DIMS, _mean, _ok, artifact_dims
from src.kg_creat.scripts.rescore_split_originality import orig_elements
from src.kg_creat.scripts.score import _artifact_elements, _draw_key

PAPER = Path("data/kg_creat/human_study/paper_scores")
LLM_RESP = Path("data/kg_creat/kombine_test30/responses")
HUMAN_RESP = Path("data/kg_creat/human_study/score_input")
OUT = Path("data/kg_creat/kombine_test30/analysis/originality_pool_robustness.json")
HUMANS = ["human_kept", "human_flagged"]
K = 5
ORIG_DIMS = [("association", "originality"), ("analogy", "originality"), ("analogy", "em_originality"),
             ("blending", "originality"), ("blending", "em_originality")]


def provider(m):
    return m.split("_")[0]


def load():
    models = sorted(d.name for d in PAPER.iterdir() if (d / "path_scores.json").exists() and not d.name.startswith("human"))
    ents = models + HUMANS
    recs = {e: json.loads((PAPER / e / "path_scores.json").read_text()) for e in ents}
    resp = {e: json.loads(((HUMAN_RESP if e.startswith("human") else LLM_RESP) / e / "responses.json").read_text())
            for e in ents}
    return models, ents, recs, resp


def collect(ents, resp):
    """Per item: surface -> contributing entities (association / base / emergent pools), and the targets to score."""
    pool_a = defaultdict(lambda: defaultdict(set))       # pid -> surface -> {entity}
    pool_b = defaultdict(lambda: defaultdict(set))
    pool_e = defaultdict(lambda: defaultdict(set))
    tgt_a = defaultdict(list)                            # pid -> [(entity, head_key, surfaces)]
    tgt_s = defaultdict(list)                            # pid -> [(entity, draw_key, base, emergent)]
    for e in ents:
        for r in resp[e]:
            pid, u, v = r["prompt_id"], r.get("u_label"), r.get("v_label")
            paths = r.get("paths") or []
            if any(paths):                               # score.build_item_element_pool
                for _k, s in _artifact_elements(paths, u, v):
                    pool_a[pid][s].add(e)
            if r.get("mode") == "baseline":              # score.score_originality (association heads)
                for idx, it in enumerate(r.get("items") or []):
                    tgt_a[pid].append((e, _draw_key(r) + (idx,), {s for _k, s in _artifact_elements(it["paths"], u, v)}))
            elif r.get("mode") in ("analogy", "blending") and r.get("items"):   # split base / emergent
                b, em = orig_elements(r["items"][0], r["mode"], u, v)
                tgt_s[pid].append((e, _draw_key(r), b, em))
                for s in b:
                    if s:
                        pool_b[pid][s].add(e)
                for s in em:
                    if s:
                        pool_e[pid][s].add(e)
    return pool_a, pool_b, pool_e, tgt_a, tgt_s


class ItemPool:
    """All surfaces of one item's pool with a precomputed cosine-distance matrix; scores surfaces under a pool mask."""

    def __init__(self, contrib, embed):
        self.surf = sorted(contrib)
        self.idx = {s: i for i, s in enumerate(self.surf)}
        self.contrib = [contrib[s] for s in self.surf]
        if not self.surf:
            self.D = np.zeros((0, 0), np.float32)
            return
        U = np.stack([embed(s) for s in self.surf]).astype(np.float32)
        U /= np.linalg.norm(U, axis=1, keepdims=True) + 1e-9
        self.D = 1.0 - U @ U.T

    def rho(self, members, exclude_identical):
        """rho for every surface, given the pool = surfaces with a contributor in `members`: the mean of its (up to)
        K smallest distances to pool surfaces -- skipping its own identical surface when `exclude_identical`
        (the paper's in-pool rule), counting an identical pool surface as distance 0 otherwise."""
        if not self.surf:
            return {}
        mask = np.array([bool(c & members) for c in self.contrib])
        if not mask.any():
            return {}
        D = np.where(mask[None, :], self.D, np.inf)
        if exclude_identical:
            np.fill_diagonal(D, np.inf)
        kk = min(K, D.shape[1])
        part = np.partition(D, kk - 1, axis=1)[:, :kk]
        out = {}
        for i, s in enumerate(self.surf):
            row = part[i][np.isfinite(part[i])]
            if len(row):
                out[s] = float(row.mean())
        return out


def score_variant(pools, tgts, members, scored, exclude_identical):
    """Return {entity: {key: (field, value)}} overwrites for entities in `scored`."""
    pa, pb, pe = pools
    ta, ts = tgts
    upd = defaultdict(dict)
    for pid, ip in pa.items():
        rh = ip.rho(members, exclude_identical)
        for e, hk, surfs in ta.get(pid, []):
            if e not in scored:
                continue
            vals = [rh[s] for s in surfs if s in rh]
            upd[e][("a",) + hk] = (sum(vals) / len(vals)) if vals else None
    for pid in set(pb) | set(pe):
        rb = pb[pid].rho(members, exclude_identical) if pid in pb else {}
        re_ = pe[pid].rho(members, exclude_identical) if pid in pe else {}
        for e, dk, b, em in ts.get(pid, []):
            if e not in scored:
                continue
            def m(sset, rh):
                v = [rh[s] for s in sset if s in rh]
                return (sum(v) / len(v)) if v else None
            upd[e][("s",) + dk] = (m(b, rb), m(em, re_))
    return upd


def apply(recs, upd):
    """Copy of `recs` with originality fields overwritten exactly where the shared-pool rescorer writes them."""
    out = copy.deepcopy(recs)
    for r in out:
        dk = _draw_key(r)
        if r.get("mode") == "baseline" and ("a",) + dk + (r["path_idx"],) in upd:
            r["originality"] = upd[("a",) + dk + (r["path_idx"],)]
        elif r.get("path_idx") == 0 and ((r.get("mode") == "analogy" and "pair_sat" in r) or
                                         (r.get("mode") == "blending" and r.get("triples"))):
            if ("s",) + dk in upd:
                r["originality"], r["em_originality"] = upd[("s",) + dk]
    return out


def composite(recs):
    raw = artifact_dims(recs)
    pt = {t: 100.0 * _mean([raw[t][k] for k in ds if _ok(raw[t].get(k))]) for t, ds in TASK_DIMS.items()}
    return {"overall": _mean(list(pt.values())), "per_task": pt,
            "orig": {f"{t}.{k}": raw[t].get(k) for t, k in ORIG_DIMS}}


def rank_agreement(base, new, ms):
    """Spearman / Kendall of overall + of each originality dimension, and the largest rank shift, over models `ms`."""
    b = [base[m]["overall"] for m in ms]
    n = [new[m]["overall"] for m in ms]
    rb = {m: i for i, m in enumerate(sorted(ms, key=lambda m: -base[m]["overall"]))}
    rn = {m: i for i, m in enumerate(sorted(ms, key=lambda m: -new[m]["overall"]))}
    shift = {m: rn[m] - rb[m] for m in ms}
    worst = max(ms, key=lambda m: abs(shift[m]))
    top8 = len(set(sorted(ms, key=lambda m: -base[m]["overall"])[:8]) & set(sorted(ms, key=lambda m: -new[m]["overall"])[:8]))
    dims = {}
    for d in base[ms[0]]["orig"]:
        x = [base[m]["orig"][d] for m in ms]
        y = [new[m]["orig"][d] for m in ms]
        ok = [(a, c) for a, c in zip(x, y) if _ok(a) and _ok(c)]
        dims[d] = float(spearmanr(*zip(*ok))[0]) if len(ok) > 3 else None
    return {"spearman": float(spearmanr(b, n)[0]), "kendall": float(kendalltau(b, n)[0]),
            "max_rank_shift": int(abs(shift[worst])), "max_shift_model": worst, "top8_overlap": top8,
            "n_models": len(ms), "orig_dim_spearman": dims,
            "max_abs_overall_change": float(max(abs(a - c) for a, c in zip(b, n)))}


def main():
    models, ents, recs, resp = load()
    embed = get_embedder()
    pa, pb, pe, ta, ts = collect(ents, resp)
    print(f"{len(models)} models + {len(HUMANS)} human groups; embedding "
          f"{len({s for d in (pa, pb, pe) for p in d.values() for s in p})} distinct surfaces ...")
    pools = ({p: ItemPool(c, embed) for p, c in pa.items()}, {p: ItemPool(c, embed) for p, c in pb.items()},
             {p: ItemPool(c, embed) for p, c in pe.items()})
    tgts = (ta, ts)
    everyone = set(ents)

    # ---- reproduction check: the paper's pool must give back the stored values
    upd = score_variant(pools, tgts, everyone, set(models) | {"human_kept"}, True)
    diffs, n = [], 0
    full = {}
    for e in models + ["human_kept"]:
        new = apply(recs[e], upd[e])
        for a, b in zip(recs[e], new):
            for f in ("originality", "em_originality"):
                if _ok(a.get(f)) and _ok(b.get(f)):
                    diffs.append(abs(a[f] - b[f])); n += 1
                elif _ok(a.get(f)) != _ok(b.get(f)):
                    diffs.append(float("inf")); n += 1
        full[e] = composite(new)
    stored = json.loads((PAPER / "composite.json").read_text())["per_model"]
    comp_diff = max(abs(full[m]["overall"] - stored[m]["overall"]) for m in models)
    repro = {"n_values": n, "max_abs_diff": float(max(diffs)), "share_exact_1e-6": float(np.mean([d < 1e-6 for d in diffs])),
             "max_abs_overall_diff_vs_stored_composite": float(comp_diff)}
    print(f"reproduction: {n} values, max |diff| {repro['max_abs_diff']:.2e}, "
          f"{100 * repro['share_exact_1e-6']:.2f}% within 1e-6; max overall diff vs stored {comp_diff:.2e}")
    base = full

    res = {"reproduction": repro, "lomo": {}, "lopo": {}, "split": [], "humans_pool": None}

    # ---- leave one model out of the pool
    for m in models:
        keep = everyone - {m}
        ms = [x for x in models if x != m]
        u = score_variant(pools, tgts, keep, set(ms), True)
        new = {x: composite(apply(recs[x], u[x])) for x in ms}
        res["lomo"][m] = rank_agreement(base, new, ms)
    # ---- leave one provider out
    for p in sorted({provider(m) for m in models}):
        drop = {m for m in models if provider(m) == p}
        ms = [x for x in models if x not in drop]
        u = score_variant(pools, tgts, everyone - drop, set(ms), True)
        new = {x: composite(apply(recs[x], u[x])) for x in ms}
        res["lopo"][p] = {**rank_agreement(base, new, ms), "n_dropped": len(drop)}
        print(f"  lopo {p:12s} drop {len(drop):2d}: spearman {res['lopo'][p]['spearman']:.3f}")

    # ---- fixed external pools: random halves of the models
    rng = random.Random(0)
    for i in range(10):
        sh = models[:]
        rng.shuffle(sh)
        A, B = set(sh[: len(sh) // 2]), set(sh[len(sh) // 2:])
        for ref, scored in ((A, B), (B, A)):
            ms = sorted(scored)
            u = score_variant(pools, tgts, ref, scored, False)
            new = {x: composite(apply(recs[x], u[x])) for x in ms}
            res["split"].append({"split": i, **rank_agreement(base, new, ms)})
    # ---- fixed external pool: human answers only
    u = score_variant(pools, tgts, set(HUMANS), set(models), False)
    new = {x: composite(apply(recs[x], u[x])) for x in models}
    res["humans_pool"] = rank_agreement(base, new, models)

    # ---- summaries
    def summ(rows):
        rows = list(rows)
        dims = {d: [r["orig_dim_spearman"][d] for r in rows if r["orig_dim_spearman"][d] is not None]
                for d in rows[0]["orig_dim_spearman"]}
        return {"n": len(rows), "spearman_min": min(r["spearman"] for r in rows),
                "spearman_median": float(np.median([r["spearman"] for r in rows])),
                "kendall_min": min(r["kendall"] for r in rows),
                "max_rank_shift": max(r["max_rank_shift"] for r in rows),
                "top8_overlap_min": min(r["top8_overlap"] for r in rows),
                "max_abs_overall_change": max(r["max_abs_overall_change"] for r in rows),
                "orig_dim_spearman_min": {d: (min(v) if v else None) for d, v in dims.items()},
                "orig_dim_spearman_median": {d: (float(np.median(v)) if v else None) for d, v in dims.items()}}
    res["summary"] = {"lomo": summ(res["lomo"].values()), "lopo": summ(res["lopo"].values()),
                      "split": summ(res["split"]), "humans_pool": summ([res["humans_pool"]])}
    OUT.write_text(json.dumps(res, indent=1))
    for k, s in res["summary"].items():
        print(f"{k:12s} n={s['n']:3d} spearman min {s['spearman_min']:.3f} median {s['spearman_median']:.3f} "
              f"kendall min {s['kendall_min']:.3f} max shift {s['max_rank_shift']} top8 min {s['top8_overlap_min']} "
              f"max |d overall| {s['max_abs_overall_change']:.2f}")
        print("             orig-dim spearman min: " + ", ".join(f"{d} {v:.3f}" for d, v in s["orig_dim_spearman_min"].items() if v is not None))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
