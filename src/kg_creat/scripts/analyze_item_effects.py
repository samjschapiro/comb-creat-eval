"""The anchor pair as the unit: what the item, rather than the model, decides.

Companion to analyze_facet_correlations.py, which holds the item fixed and varies the model. Here the
model is averaged out and the 30 anchor pairs vary. Three things:

  1. ANCHOR DISTANCE d(u,v) against each dimension's item mean. The headline is that distant anchors
     lower blend originality -- the opposite of the intuition that remote pairs breed novelty.
  2. WHY. Originality is pool-relative over an artifact's non-anchor ELEMENTS (score.py:84), so the
     candidate mechanisms are all properties of the item's element pool. Four are tested and three
     rejected; the surviving one is POLARISATION -- with distant anchors each element belongs clearly
     to one side, the pool splits into two lobes, and every element then has near neighbours inside
     its own lobe.
  3. DOMAINS. The pool is domain-tagged, so each dimension is also broken out by anchor domain.
     Descriptive only: 3-6 anchors per domain.

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_item_effects
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.kg_creat.embed import get_embedder

SCORES = "data/kg_creat/kombine_test30/scores"
RESP = "data/kg_creat/kombine_test30/responses"
OUT = Path("data/kg_creat/kombine_test30/analysis/item_effects.json")
MODE = {"association": "baseline", "analogy": "analogy", "blending": "blending"}
DIMS = ["utility", "surprise", "originality", "em_originality"]


def load():
    """Per (task, item): every model's dimension values, the domains, and the element pool."""
    vals = defaultdict(lambda: defaultdict(list))
    dom, elems = {}, defaultdict(set)
    for f in sorted(glob.glob(f"{SCORES}/*/path_scores.json")):
        for r in json.load(open(f)):
            task = next((t for t, m in MODE.items() if r.get("mode") == m), None)
            if task is None or not r.get("u_label"):
                continue
            if task == "analogy" and "pair_sat" not in r:
                continue
            if task != "analogy" and not r.get("triples"):
                continue
            key = (task, (r["u_label"], r["v_label"]))
            ok = r.get("pair_sat") is True if task == "analogy" else r.get("sat") is True
            vals[key]["utility"].append(float(ok))
            for dim, field in (("surprise", "R"), ("originality", "originality"),
                               ("em_originality", "em_originality")):
                if r.get(field) is not None:
                    vals[key][dim].append(float(r[field]))
            dom[key[1]] = (r.get("domain_u"), r.get("domain_v"))
    for f in sorted(glob.glob(f"{RESP}/*/responses.json")):
        for r in json.load(open(f)):
            if r.get("mode") != "blending" or not r.get("items"):
                continue
            item = (r["u_label"], r["v_label"])
            anchors = {item[0].lower(), item[1].lower()}
            for t in (r["items"][0].get("paths") or [[]])[0]:
                if len(t) >= 3:
                    elems[item].update(s for s in (t[0], t[2]) if str(s).lower() not in anchors)
    return vals, dom, elems


def main():
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    vec = lambda s: un(np.asarray(embed(s), float))
    vals, dom, elems = load()
    out = {"anchor_distance": {}, "mechanism": {}, "domains": {}}

    print("ANCHOR DISTANCE d(u,v) vs the item mean of each dimension")
    for task in MODE:
        items = sorted({i for (t, i) in vals if t == task})
        D = np.array([1 - float(vec(u) @ vec(v)) for u, v in items])
        for dim in DIMS:
            ys = [np.mean(vals[(task, i)][dim]) for i in items if vals[(task, i)][dim]]
            if len(ys) < len(items):
                continue
            y = np.array(ys)
            r, p = pearsonr(D, y)
            rho, prho = spearmanr(D, y)
            loo = [pearsonr(np.delete(D, i), np.delete(y, i))[1] for i in range(len(D))]
            out["anchor_distance"][f"{task}.{dim}"] = {
                "r": round(float(r), 3), "p": float(p), "spearman_rho": round(float(rho), 3),
                "spearman_p": float(prho), "loo_n_sig": int(sum(q < 0.05 for q in loo)),
                "loo_worst_p": float(max(loo)), "n_items": len(items),
                "item_mean_range": [round(float(y.min()), 3), round(float(y.max()), 3)]}
            if p < 0.05:
                print(f"  {task:12s} {dim:15s} r={r:+.2f} (p={p:.4f}) rho={rho:+.2f} | "
                      f"LOO {sum(q < 0.05 for q in loo)}/{len(D)} | range {y.min():.3f}-{y.max():.3f}")

    # ---- why: properties of the element pool originality is scored against ----
    items = sorted({i for (t, i) in vals if t == "blending"})
    D = np.array([1 - float(vec(u) @ vec(v)) for u, v in items])
    O = np.array([np.mean(vals[("blending", i)]["originality"]) for i in items])
    pol, hug, sparsity = [], [], []
    for it in items:
        a, b = vec(it[0]), vec(it[1])
        V = np.vstack([vec(s) for s in elems[it]])
        da, db = 1 - V @ a, 1 - V @ b
        pol.append(float(np.mean(np.abs(da - db))))       # how one-sided the elements are
        hug.append(float(np.mean(np.minimum(da, db))))    # how close they sit to an anchor at all
        S = V @ V.T
        np.fill_diagonal(S, -9)
        sparsity.append(float(np.mean(1 - np.sort(S, axis=1)[:, -5:].mean(axis=1))))
    pol, hug, sparsity = map(np.array, (pol, hug, sparsity))

    def partial(x, y, z):
        rx = x - np.polyval(np.polyfit(z, x, 1), z)
        ry = y - np.polyval(np.polyfit(z, y, 1), z)
        return pearsonr(rx, ry)

    print("\nMECHANISM (blending, n = 30 items)")
    for name, z in (("polarisation", pol), ("anchor proximity", hug), ("pool sparsity", sparsity)):
        r1, p1 = pearsonr(D, z)
        r2, p2 = pearsonr(z, O)
        r3, p3 = partial(D, O, z)
        out["mechanism"][name] = {"distance_to_mediator": [round(float(r1), 3), float(p1)],
                                  "mediator_to_originality": [round(float(r2), 3), float(p2)],
                                  "partial_distance_originality": [round(float(r3), 3), float(p3)]}
        print(f"  {name:17s} d(u,v)~m {r1:+.2f} (p={p1:.3f}) | m~originality {r2:+.2f} (p={p2:.3f}) | "
              f"partial r(d, orig | m) {r3:+.2f} (p={p3:.3f})")

    print("\nDOMAINS (blending item means; descriptive, 3-6 anchors each)")
    per = defaultdict(lambda: defaultdict(list))
    for i, it in enumerate(items):
        for d in dom[it]:
            per[d]["originality"].append(O[i])
            per[d]["surprise"].append(np.mean(vals[("blending", it)]["surprise"]))
            per[d]["utility"].append(np.mean(vals[("blending", it)]["utility"]))
            per[d]["distance"].append(D[i])
    print(f"  {'domain':14s} {'n':>3s} {'orig':>7s} {'surprise':>9s} {'utility':>8s} {'d(u,v)':>7s}")
    for d, v in sorted(per.items(), key=lambda kv: -np.mean(kv[1]["utility"])):
        if len(v["originality"]) < 3:
            continue
        out["domains"][str(d)] = {k: round(float(np.mean(x)), 3) for k, x in v.items()}
        out["domains"][str(d)]["n_anchors"] = len(v["originality"])
        print(f"  {str(d):14s} {len(v['originality']):3d} {np.mean(v['originality']):7.3f} "
              f"{np.mean(v['surprise']):9.3f} {np.mean(v['utility']):8.2f} {np.mean(v['distance']):7.2f}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
