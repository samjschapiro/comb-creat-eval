"""Cross-task correlations: what the Kombine scoring dimensions measure, and whether any of them is a
model-level trait that carries from one task to the next.

Everything is a Pearson correlation ACROSS THE MODEL POOL (n=21, one point per model).

The dimensions are read from `path_scores.json` as UNGATED means -- averaged over every artifact the
model produced, pass or fail -- rather than the utility-gated values the composite scores with, and
rather than means conditioned on passing. Gating multiplies every dimension by one 0/1 mask, so gated
dimensions correlate at r > 0.99 and the matrix says only "utility correlates with utility";
conditioning instead averages each model over its own passing subset, so models are described on
different samples. The ungated mean holds the denominator fixed and lets utility be what it is: a
separate dimension, the rate at which the model got it right.

Three questions:

  1. Do the three TASKS measure the same thing? (correlation between per-task composites)
  2. Do the DIMENSIONS cluster by dimension or by task? (the full facet matrix)
  3. Is originality a trait that transfers? Base (the scored artifact) and emergent (the invention)
     originality are scored separately for analogy and blending, giving a 2x2 of cross-task cells --
     and the answer is checked against overall leaderboard rank, since a "trait" that is just
     capability is not a trait.

A pool this size gives a large critical |r| for p < 0.05: individual coefficients are noisy and the structure carries the
claims, so every reported cell also gets its p-value in the JSON.

    .venv/bin/python -m src.kg_creat.scripts.analyze_facet_correlations
"""
import glob
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist, pearsonr, spearmanr

SCORES = Path("data/kg_creat/kombine_test30/scores/composite.json")
SCORE_DIR = Path("data/kg_creat/kombine_test30/scores")
OUT = Path("data/kg_creat/kombine_test30/analysis/facet_correlations.json")
FIG = Path("docs/reports/2026-09-02_kg_creat_facet_correlations/figures/fig_facet_corr.png")
FIG_SMALL = FIG.with_name("fig_facet_corr_reduced.png")   # the three base dimensions only, for the paper
MODE = {"association": "baseline", "analogy": "analogy", "blending": "blending"}
TASKS = ["association", "analogy", "blending"]
# Blend surprise is S_bl = mean distance from each input to the blend's GENERIC SPACE g, which the
# model writes -- so it varies by model (21 distinct values per item, within-item SD 0.05) and belongs
# here. An older scorer set it to d_cos(u, v), fixed per item; comments to that effect are stale.
DIMS = {"association": ["utility", "surprise", "originality"],
        "analogy": ["utility", "surprise", "originality", "em_originality", "em_utility", "em_integration"],
        "blending": ["utility", "surprise", "originality", "em_originality", "em_utility", "em_integration"]}
LABEL = {"utility": "utility", "surprise": "surprise", "originality": "originality (base)",
         "em_originality": "originality (emergent)", "em_utility": "emergent utility",
         "em_integration": "integration quality"}


def ungated_dims(model: str) -> dict:
    """Per-task dimensions for one model, averaged over ALL of its artifacts.

    Neither gated nor conditional. Gating multiplies every dimension by the same 0/1 utility mask, so
    gated dimensions correlate at r > 0.99 by construction. Conditioning on the artifacts that passed
    instead averages each model over a different, self-selected subsample -- a model that passes 40% of
    the time is being described by its own best 40%. The ungated mean keeps one denominator for every
    model: how remote, original and well-integrated this model's output is, whether or not it was
    valid. Utility stays the pass rate, which is the thing utility actually measures.
    """
    recs = json.loads((SCORE_DIR / model / "path_scores.json").read_text())
    out = {}
    for task, mode in MODE.items():
        rs = [r for r in recs if r.get("mode") == mode]
        if mode == "analogy":
            arts = [r for r in rs if "pair_sat" in r]
            ok = [r for r in arts if r.get("pair_sat") is True]
        else:
            arts = [r for r in rs if r.get("triples")]
            ok = [r for r in arts if r.get("sat") is True]

        def dim(key, f=lambda v: float(v)):         # over every artifact, pass or fail
            vals = [f(r[key]) for r in arts if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        d = {"utility": (len(ok) / len(arts)) if arts else float("nan"),
             "surprise": dim("R"), "originality": dim("originality")}
        if task in ("analogy", "blending"):
            d["em_originality"] = dim("em_originality")
        if task == "analogy":
            d["em_utility"] = dim("invention_utility", lambda v: float(bool(v)))
            d["em_integration"] = dim("invention_integration", lambda v: float(bool(v)))
        elif task == "blending":
            d["em_utility"] = dim("blend_utility", lambda v: float(bool(v)))
            d["em_integration"] = dim("blend_integration", lambda v: (float(v) - 1) / 2)
        out[task] = d
    return out


def main():
    c = json.loads(SCORES.read_text())
    models = c["models"]
    pm = c["per_model"]
    n = len(models)

    cond = {m: ungated_dims(m) for m in models}
    task_v = {t: np.array([pm[m]["per_task"][t] for m in models]) for t in TASKS}
    overall = np.array([pm[m]["overall"] for m in models])
    facet_v, facets = {}, []
    for t in TASKS:
        for d in DIMS[t]:
            key = (t, d)
            facet_v[key] = np.array([cond[m][t][d] for m in models])
            facets.append(key)
    bad = [f for f in facets if not np.all(np.isfinite(facet_v[f]))]
    if bad:
        raise ValueError(f"FATAL: undefined dimension(s) {bad} -- a model produced no artifact carrying "
                         f"that value; decide explicitly how to treat it rather than averaging nothing")

    def rp(x, y):
        r, p = pearsonr(x, y)
        return {"r": round(float(r), 3), "p": float(p)}

    # the p < 0.05 threshold depends on n, so derive it rather than hardcoding a pool size
    from scipy.stats import t as _t
    _tc = _t.ppf(0.975, n - 2)
    _rc = float(_tc / np.sqrt(_tc ** 2 + n - 2))
    out = {"n_models": n, "models": models,
           "significance_note": f"|r| >= {_rc:.2f} is p < 0.05 at n = {n}",
           "r_crit_05": round(_rc, 3)}

    print(f"n = {n} models\n\nTASK-LEVEL (per-task composite)")
    out["task_level"] = {}
    for a, b in combinations(TASKS, 2):
        s = rp(task_v[a], task_v[b])
        out["task_level"][f"{a}~{b}"] = s
        print(f"  {a:12s} ~ {b:12s} r = {s['r']:+.2f}  (p = {s['p']:.3f})")

    print("\nFACET MATRIX (每 pair; printed where |r| >= 0.43)".replace("每", "every"))
    out["facet_matrix"] = {}
    for fa, fb in combinations(facets, 2):
        s = rp(facet_v[fa], facet_v[fb])
        out["facet_matrix"][f"{fa[0]}.{fa[1]}~{fb[0]}.{fb[1]}"] = s
        if abs(s["r"]) >= 0.43:
            print(f"  {fa[0][:5]}.{fa[1]:15s} ~ {fb[0][:5]}.{fb[1]:15s} r = {s['r']:+.2f} (p = {s['p']:.3f})")

    # within-task utility vs the two novelty dimensions: the tradeoff claim
    print("\nUTILITY vs NOVELTY, within task")
    out["utility_vs_novelty"] = {}
    for t in TASKS:
        for d in [x for x in DIMS[t] if x in ("surprise", "originality", "em_originality")]:
            s = rp(facet_v[(t, "utility")], facet_v[(t, d)])
            out["utility_vs_novelty"][f"{t}.{d}"] = s
            print(f"  {t:12s} utility ~ {d:15s} r = {s['r']:+.2f} (p = {s['p']:.3f})")

    # does originality transfer across tasks, and is it just capability?
    print("\nORIGINALITY ACROSS TASKS (analogy x blending)")
    out["originality_transfer"] = {}
    for a in ("originality", "em_originality"):
        for b in ("originality", "em_originality"):
            s = rp(facet_v[("analogy", a)], facet_v[("blending", b)])
            out["originality_transfer"][f"analogy.{a}~blending.{b}"] = s
            print(f"  analogy {a:15s} ~ blending {b:15s} r = {s['r']:+.2f} (p = {s['p']:.3f})")
    for t in ("analogy", "blending"):
        s = rp(facet_v[(t, "originality")], facet_v[(t, "em_originality")])
        out["originality_transfer"][f"{t}.base~{t}.emergent"] = s
        print(f"  within {t:11s} base ~ emergent          r = {s['r']:+.2f} (p = {s['p']:.3f})")

    em_mean = np.mean([facet_v[("analogy", "em_originality")], facet_v[("blending", "em_originality")]], axis=0)
    s = rp(em_mean, overall)
    rho, prho = spearmanr(em_mean, overall)
    out["emergent_originality_vs_capability"] = {**s, "spearman_rho": round(float(rho), 3),
                                                 "spearman_p": float(prho)}
    order = np.argsort(-em_mean)
    out["most_emergent_original"] = [models[i] for i in order[:5]]
    out["their_overall_ranks"] = [int(np.where(np.argsort(-overall) == i)[0][0]) + 1 for i in order[:5]]
    print(f"\nMEAN EMERGENT ORIGINALITY ~ OVERALL COMPOSITE  r = {s['r']:+.2f} (p = {s['p']:.3f}), "
          f"rho = {rho:+.2f}")
    print("  most emergent-original models: " +
          ", ".join(f"{models[i]} (rank {out['their_overall_ranks'][k]})" for k, i in enumerate(order[:5])))

    out["associative_hypothesis"] = associative_hypothesis(pm, models, facet_v)

    from src.kg_creat.embed import get_embedder
    out["per_item"] = per_item(cell_means(), get_embedder("mlx-community/all-MiniLM-L6-v2-4bit"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")
    plot(facets, facet_v, FIG)
    plot(facets, facet_v, FIG_SMALL, compact=True,
         dims={t: [d for d in ("utility", "surprise", "originality") if d in DIMS[t]] for t in TASKS})




# ---------------------------------------------------------------- the item as the unit
FIELD = {"surprise": ("R", float), "originality": ("originality", float),
         "em_originality": ("em_originality", float)}


def cell_means():
    """cells[(task, dim)][model][item] -- one value per model per anchor pair, so the same numbers can
    be read down the models (who is more original?) or across the items (which pairs elicit more?)."""
    cells = defaultdict(lambda: defaultdict(dict))
    for f in sorted(glob.glob(f"{SCORE_DIR}/*/path_scores.json")):
        m = f.split("/")[-2]
        recs = json.load(open(f))
        for task, mode in MODE.items():
            byitem = defaultdict(list)
            for r in recs:
                if r.get("mode") != mode:
                    continue
                if mode == "analogy":
                    if "pair_sat" not in r:
                        continue
                    ok = r.get("pair_sat") is True
                else:
                    if not r.get("triples"):
                        continue
                    ok = r.get("sat") is True
                byitem[(r["u_label"], r["v_label"])].append((r, ok))
            for item, rs in byitem.items():
                cells[(task, "utility")][m][item] = float(np.mean([o for _, o in rs]))
                for dim in DIMS[task]:
                    if dim == "utility":
                        continue
                    if dim in FIELD:
                        key, f_ = FIELD[dim]
                    elif dim == "em_utility":
                        key, f_ = ("invention_utility" if task == "analogy" else "blend_utility",
                                   lambda v: float(bool(v)))
                    else:
                        key = "invention_integration" if task == "analogy" else "blend_integration"
                        f_ = (lambda v: float(bool(v))) if task == "analogy" else (lambda v: (float(v) - 1) / 2)
                    vals = [f_(r[key]) for r, _ in rs if r.get(key) is not None]
                    if vals:
                        cells[(task, dim)][m][item] = float(np.mean(vals))
    return cells


def item_means(cells, task, dim, min_models=15):
    """Item means over the models that answered that item (an item needs `min_models` to count)."""
    d = cells[(task, dim)]
    items = sorted({i for m in d for i in d[m] if sum(i in d[m2] for m2 in d) >= min_models})
    return items, np.array([np.mean([d[m][i] for m in d if i in d[m]]) for i in items])


def per_item(cells, embed):
    """Three questions the model-level matrix cannot answer: how much of each dimension is the anchor
    pair rather than the model, whether a pair that is generous to one task is generous to the next,
    and whether the within-task relationships hold when items rather than models vary."""
    out = {"variance_share": {}, "analogy_vs_blending": {}, "utility_vs_novelty": {},
           "anchor_distance": {}}

    print("\n\nTHE ITEM AS THE UNIT\n\nVARIANCE SHARE (of each dimension, two-way)")
    print(f"  {'task.dimension':32s} {'item':>6s} {'model':>7s} {'resid':>7s}")
    for task in TASKS:
        for dim in DIMS[task]:
            d = cells[(task, dim)]
            models = sorted(d)
            items = sorted({i for m in d for i in d[m]})
            M = np.array([[d[m].get(i, np.nan) for i in items] for m in models], float)
            M = M[:, ~np.isnan(M).any(axis=0)]              # items every model answered
            v_i = M.mean(axis=0).var(ddof=1)
            v_m = M.mean(axis=1).var(ddof=1)
            v_r = (M - M.mean(axis=0)[None, :] - M.mean(axis=1)[:, None] + M.mean()).var(ddof=1)
            tot = v_i + v_m + v_r
            out["variance_share"][f"{task}.{dim}"] = {"item_pct": 100*v_i/tot, "model_pct": 100*v_m/tot,
                                                      "residual_pct": 100*v_r/tot, "n_items": M.shape[1]}
            print(f"  {task + '.' + dim:32s} {100*v_i/tot:5.1f}% {100*v_m/tot:6.1f}% {100*v_r/tot:6.1f}%")

    print("\nSAME 30 ANCHOR PAIRS: analogy vs blending, item means")
    for dim in DIMS["blending"]:
        ia, xa = item_means(cells, "analogy", dim)
        ib, xb = item_means(cells, "blending", dim)
        common = sorted(set(ia) & set(ib))
        x = np.array([xa[ia.index(i)] for i in common]); y = np.array([xb[ib.index(i)] for i in common])
        r, p_ = pearsonr(x, y)
        out["analogy_vs_blending"][dim] = {"r": round(float(r), 3), "p": float(p_), "n_items": len(common)}
        print(f"  {dim:16s} r = {r:+.2f} (p = {p_:.4f}, n = {len(common)})")

    print("\nUTILITY vs NOVELTY, items varying (the model-level values are above)")
    for task in TASKS:
        for dim in [d for d in DIMS[task] if d in ("surprise", "originality", "em_originality")]:
            iu, xu = item_means(cells, task, "utility")
            io, xo = item_means(cells, task, dim)
            common = sorted(set(iu) & set(io))
            x = np.array([xu[iu.index(i)] for i in common]); y = np.array([xo[io.index(i)] for i in common])
            r, p_ = pearsonr(x, y)
            out["utility_vs_novelty"][f"{task}.{dim}"] = {"r": round(float(r), 3), "p": float(p_),
                                                          "n_items": len(common)}
            print(f"  {task:12s} utility ~ {dim:15s} r = {r:+.2f} (p = {p_:.4f}, n = {len(common)})")

    print("\nANCHOR DISTANCE d(u,v) vs the item means (printed at |r| >= 0.36, p < 0.05 at n = 30)")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    for task in TASKS:
        items, _ = item_means(cells, task, "utility")
        D = np.array([1 - float(un(np.asarray(embed(u), float)) @ un(np.asarray(embed(v), float)))
                      for u, v in items])
        for dim in DIMS[task]:
            io, y = item_means(cells, task, dim)
            common = sorted(set(items) & set(io))
            xa = np.array([D[items.index(i)] for i in common])
            ya = np.array([y[io.index(i)] for i in common])
            r, p_ = pearsonr(xa, ya)
            out["anchor_distance"][f"{task}.{dim}"] = {"r": round(float(r), 3), "p": float(p_),
                                                       "n_items": len(common)}
            if abs(r) >= 0.36:
                print(f"  {task:12s} d(u,v) ~ {dim:15s} r = {r:+.2f} (p = {p_:.4f}, n = {len(common)})")
    return out


def plot(facets, facet_v, out_png, dims=None, compact=False):
    """Facet correlation heatmap. Correlation is a POLARITY measure, so the palette is diverging --
    two hues either side of a neutral midpoint at r = 0, symmetric limits, never a rainbow. Cells that
    clear p < 0.05 carry their value; the rest stay unlabelled so the eye reads the block structure
    rather than every number.

    ``compact`` renders the camera-ready half-column version: drawn to stay legible at half the
    text width, with the always-1.0 diagonal dropped and the column tick labels removed (rows and
    columns carry the same dimensions in the same order, and the task band names the blocks)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                         "text.color": "#222222"})
    TASK_COL = {"association": "#A8476A", "analogy": "#9A7D2E", "blending": "#3F6F8F"}
    dims = dims or DIMS
    facets = [f for f in facets if f[1] in dims[f[0]]]
    k = len(facets)
    M = np.eye(k)
    for i in range(k):
        for j in range(k):
            if i != j:
                M[i, j] = pearsonr(facet_v[facets[i]], facet_v[facets[j]])[0]
    # the diagonal is 1.0 by construction; in the compact figure it is dropped so the eye is not
    # drawn to a band that carries no information.
    mask = np.triu(np.ones_like(M, dtype=bool), 1 if not compact else 0)
    M = np.ma.array(M, mask=mask)
    n_models = len(next(iter(facet_v.values())))
    tc = t_dist.ppf(0.975, n_models - 2)
    r_crit = float(tc / np.sqrt(tc ** 2 + n_models - 2))

    # In a STRICT lower triangle the first row and the last column hold no cells, so the compact
    # figure drops them rather than printing an empty labelled band.
    off = 1 if compact else 0
    D = M[off:, :k - off] if compact else M
    if compact:
        fig, ax = plt.subplots(figsize=(0.34 * k + 2.1, 0.34 * k + 1.2))
    else:
        fig, ax = plt.subplots(figsize=(0.71 * k + 2.0, 0.62 * k + 2.6))
    im = ax.imshow(D, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(k):
        for j in range(i + 1):
            if i == j:
                continue
            r = M[i, j]
            if abs(r) >= r_crit:
                ax.text(j, i - off, f"{r:+.2f}".replace("+0.", ".").replace("-0.", "−."),
                        ha="center", va="center", fontsize=15 if compact else 13,
                        fontweight="bold" if compact else "normal",
                        color="white" if abs(r) > 0.62 else "#222222")
    edges = np.cumsum([len(dims[t]) for t in TASKS])[:-1]
    for e in edges:
        ax.axhline(e - 0.5 - off, color="white", lw=2.5)
        ax.axvline(e - 0.5, color="white", lw=2.5)
    labels = [(d if compact else LABEL[d]) for t in TASKS for d in dims[t]]
    cols = [TASK_COL[t] for t in TASKS for _ in dims[t]]
    ax.set_yticks(range(k - off))
    ax.set_yticklabels(labels[off:], fontsize=11 if compact else 12)
    for lab, c in zip(ax.get_yticklabels(), cols[off:]):
        lab.set_color(c)
    if compact:                     # columns repeat the rows; the task band identifies the blocks
        ax.set_xticks([])
    else:
        ax.set_xticks(range(k))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
        for lab, c in zip(ax.get_xticklabels(), cols):
            lab.set_color(c)
    start, ncol, ybar = 0, k - off, k - off - 0.3
    for t in TASKS:                                          # task band under the columns
        end = min(start + len(dims[t]), ncol)
        if start >= ncol:
            break
        ax.plot([start - 0.42, end - 0.58], [ybar, ybar], color=TASK_COL[t],
                lw=2.5 if compact else 3, clip_on=False)
        ax.text((start + end - 1) / 2, ybar + 0.35, t, ha="center", va="top",
                fontsize=11 if compact else 12.5,
                color=TASK_COL[t], fontweight="bold", clip_on=False)
        start = end
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, shrink=0.62,
                      ticks=[-1, -0.5, 0, 0.5, 1] if not compact else [-1, 0, 1])
    cb.set_label(f"Pearson $r$ across the {n_models} models", fontsize=9 if compact else 11)
    cb.ax.tick_params(labelsize=9 if compact else 10)
    cb.outline.set_visible(False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_png.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    print(f"wrote {out_png}")



def associative_hypothesis(pm, models, facet_v):
    """Mednick's associative account, as a test the benchmark can fail.

    If associative ability is the substrate of combinatorial creativity -- the claim behind the Remote
    Associates Test and its use as a creativity measure -- then a model's association score should
    predict BOTH of the tasks that build on association: analogy (structure mapping) and blending
    (conceptual integration). The two correlations share the association variable, so the comparison
    is a DEPENDENT, overlapping one: Steiger's z with Williams' correction, which uses the third
    correlation r(analogy, blending) to account for the shared term. Fisher intervals are reported
    because at n = 21 a non-significant r is not evidence of no relationship without one.
    """
    def fisher_ci(r, n, a=0.05):
        z = np.arctanh(r); se = 1 / np.sqrt(n - 3); c = 1.959963985 * se
        return float(np.tanh(z - c)), float(np.tanh(z + c))

    def williams(r12, r13, r23, n):
        """Is r12 (association~analogy) larger than r13 (association~blending)? Dependent, overlapping."""
        d = r12 - r13
        det = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23
        rbar = (r12 + r13) / 2
        t = d * np.sqrt((n - 1) * (1 + r23) /
                        (2 * det * (n - 1) / (n - 3) + rbar**2 * (1 - r23)**3))
        from scipy.stats import t as tdist
        return float(t), float(2 * (1 - tdist.cdf(abs(t), n - 3)))

    n = len(models)
    out = {"n_models": n, "levels": {}}
    print("\n\nTHE ASSOCIATIVE HYPOTHESIS (does association predict combinatorial creativity?)")
    series = {"composite": {t: np.array([pm[m]["per_task"][t] for m in models]) for t in TASKS}}
    for dim in ("utility", "originality"):
        series[dim] = {t: facet_v[(t, dim)] for t in TASKS}
    for level, s in series.items():
        r12 = pearsonr(s["association"], s["analogy"])[0]
        r13 = pearsonr(s["association"], s["blending"])[0]
        r23 = pearsonr(s["analogy"], s["blending"])[0]
        tt, pp = williams(r12, r13, r23, n)
        out["levels"][level] = {
            "assoc_analogy": {"r": round(float(r12), 3), "ci": [round(x, 3) for x in fisher_ci(r12, n)],
                              "p": float(pearsonr(s["association"], s["analogy"])[1])},
            "assoc_blending": {"r": round(float(r13), 3), "ci": [round(x, 3) for x in fisher_ci(r13, n)],
                               "p": float(pearsonr(s["association"], s["blending"])[1])},
            "analogy_blending": {"r": round(float(r23), 3)},
            "williams_t": round(tt, 3), "williams_p": pp}
        print(f"  {level:12s} assoc~analogy {r12:+.2f} [{fisher_ci(r12, n)[0]:+.2f},{fisher_ci(r12, n)[1]:+.2f}]"
              f"  assoc~blending {r13:+.2f} [{fisher_ci(r13, n)[0]:+.2f},{fisher_ci(r13, n)[1]:+.2f}]"
              f"  difference: t = {tt:+.2f}, p = {pp:.3f}")

    # the jagged cases: models whose association rank and blending rank disagree most
    ra = np.argsort(np.argsort(-np.array([pm[m]["per_task"]["association"] for m in models])))
    rb = np.argsort(np.argsort(-np.array([pm[m]["per_task"]["blending"] for m in models])))
    gap = rb - ra
    out["largest_rank_gaps"] = [{"model": models[i], "association_rank": int(ra[i]) + 1,
                                 "blending_rank": int(rb[i]) + 1} for i in np.argsort(-np.abs(gap))[:5]]
    print("  models whose association and blending ranks disagree most:")
    for e in out["largest_rank_gaps"]:
        print(f"    {e['model']:34s} association #{e['association_rank']:<3d} blending #{e['blending_rank']}")
    return out

if __name__ == "__main__":
    main()
