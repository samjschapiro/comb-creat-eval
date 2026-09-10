"""Shared abstraction failure as a model x item grid.

Each cell is one blend: filled where the 3-judge panel rejected the generic space (the schema is
instantiated by one input only), open where it was accepted. Rows are frontier models sorted by
failure rate, columns are the 30 anchor pairs sorted by difficulty. Three things are then marginals of
the same grid:

  * no column is fully filled -- every anchor pair was solved by some model, so the failure is not the
    item being impossible;
  * rows differ enormously (17%-70%), and not in leaderboard order;
  * columns differ too, so item difficulty is real but never total.

The panel is a majority of three judges at ICC 0.48-0.65, so a filled cell is a moderately reliable
verdict, not ground truth -- said in the caption rather than implied away by a crisp binary grid.

    .venv/bin/python -m src.kg_creat.scripts.plot_abstraction_failure
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.kg_creat.scripts.analyze_failure_modes import FRONTIER, SCORES
from src.kg_creat.scripts.plot_multiples_matrix import brand_logos, _disp
from src.kg_creat.scripts.plot_radar import _provider as _radar_prov

RESP = Path("data/kg_creat/kombine_test30/responses")
OUT = Path("docs/reports/2026-09-03_kg_creat_frontier_failures/figures/fig_abstraction_failure.png")
FILL = "#A8476A"          # rejected -- the failure is the marked state here, so it gets the warm hue
OK = "#2F7D6E"
CROSS = "#D55E00"         # the by-item grid fills a rejected cell in Okabe-Ito vermilion: reads as red to
                          # normal vision and stays bright and distinct under protan/deutan deficiencies
CONTRAST_ITEM = ("X-rays", "Nuclear fission")
# The by-item grid drops this anchor pair and every easier one (rows are sorted hardest first), to
# keep the paper figure to the items where the failure is common. The caption must say so.
BY_ITEM_CUT = ("Christianity", "Beauty")

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "text.color": "#222222"})


def load():
    g_by = {}
    for f in sorted(RESP.glob("*/responses.json")):
        for r in json.loads(f.read_text()):
            if r.get("mode") == "blending" and r.get("items"):
                g_by[(f.parent.name, r["u_label"], r["v_label"])] = r["items"][0]
    cell = {}
    for f in sorted(SCORES.glob("*/path_scores.json")):
        m = f.parent.name
        if m not in FRONTIER:
            continue
        for r in json.loads(f.read_text()):
            # THE GATE is `generic_ok` (blend utility U_bl). `blend_integration` (scope) is a
            # SEPARATE panel field graded given a schema, and the two disagree on 23% of blends --
            # an earlier version of this figure filled cells on scope == 1 and mislabelled it.
            if r.get("mode") == "blending" and r.get("generic_ok") is not None:
                cell[(m, r["u_label"], r["v_label"])] = not bool(r["generic_ok"])
    return cell, g_by


def main():
    cell, _ = load()
    models = sorted({k[0] for k in cell})
    items = sorted({(k[1], k[2]) for k in cell})
    M = np.array([[cell.get((m, u, v), np.nan) for (u, v) in items] for m in models], float)

    row_rate = np.nanmean(M, axis=1)
    col_rate = np.nanmean(M, axis=0)
    ri, ci = np.argsort(row_rate), np.argsort(-col_rate)
    M, models = M[ri][:, ci], [models[i] for i in ri]
    items = [items[j] for j in ci]
    row_rate, col_rate = row_rate[ri], col_rate[ci]
    nr, nc = M.shape

    # Sized for ICLR's 5.5in \textwidth: at \linewidth this scales by 5.5/9.2 = 0.6, so the 10pt
    # column labels land near 6pt on the page. A wider drawing renders them unreadably small.
    fig = plt.figure(figsize=(9.2, 5.8))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(nr - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            if np.isnan(v):
                continue
            if v:
                ax.add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, facecolor=to_rgba(FILL, .9),
                                           edgecolor="white", linewidth=0.7, zorder=3))
            else:
                ax.add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, facecolor="none",
                                           edgecolor=to_rgba(OK, .55), linewidth=1.1, zorder=2))
    logos = brand_logos()
    for i, m in enumerate(models):
        ax.text(-1.45, i, _disp(m), ha="right", va="center", fontsize=12)
        img = logos.get(_radar_prov(m))
        if img is not None:
            ab = AnnotationBbox(OffsetImage(img, zoom=0.027, alpha=0.95), (-0.95, i), frameon=False,
                                box_alignment=(0.5, 0.5), annotation_clip=False)
            ab.set_clip_on(False); ab.set_zorder(6)
            ax.add_artist(ab)
        ax.text(nc - 0.35, i, f"{100*row_rate[i]:.0f}%", ha="left", va="center", fontsize=11,
                color="#5C6472", family="monospace")
    for j, (u, v) in enumerate(items):
        lab = f"{u} + {v}"
        # One colour for every pair: the columns are already ordered by difficulty and the marginal
        # row prints each rate, so a two-tone split added no information and implied a hard class
        # boundary at an arbitrary cutoff.
        ax.text(j, -0.75, lab if len(lab) <= 40 else lab[:38] + "…", rotation=55, rotation_mode="anchor",
                ha="left", va="baseline", fontsize=10, color="#14161B")
    ax.text(nc - 0.35, -1.0, "rejected", ha="left", va="center", fontsize=10.5, color="#5C6472")
    ax.plot([-0.45, nc - 0.55], [nr - 0.35, nr - 0.35], color="#C9CFD8", lw=1)
    for j in range(nc):
        ax.text(j, nr + 0.15, f"{int(round(100*col_rate[j]))}", ha="center", va="center", fontsize=9,
                color="#7A7F88", family="monospace", clip_on=False)
    ax.text(-1.45, nr + 0.15, "rejected % per pair", ha="right", va="center", fontsize=10,
            color="#7A7F88", clip_on=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    print(f"saved {OUT}  ({nr} models x {nc} anchor pairs; "
          f"columns fully rejected: {int((col_rate == 1).sum())})")
    plt.close(fig)
    by_item(M.T, items, models, col_rate, row_rate, logos)


def by_item(M, items, models, item_rate, model_rate, logos):
    """The same grid transposed: anchor pairs as rows (horizontal text on the left), models as
    columns (names under the grid), so the item difficulty reads down the page and the reader can
    find a model by name. Rows are ordered hardest first, columns best model first."""
    if BY_ITEM_CUT is not None:
        if BY_ITEM_CUT not in items:
            raise ValueError(f"FATAL: BY_ITEM_CUT {BY_ITEM_CUT} is not an anchor pair in the data")
        k, n_all = items.index(BY_ITEM_CUT), len(items)
        M, items, item_rate = M[:k], items[:k], item_rate[:k]
        print(f"  by-item grid cut at {BY_ITEM_CUT}: keeping the {k} hardest of {n_all} "
              f"anchor pairs (rejection >= {100 * item_rate[-1]:.0f}%)")
    nr, nc = M.shape
    fig = plt.figure(figsize=(10.0, 1.8 + 0.42 * nr))
    ax = fig.add_subplot(111)
    ax.set_xlim(-0.5, nc - 0.5); ax.set_ylim(nr - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            if np.isnan(v):
                continue
            # every answered cell is a black-outlined square; a rejected generic space is filled red
            ax.add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, facecolor=CROSS if v else "none",
                                       edgecolor="black", linewidth=1.5, zorder=2))
    for i, (u, v) in enumerate(items):
        ax.text(-0.9, i, f"{u} + {v}", ha="right", va="center", fontsize=15, color="black", fontweight="bold")
    for j, m in enumerate(models):
        # the provider mark sits just under the grid, and the model name hangs from it
        img = logos.get(_radar_prov(m))
        if img is not None:
            ab = AnnotationBbox(OffsetImage(img, zoom=0.048, alpha=0.95), (j, nr + 0.1), frameon=False,
                                box_alignment=(0.5, 0.5), annotation_clip=False)
            ab.set_clip_on(False); ab.set_zorder(6)
            ax.add_artist(ab)
        ax.text(j + 0.15, nr + 0.75, _disp(m), rotation=55, rotation_mode="anchor", ha="right", va="top",
                fontsize=14, color="black")
    out = OUT.with_name("fig_abstraction_failure_by_item")
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}  ({nr} anchor pairs x {nc} models)")


if __name__ == "__main__":
    main()
