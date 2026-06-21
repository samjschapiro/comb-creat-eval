"""Appendix companion to make_effort_temp_boxplots.py: a 4x3 grid that decomposes the Overall
composite into its FOUR facets. Rows = surprise, coherence, realism, diversity (raw units);
columns = the three interventions (reasoning effort, sampling temperature, prompting strategy).
Each dot is one (model x condition) cell; the dashed line is the BEST-8 human value for that
facet (apples-to-apples with the best-config cells, as in the main figure).

Shows WHICH facet (if any) each intervention moves -- e.g. in-context regen lifts only diversity.

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/make_facet_grid.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plot_twist.scripts.make_effort_temp_boxplots import (
    effort_cells, temp_cells, strategy_cells, human_topN_facets,
    BOX_COLS, HUMAN_COL, OUT, FIG,
)

FACET_ROWS = [("mean_surprise", "Surprise (1-5)"), ("mean_coherence", "Coherence (1-5)"),
              ("mean_realism", "Realism (1-5)"), ("div", "Diversity")]
COL_TITLES = ["Reasoning effort", "Sampling temperature", "Prompting strategy"]


def cell(ax, levels, labels, by, hline, show_x):
    data = [by[lv] for lv in levels]
    bp = ax.boxplot(data, positions=range(len(levels)), widths=0.55, patch_artist=True, vert=True,
                    medianprops=dict(color="black", lw=1.5), whiskerprops=dict(color="#444"),
                    capprops=dict(color="#444"), flierprops=dict(marker="", alpha=0))
    for patch, col in zip(bp["boxes"], BOX_COLS):
        patch.set_facecolor((*col[:3], 0.35)); patch.set_edgecolor("#333"); patch.set_linewidth(0.9)
    rng = np.random.default_rng(0)
    for i, lv in enumerate(levels):
        ys = by[lv]
        jx = i + (rng.random(len(ys)) - 0.5) * 0.18
        ax.scatter(jx, ys, s=16, color=BOX_COLS[i], edgecolor="#333", linewidth=0.3,
                   alpha=0.45, zorder=3)
    hl = ax.axhline(hline, color=HUMAN_COL, lw=1.6, ls="--", zorder=4)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(labels if show_x else [""] * len(levels))
    return hl


def main():
    hf = human_topN_facets(8)
    panels = [effort_cells(), temp_cells(), strategy_cells()]
    fig, axes = plt.subplots(4, 3, figsize=(9.5, 11), sharey="row")
    hl = None
    for ri, (fkey, flabel) in enumerate(FACET_ROWS):
        for ci, (cells, keyf, levels, labels) in enumerate(panels):
            ax = axes[ri, ci]
            by = {lv: [c[fkey] for c in cells
                       if c[keyf] == lv and np.isfinite(c.get(fkey, np.nan))] for lv in levels}
            hl = cell(ax, levels, labels, by, hf[fkey], show_x=(ri == 3))
            if ri == 0:
                ax.set_title(COL_TITLES[ci], fontweight="bold", pad=8)
            if ci == 0:
                ax.set_ylabel(flabel)
    axes[0, 2].legend([hl], ["human (best 8)"], loc="lower right", fontsize=9, frameon=False)
    fig.suptitle("Per-facet effect of each intervention (dot = one model$\\times$condition cell)",
                 fontweight="bold", y=0.995)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (OUT / "effort_temp_facets.pdf", FIG / "effort_temp_facets.pdf", OUT / "effort_temp_facets.png"):
        fig.savefig(d)
    plt.close(fig)
    print(f"saved -> {FIG/'effort_temp_facets.pdf'}")


if __name__ == "__main__":
    main()
