"""The tau-inventive-multiples table as a figure: rate of multiples at tau = 1, 2, 3.

Three panels from `tau_curve` in inventive_multiples.json, one row, text width:
  (a) % of all inventions that belong to at least one tau-multiple;
  (b) % of invention pairs that are tau-multiples, blends vs analogies (log scale; the blend/analogy
      ratio printed above each tau), in the paper's task colours;
  (c) the same split by whether the two models share a provider family (same/different ratio above).
Bars carry their value; a zero rate is drawn as no bar with "0" written at the axis.

    .venv/bin/python -m src.kg_creat.scripts.plot_tau_multiples
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

SRC = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
# One palette for all three panels, shared with Table 4's row tint and Figure 8: dark slate for the first bar, light grey for the second
BLEND, ANALOGY = "#486878", "#C9CDD1"
SAME, DIFF = "#486878", "#C9CDD1"
INV = "#486878"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "font.size": 16,
                     "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black"})
YMIN = 0.03                                      # log-axis floor; a zero rate is written just above it


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)
    ax.grid(axis="y", color="#E6E6E6", lw=0.8, zorder=0); ax.set_axisbelow(True)


def title_legend(ax, title, handles=None):
    """Panel title at the left; the legend inside the axes at the upper right, above the low tau = 3 bars and the ratio row."""
    ax.set_title(title, loc="left", fontsize=16, pad=8)
    if handles:
        ax.legend(handles=handles, frameon=False, fontsize=12.5, loc="upper center", ncol=2, handlelength=1.0,
                  handletextpad=0.5, columnspacing=1.2, borderaxespad=0.1)


def bars(ax, taus, left, right, left_lab, right_lab, cl, cr, ratio=True, ytick_labels=True):
    x = np.arange(len(taus)); w = 0.36
    L, Rv = np.array(left, float), np.array(right, float)
    for off, vals, c, side in ((-w / 2, L, cl, -1), (w / 2, Rv, cr, +1)):
        pos = x + off
        ax.bar(pos[vals > 0], vals[vals > 0], w, color=c, edgecolor="none", zorder=3)
        for k, (xi, vi) in enumerate(zip(pos, vals)):
            y = (vi * 1.12) if vi > 0 else YMIN * 1.12
            txt = "0%" if vi == 0 else (f"{vi:.2f}%" if vi < 1 else f"{vi:.1f}%")
            other = Rv[k] if side < 0 else L[k]
            nudge = side * 0.06 if other > vi else 0.0        # the shorter bar's label steps away from its taller neighbour
            ax.text(xi + nudge, y, txt, ha="center", va="bottom", fontsize=10, color="black", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none", alpha=0.9))
    if ratio:                                                  # the left/right ratio, one row at a fixed height above each pair
        for xi, (l, r) in enumerate(zip(left, right)):
            if r > 0:
                ax.text(xi, RATIO_Y, f"{l / r:.1f}$\\times$", ha="center", va="center", fontsize=13, fontweight="bold", color="black")
    ax.set_xticks(x); ax.set_xticklabels([f"$\\tau={t}$" for t in taus], fontsize=14)
    ax.set_yscale("log"); ax.set_ylim(YMIN, 260)
    ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1%", "1%", "10%", "100%"] if ytick_labels else [], fontsize=13)
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    style(ax)
    return [Patch(color=cl, label=left_lab), Patch(color=cr, label=right_lab)]


RATIO_Y = 48                                      # the ratio row sits on one line, above the tallest bar and below the legend


def main():
    d = json.loads(SRC.read_text())
    rows = {r["tau"]: r for r in d["tau_curve"]}
    taus = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.7), gridspec_kw={"width_ratios": [1.0, 1.1, 1.0], "wspace": 0.25})
    ax = axes[0]; v = [rows[t]["inventions_pct"] for t in taus]; x = np.arange(3)
    ax.bar(x, v, 0.55, color=INV, edgecolor="none", zorder=3)
    for xi, vi in zip(x, v): ax.text(xi, vi + 1.5, f"{vi:.1f}%", ha="center", va="bottom", fontsize=10, color="black")
    ax.set_xticks(x); ax.set_xticklabels([f"$\\tau={t}$" for t in taus], fontsize=14); ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80]); ax.set_yticklabels([f"{t}%" for t in (0, 20, 40, 60, 80)], fontsize=13)
    ax.set_ylabel("% of inventions", fontsize=13)
    style(ax); title_legend(ax, "(a) Inventions in a multiple")
    h = bars(axes[1], taus, [rows[t]["blending_pct"] for t in taus], [rows[t]["analogy_pct"] for t in taus], "Blend", "Analogy", BLEND, ANALOGY)
    axes[1].set_ylabel("% of invention pairs", fontsize=13)
    title_legend(axes[1], "(b) Pairs in multiples by task", h)
    h = bars(axes[2], taus, [rows[t]["same_provider_pct"] for t in taus], [rows[t]["cross_provider_pct"] for t in taus], "Same", "Different", SAME, DIFF, ytick_labels=False)
    title_legend(axes[2], "(c) Pairs in multiples by family", h)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_tau_multiples.{ext}", dpi=220, bbox_inches="tight")
    print("saved", OUT / "fig_tau_multiples.{png,pdf}")
    for t in taus:
        r = rows[t]; print(f"tau={t}: blend {r['blending_pct']:.2f} analogy {r['analogy_pct']:.2f} same {r['same_provider_pct']:.2f} diff {r['cross_provider_pct']:.2f} inv {r['inventions_pct']:.1f}")


if __name__ == "__main__":
    main()
