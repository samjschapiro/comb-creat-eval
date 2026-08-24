"""Camera-ready model-profile grid for the top-N Kombine models.

A 3-row grid, 2 columns per task (6 columns total):
  Row 1: a per-task RANKING of the models by composite score (z-composite shown in parentheses),
         spanning that task's two columns.
  Row 2: two dimension charts per task (utility, surprise), directly beneath the task.
  Row 3: up to two dimension charts per task (originality, and emergent where it exists).
Dimension bars use RAW scores (all non-negative), with a shared y-scale per dimension across tasks.
Model colors come from the batlow scientific colormap. Font is Nimbus Roman to match the paper.

    python src/kg_creat/scripts/plot_profiles.py data/kg_creat/kombine_v2/scores/composite.json \\
        papers/kg_creat-iclr/media/profiles_grid
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from cmcrameri import cm as cmc

DISPLAY = {
    "openai_gpt-5": "gpt-5", "openai_gpt-5-6-sol": "gpt-5.6-sol", "openai_gpt-4-1": "gpt-4.1",
    "openai_gpt-4o-mini": "gpt-4o-mini", "anthropic_claude-sonnet-4-5": "claude-sonnet-4.5",
    "google_gemini-2-5-flash": "gemini-2.5-flash", "google_gemini-3-1-pro-preview": "gemini-3.1-pro",
    "google_gemini-3-7-flash": "gemini-3.7-flash", "qwen_qwen3-max": "qwen3-max",
}
# (task label, internal key, [row2 dims], [row3 dims])
TASKS = [
    ("Association", "association", ["utility", "surprise"], ["originality"]),
    ("Analogy", "analogy", ["utility", "surprise"], ["originality", "emergent"]),
    ("Blending", "blending", ["utility", "surprise"], ["originality", "emergent"]),
]
DIM_LABEL = {"utility": "Utility", "surprise": "Surprise", "originality": "Originality",
             "emergent": "Emergent"}


def _bars(ax, vals, colors, title, show_y):
    x = np.arange(len(vals))
    ax.axhline(50, color="#CFCFCF", lw=0.8, ls=(0, (4, 3)), zorder=1)  # median (50th pctile)
    ax.bar(x, vals, 0.72, color=colors, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([])
    ax.grid(axis="y", color="#DDDDDD", lw=0.6, alpha=0.8, zorder=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    if show_y:
        ax.set_yticklabels(["0", "25", "50", "75", "100"])
        ax.tick_params(axis="y", labelsize=9, color="#888888")
        ax.set_ylabel("percentile", fontsize=11)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=15, pad=8)


def main(composite_path, out_stem, top_n):
    plt.rcParams.update({"font.family": "Nimbus Roman", "mathtext.fontset": "custom",
                         "mathtext.rm": "Nimbus Roman", "axes.linewidth": 0.8, "font.size": 12})
    c = json.loads(Path(composite_path).read_text())
    models, top = c["ranking"], c["ranking"][:top_n]
    colors = [cmc.batlow(x) for x in np.linspace(0.08, 0.92, len(top))]
    cmap = {m: colors[i] for i, m in enumerate(top)}

    raw = {(t[1], d): {m: c["per_model"][m]["raw"].get(t[1], {}).get(d)
                       for m in models}
           for t in TASKS for d in t[2] + t[3]}
    comp = {t[1]: {m: c["per_model"][m]["per_task"][t[1]] for m in models} for t in TASKS}
    # percentile of each model's raw score among ALL evaluated models, per (task, dimension)
    def pctl(v, arr):
        return 100.0 * (sum(a < v for a in arr) + 0.5 * sum(a == v for a in arr)) / len(arr)
    pct = {}
    for (key, d), vals in raw.items():
        allv = [x for x in vals.values() if x is not None]
        pct[(key, d)] = {m: (pctl(vals[m], allv) if vals[m] is not None else None) for m in models}

    fig = plt.figure(figsize=(15.0, 10.5))
    # 12 columns = 4 per task, each dimension panel spans 2; a lone row-3 panel spans the middle two.
    gs = GridSpec(3, 12, figure=fig, height_ratios=[1.15, 1.0, 1.0], hspace=0.42, wspace=1.1)

    for ti, (label, key, r2, r3) in enumerate(TASKS):
        base = 4 * ti
        letter = "abc"[ti]
        # Row 1: clean ranked mini-leaderboard (rank, colour chip, model, right-aligned z)
        axr = fig.add_subplot(gs[0, base:base + 4])
        axr.axis("off")
        axr.set_title(f"({letter}) {label}", fontsize=27, pad=4)
        axr.plot([0.05, 0.97], [0.965, 0.965], color="#CCCCCC", lw=0.8,
                 transform=axr.transAxes, clip_on=False)
        ranked = sorted(top, key=lambda m: comp[key][m], reverse=True)
        y0, dy = 0.80, 0.175
        for i, m in enumerate(ranked):
            y = y0 - i * dy
            axr.text(0.05, y, f"{i+1}", fontsize=14, color="#777777",
                     ha="left", va="center", transform=axr.transAxes)
            axr.plot([0.155], [y], marker="s", ms=11, color=cmap[m],
                     transform=axr.transAxes, clip_on=False)
            axr.text(0.24, y, DISPLAY.get(m, m), fontsize=14.5, color="#1a1a1a",
                     ha="left", va="center", transform=axr.transAxes)
            axr.text(0.97, y, f"{comp[key][m]:+.2f}", fontsize=14.5, color="#1a1a1a",
                     ha="right", va="center", transform=axr.transAxes)
        # Rows 2 and 3: percentile panels, bars sorted DESCENDING, labelled (letter.n).
        # 2 dims fill the block; a single dim is centered under the pair above.
        for row, dims in ((1, r2), (2, r3)):
            for j, d in enumerate(dims):
                cs = slice(base + 2 * j, base + 2 * j + 2) if len(dims) == 2 \
                    else slice(base + 1, base + 3)
                n = (0 if row == 1 else len(r2)) + j + 1
                ax = fig.add_subplot(gs[row, cs])
                show_y = (ti == 0 and j == 0)  # leftmost panel of each row (shared 0-100 scale)
                pairs = sorted(((pct[(key, d)][m], cmap[m]) for m in top), key=lambda t: -t[0])
                _bars(ax, [p[0] for p in pairs], [p[1] for p in pairs],
                      f"({letter}.{n}) {DIM_LABEL[d]}", show_y)

    handles = [Patch(color=cmap[m], label=DISPLAY.get(m, m)) for m in top]
    fig.legend(handles=handles, loc="lower center", ncol=len(top), frameon=False, fontsize=16,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.6, handlelength=1.2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", dpi=300)
    print(f"Wrote {out_stem}.pdf / .png  (top {top_n}: {', '.join(DISPLAY.get(m, m) for m in top)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("composite_path")
    ap.add_argument("out_stem")
    ap.add_argument("--top_n", type=int, default=5)
    a = ap.parse_args()
    main(a.composite_path, a.out_stem, a.top_n)
