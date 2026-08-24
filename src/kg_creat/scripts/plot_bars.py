"""Camera-ready grouped-bar (model-profile) plots for the top-N Kombine models -- ONE panel per task.

Bar analog of plot_radar.py. Each task gets a panel whose x-axis is that task's scoring dimensions
(association: utility, surprise, originality; analogy and blending add emergent). Within each dimension
group, one bar per model gives its gated score, z-scored across all models (0 = field mean, marked by
the dashed line; +1 = one SD above). All panels share one y-scale. Font is Nimbus Roman to match the paper.

    python src/kg_creat/scripts/plot_bars.py data/kg_creat/kombine_v2/scores/composite.json \\
        papers/kg_creat-iclr/media/bar_profiles
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DISPLAY = {
    "openai_gpt-5": "gpt-5", "openai_gpt-5-6-sol": "gpt-5.6-sol", "openai_gpt-4-1": "gpt-4.1",
    "openai_gpt-4o-mini": "gpt-4o-mini", "anthropic_claude-sonnet-4-5": "claude-sonnet-4.5",
    "google_gemini-2-5-flash": "gemini-2.5-flash", "google_gemini-3-1-pro-preview": "gemini-3.1-pro",
    "google_gemini-3-7-flash": "gemini-3.7-flash", "qwen_qwen3-max": "qwen3-max",
}
# (task label, internal key, dimensions on that task's panel)
TASKS = [
    ("Association", "association", ["utility", "surprise", "originality"]),
    ("Analogy", "analogy", ["utility", "surprise", "originality", "emergent"]),
    ("Blending", "blending", ["utility", "surprise", "originality", "emergent"]),
]
DIM_LABEL = {"utility": "Utility", "surprise": "Surprise", "originality": "Originality",
             "emergent": "Emergent"}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def main(composite_path, out_stem, top_n):
    plt.rcParams.update({"font.family": "Nimbus Roman", "mathtext.fontset": "custom",
                         "mathtext.rm": "Nimbus Roman", "axes.linewidth": 0.8, "font.size": 12})
    c = json.loads(Path(composite_path).read_text())
    models, top = c["ranking"], c["ranking"][:top_n]

    # First pass: z-score each (task, dimension) across all models; collect the range over the plotted
    # top-N so all panels share one y-scale.
    panels, zvals = [], []
    for label, key, dims in TASKS:
        raw = {m: c["per_model"][m]["raw"].get(key, {}) for m in models}
        z = {}
        for d in dims:
            col = np.array([raw[m].get(d, np.nan) for m in models], float)
            mu, sd = np.nanmean(col), np.nanstd(col)
            z[d] = (col - mu) / sd if sd > 1e-9 else np.zeros_like(col)
        panels.append((label, dims, z))
        zvals += [z[d][models.index(m)] for m in top for d in dims]
    ymin, ymax = min(zvals) - 0.3, max(zvals) + 0.3

    fig, axes = plt.subplots(1, len(TASKS), figsize=(15.0, 5.0), sharey=True,
                             gridspec_kw=dict(width_ratios=[len(d) for _, _, d in TASKS]))
    width = 0.8 / len(top)
    for ax, (label, dims, z) in zip(axes, panels):
        x = np.arange(len(dims))
        for i, m in enumerate(top):
            vals = [z[d][models.index(m)] for d in dims]
            off = (i - (len(top) - 1) / 2) * width
            ax.bar(x + off, vals, width, color=COLORS[i], label=DISPLAY.get(m, m), zorder=3)
        ax.axhline(0, color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=2)  # field mean
        ax.set_xticks(x)
        ax.set_xticklabels([DIM_LABEL[d] for d in dims], fontsize=13)
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis="y", labelsize=9, color="#888888")
        ax.grid(axis="y", color="#BBBBBB", lw=0.6, alpha=0.8, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#BBBBBB")
        ax.spines["bottom"].set_color("#BBBBBB")
        ax.set_title(label, fontsize=30, pad=14)
    axes[0].set_ylabel("score ($z$ across models)", fontsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(top), frameon=False, fontsize=16,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handlelength=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
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
