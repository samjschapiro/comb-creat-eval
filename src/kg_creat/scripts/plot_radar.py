"""Camera-ready radar (model-profile) plots for the top-N Kombine models -- ONE radar per task.

Each task gets its own radar whose axes are that task's scoring dimensions (association: utility,
surprise, originality; analogy and blending add emergent). A model's value on each axis is its gated
score for that task, z-scored across all models (0 = field mean, marked by the dashed ring; +1 = one
SD above). All three radars share one radial scale. Font is Nimbus Roman to match the paper.

    python src/kg_creat/scripts/plot_radar.py data/kg_creat/kombine_v2/scores/composite.json \\
        papers/kg_creat-iclr/media/radar_profiles
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
# (task label, internal key, dimensions on that task's radar)
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
    # top-N so all three radars share one radial scale.
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
    ymin, ymax = min(zvals) - 0.4, max(zvals) + 0.4
    yticks = list(range(int(np.ceil(ymin)), int(np.floor(ymax)) + 1))

    fig, axes = plt.subplots(1, len(TASKS), figsize=(13.5, 5.0), subplot_kw=dict(polar=True))
    for ax, (label, dims, z) in zip(axes, panels):
        N = len(dims)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(ymin, ymax)
        # emphasize the z = 0 ring (the field mean across all models)
        fine = np.linspace(0, 2 * np.pi, 200)
        ax.plot(fine, np.zeros_like(fine), color="#666666", lw=0.9, ls=(0, (4, 3)), zorder=2)
        for i, m in enumerate(top):
            v = [z[d][models.index(m)] for d in dims]
            v += v[:1]
            ax.plot(angles, v, color=COLORS[i], lw=1.8, label=DISPLAY.get(m, m), zorder=3)
            ax.fill(angles, v, color=COLORS[i], alpha=0.06, zorder=1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([DIM_LABEL[d] for d in dims], fontsize=12)
        ax.tick_params(axis="x", pad=6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t:+d}" if t else "0" for t in yticks], fontsize=8, color="#888888")
        ax.grid(color="#BBBBBB", lw=0.6, alpha=0.8)
        ax.spines["polar"].set_color("#BBBBBB")
        ax.set_title(label, fontsize=30, pad=22)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(top), frameon=False, fontsize=16,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handlelength=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
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
