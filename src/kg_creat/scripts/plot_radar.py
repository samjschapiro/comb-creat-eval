"""Camera-ready radar (model-profile) plots for the top-N Kombine models -- ONE radar per task.

Each task gets its own radar whose axes are that task's scoring dimensions (association: utility,
surprise, originality; analogy and blending add emergent). A model's value on each axis is its gated
score for that task, min-max normalized across all models so every axis runs 0 (weakest model) to 1
(strongest). Font is Nimbus Roman to match the paper (\\usepackage{times}).

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

    fig, axes = plt.subplots(1, len(TASKS), figsize=(13.5, 5.0), subplot_kw=dict(polar=True))
    for ax, (label, key, dims) in zip(axes, TASKS):
        # per-axis min-max normalization across all models for this task
        raw = {m: c["per_model"][m]["raw"].get(key, {}) for m in models}
        col = {d: np.array([raw[m].get(d, np.nan) for m in models], float) for d in dims}
        norm = {d: (col[d] - np.nanmin(col[d])) / (np.nanmax(col[d]) - np.nanmin(col[d]) + 1e-9)
                for d in dims}
        N = len(dims)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for i, m in enumerate(top):
            v = [norm[d][models.index(m)] for d in dims]
            v += v[:1]
            ax.plot(angles, v, color=COLORS[i], lw=1.8, label=DISPLAY.get(m, m), zorder=3)
            ax.fill(angles, v, color=COLORS[i], alpha=0.06, zorder=1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([DIM_LABEL[d] for d in dims], fontsize=12)
        ax.tick_params(axis="x", pad=6)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels([])
        ax.grid(color="#BBBBBB", lw=0.6, alpha=0.8)
        ax.spines["polar"].set_color("#BBBBBB")
        ax.set_title(label, fontsize=15, pad=18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(top), frameon=False, fontsize=12,
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
