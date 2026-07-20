"""Plot novelty (ideation) vs utility/success-rate (execution), one panel per constraint type.

Downstream of score.py. For the syntactic constraints (exclusion/inclusion/categorical/
ordering) each model is drawn as an arrow from its no-constraint baseline to its constrained
position, so the arrow direction reads directly: down-right = "more novel, less compliant"
(the ideation-execution gap); up-left = "plays it safe". Regime-B panels (analogy/blending)
have no baseline, so models are plotted as points.

    .venv_mlx/bin/python src/kg_creat/scripts/plot_novelty_utility.py data/kg_creat/scores_pilot_local
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Categorical color by model identity (CVD-safe blue/orange pair); fixed order, not cycled.
MODEL_COLORS = ["#2563EB", "#EA580C", "#059669", "#7C3AED"]
A_MODES = ["exclusion", "inclusion", "categorical", "ordering"]
B_MODES = ["analogy", "blending"]
INK, MUTED, GRID = "#1f2933", "#66727f", "#e3e8ee"


def _short(model_key: str) -> str:
    for tag in ("3B", "7B", "14B", "32B", "72B"):
        if tag in model_key:
            fam = "Qwen2.5" if "Qwen2" in model_key else model_key.split("_")[0]
            return f"{fam}-{tag}"
    return model_key


def main(scores_dir):
    scores_dir = Path(scores_dir)
    summ = json.loads((scores_dir / "scores_summary.json").read_text())
    models = list(summ.keys())
    labels = [_short(m) for m in models]

    modes = A_MODES + B_MODES
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, mode in zip(axes, modes):
        ax.set_title(mode.capitalize(), fontsize=12, color=INK, pad=8)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)

        for mi, model in enumerate(models):
            pm = summ[model]["per_mode"]
            c = MODEL_COLORS[mi % len(MODEL_COLORS)]
            cell = pm.get(mode)
            if not cell or cell["R_emit"] is None or cell["sat_rate"] is None:
                continue
            cx, cy = cell["R_emit"], cell["sat_rate"]
            if mode in A_MODES and "baseline" in pm and pm["baseline"]["sat_rate"] is not None:
                bx, by = pm["baseline"]["R_emit"], pm["baseline"]["sat_rate"]
                ax.annotate("", xy=(cx, cy), xytext=(bx, by),
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, alpha=0.9), zorder=3)
                ax.scatter([bx], [by], s=42, facecolors="white", edgecolors=c, linewidths=1.8, zorder=4)
            ax.scatter([cx], [cy], s=70, color=c, zorder=5, label=labels[mi] if ax is axes[0] else None)

    for ax in axes[3:]:
        ax.set_xlabel("Novelty  (mean embedding remoteness)", fontsize=10, color=MUTED)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Utility  (success rate)", fontsize=10, color=MUTED)

    handles, leg = axes[0].get_legend_handles_labels()
    # add a baseline glyph explainer
    from matplotlib.lines import Line2D
    handles = handles + [Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                                markeredgecolor=MUTED, markersize=8, label="(open = no-constraint baseline)")]
    fig.legend(handles=handles, loc="upper center", ncol=len(labels) + 1, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Novelty vs. utility per constraint type  —  arrow = baseline → constrained",
                 fontsize=13, color=INK, y=1.08, x=0.5)
    fig.tight_layout(rect=[0, 0, 1, 1.0])

    out = scores_dir / "novelty_vs_utility.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_pilot_local")
