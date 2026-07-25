"""Headline figure: creativity's two mechanisms — a novelty gain and an adherence loss.

Creativity factorises: E[R*U] = R_valid * adherence, its two multiplicative factors. This figure
shows how each constraint moves them, pooled as the mean over models of the per-model % change vs
its own baseline. Every constraint raises the novelty of successful paths a little (green) and cuts
the adherence rate a lot (red); net creativity is their per-model product and is shown in the net
figure. The two are drawn as separate measured changes, not summed — percentage changes do not add.

Built to the Nature branded-journals artwork guide: 88 mm single-column, Arial, 5-7 pt, vector PDF.

    .venv_mlx/bin/python src/kg_creat/scripts/fig_creativity_mechanism.py data/kg_creat/scores_regimeA_all
"""

import json
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

MM = 1 / 25.4
COL1 = 88 * MM

# Ordered best-to-worst net effect, so the rows read top-to-bottom as a severity ranking.
MODES = ["categorical", "exclusion", "inclusion_rare", "inclusion"]
LABEL = {"categorical": "Categorical", "exclusion": "Exclusion",
         "inclusion_rare": "Inclusion (rare)", "inclusion": "Inclusion (common)"}

INK, MUTED = "#1f2933", "#5b6672"
NOV, ADH = "#2F7D4F", "#B3402F"     # novelty gain (green), adherence loss (red)


def main(scores_dir, outdir):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6.5,
        "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    S = json.loads((Path(scores_dir) / "scores_summary.json").read_text())
    # Paired pooling: mean over models of the per-model % change vs its own baseline. Novelty and
    # adherence are the two multiplicative factors of creativity (E[R·U] = R_valid × adherence);
    # they are shown as two separate measured changes rather than summed, because percentage
    # changes do not add — net creativity is their per-model product and lives in the net figure.
    def dpct(m, field):
        v = [S[k]["per_mode"][m][field] / S[k]["per_mode"]["baseline"][field] - 1
             for k in S if S[k]["per_mode"]["baseline"][field] and S[k]["per_mode"][m][field]]
        return 100 * mean(v)
    nov = {m: dpct(m, "R_valid") for m in MODES}
    adh = {m: dpct(m, "sat_rate") for m in MODES}
    net = {m: dpct(m, "creativity") for m in MODES}   # for the console readout only

    fig, ax = plt.subplots(figsize=(COL1, 40 * MM), layout="constrained")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2, pad=1.5)
    ax.axvline(0, color=INK, lw=0.7, zorder=4)

    h = 0.34
    for i, m in enumerate(MODES):
        ax.barh(i - h / 1.65, nov[m], height=h, color=NOV, zorder=3)         # novelty gain, rightward
        ax.barh(i + h / 1.65, adh[m], height=h, color=ADH, zorder=3)         # adherence loss, leftward
        ax.text(nov[m] + 1.2, i - h / 1.65, f"+{nov[m]:.0f}%", va="center", ha="left",
                fontsize=5.5, color=NOV)
        ax.text(adh[m] - 1.2, i + h / 1.65, f"{adh[m]:.0f}%", va="center", ha="right",
                fontsize=5.5, color=ADH)

    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([LABEL[m] for m in MODES], color=INK)
    ax.set_ylim(len(MODES) - 0.45, -0.6)
    ax.set_xlim(-62, 20)
    ax.set_xticks([-60, -40, -20, 0])
    ax.set_xticklabels(["−60", "−40", "−20", "0"])
    ax.set_xlabel("Change vs. unconstrained baseline (%)")
    ax.legend(handles=[Patch(facecolor=NOV, label="novelty of successful paths"),
                       Patch(facecolor=ADH, label="adherence rate")],
              loc="upper left", frameon=False, handlelength=1.1, handletextpad=0.4,
              labelspacing=0.25, borderpad=0.0, bbox_to_anchor=(0.0, 0.99))

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = outdir / f"fig_creativity_mechanism.{ext}"
        fig.savefig(f, dpi=600, facecolor="white")
        print(f"saved {f}")
    print("\nvalues:")
    for m in MODES:
        print(f"  {LABEL[m]:20s} novelty {nov[m]:+.1f}%  adherence {adh[m]:+.1f}%  net {net[m]:+.1f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_regimeA_all",
         sys.argv[2] if len(sys.argv) > 2 else "papers/kg_creat-iclr/media")
