"""Camera-ready figure: creativity cost by constraint type. Nature Machine Intelligence spec.

Built to the Nature branded-journals artwork guide: 88 mm single-column width, Arial (sans-serif)
throughout with all text between 5 and 7 pt, RGB, and vector PDF for line art. The PDF is the
submission artefact; the PNG is only for on-screen review (Nature does not accept bitmap formats
for vector artwork).

The unit plotted is the MODEL, not the endpoint bundle: each mark is one model's mean paired
difference over the 30 shared endpoint bundles. Plotting the ~240 raw per-bundle differences
instead would spread bundle-level noise of +/-0.5 -- each bundle's creativity comes from only
5 paths -- across an effect that lives in a 0.17-wide band.

    .venv_mlx/bin/python src/kg_creat/scripts/fig_creativity_camera.py data/kg_creat/scores_regimeA_all
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
COL1 = 88 * MM          # Nature single-column width

# Least to most damaging, so the rows read top-to-bottom as a severity ranking.
MODES = ["categorical", "exclusion", "inclusion_rare", "inclusion", "ordering"]
LABEL = {"categorical": "Categorical", "exclusion": "Exclusion",
         "inclusion_rare": "Inclusion (rare)", "inclusion": "Inclusion (common)",
         "ordering": "Ordering"}

INK, MUTED, DOT = "#1f2933", "#5b6672", "#98a2b0"


def main(scores_dir, outdir):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        # Nature: maximum text size 7 pt, minimum 5 pt.
        "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.0,
        # No bbox="tight": it resizes the canvas, and the width must be exactly 88 mm.
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    summ = json.loads((Path(scores_dir) / "scores_summary.json").read_text())
    vals = {m: np.array([summ[k]["two_by_two"][m]["mean_dcreativity"] for k in summ]) for m in MODES}

    fig, ax = plt.subplots(figsize=(COL1, 38 * MM), layout="constrained")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2, pad=1.5)
    ax.axvline(0, color=INK, lw=0.6, zorder=4)

    for i, m in enumerate(MODES):
        v = vals[m]
        ax.scatter(v, np.full(len(v), i), s=9, color=DOT, linewidths=0, zorder=5)
        ax.plot([v.mean()], [i], marker="D", ms=3.4, color=INK, mec="white", mew=0.5, zorder=6)

    ax.set_yticks(range(len(MODES)))
    ax.set_yticklabels([LABEL[m] for m in MODES], color=INK)
    ax.set_ylim(len(MODES) - 0.4, -0.6)
    ax.set_xlim(-0.29, 0.10)
    ax.set_xticks([-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05])
    ax.set_xlabel("Paired \u0394 creativity vs. unconstrained baseline")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = outdir / f"fig_creativity_by_constraint.{ext}"
        fig.savefig(f, dpi=600, facecolor="white")
        print(f"saved {f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_regimeA_all",
         sys.argv[2] if len(sys.argv) > 2 else "papers/kg_creat-iclr/media")
