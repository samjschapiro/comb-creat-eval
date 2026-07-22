"""Camera-ready figure: creativity cost by constraint type, and the difficulty control.

Emits PDF (for LaTeX) + PNG (for review) sized to an ICLR \\linewidth. Typography is set in Times
to match the body text, and no in-figure titles are drawn -- the explanation belongs in the
caption, so the panels carry only what a reader needs while looking at the marks.

Panel (a) the paired effect: every constrained cell was administered on the SAME endpoint bundles
as the unconstrained baseline, so each point is one model on one endpoint pair, differenced
against itself. Showing all ~240 differences (not just the mean) is the point: it is what
distinguishes a constraint that harms nearly every pair from one that is close to a coin flip.

Panel (b) the control: creativity cost could just track how restrictive a constraint is. Plotting
the effect against how much of the models' own default behaviour each constraint rules out shows
it does not -- rare-inclusion and ordering rule out the same share and land far apart.

    .venv_mlx/bin/python src/kg_creat/scripts/fig_creativity_camera.py data/kg_creat/scores_regimeA_all
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from src.kg_creat.scripts.paired_analysis import collect, paired, wilcoxon  # noqa: E402

# Effect order: least to most damaging, so the panel reads top-to-bottom as a severity ranking.
MODES = ["categorical", "exclusion", "inclusion_rare", "inclusion", "ordering"]
LABEL = {"categorical": "Categorical", "exclusion": "Exclusion",
         "inclusion_rare": "Inclusion (rare class)", "inclusion": "Inclusion (common class)",
         "ordering": "Ordering"}
# Share of the models' own unconstrained paths each administered constraint would rule out.
# Categorical is measured on a different scale (entity types over graph routes) and is therefore
# drawn hollow in panel (b) rather than being silently ranked against the others.
BITES = {"exclusion": 0.506, "inclusion": 0.907, "inclusion_rare": 0.992, "ordering": 0.986}

INK, MUTED, GRID = "#1f2933", "#5b6672", "#dfe4ea"
ACCENT, WARN = "#2563EB", "#B3402F"   # matched-pair highlight, and the catastrophic constraint
DOT = "#9aa5b1"


def _rc():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,   # embed TrueType so the PDF is editable/searchable, not bitmapped
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
    })


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=2.5, pad=2)
    for lbl in ax.get_yticklabels() + ax.get_xticklabels():
        lbl.set_color(INK)


def stats(diffs):
    d = np.asarray(diffs, dtype=float)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(len(d))
    return m, m - 1.96 * se, m + 1.96 * se, float((d < 0).mean()), len(d)


def main(scores_dir, outdir):
    _rc()
    data = collect(scores_dir)
    base = data["baseline"]
    rows = {m: stats(paired(data[m], base)) for m in MODES}
    diffs = {m: np.asarray(paired(data[m], base), dtype=float) for m in MODES}

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(5.5, 2.45), gridspec_kw={"width_ratios": [1.32, 1], "wspace": 0.42})

    # ---------------- (a) paired effect per constraint type ----------------
    _style(axA)
    rng = np.random.default_rng(0)   # jitter only; seeded so the figure is reproducible
    X_LO, X_HI = -0.58, 0.46
    X_PCT = 0.72                     # readout column, placed clear of the point cloud
    ypos = {m: i for i, m in enumerate(MODES)}
    axA.axvline(0, color=INK, lw=0.7, zorder=2)
    for m in MODES:
        y = ypos[m]
        d = np.clip(diffs[m], X_LO + 0.01, X_HI - 0.01)   # keep the few outliers inside the frame
        axA.scatter(d, y + rng.uniform(-0.19, 0.19, len(d)), s=1.5, color=DOT,
                    alpha=0.40, linewidths=0, zorder=3, rasterized=True)
        mean, lo, hi, frac, n = rows[m]
        col = WARN if m == "ordering" else INK
        # 95% CIs are narrower than the marker at this scale (all within +/-0.026), so the dot
        # IS the interval to the eye; exact bounds belong in the caption rather than the glyph.
        axA.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="butt", zorder=5)
        axA.plot([mean], [y], "o", ms=4.4, color=col, mec="white", mew=0.8, zorder=6)
        axA.text(X_PCT, y, f"{100*frac:.0f}%", va="center", ha="right", fontsize=7.5,
                 color=col)

    axA.set_yticks(range(len(MODES)))
    axA.set_yticklabels([LABEL[m] for m in MODES])
    axA.invert_yaxis()
    axA.set_xlim(X_LO, X_PCT)
    axA.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    axA.set_ylim(len(MODES) - 0.40, -0.72)
    axA.spines["bottom"].set_bounds(X_LO, X_HI)   # axis line stops before the readout column
    axA.set_xlabel("Paired $\\Delta$ creativity vs. same endpoints unconstrained")
    axA.text(X_PCT, -0.68, "pairs\nharmed", va="top", ha="right", fontsize=6.8,
             color=MUTED, style="italic", linespacing=1.15)
    axA.set_title("(a)", loc="left", color=INK, fontweight="bold", pad=3)

    # ---------------- (b) effect vs. restrictiveness ----------------
    _style(axB)
    axB.axhline(0, color=INK, lw=0.7, zorder=2)
    for m in ("exclusion", "inclusion", "inclusion_rare", "ordering"):
        mean = rows[m][0]
        col = WARN if m == "ordering" else INK
        axB.plot([BITES[m]], [mean], "o", ms=5, color=col, mec="white", mew=0.7, zorder=6)
    # categorical: same effect axis, but its restrictiveness is on a different scale -> hollow
    axB.plot([0.522], [rows["categorical"][0]], "o", ms=5, mfc="white", mec=INK, mew=0.9, zorder=6)

    axB.annotate("Exclusion", (BITES["exclusion"], rows["exclusion"][0]),
                 xytext=(BITES["exclusion"], rows["exclusion"][0] - 0.011),
                 fontsize=7, color=MUTED, ha="center", va="top")
    # Asterisk rather than a legend entry: a legend swatch here reads as another data point.
    axB.annotate("Categorical$^{*}$", (0.522, rows["categorical"][0]),
                 xytext=(0.552, rows["categorical"][0] + 0.002),
                 fontsize=7, color=MUTED, ha="left", va="bottom")
    axB.annotate("Inclusion\n(common)", (BITES["inclusion"], rows["inclusion"][0]),
                 xytext=(BITES["inclusion"] - 0.012, rows["inclusion"][0] - 0.009),
                 fontsize=7, color=MUTED, ha="center", va="top", linespacing=1.15)

    # The matched-difficulty contrast: two constraints ruling out the same share of default
    # behaviour, differenced against each other on shared endpoints -- so the gap is type, not
    # restrictiveness. This bracket is the panel's actual claim.
    x_r, x_o = BITES["inclusion_rare"], BITES["ordering"]
    y_r, y_o = rows["inclusion_rare"][0], rows["ordering"][0]
    xb = 1.075
    for xs, ys in ((x_r, y_r), (x_o, y_o)):
        axB.plot([xs + 0.012, xb], [ys, ys], color=ACCENT, lw=0.6, ls=(0, (3, 2)), zorder=4)
    axB.annotate("", xy=(xb, y_o), xytext=(xb, y_r),
                 arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=0.9,
                                 shrinkA=0, shrinkB=0, mutation_scale=7))
    axB.text(xb + 0.022, (y_r + y_o) / 2, "$-0.100$\n$p\\!<\\!10^{-17}$", color=ACCENT,
             fontsize=7, ha="left", va="center", linespacing=1.4)
    axB.annotate("Inclusion\n(rare)", (x_r, y_r), xytext=(x_r - 0.028, y_r + 0.004),
                 fontsize=7, color=MUTED, ha="right", va="bottom", linespacing=1.15)
    axB.annotate("Ordering", (x_o, y_o), xytext=(x_o - 0.028, y_o),
                 fontsize=7, color=WARN, ha="right", va="center")

    axB.set_xlim(0.44, 1.30)
    axB.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axB.set_ylim(-0.212, 0.028)
    axB.spines["bottom"].set_bounds(0.44, 1.02)
    axB.set_xlabel("Share of models' default paths ruled out")
    axB.set_ylabel("Paired $\\Delta$ creativity")
    axB.set_title("(b)", loc="left", color=INK, fontweight="bold", pad=3)
    axB.text(0.02, 0.03, "$^{*}$restrictiveness on a different scale", transform=axB.transAxes,
             fontsize=6.5, color=MUTED, style="italic", ha="left", va="bottom")

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f = outdir / f"fig_creativity_by_constraint.{ext}"
        fig.savefig(f, dpi=400, facecolor="white")
        print(f"saved {f}")

    print("\nvalues drawn:")
    for m in MODES:
        mean, lo, hi, frac, n = rows[m]
        p = wilcoxon(diffs[m])
        print(f"  {LABEL[m]:24s} Δ={mean:+.4f} [{lo:+.4f},{hi:+.4f}] harmed={100*frac:.1f}% "
              f"n={n} p={p:.2e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_regimeA_all",
         sys.argv[2] if len(sys.argv) > 2 else "papers/kg_creat-iclr/media")
