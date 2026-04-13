"""Generate camera-ready figures for the ICCC paper.

Uses the correlation_analysis.json and all_scores.json from the scoring pipeline
plus benchmarks.json to produce four publication-quality figures:

    fig1_correlation_matrix.pdf  — metrics × benchmarks heatmap with significance
    fig2_pace_scatter.pdf        — PACE vs Arena CW with model labels
    fig3_hivemind_direction.pdf  — creativity metrics vs Hivemind (2 dimensions)
    fig4_cdat_by_temperature.pdf — CDAT temperature sensitivity

Usage:
    uv run python src/dat_eval/scripts/make_figures.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cmcrameri import cm as cmc

# --- Camera-ready styling ---
mpl.rcParams.update({
    # Clean sans-serif — Helvetica is the scientific-publication standard.
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica:italic",
    "mathtext.bf": "Helvetica:bold",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "pdf.fonttype": 42,   # TrueType fonts (editable in Illustrator)
    "ps.fonttype": 42,
})

# --- Batlow (Crameri) color scheme ---
# Sequential perceptually-uniform colormap, used throughout for consistency.
# For diverging needs (correlation heatmap), we use `vik` — the Crameri diverging
# counterpart — which pairs visually with batlow while giving the +/- signal the
# dedicated hue contrast a correlation plot needs.
CMAP_SEQ = cmc.batlow       # sequential (for gradient fills)
CMAP_DIV = cmc.vik          # diverging  (for correlation heatmap with natural zero)
CMAP_CAT = cmc.batlowS      # categorical (100 distinct colors sampled from batlow)

# Pick 4 well-separated categorical colors from Batlow for the four metrics.
# Sampling at low-mid-high of the perceptual range gives distinct-but-harmonious tones.
_BATLOW_SAMPLES = CMAP_SEQ(np.linspace(0.10, 0.88, 4))
C_DAT     = _BATLOW_SAMPLES[0]   # deep purple-blue
C_CNOV    = _BATLOW_SAMPLES[1]   # teal
C_CAPP    = _BATLOW_SAMPLES[2]   # olive-green
C_PACE    = _BATLOW_SAMPLES[3]   # warm yellow

# Legacy aliases used by old plotting code paths — kept consistent with Batlow.
C_BLUE   = C_PACE      # "highlight" color: warm yellow from Batlow's bright end
C_ORANGE = _BATLOW_SAMPLES[1]
C_GREEN  = _BATLOW_SAMPLES[2]
C_RED    = CMAP_DIV(0.92)   # vik's red end for trend lines (paired with batlow)
C_PURPLE = _BATLOW_SAMPLES[0]
C_GREY   = "#4d4d4d"


FIGS_DIR = Path(__file__).parent.parent.parent.parent / "docs" / "reports" / "2026-04-12_preliminary_correlations" / "figures"
RESULTS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "dat_eval" / "run_v1" / "downstream" / "scores_v1" / "results"
BENCH_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "comb_eval" / "benchmarks.json"


def load_data():
    with open(RESULTS_DIR / "correlation_analysis.json") as f:
        corr = json.load(f)
    with open(RESULTS_DIR / "all_scores.json") as f:
        scores = json.load(f)
    with open(BENCH_PATH) as f:
        benchmarks = json.load(f)
    return corr, scores, benchmarks


def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


# --- Figure 1: correlation matrix heatmap ---
def fig1_correlation_matrix(corr):
    """Heatmap: rows = metrics, cols = benchmarks.

    Cells show Spearman rho with significance stars. Color = rho, -1 (red) to
    +1 (blue). Annotated with n in footnote.
    """
    metrics = ["dat", "cdat_novelty", "cdat_appropriateness", "pace"]
    metric_labels = ["DAT", "CDAT Novelty", "CDAT Appropriateness", "PACE"]

    benchmarks_col = [
        ("vs_arena_cw", "Arena\nCreative\nWriting"),
        ("partial_cw_control_overall", "Arena CW\n| Overall\n(partial)"),
        ("vs_eq_bench_cw", "EQ-Bench\nCreative\nWriting"),
        ("vs_hivemind_intra_sim", "Hivemind\nIntra-Sim\n(neg. expected)"),
    ]

    n_m = len(metrics)
    n_b = len(benchmarks_col)

    rho_matrix = np.full((n_m, n_b), np.nan)
    p_matrix = np.full((n_m, n_b), np.nan)
    n_matrix = np.full((n_m, n_b), 0, dtype=int)

    for i, m in enumerate(metrics):
        for j, (bkey, _) in enumerate(benchmarks_col):
            entry = corr.get(m, {}).get(bkey)
            if entry is None:
                continue
            rho_matrix[i, j] = entry["spearman_rho"]
            p_matrix[i, j] = entry["p_value"]
            n_matrix[i, j] = entry.get("n_models", 0)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    im = ax.imshow(rho_matrix, vmin=-1, vmax=1, cmap=CMAP_DIV, aspect="auto")

    ax.set_xticks(range(n_b))
    ax.set_xticklabels([b[1] for b in benchmarks_col], fontsize=9)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(metric_labels, fontsize=10)

    # Cell labels
    for i in range(n_m):
        for j in range(n_b):
            rho = rho_matrix[i, j]
            if np.isnan(rho):
                continue
            p = p_matrix[i, j]
            n = n_matrix[i, j]
            stars = sig_stars(p)
            color = "white" if abs(rho) > 0.5 else "black"
            label = f"{rho:+.2f}{stars}\nn={n}"
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8.5)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Spearman $\\rho$", fontsize=10)

    ax.set_title("Creativity metrics × benchmarks (Spearman $\\rho$)", fontsize=11, pad=10)

    fig.text(0.01, -0.02, "* p < .05    ** p < .01    *** p < .001", fontsize=8, color=C_GREY)

    out = FIGS_DIR / "fig1_correlation_matrix.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


def _short_label(mk: str) -> str:
    return (mk.replace("anthropic_", "").replace("openai_", "")
              .replace("meta-llama_", "").replace("google_", "")
              .replace("qwen_", "").replace("mistralai_", "")
              .replace("deepseek_", "").replace("cohere_", "")
              .replace("nvidia_", "").replace("microsoft_", ""))


def _scatter_panel(
    ax,
    scores,
    benchmarks,
    metric_key: str,
    metric_label: str,
    benchmark_key: str = "arena_cw",
    benchmark_label: str = "Arena Creative Writing Elo",
    point_color: str = C_BLUE,
    n_labels: int = 6,
):
    """Draw a single metric-vs-benchmark scatter panel on the given ax."""
    from scipy.stats import spearmanr
    xs, ys, labels = [], [], []
    for mk, sc in scores.items():
        val = sc.get(metric_key)
        if val is None or val == 0:
            continue
        if mk not in benchmarks or benchmark_key not in benchmarks[mk]:
            continue
        xs.append(val)
        ys.append(benchmarks[mk][benchmark_key])
        labels.append(_short_label(mk))

    if not xs:
        ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                transform=ax.transAxes, color=C_GREY)
        ax.set_xlabel(metric_label)
        ax.set_ylabel(benchmark_label)
        return

    xs = np.array(xs)
    ys = np.array(ys)

    ax.scatter(xs, ys, s=28, color=point_color, alpha=0.78,
               edgecolor="white", linewidth=0.5, zorder=3)

    rho, pval = spearmanr(xs, ys)

    # Linear fit (cosmetic)
    order = np.argsort(xs)
    ax.plot(xs[order], np.poly1d(np.polyfit(xs, ys, 1))(xs[order]),
            color=C_RED, linewidth=1.2, linestyle="--", alpha=0.65, zorder=2)

    # Light label set: top/bottom by each axis
    label_set = set()
    label_set.add(int(np.argmax(xs)))
    label_set.add(int(np.argmin(xs)))
    label_set.add(int(np.argmax(ys)))
    label_set.add(int(np.argmin(ys)))
    # Plus evenly-spaced samples along the Arena CW axis
    sorted_by_y = np.argsort(ys)
    step = max(1, len(ys) // max(1, n_labels - 2))
    label_set.update(sorted_by_y[::step].tolist())

    x_med = np.median(xs)
    y_med = np.median(ys)
    for idx in label_set:
        lx, ly = xs[idx], ys[idx]
        # Pick offset direction based on quadrant
        if lx > x_med and ly > y_med:
            dx, dy, ha, va = -4, 4, "right", "bottom"
        elif lx > x_med:
            dx, dy, ha, va = 4, -4, "left", "top"
        elif ly > y_med:
            dx, dy, ha, va = 4, 4, "left", "bottom"
        else:
            dx, dy, ha, va = 4, -4, "left", "top"
        ax.annotate(labels[idx], (lx, ly),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.8, color=C_GREY,
                    ha=ha, va=va,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.3))

    ax.set_xlabel(metric_label)
    ax.set_ylabel(benchmark_label)

    sig = sig_stars(pval)
    ax.set_title(f"{metric_label}   $\\rho$ = {rho:+.3f}{sig}  (n={len(xs)})",
                 fontsize=10)

    # Breathing room
    xr = xs.max() - xs.min()
    if xr > 0:
        ax.set_xlim(xs.min() - 0.05 * xr, xs.max() + 0.05 * xr)


# Panel-color mapping shared by all metric grids — Batlow-sampled
_METRIC_PANELS = [
    ("dat",                 "DAT score",              C_DAT),
    ("cdat_novelty",        "CDAT Novelty",           C_CNOV),
    ("cdat_appropriateness","CDAT Appropriateness",   C_CAPP),
    ("pace",                "PACE score",             C_PACE),
]


def _grid_scatter(scores, benchmarks, benchmark_key, benchmark_label,
                   suptitle, outname, suptitle_y=1.005):
    """Render a 2x2 scatter grid for the four metrics against the given benchmark."""
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    flat_axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    for ax, (metric, label, color) in zip(flat_axes, _METRIC_PANELS):
        _scatter_panel(
            ax, scores, benchmarks, metric, label,
            benchmark_key=benchmark_key,
            benchmark_label=benchmark_label,
            point_color=color,
        )

    fig.suptitle(suptitle, fontsize=12, y=suptitle_y)
    fig.tight_layout()

    out = FIGS_DIR / outname
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


def fig2_all_metrics_scatter(scores, benchmarks):
    _grid_scatter(
        scores, benchmarks,
        benchmark_key="arena_cw",
        benchmark_label="Arena Creative Writing Elo",
        suptitle="Creativity metrics vs Arena Creative Writing",
        outname="fig2_all_metrics_scatter.pdf",
    )


def fig2b_all_metrics_vs_eqbench(scores, benchmarks):
    _grid_scatter(
        scores, benchmarks,
        benchmark_key="eq_bench_cw",
        benchmark_label="EQ-Bench Creative Writing Elo",
        suptitle="Creativity metrics vs EQ-Bench Creative Writing",
        outname="fig2b_all_metrics_vs_eqbench.pdf",
    )


def fig2c_all_metrics_vs_hivemind(scores, benchmarks):
    _grid_scatter(
        scores, benchmarks,
        benchmark_key="hivemind_intra_sim",
        benchmark_label="Hivemind intra-model similarity\n(higher = more homogeneous)",
        suptitle=("Creativity metrics vs Hivemind homogeneity\n"
                  "(valid creativity metrics should correlate NEGATIVELY)"),
        outname="fig2c_all_metrics_vs_hivemind.pdf",
        suptitle_y=1.015,
    )


# --- Figure 2 (combined): 4 metrics x 3 benchmarks grid ---
def fig2_combined_grid(scores, benchmarks):
    """Single figure: rows = metrics (DAT / CDAT Nov / CDAT App / PACE),
    columns = benchmarks (Arena CW / EQ-Bench CW / Hivemind).

    Layout convention: x-axis = metric (the predictor), y-axis = benchmark
    (the outcome). This means every panel in a given row shares the same
    x-axis scale (same metric); every panel in a given column shares the
    same y-axis scale (same benchmark). We therefore share axes within
    rows and columns, hide redundant tick labels, and place metric/benchmark
    labels only at grid edges.
    """
    from scipy.stats import spearmanr
    column_specs = [
        ("arena_cw",            "Arena CW Elo"),
        ("eq_bench_cw",         "EQ-Bench CW Elo"),
        ("mazur_cw_v2",         "Mazur CW v2 (0-10)"),
        ("hivemind_intra_sim",  "Hivemind intra-similarity"),
    ]
    n_cols = len(column_specs)
    n_rows = len(_METRIC_PANELS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 9.0),
                              sharex="row", sharey="col")

    for row_idx, (metric_key, metric_label, color) in enumerate(_METRIC_PANELS):
        for col_idx, (bench_key, bench_label) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]

            # Collect paired data for this metric / benchmark combo
            xs, ys, labels = [], [], []
            for mk, sc in scores.items():
                val = sc.get(metric_key)
                if val is None or val == 0:
                    continue
                if mk not in benchmarks or bench_key not in benchmarks[mk]:
                    continue
                xs.append(val)
                ys.append(benchmarks[mk][bench_key])
                labels.append(_short_label(mk))

            if not xs:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                        transform=ax.transAxes, color=C_GREY, fontsize=9)
                continue

            xs = np.array(xs)
            ys = np.array(ys)

            ax.scatter(xs, ys, s=22, color=color, alpha=0.78,
                       edgecolor="white", linewidth=0.4, zorder=3)

            rho, pval = spearmanr(xs, ys)
            order = np.argsort(xs)
            ax.plot(xs[order], np.poly1d(np.polyfit(xs, ys, 1))(xs[order]),
                    color=C_RED, linewidth=1.0, linestyle="--",
                    alpha=0.6, zorder=2)

            # Per-panel stat readout in the corner
            stars = sig_stars(pval)
            ax.text(0.03, 0.97,
                    f"$\\rho$ = {rho:+.2f}{stars}\n$n$ = {len(xs)}",
                    transform=ax.transAxes, fontsize=8.0,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=1.5))

            # Label 4 extreme models per panel so the reader can anchor which
            # dots are which. Pick the top/bottom on both axes, deduped.
            idx_set = set()
            idx_set.add(int(np.argmax(ys)))   # best on benchmark
            idx_set.add(int(np.argmin(ys)))   # worst on benchmark
            idx_set.add(int(np.argmax(xs)))   # highest metric
            idx_set.add(int(np.argmin(xs)))   # lowest metric

            x_med = np.median(xs)
            y_med = np.median(ys)
            for idx in idx_set:
                lx, ly = xs[idx], ys[idx]
                if lx >= x_med and ly >= y_med:
                    dx, dy, ha, va = -3, 3, "right", "bottom"
                elif lx >= x_med:
                    dx, dy, ha, va = -3, -3, "right", "top"
                elif ly >= y_med:
                    dx, dy, ha, va = 3, 3, "left", "bottom"
                else:
                    dx, dy, ha, va = 3, -3, "left", "top"
                ax.annotate(
                    labels[idx], (lx, ly),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=5.8, color=C_GREY, ha=ha, va=va,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.6, pad=0.3),
                    zorder=4,
                )

            # No per-panel axis labels — we use row/column titles instead.
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=8)

    # Column titles (benchmark names) on the top row.
    for col_idx, (_, bench_label) in enumerate(column_specs):
        axes[0, col_idx].set_title(bench_label, fontsize=10.5, pad=8,
                                    weight="bold")

    # Row titles (metric names) on the far left, outside the axes.
    for row_idx, (_, metric_label, _) in enumerate(_METRIC_PANELS):
        axes[row_idx, 0].annotate(
            metric_label,
            xy=(-0.28, 0.5), xycoords="axes fraction",
            ha="center", va="center",
            fontsize=10.5, weight="bold", rotation=90,
        )

    fig.suptitle(
        "Creativity metrics (rows) vs external benchmarks (columns)",
        fontsize=11, y=1.003,
    )
    fig.tight_layout(rect=[0.05, 0, 1, 0.99])

    out = FIGS_DIR / "fig2_combined_grid.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


# --- Figure 3: Hivemind direction check ---
def fig3_hivemind(corr):
    """Bar chart showing signed correlation of each creativity metric with
    Hivemind intra-model similarity. A valid creativity metric should be NEGATIVE
    (higher creativity = more diverse output = lower intra-model similarity).

    Positive bars indicate metrics that are anti-correlated with diversity.
    """
    bars = [
        ("DAT", "dat"),
        ("CDAT Novelty", "cdat_novelty"),
        ("CDAT Appropriateness", "cdat_appropriateness"),
        ("PACE", "pace"),
    ]

    # Use the two ends of the diverging Crameri `vik` map to encode direction:
    # cool (blue) end = correct direction (negative, predicts diversity);
    # warm (red) end = wrong direction (positive, predicts homogeneity).
    col_correct = CMAP_DIV(0.15)
    col_wrong = CMAP_DIV(0.85)

    rhos, ps, ns, colors = [], [], [], []
    for _, key in bars:
        entry = corr.get(key, {}).get("vs_hivemind_intra_sim")
        if entry is None:
            rhos.append(0); ps.append(1); ns.append(0); colors.append(C_GREY)
            continue
        rho = entry["spearman_rho"]
        rhos.append(rho)
        ps.append(entry["p_value"])
        ns.append(entry["n_models"])
        if rho < -0.1:
            colors.append(col_correct)
        elif rho > 0.1:
            colors.append(col_wrong)
        else:
            colors.append(C_GREY)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    xpos = np.arange(len(bars))
    bars_plotted = ax.bar(xpos, rhos, color=colors, alpha=0.92, edgecolor="white", linewidth=0.6)

    # Place labels above bars (for positive bars) or below bars (for negative bars)
    # using a constant offset from the zero line for consistency.
    for i, (rho, p, n) in enumerate(zip(rhos, ps, ns)):
        stars = sig_stars(p)
        if rho >= 0:
            y = rho + 0.04
            va = "bottom"
        else:
            y = rho - 0.04
            va = "top"
        ax.text(i, y, f"{rho:+.2f}{stars}\nn={n}",
                ha="center", va=va, fontsize=8.5,
                color=colors[i] if abs(rho) > 0.1 else C_GREY,
                weight="bold")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(-0.1, color=C_GREY, linewidth=0.3, linestyle=":", alpha=0.5)
    ax.axhline(0.1, color=C_GREY, linewidth=0.3, linestyle=":", alpha=0.5)

    ax.set_xticks(xpos)
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("Spearman $\\rho$ vs Hivemind intra-model similarity")
    ax.set_ylim(-0.85, 0.85)
    ax.set_title("Do creativity metrics predict output diversity?")

    # Direction guide, placed in the right margin away from bars
    ax.text(1.02, -0.35, "$\\downarrow$ negative = correlates\nwith diverse output\n(expected for a\ncreativity metric)",
            transform=ax.get_yaxis_transform(), fontsize=8,
            color=col_correct, va="top", ha="left")
    ax.text(1.02, 0.35, "$\\uparrow$ positive = correlates\nwith homogeneous output\n(unexpected for a\ncreativity metric)",
            transform=ax.get_yaxis_transform(), fontsize=8,
            color=col_wrong, va="bottom", ha="left")

    out = FIGS_DIR / "fig3_hivemind_direction.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


# --- Figure 4: CDAT by temperature ---
def fig4_cdat_by_temperature(corr):
    """How CDAT correlations change across temperatures (1.0, 1.5, 2.0).

    Shows the novelty-appropriateness tradeoff and the Hivemind correlation at
    each temperature. Illustrates that higher temp doesn't fix the problem.
    """
    temps = ["1.0", "1.5", "2.0"]
    n_temps = len(temps)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)

    # Panel A: vs Arena CW
    ax = axes[0]
    approp_rhos = []
    novelty_rhos = []
    approp_ps = []
    novelty_ps = []
    for t in temps:
        a_entry = corr.get(f"cdat_approp_t{t}", {}).get("vs_arena_cw", {})
        n_entry = corr.get(f"cdat_novelty_t{t}", {}).get("vs_arena_cw", {})
        approp_rhos.append(a_entry.get("spearman_rho", 0))
        novelty_rhos.append(n_entry.get("spearman_rho", 0))
        approp_ps.append(a_entry.get("p_value", 1))
        novelty_ps.append(n_entry.get("p_value", 1))

    xpos = np.arange(n_temps)
    w = 0.35
    ax.bar(xpos - w/2, approp_rhos, w, color=C_CAPP, label="Appropriateness", alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.bar(xpos + w/2, novelty_rhos, w, color=C_CNOV, label="Novelty", alpha=0.9, edgecolor="white", linewidth=0.5)
    for i, (a, n, pa, pn) in enumerate(zip(approp_rhos, novelty_rhos, approp_ps, novelty_ps)):
        a_y = a + 0.035 if a >= 0 else a - 0.035
        a_va = "bottom" if a >= 0 else "top"
        ax.text(i - w/2, a_y, f"{a:+.2f}{sig_stars(pa)}",
                ha="center", va=a_va, fontsize=8, color=C_CAPP, weight="bold")
        n_y = n + 0.035 if n >= 0 else n - 0.035
        n_va = "bottom" if n >= 0 else "top"
        ax.text(i + w/2, n_y, f"{n:+.2f}{sig_stars(pn)}",
                ha="center", va=n_va, fontsize=8, color=C_CNOV, weight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"$T={t}$" for t in temps])
    ax.set_ylabel("Spearman $\\rho$")
    ax.set_title("CDAT vs Arena Creative Writing", fontsize=10)
    ax.legend(loc="lower left", frameon=False, fontsize=8)

    # Panel B: vs Hivemind
    ax = axes[1]
    approp_rhos = []
    novelty_rhos = []
    approp_ps = []
    novelty_ps = []
    for t in temps:
        a_entry = corr.get(f"cdat_approp_t{t}", {}).get("vs_hivemind_intra_sim", {})
        n_entry = corr.get(f"cdat_novelty_t{t}", {}).get("vs_hivemind_intra_sim", {})
        approp_rhos.append(a_entry.get("spearman_rho", 0))
        novelty_rhos.append(n_entry.get("spearman_rho", 0))
        approp_ps.append(a_entry.get("p_value", 1))
        novelty_ps.append(n_entry.get("p_value", 1))

    ax.bar(xpos - w/2, approp_rhos, w, color=C_CAPP, label="Appropriateness", alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.bar(xpos + w/2, novelty_rhos, w, color=C_CNOV, label="Novelty", alpha=0.9, edgecolor="white", linewidth=0.5)
    for i, (a, n, pa, pn) in enumerate(zip(approp_rhos, novelty_rhos, approp_ps, novelty_ps)):
        a_y = a + 0.035 if a >= 0 else a - 0.035
        a_va = "bottom" if a >= 0 else "top"
        ax.text(i - w/2, a_y, f"{a:+.2f}{sig_stars(pa)}",
                ha="center", va=a_va, fontsize=8, color=C_CAPP, weight="bold")
        n_y = n + 0.035 if n >= 0 else n - 0.035
        n_va = "bottom" if n >= 0 else "top"
        ax.text(i + w/2, n_y, f"{n:+.2f}{sig_stars(pn)}",
                ha="center", va=n_va, fontsize=8, color=C_CNOV, weight="bold")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"$T={t}$" for t in temps])
    ax.set_title("CDAT vs Hivemind homogeneity", fontsize=10)

    axes[0].set_ylim(-0.85, 0.85)

    fig.suptitle("CDAT sub-scores across temperature", fontsize=11, y=1.01)
    fig.text(0.01, -0.04, "* p < .05    ** p < .01    *** p < .001", fontsize=8, color=C_GREY)

    out = FIGS_DIR / "fig4_cdat_by_temperature.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


def fig_inter_metric_triangle(corr):
    """Triangular heatmap of inter-metric Spearman correlations.

    Shows the lower triangle of the 4x4 metric-by-metric correlation matrix
    (DAT, CDAT Novelty, CDAT Appropriateness, PACE). Upper triangle is
    blanked out because it would be redundant. Cells annotated with the
    Spearman rho and significance stars.
    """
    metrics = ["dat", "cdat_novelty", "cdat_appropriateness", "pace"]
    labels = ["DAT", "CDAT Nov.", "CDAT App.", "PACE"]
    n = len(metrics)

    # Pull pairwise values from corr["inter_metric"]
    inter = corr.get("inter_metric", {})
    mat = np.full((n, n), np.nan)
    pmat = np.full((n, n), np.nan)
    for i, mi in enumerate(metrics):
        for j, mj in enumerate(metrics):
            if i == j:
                mat[i, j] = 1.0
                pmat[i, j] = 0.0
                continue
            if i < j:
                continue  # upper triangle — blank
            key_a = f"{mj}_vs_{mi}"
            key_b = f"{mi}_vs_{mj}"
            entry = inter.get(key_a) or inter.get(key_b)
            if entry is None:
                continue
            mat[i, j] = entry["spearman_rho"]
            pmat[i, j] = entry["p_value"]

    # Mask upper triangle for plotting
    display = np.where(np.isnan(mat), 0.0, mat)
    mask = np.isnan(mat)

    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    im = ax.imshow(display, vmin=-1, vmax=1, cmap=CMAP_SEQ, aspect="equal")

    # White-out upper triangle (and diagonal if desired; keeping diagonal
    # shaded to indicate self-identity)
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                            color="white", zorder=2))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate lower triangle with rho and stars
    for i in range(n):
        for j in range(n):
            if i < j or np.isnan(mat[i, j]):
                continue
            if i == j:
                txt = "—"
                color = "black"
            else:
                stars = sig_stars(pmat[i, j])
                txt = f"{mat[i, j]:+.2f}{stars}"
                # Batlow is dark at low values (purple-blue), bright at high
                # (yellow). Use white for dark cells (rho < ~-0.2), black
                # otherwise.
                color = "white" if mat[i, j] < -0.2 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=color, zorder=3)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label("Spearman $\\rho$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Inter-metric correlations", fontsize=9.5, pad=4)
    ax.tick_params(axis="both", which="major", length=0)

    fig.tight_layout()
    out = FIGS_DIR / "fig_inter_metric.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    corr, scores, benchmarks = load_data()

    fig1_correlation_matrix(corr)
    fig2_combined_grid(scores, benchmarks)
    fig4_cdat_by_temperature(corr)
    fig_inter_metric_triangle(corr)

    print(f"\nAll figures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
