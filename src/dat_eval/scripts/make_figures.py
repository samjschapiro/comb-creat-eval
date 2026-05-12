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
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
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

# Per-test categorical palette (Okabe-Ito — colorblind-safe, picked so DAT
# and CDAT are maximally separated rather than adjacent samples of a
# sequential map). Five entries covering the full headline-figure test set:
# DAT, CDAT (gated), CDAT-N (ungated novelty), CDAT-A (ungated approp.), PACE.
C_DAT   = "#0072B2"   # blue
C_CDAT  = "#D55E00"   # vermillion (gated CDAT)
C_CNOV  = "#009E73"   # bluish green (CDAT novelty)
C_CAPP  = "#CC79A7"   # reddish purple (CDAT appropriateness)
C_PACE  = "#E69F00"   # orange

# Legacy aliases.
C_BLUE   = C_DAT
C_ORANGE = C_PACE
C_GREEN  = C_CNOV
C_RED    = CMAP_DIV(0.92)   # vik's red end for trend lines
C_PURPLE = C_CAPP
C_GREY   = "#4d4d4d"


FIGS_DIR = Path(__file__).parent.parent.parent.parent / "docs" / "reports" / "2026-04-12_preliminary_correlations" / "figures"
PAPER_FIGS_DIR = Path(__file__).parent.parent.parent.parent / "papers" / "iccc-2026" / "figures"
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


def load_composite_scores() -> dict:
    """Build per-model composite metric scores by averaging across embeddings.

    For each embedder in multi_embed_scores.json, z-score its metric values
    across models, then for each model take the mean z-score across all three
    embedders. Returns the same shape as all_scores.json (model -> metric ->
    composite score) so existing plotting code can use it interchangeably.
    """
    me_path = RESULTS_DIR / "multi_embed_scores.json"
    if not me_path.exists():
        return {}
    with open(me_path) as f:
        me = json.load(f)
    embs = sorted(me.keys())
    tasks = ["dat", "cdat", "cdat_novelty", "cdat_appropriateness", "pace"]
    models = sorted({m for emb in embs for m in me[emb]})

    composite: dict[str, dict[str, float]] = {}
    for t in tasks:
        stats = {}
        for emb in embs:
            vals = [me[emb].get(m, {}).get(t) for m in models]
            vals = [v for v in vals if v is not None
                    and not (isinstance(v, float) and (np.isnan(v) or v == 0))]
            if not vals:
                continue
            stats[emb] = (float(np.mean(vals)), float(np.std(vals)) or 1.0)
        for m in models:
            zs = []
            for emb in embs:
                if emb not in stats:
                    continue
                v = me[emb].get(m, {}).get(t)
                if v is None or (isinstance(v, float) and (np.isnan(v) or v == 0)):
                    continue
                mean, std = stats[emb]
                zs.append((v - mean) / std)
            if zs:
                composite.setdefault(m, {})[t] = float(np.mean(zs))
    return composite


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


def _family(mk: str) -> str:
    """Map a model key to its provider/family label."""
    if mk.startswith("anthropic_"):   return "Anthropic"
    if mk.startswith("openai_"):      return "OpenAI"
    if mk.startswith("google_"):      return "Google"
    if mk.startswith("meta-llama_"):  return "Meta"
    if mk.startswith("mistralai_"):   return "Mistral"
    if mk.startswith("qwen_"):        return "Qwen"
    if mk.startswith("deepseek_"):    return "DeepSeek"
    if mk.startswith("cohere_"):      return "Cohere"
    if mk.startswith("nvidia_"):      return "NVIDIA"
    if mk.startswith("microsoft_"):   return "Microsoft"
    return "Other"


# Family order used when sorting models into the legend so related
# models appear together (and get visually similar hues from the
# perceptually-smooth Batlow palette).
_FAMILY_ORDER = [
    "OpenAI", "Anthropic", "Google", "Meta", "Mistral",
    "Qwen", "DeepSeek", "Cohere", "NVIDIA", "Microsoft",
]


def _build_model_color_map(model_keys) -> dict:
    """Assign each model a unique colour from the Batlow categorical palette.
    Models are ordered by family (then alphabetically within family) so
    adjacent colours land on related models.
    """
    ordered = sorted(
        model_keys,
        key=lambda mk: (
            _FAMILY_ORDER.index(_family(mk)) if _family(mk) in _FAMILY_ORDER else len(_FAMILY_ORDER),
            mk,
        ),
    )
    n = len(ordered)
    samples = CMAP_CAT(np.linspace(0.0, 1.0, n))
    return {mk: samples[i] for i, mk in enumerate(ordered)}


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
    ("cdat",                "CDAT (gated novelty)",   C_CDAT),
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
    """Semi-partial scatter grid using composite (across-embedding) scores.

    Rows = creativity metrics (DAT / CDAT gated / PACE).
    Columns = benchmarks (Arena CW / EQ-B. / Mazur / Hivemind diversity).

    Per-model metric values are composite z-scores across GloVe, FastText,
    and SBERT (the 'Overall' block of Table~\\ref{tab:correlations}).
    Each panel plots raw metric vs benchmark residualised against
    [Arena Overall, MMLU-Pro]. Panel stat is the semi-partial Pearson
    r — matches the Overall block of Table~\\ref{tab:correlations}.
    """
    from scipy.stats import pearsonr

    composite = load_composite_scores()
    if composite:
        scores = composite
    column_specs = [
        ("arena_cw",            "Arena CW (capability-adjusted)"),
        ("eq_bench_cw",         "EQ-Bench CW (capability-adjusted)"),
        ("mazur_cw_v2",         "Mazur CW v2 (capability-adjusted)"),
        ("hivemind_diversity",  "Hivemind diversity (capability-adjusted)"),
    ]
    n_cols = len(column_specs)
    n_rows = len(_METRIC_PANELS)

    model_color = _build_model_color_map(scores.keys())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 11.5),
                              sharex="row", sharey="col")

    models_seen = set()
    for row_idx, (metric_key, metric_label, _color) in enumerate(_METRIC_PANELS):
        for col_idx, (bench_key, bench_label) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]

            # Collect paired data for this metric / benchmark combo,
            # restricted to models with BOTH capability proxies available.
            xs, ys, ao, mp, labels, mks = [], [], [], [], [], []
            for mk, sc in scores.items():
                val = sc.get(metric_key)
                if val is None or val == 0:
                    continue
                if mk not in benchmarks or bench_key not in benchmarks[mk]:
                    continue
                bench_row = benchmarks[mk]
                ao_v = bench_row.get("arena_overall")
                mp_v = bench_row.get("mmlu_pro")
                if ao_v is None or mp_v is None:
                    continue
                xs.append(val)
                ys.append(bench_row[bench_key])
                ao.append(ao_v)
                mp.append(mp_v)
                labels.append(_short_label(mk))
                mks.append(mk)
                models_seen.add(mk)

            if len(xs) < 5:
                ax.text(0.5, 0.5, "(n < 5)", ha="center", va="center",
                        transform=ax.transAxes, color=C_GREY, fontsize=9)
                continue

            xs_raw = np.array(xs)
            ys_raw = np.array(ys)
            Z = np.column_stack([np.ones(len(xs_raw)),
                                  np.array(ao), np.array(mp)])

            # Semi-partial: residualise only the benchmark (Y); keep metric raw
            xs = xs_raw
            beta_y, *_ = np.linalg.lstsq(Z, ys_raw, rcond=None)
            ys = ys_raw - Z @ beta_y

            point_colors = [model_color[mk] for mk in mks]
            ax.scatter(xs, ys, s=28, c=point_colors, alpha=0.9,
                       edgecolor="white", linewidth=0.4, zorder=3)

            # Semi-partial Pearson r: raw metric vs benchmark residual
            rho, pval = pearsonr(xs, ys)
            order = np.argsort(xs)
            ax.plot(xs[order], np.poly1d(np.polyfit(xs, ys, 1))(xs[order]),
                    color=C_RED, linewidth=1.0, linestyle="--",
                    alpha=0.6, zorder=2)

            # Y=0 reminds the reader the benchmark axis is a residual
            ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.6, zorder=1)

            # Per-panel stat readout in the corner: semi-partial Pearson r.
            stars = sig_stars(pval)
            ax.text(0.03, 0.97,
                    f"$r_{{\\mathrm{{semi}}}} = {rho:+.2f}{stars}$\n$n$ = {len(xs)}",
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

    # Shared per-model legend at the bottom: one entry per model that
    # actually appears in at least one panel, ordered by family.
    models_for_legend = sorted(
        models_seen,
        key=lambda mk: (
            _FAMILY_ORDER.index(_family(mk)) if _family(mk) in _FAMILY_ORDER else len(_FAMILY_ORDER),
            mk,
        ),
    )
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=model_color[mk], markersize=6,
                   markeredgecolor="white", markeredgewidth=0.4,
                   label=_short_label(mk))
        for mk in models_for_legend
    ]
    n_cols_legend = 5
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.005),
               ncol=n_cols_legend, frameon=False, fontsize=10,
               handletextpad=0.4, columnspacing=1.4,
               labelspacing=0.35)

    fig.tight_layout(rect=[0.05, 0.18, 1, 1])

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


def fig_correlation_summary_heatmap(corr):
    """Heatmap replacement for the tabular correlation summary.

    Layout: 4 metrics (rows) x 8 cells (cols), where the 8 cols alternate
    Simple and joint-Partial for each of the four benchmarks (Arena CW,
    EQ-B., Mazur, Hivemind diversity). Each cell is coloured by Spearman rho
    and annotated with the rho and its significance stars. The joint partial
    uses Arena Overall + MMLU-Pro as controls.
    """
    metrics = ["dat", "cdat_novelty", "cdat_appropriateness", "pace"]
    metric_labels = ["DAT", "CDAT Nov.", "CDAT App.", "PACE"]

    benchmarks = [
        ("arena_cw",           "vs_arena_cw",           "partial_cw_control_both",       "Arena"),
        ("eq_bench_cw",        "vs_eq_bench_cw",        "partial_eqbench_control_both",  "EQ-B."),
        ("mazur_cw_v2",        "vs_mazur_cw",           "partial_mazur_control_both",    "Mazur"),
        ("hivemind_diversity", "vs_hivemind_diversity", "partial_hivemind_control_both", "Hive."),
    ]

    n_metrics = len(metrics)
    n_cols = 2 * len(benchmarks)
    rho_mat = np.full((n_metrics, n_cols), np.nan)  # holds Pearson r
    p_mat = np.full((n_metrics, n_cols), np.nan)
    n_mat = np.full((n_metrics, n_cols), np.nan)

    for i, m in enumerate(metrics):
        for j, (_, simple_key, partial_key, _) in enumerate(benchmarks):
            for col_offset, key in enumerate([simple_key, partial_key]):
                entry = corr.get(m, {}).get(key)
                if entry is None:
                    continue
                rho_mat[i, 2 * j + col_offset] = entry["pearson_r"]
                p_mat[i, 2 * j + col_offset]   = entry["pearson_p"]
                n_mat[i, 2 * j + col_offset]   = entry["n_models"]

    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    im = ax.imshow(rho_mat, vmin=-0.8, vmax=0.8, cmap=CMAP_SEQ, aspect="auto")

    # Column labels on top: alternating "Simp." / "Part."
    col_sub_labels = ["Simp.", "Part."] * len(benchmarks)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_sub_labels, fontsize=8)
    ax.tick_params(axis="x", which="major", length=0, pad=2)

    # Benchmark-group labels above the subcolumn labels
    for j, (_, _, _, bench_label) in enumerate(benchmarks):
        x_center = 2 * j + 0.5
        ax.text(x_center, -1.2, bench_label,
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                transform=ax.transData)

    # Row labels
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metric_labels, fontsize=9)
    ax.tick_params(axis="y", which="major", length=0)

    # Vertical dividers between benchmark groups
    for j in range(1, len(benchmarks)):
        ax.axvline(2 * j - 0.5, color="white", linewidth=2.0, zorder=4)

    # Annotate cells with rho + stars; white text on dark cells.
    for i in range(n_metrics):
        for j in range(n_cols):
            v = rho_mat[i, j]
            if np.isnan(v):
                continue
            stars = sig_stars(p_mat[i, j])
            txt = f"{v:+.2f}{stars}"
            # Batlow is dark at low values and bright at high values. White
            # text reads best on the dark (low/negative) end.
            color = "white" if v < -0.1 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.8, color=color, zorder=3)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Pearson $r$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Footer: n per column group (Simple / Partial)
    ns = {j: int(np.nanmax(n_mat[:, 2 * j])) if not np.all(np.isnan(n_mat[:, 2 * j])) else "-"
          for j in range(len(benchmarks))}
    nps = {j: int(np.nanmax(n_mat[:, 2 * j + 1])) if not np.all(np.isnan(n_mat[:, 2 * j + 1])) else "-"
           for j in range(len(benchmarks))}
    footer = "   ".join(
        f"{lbl}: n={ns[j]}/{nps[j]}"
        for j, (_, _, _, lbl) in enumerate(benchmarks)
    )
    ax.text(0.0, 1.18, footer,
            transform=ax.transAxes, fontsize=7, color=C_GREY, va="bottom")

    fig.tight_layout()
    out = FIGS_DIR / "fig_correlation_summary.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


def _draw_corr_triangle(ax, mat, labels, rotation=20, label_fs=14,
                         cell_fs=14, aspect="auto"):
    """Lower-triangular Pearson correlation heatmap with cell annotations."""
    n = len(labels)
    upper_mask = ~np.tri(n, dtype=bool)
    display = np.ma.masked_array(np.where(np.isnan(mat), 0.0, mat),
                                 mask=upper_mask)
    im = ax.imshow(display, vmin=-1, vmax=1, cmap="coolwarm", aspect=aspect)
    # Per-cell black borders on the lower-triangular cells. No separate
    # rectangle around the full NxN area — cells along the staircase edge
    # of the lower triangle naturally form the visible outer outline.
    from matplotlib.patches import Rectangle
    for i in range(n):
        for j in range(i + 1):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                    fill=False, edgecolor="black",
                                    linewidth=0.2, zorder=4))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=rotation,
                       ha=("center" if rotation == 0 else "right"),
                       fontsize=label_fs)
    ax.set_yticklabels(labels, fontsize=label_fs)
    for i in range(n):
        for j in range(n):
            if j > i:
                continue
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt = "—" if i == j else f"{v:+.2f}"
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=cell_fs, color=color, zorder=3)
    ax.tick_params(axis="both", which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def fig_benchmark_correlations(benchmarks):
    """Side-by-side Pearson correlation triangles (two-column figure).

      (a) Inter-benchmark: 8x8 over capability proxies + outcome benchmarks.
      (b) Inter-test: 5x5 over DAT, CDAT, CDAT-N, CDAT-A, PACE composite
          z-scores across GloVe, FastText, and SBERT.
    """
    from scipy.stats import pearsonr
    from matplotlib.gridspec import GridSpec

    keys_a = ["arena_overall", "mmlu_pro", "arena_cw", "eq_bench_cw",
              "mazur_cw_v2", "hivemind_diversity", "noveltybench_utility",
              "liveideabench"]
    labels_a = ["Arena Ovr", "MMLU-Pro",
                "Arena CW", "EqBench CW", "Mazur V2",
                "HiveMind", "NoveltyBench", "LiveIdea"]
    na = len(keys_a)
    mat_a = np.full((na, na), np.nan)
    for i, ki in enumerate(keys_a):
        for j, kj in enumerate(keys_a):
            xs, ys = [], []
            for _, v in benchmarks.items():
                if ki in v and kj in v:
                    xs.append(v[ki]); ys.append(v[kj])
            if len(xs) >= 3:
                r, _ = pearsonr(xs, ys)
                mat_a[i, j] = r

    composite = load_composite_scores()
    # RAT scores (zero-shot strict accuracy). Pilot wins on overlap; keys
    # converted from OR-id form to the ormap form used in `composite`.
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    rat_orm: dict[str, float] = {}
    def _ormap(s: str) -> str:
        return s.replace("/", "_").replace(".", "-").replace(":", "_")
    for _path in [
        PROJECT_ROOT / "data/new_tests/rat/expansion_v1/summary.json",
        PROJECT_ROOT / "data/new_tests/rat/pilot_v1/summary.json",
    ]:
        if _path.exists():
            with open(_path) as f:
                _rsum = json.load(f)
            for _m, _s in _rsum.items():
                if _s.get("n_total", 0) > 0 and _s.get("n_errors", 0) < _s["n_total"]:
                    rat_orm[_ormap(_m)] = _s["zs_accuracy_strict"]

    tasks = ["dat", "cdat", "cdat_novelty", "cdat_appropriateness", "pace"]
    labels_b = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "PACE", "RAT"]
    label_to_task = dict(zip(labels_b, tasks + [None]))  # None means RAT (special-cased)

    def get_score(label: str, model: str):
        if label == "RAT":
            return rat_orm.get(model)
        tk = label_to_task[label]
        v = composite.get(model, {}).get(tk)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return v

    all_b_models = set(composite.keys()) | set(rat_orm.keys())
    nb = len(labels_b)
    mat_b = np.full((nb, nb), np.nan)
    for i, ti in enumerate(labels_b):
        for j, tj in enumerate(labels_b):
            xs, ys = [], []
            for m in all_b_models:
                vi = get_score(ti, m); vj = get_score(tj, m)
                if vi is None or vj is None:
                    continue
                xs.append(vi); ys.append(vj)
            if len(xs) >= 3:
                r, _ = pearsonr(xs, ys)
                mat_b[i, j] = r

    fig = plt.figure(figsize=(15.0, 6.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[na, nb], wspace=0.20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    im = _draw_corr_triangle(ax1, mat_a, labels_a, rotation=20,
                              label_fs=14, cell_fs=14)
    _draw_corr_triangle(ax2, mat_b, labels_b, rotation=20,
                        label_fs=14, cell_fs=14, aspect="auto")

    # Resize panel (b) so each cell matches panel (a)'s physical size
    # (no vertical stretching), anchored to the bottom of the gridspec
    # slot so heatmap bottoms align with panel (a)'s.
    fig.canvas.draw()
    pos1 = ax1.get_position()
    cell_size = pos1.height / na
    pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos1.y0, pos2.width, cell_size * nb])

    ax1.set_title("(a) Inter-benchmark correlations",
                   fontsize=19, loc="left", pad=6)
    ax2.set_title("(b) Inter-test correlations",
                   fontsize=19, loc="left", pad=6)

    cbar = fig.colorbar(im, ax=[ax1, ax2], shrink=0.7, pad=0.03)
    cbar.set_label("Pearson $r$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    for out_dir in [FIGS_DIR, PAPER_FIGS_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_benchmark_correlations.pdf"
        plt.savefig(out, bbox_inches="tight")
        plt.savefig(out.with_suffix(".png"), bbox_inches="tight")
        print(f"Saved {out}")
    plt.close()


def fig_inter_metric_triangle(corr):
    """Triangular heatmap of inter-metric Spearman correlations.

    Shows the lower triangle of the 4x4 metric-by-metric correlation matrix
    (DAT, CDAT Novelty, CDAT Appropriateness, PACE). Upper triangle is
    blanked out because it would be redundant. Cells annotated with the
    Spearman rho and significance stars.
    """
    metrics = ["dat", "cdat", "pace"]
    labels = ["DAT", "CDAT", "PACE"]
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
            mat[i, j] = entry.get("pearson_r", entry["spearman_rho"])
            pmat[i, j] = entry.get("pearson_p", entry["p_value"])

    # Mask upper triangle for plotting
    display = np.where(np.isnan(mat), 0.0, mat)
    mask = np.isnan(mat)

    fig, ax = plt.subplots(figsize=(3.3, 3.2))
    # Batlow (sequential, perceptually uniform): dark at -1, bright at +1.
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
                # values (yellow-orange). White text on dark (negative) cells.
                color = "white" if mat[i, j] < -0.2 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=color, zorder=3)

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.04)
    cbar.set_label("Pearson $r$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Inter-metric correlations", fontsize=9.5, pad=4)
    ax.tick_params(axis="both", which="major", length=0)

    fig.tight_layout()
    out = FIGS_DIR / "fig_inter_metric.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


def fig_scatter_by_embedding(benchmarks):
    """Non-residualised scatter grid overlaying all three embeddings.

    Rows = creativity metrics. Columns = benchmarks. Within each embedding,
    metric scores are z-scored across models so the three embedding clouds
    overlay on a common x-axis (raw benchmarks on y). Each point's colour
    identifies the model (same per-model colour map as fig2_combined_grid);
    each point's marker shape identifies the embedding (o GloVe, ^ FastText,
    s SBERT).
    """
    from scipy.stats import pearsonr

    me_path = RESULTS_DIR / "multi_embed_scores.json"
    if not me_path.exists():
        print("Skipping fig_scatter_by_embedding: multi_embed_scores.json missing.")
        return
    with open(me_path) as f:
        me = json.load(f)

    emb_specs = [
        ("glove",    "GloVe",    "o"),
        ("fasttext", "FastText", "^"),
        ("sbert",    "SBERT",    "s"),
    ]

    column_specs = [
        ("arena_cw",           "Arena CW"),
        ("eq_bench_cw",        "EQ-Bench CW"),
        ("mazur_cw_v2",        "Mazur V2"),
        ("hivemind_diversity", "Hivemind Div."),
    ]
    n_rows = len(_METRIC_PANELS)
    n_cols = len(column_specs)

    # Build a stable per-model colour map across every model present in any
    # embedding's scored set.
    all_models = {m for emb, _, _ in emb_specs for m in me.get(emb, {})}
    model_color = _build_model_color_map(all_models)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12.0, 11.5),
                              sharex="row", sharey="col")

    # Pre-compute per-embedding z-score stats for each metric so clouds overlay.
    tasks = [p[0] for p in _METRIC_PANELS]
    zstats = {}
    for t in tasks:
        for emb, _, _ in emb_specs:
            vals = [me[emb].get(m, {}).get(t) for m in all_models]
            vals = [v for v in vals if v is not None
                    and not (isinstance(v, float) and (np.isnan(v) or v == 0))]
            if vals:
                zstats[(t, emb)] = (float(np.mean(vals)), float(np.std(vals)) or 1.0)

    models_seen = set()
    for row_idx, (metric_key, metric_label, _) in enumerate(_METRIC_PANELS):
        for col_idx, (bench_key, _) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            per_emb_stats = []
            for emb_key, emb_label, emb_marker in emb_specs:
                mu_sigma = zstats.get((metric_key, emb_key))
                if mu_sigma is None:
                    continue
                mu, sigma = mu_sigma
                xs, ys, mks = [], [], []
                for mk, srow in me[emb_key].items():
                    v = srow.get(metric_key)
                    if v is None or (isinstance(v, float) and (np.isnan(v) or v == 0)):
                        continue
                    if mk not in benchmarks or bench_key not in benchmarks[mk]:
                        continue
                    xs.append((v - mu) / sigma)
                    ys.append(benchmarks[mk][bench_key])
                    mks.append(mk)
                    models_seen.add(mk)
                if not xs:
                    continue
                xs_a = np.asarray(xs)
                ys_a = np.asarray(ys)
                point_colors = [model_color[mk] for mk in mks]
                ax.scatter(xs_a, ys_a, s=20, c=point_colors, marker=emb_marker,
                           alpha=0.8, edgecolor="white", linewidth=0.3, zorder=3)
                if len(xs) >= 5:
                    r, _ = pearsonr(xs_a, ys_a)
                    per_emb_stats.append((emb_label, r, len(xs)))

            if per_emb_stats:
                lines = []
                for emb_label, r, n in per_emb_stats:
                    lines.append(f"{emb_label[:1]}: {r:+.2f} ($n$={n})")
                ax.text(0.03, 0.97, "\n".join(lines),
                        transform=ax.transAxes, fontsize=7.5,
                        verticalalignment="top",
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.78, pad=1.3))

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", which="major", labelsize=8)

    for col_idx, (_, bench_label) in enumerate(column_specs):
        axes[0, col_idx].set_title(bench_label, fontsize=10.5, pad=8, weight="bold")
    for row_idx, (_, metric_label, _) in enumerate(_METRIC_PANELS):
        axes[row_idx, 0].annotate(
            metric_label, xy=(-0.28, 0.5), xycoords="axes fraction",
            ha="center", va="center", fontsize=10.5, weight="bold", rotation=90,
        )

    # Embedding-marker legend at the top (shape conveys embedding).
    emb_handles = [
        plt.Line2D([0], [0], marker=m, linestyle="",
                   color=C_GREY, markersize=7, markeredgecolor="white",
                   markeredgewidth=0.4, label=lbl)
        for _, lbl, m in emb_specs
    ]
    fig.legend(handles=emb_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=3, frameon=False, fontsize=10)

    # Per-model colour legend at the bottom (same ordering as fig2_combined_grid).
    models_for_legend = sorted(
        models_seen,
        key=lambda mk: (
            _FAMILY_ORDER.index(_family(mk)) if _family(mk) in _FAMILY_ORDER else len(_FAMILY_ORDER),
            mk,
        ),
    )
    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=model_color[mk], markersize=6,
                   markeredgecolor="white", markeredgewidth=0.4,
                   label=_short_label(mk))
        for mk in models_for_legend
    ]
    fig.legend(handles=model_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.005),
               ncol=5, frameon=False, fontsize=10,
               handletextpad=0.4, columnspacing=1.4,
               labelspacing=0.35)

    fig.tight_layout(rect=[0.05, 0.18, 1, 0.985])

    out = FIGS_DIR / "fig_scatter_by_embedding.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


def fig_validity_specificity(benchmarks):
    """2x2 scatter: Validity (raw Pearson r) vs Specificity (semi-partial
    Pearson r controlling for Arena Overall + MMLU-Pro), one panel per
    benchmark. Colour = test (DAT / CDAT / PACE), marker = embedding
    (GloVe / FastText / SBERT / Overall composite). Points significant
    on BOTH axes (p < .05, expected direction) are drawn with a thick
    black edge. Figural analogue of Table~\\ref{tab:correlations}.
    """
    from scipy.stats import pearsonr
    from matplotlib.lines import Line2D

    me_path = RESULTS_DIR / "multi_embed_scores.json"
    if not me_path.exists():
        print("Skipping fig_validity_specificity: multi_embed_scores.json missing.")
        return
    with open(me_path) as f:
        me_scores = json.load(f)
    composite = load_composite_scores()

    # (key, label, marker, size)
    embedders = [
        ("glove",    "GloVe",    "o", 60),
        ("fasttext", "FastText", "^", 60),
        ("sbert",    "SBERT",    "s", 60),
        ("overall",  "Overall",  "D", 90),   # diamond, larger: headline row
    ]
    tests = [
        ("dat",  "DAT",  C_DAT),
        ("cdat", "CDAT", C_CDAT),
        ("pace", "PACE", C_PACE),
    ]
    benchs = [
        ("arena_cw",           "Arena CW"),
        ("eq_bench_cw",        "EQ-Bench CW"),
        ("mazur_cw_v2",        "Mazur CW v2"),
        ("hivemind_diversity", "Hivemind diversity"),
    ]

    # Per (test, embedding, benchmark): raw r + semi-partial r + p-values.
    records = {}
    for emb_key, _, _, _ in embedders:
        score_src = composite if emb_key == "overall" else me_scores.get(emb_key, {})
        for test_key, _, _ in tests:
            for bench_key, _ in benchs:
                xs, ys, ao, mp = [], [], [], []
                for mk, sc in score_src.items():
                    v = sc.get(test_key)
                    if v is None or (isinstance(v, float) and (np.isnan(v) or v == 0)):
                        continue
                    b = benchmarks.get(mk, {})
                    if bench_key not in b or "arena_overall" not in b or "mmlu_pro" not in b:
                        continue
                    xs.append(v); ys.append(b[bench_key])
                    ao.append(b["arena_overall"]); mp.append(b["mmlu_pro"])
                if len(xs) < 5:
                    records[(test_key, emb_key, bench_key)] = None
                    continue
                xs_a = np.asarray(xs, float)
                ys_a = np.asarray(ys, float)
                Z = np.column_stack([np.ones(len(xs_a)),
                                      np.asarray(ao, float),
                                      np.asarray(mp, float)])
                val_r, val_p = pearsonr(xs_a, ys_a)
                beta, *_ = np.linalg.lstsq(Z, ys_a, rcond=None)
                y_resid = ys_a - Z @ beta
                spec_r, spec_p = pearsonr(xs_a, y_resid)
                records[(test_key, emb_key, bench_key)] = {
                    "val_r": float(val_r), "val_p": float(val_p),
                    "spec_r": float(spec_r), "spec_p": float(spec_p),
                }

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.7),
                              sharex=True, sharey=True)

    for ax, (bench_key, bench_label) in zip(axes.flat, benchs):
        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)

        import matplotlib.transforms as mtransforms
        star_trans = mtransforms.offset_copy(ax.transData, fig=fig,
                                             x=7, y=6, units="points")
        for test_key, _, test_color in tests:
            for emb_key, _, marker, size in embedders:
                rec = records.get((test_key, emb_key, bench_key))
                if rec is None:
                    continue
                both_sig = (rec["val_p"] < 0.05 and rec["spec_p"] < 0.05
                            and rec["val_r"] > 0 and rec["spec_r"] > 0)
                ax.scatter(
                    rec["val_r"], rec["spec_r"],
                    marker=marker, s=size, c=[test_color],
                    edgecolor="white", linewidth=0.5,
                    zorder=3, alpha=0.95,
                )
                # Small gold star offset to the upper-right of both-sig cells
                if both_sig:
                    ax.scatter(
                        rec["val_r"], rec["spec_r"],
                        marker="*", s=55,
                        facecolor="#ffcc00", edgecolor="black",
                        linewidth=0.5, zorder=5, transform=star_trans,
                    )

        ax.set_title(bench_label, fontsize=10.5)
        ax.tick_params(axis="both", labelsize=8.5)

    # Shared axes: extra pad on top/right so offset sig stars stay inside.
    all_val = [rec["val_r"] for rec in records.values() if rec]
    all_spec = [rec["spec_r"] for rec in records.values() if rec]
    xpad = 0.05 * (max(all_val) - min(all_val))
    ypad = 0.05 * (max(all_spec) - min(all_spec))
    for ax in axes.flat:
        ax.set_xlim(min(all_val) - xpad, max(all_val) + 2.0 * xpad)
        ax.set_ylim(min(all_spec) - ypad, max(all_spec) + 2.0 * ypad)

    # Axis labels only on outer edge for shared layout
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Validity  $r$", fontsize=10.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Specificity  $r \mid g$", fontsize=10.5)

    # --- Legends ---
    test_handles = [Line2D([], [], marker="o", linestyle="none",
                           markerfacecolor=c, markeredgecolor="white",
                           markeredgewidth=0.5, markersize=8, label=lbl)
                    for (_, lbl, c) in tests]
    emb_handles = [Line2D([], [], marker=m, linestyle="none",
                          markerfacecolor=C_GREY, markeredgecolor="white",
                          markeredgewidth=0.5,
                          markersize=(10 if k == "overall" else 8),
                          label=lbl)
                   for (k, lbl, m, _) in embedders]
    sig_handle = [Line2D([], [], marker="*", linestyle="none",
                         markerfacecolor="#ffcc00", markeredgecolor="black",
                         markeredgewidth=0.5, markersize=9,
                         label=r"sig. on both ($p\!<\!.05$)")]

    # Single combined legend: row 1 = tests + sig indicator,
    # row 2 = embeddings. ncol=4 fills across rows first.
    combined = test_handles + sig_handle + emb_handles
    fig.legend(handles=combined,
               loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=4, frameon=False, fontsize=8.5,
               handletextpad=0.3, columnspacing=1.2,
               labelspacing=0.3)

    fig.tight_layout(rect=[0, 0.16, 1, 1])
    out = FIGS_DIR / "fig_validity_specificity.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")


def _benchmark_signed_R(bench_key: str, BMARKS: dict) -> float | None:
    """Signed multiple correlation R of bench_key on the capability stack
    g = (Arena Overall, MMLU-Pro). Sign follows the dominant Arena Overall
    direction (matches fig_specificity_ceilings). Returns None if the
    intersection has fewer than 5 models.
    """
    ms = [m for m, d in BMARKS.items()
          if bench_key in d and "arena_overall" in d and "mmlu_pro" in d]
    if len(ms) < 5:
        return None
    ys = np.array([BMARKS[m][bench_key] for m in ms], dtype=float)
    A = np.array([[BMARKS[m]["arena_overall"], BMARKS[m]["mmlu_pro"]]
                  for m in ms], dtype=float)
    A1 = np.column_stack([np.ones(len(ys)), A])
    beta = np.linalg.lstsq(A1, ys, rcond=None)[0]
    yhat = A1 @ beta
    R2 = 1.0 - np.sum((ys - yhat) ** 2) / np.sum((ys - ys.mean()) ** 2)
    R = float(np.sqrt(max(0.0, R2)))
    if np.corrcoef(A[:, 0], ys)[0, 1] < 0:
        R = -R
    return R


def _panel_avg_ceiling(R_list: list[float], v_grid: np.ndarray) -> np.ndarray:
    """Mean across a panel's benchmarks of the per-benchmark specificity ceiling
    upper envelope ceiling_b(v) = v sqrt(1 - R_b^2) + |R_b| sqrt(1 - v^2). The
    average is taken at each v (mean-of-ceilings, not ceiling-of-mean-R), so a
    single test that hit every benchmark's individual ceiling would land on
    this curve."""
    if not R_list:
        return np.zeros_like(v_grid)
    upper = np.zeros_like(v_grid)
    root1mv2 = np.sqrt(np.clip(1 - v_grid ** 2, 0, None))
    for R in R_list:
        upper = upper + v_grid * np.sqrt(max(0.0, 1 - R ** 2)) + abs(R) * root1mv2
    return upper / len(R_list)


def _panel_max_ceiling(R_list: list[float], v_grid: np.ndarray) -> np.ndarray:
    """Single curve from the panel benchmark with the largest |R|. Its
    ceiling sits above all the others' across the bulk of the moderate-v
    range where observed test points cluster, so it serves as the
    construct-level ceiling reference (rather than an elementwise envelope
    or a mean-of-curves)."""
    if not R_list:
        return np.zeros_like(v_grid)
    R = max(R_list, key=abs)
    root1mv2 = np.sqrt(np.clip(1 - v_grid ** 2, 0, None))
    return v_grid * np.sqrt(max(0.0, 1 - R ** 2)) + abs(R) * root1mv2


def fig_headline():
    """Two-panel headline scatter pulling the ``Overall'' (mean z-score
    across GloVe / FastText / SBERT) block from Table~1. Left panel =
    Creative Writing benchmarks (Arena CW, EQ-Bench CW, Mazur CW v2),
    right panel = Divergent Thinking benchmarks (Hivemind diversity,
    NoveltyBench utility). Colours encode tests; small translucent
    circles are per-benchmark cells and the large black-outlined circle
    per test is the within-panel benchmark average (``Overall'').

    Each panel also draws the construct-level theoretical specificity
    ceiling: the unweighted mean of the per-benchmark ceiling curves
    ceiling(v) = v sqrt(1 - R^2) + |R| sqrt(1 - v^2), where R is the
    signed multiple correlation of each benchmark on the capability
    stack g = (Arena Overall, MMLU-Pro). Computed from benchmarks.json
    so it tracks any future benchmark coverage updates.

    Saved to both the report's figures directory and the ICCC paper's
    figures directory so the paper can reference the same file via a
    ``figure*`` (two-column) environment.
    """
    from matplotlib.lines import Line2D

    figsize = (12.6, 4.8)
    title_fs, axis_fs, tick_fs = 14.0, 10.5, 9.0
    leg_fs, leg_title_fs, annotate_fs = 12.0, 12.0, 8.5
    s_ind, s_overall, s_star = 38, 170, 40
    overall_edge_lw, ind_sig_lw, ind_nosig_lw = 1.3, 0.9, 0.4
    leg_ms_test, leg_ms_overall, leg_ms_star, leg_ms_outl = 12, 14, 13, 9
    star_off_pts = 6
    rect = [0, 0.09, 1, 1]
    out_dirs = [FIGS_DIR, PAPER_FIGS_DIR]

    # Colors encode tests. 5 well-separated categorical samples from Batlow
    # for the original semantic-distance tests; RAT gets a distinct warm
    # brown so it reads as a different test family at a glance.
    test_samples = CMAP_SEQ(np.linspace(0.05, 0.82, 5))
    test_colors = {
        "DAT":      test_samples[0],
        "CDAT":     test_samples[1],
        "CDAT-N":   test_samples[2],
        "CDAT-A":   test_samples[3],
        "PACE":     test_samples[4],
        "RAT":      "#8C5A3D",
    }
    test_order = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "PACE", "RAT"]

    cw_benchmarks = ["Arena CW", "EQ-Bench CW", "Mazur CW v2"]
    dt_benchmarks = ["Hivemind Div.", "NovBench Util."]
    si_benchmarks = ["LiveIdeaBench"]

    # Per-panel benchmark keys for the theoretical-ceiling computation.
    panel_bench_keys = {
        "Creative Writing":    ["arena_cw", "eq_bench_cw", "mazur_cw_v2"],
        "Divergent Thinking":  ["hivemind_diversity", "noveltybench_utility"],
        "Scientific Ideation": ["liveideabench"],
    }
    with open(BENCH_PATH) as _f:
        _BMARKS = json.load(_f)
    panel_R = {}
    for panel, keys in panel_bench_keys.items():
        Rs = []
        for k in keys:
            R = _benchmark_signed_R(k, _BMARKS)
            if R is not None:
                Rs.append(R)
        panel_R[panel] = Rs

    # Overall (mean z-score across 3 embeddings) block of Table 1.
    # Each entry: (validity r, specificity r | g, val_sig, spec_sig).
    # Significance flags follow the paper's bolding (p < 0.05, any direction).
    cw_data = {
        "DAT":      [(+0.44, +0.08, True,  False),
                     (+0.71, +0.50, True,  True),
                     (+0.59, +0.50, True,  True)],
        "CDAT":     [(-0.13, +0.28, False, False),
                     (-0.06, +0.13, False, False),
                     (+0.07, +0.39, False, False)],
        "CDAT-N":   [(-0.18, +0.23, False, False),
                     (-0.14, +0.15, False, False),
                     (+0.09, +0.35, False, False)],
        "CDAT-A":   [(+0.54, -0.12, True,  False),
                     (+0.47, -0.02, True,  False),
                     (+0.24, -0.21, False, False)],
        "PACE":     [(+0.72, +0.05, True,  False),
                     (+0.70, +0.20, True,  False),
                     (+0.75, +0.18, True,  False)],
        "RAT":      [(+0.76, -0.03, True,  False),
                     (+0.57, -0.04, True,  False),
                     (+0.50, +0.08, True,  False)],
    }
    dt_data = {
        "DAT":      [(+0.33, +0.26, False, False),
                     (+0.15, -0.26, False, False)],
        "CDAT":     [(+0.25, +0.19, False, False),
                     (+0.60, +0.57, False, False)],
        "CDAT-N":   [(+0.24, +0.17, False, False),
                     (+0.54, +0.46, True,  False)],
        "CDAT-A":   [(-0.39, -0.16, False, False),
                     (-0.67, -0.40, True,  False)],
        "PACE":     [(-0.05, +0.37, False, False),
                     (+0.18, -0.06, False, False)],
        "RAT":      [(-0.55, +0.05, True,  False),
                     (-0.30, -0.05, False, False)],
    }
    # Scientific Ideation (LiveIdeaBench, n=17): no test reaches p<.05.
    si_data = {
        "DAT":      [(-0.01, +0.28, False, False)],
        "CDAT":     [(+0.06, +0.26, False, False)],
        "CDAT-N":   [(-0.09, +0.11, False, False)],
        "CDAT-A":   [(+0.16, -0.01, False, False)],
        "PACE":     [(+0.07, -0.07, False, False)],
        "RAT":      [(+0.30, +0.12, False, False)],
    }

    # Per-panel label placement for the black "Overall" composite points.
    # (dx, dy, horizontal-align, vertical-align) in data coordinates.
    label_offsets = {
        "Creative Writing": {
            "DAT":      (+0.030, +0.008, "left",   "center"),
            "CDAT":     ( 0.000, +0.042, "center", "bottom"),
            "CDAT-N":   ( 0.000, -0.042, "center", "top"),
            "CDAT-A":   ( 0.000, -0.042, "center", "top"),
            "PACE":     (-0.030, +0.008, "right",  "center"),
            "RAT":      (+0.030, -0.014, "left",   "center"),
        },
        "Divergent Thinking": {
            "DAT":      ( 0.000, -0.042, "center", "top"),
            "CDAT":     (+0.030, +0.008, "left",   "center"),
            "CDAT-N":   (-0.030, +0.008, "right",  "center"),
            "CDAT-A":   ( 0.000, +0.048, "center", "bottom"),
            "PACE":     ( 0.000, -0.048, "center", "top"),
            "RAT":      (-0.030, +0.008, "right",  "center"),
        },
        "Scientific Ideation": {
            "DAT":      ( 0.000, +0.048, "center", "bottom"),
            "CDAT":     (+0.030, +0.008, "left",   "center"),
            "CDAT-N":   ( 0.000, -0.048, "center", "top"),
            "CDAT-A":   (+0.030, +0.008, "left",   "center"),
            "PACE":     (-0.030, +0.008, "right",  "center"),
            "RAT":      (+0.030, +0.008, "left",   "center"),
        },
    }

    fig, (ax_l, ax_m, ax_r) = plt.subplots(
        1, 3, figsize=figsize, sharex=True, sharey=True,
    )
    import matplotlib.transforms as mtransforms

    from matplotlib.patches import Rectangle

    def _plot(ax, data, benchmarks, title):
        # Soft green wash in the positive-positive quadrant: the region a
        # useful creativity test should occupy (positive validity AND
        # positive specificity). Rectangle extends well beyond plausible
        # axis limits; matplotlib clips to the axes.
        ax.add_patch(Rectangle(
            (0, 0), 10, 10,
            facecolor="#7ec587", alpha=0.10,
            edgecolor="none", zorder=-1,
        ))
        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)

        # Construct-level theoretical specificity ceiling (mean of the
        # per-benchmark ceiling curves over this panel's benchmarks).
        v_grid = np.linspace(-1, 1, 400)
        R_list = panel_R.get(title, [])
        if R_list:
            ceil_curve = _panel_avg_ceiling(R_list, v_grid)
            ax.plot(v_grid, ceil_curve, color="black", linewidth=1.0,
                    linestyle="-", alpha=0.85, zorder=1)
        # Offset transform for the tiny gold "both-sig" star sitting to the
        # upper-right of its parent marker.
        star_trans = mtransforms.offset_copy(
            ax.transData, fig=fig,
            x=star_off_pts, y=star_off_pts, units="points",
        )
        for test in test_order:
            pts = data[test]
            color = test_colors[test]
            for bench_label, (val, spec, val_sig, spec_sig) in zip(benchmarks, pts):
                any_sig = val_sig or spec_sig
                both_pos_sig = (val_sig and spec_sig
                                and val > 0 and spec > 0)
                ax.scatter(
                    val, spec,
                    marker="o", s=s_ind,
                    c=[color],
                    edgecolor=("black" if any_sig else "white"),
                    linewidth=(ind_sig_lw if any_sig else ind_nosig_lw),
                    zorder=3, alpha=0.55,
                )
                if both_pos_sig:
                    ax.scatter(
                        val, spec,
                        marker="*", s=s_star,
                        facecolor="#ffcc00", edgecolor="black",
                        linewidth=0.4, zorder=5, transform=star_trans,
                    )
            # Within-panel "Overall" composite = unweighted mean across benchmarks.
            vals = [v for v, _, *_ in pts]
            specs = [s for _, s, *_ in pts]
            mx, my = float(np.mean(vals)), float(np.mean(specs))
            ax.scatter(
                mx, my,
                marker="o", s=s_overall,
                c=[color],
                edgecolor="black", linewidth=overall_edge_lw,
                zorder=4,
            )
            dx, dy, ha, va = label_offsets[title][test]
            ax.annotate(
                test, xy=(mx, my), xytext=(mx + dx, my + dy),
                ha=ha, va=va, fontsize=annotate_fs, fontweight="bold",
                color="black", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=1.0, alpha=0.85),
            )
        ax.set_title(title, fontsize=title_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.set_xlabel(r"Validity  ($r$)", fontsize=axis_fs)

    _plot(ax_l, cw_data, cw_benchmarks, "Creative Writing")
    _plot(ax_m, dt_data, dt_benchmarks, "Divergent Thinking")
    _plot(ax_r, si_data, si_benchmarks, "Scientific Ideation")
    ax_l.set_ylabel(r"Specificity  ($r \mid g$)", fontsize=axis_fs)

    # Axis limits: include composite points too.
    all_vals, all_specs = [], []
    for data in (cw_data, dt_data, si_data):
        for pts in data.values():
            for v, s, *_ in pts:
                all_vals.append(v); all_specs.append(s)
            all_vals.append(float(np.mean([v for v, _, *_ in pts])))
            all_specs.append(float(np.mean([s for _, s, *_ in pts])))
    xpad = 0.12 * (max(all_vals) - min(all_vals))
    ypad = 0.10 * (max(all_specs) - min(all_specs))
    xlim_lo = min(all_vals) - xpad
    xlim_hi = max(all_vals) + xpad

    # Stretch ymax so the ceiling curve is fully visible within the data x-range.
    v_for_ceil = np.linspace(xlim_lo, xlim_hi, 200)
    ceil_maxes = [
        _panel_avg_ceiling(Rs, v_for_ceil).max()
        for Rs in panel_R.values() if Rs
    ]
    ymax_data = max(all_specs) + ypad
    ymax = max(ymax_data, (max(ceil_maxes) + 0.04) if ceil_maxes else ymax_data)
    ymin = min(all_specs) - ypad
    for ax in (ax_l, ax_m, ax_r):
        ax.set_xlim(xlim_lo, xlim_hi)
        ax.set_ylim(ymin, ymax)

    # --- Single flat Test legend at the bottom. Indicator conventions
    # (Overall marker, both-axes star, one-axis outline) are explained
    # in the figure caption rather than in the figure itself. ---
    test_handles = [
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor=test_colors[t], markeredgecolor="black",
               markeredgewidth=1.0, markersize=leg_ms_test, label=t)
        for t in test_order
    ]
    ceiling_handle = Line2D([], [], color="black", linewidth=1.0,
                             linestyle="-", label="theoretical ceiling")
    leg_tests = fig.legend(
        handles=[*test_handles, ceiling_handle],
        loc="lower center", bbox_to_anchor=(0.5, 0.00),
        ncol=6, frameon=False, fontsize=leg_fs,
        handletextpad=0.3, columnspacing=1.6,
    )
    leg_tests._legend_box.align = "center"

    fig.tight_layout(rect=rect)
    for out_dir in out_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_headline.pdf"
        plt.savefig(out)
        plt.savefig(out.with_suffix(".png"))
        print(f"Saved {out}")
    plt.close()


def fig_specificity_ceilings():
    """Four-panel plot of the theoretical specificity ceiling per benchmark.

    For a test X with validity v = r(X,Y) against benchmark Y whose
    correlation with a scalar capability proxy Z is c = r(Y,Z), the
    maximum attainable specificity in the expected direction is
        max r(X, Y | Z) = v * sqrt(1 - c^2) + |c| * sqrt(1 - v^2),
    which follows from the PSD bound
        |r(X,Z) - r(Y,Z)| <= sqrt(2 * (1 - r(X,Y))).
    The "feasible" specificity region (between min and max envelopes)
    is shaded; the Overall-block cells from Table 1 are overlaid.
    """
    from matplotlib.lines import Line2D

    test_samples = CMAP_SEQ(np.linspace(0.05, 0.82, 5))
    test_colors = {
        "DAT":      test_samples[0],
        "CDAT":     test_samples[1],
        "CDAT-N":   test_samples[2],
        "CDAT-A":   test_samples[3],
        "PACE":     test_samples[4],
    }
    test_order = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "PACE"]

    # Recompute (validity, specificity) for each (test, benchmark) on the
    # same n-subset that defines the multiple-R for that benchmark --- the
    # subset of models with Y, Arena Overall, and MMLU-Pro all present. This
    # keeps the lens and the observed points on the same joint distribution,
    # which matters at small n (NovBench Util. has n=11 and the Table 1
    # numbers mix validity-n with specificity-n, enough to push points just
    # outside the population lens).
    me_path = RESULTS_DIR / "multi_embed_scores.json"
    with open(me_path) as f:
        me = json.load(f)
    with open(BENCH_PATH) as f:
        BMARKS = json.load(f)

    # Composite = mean z-score across the 3 embeddings.
    embs = sorted(me.keys())
    tasks_all = ["dat", "cdat", "cdat_novelty", "cdat_appropriateness", "pace"]
    all_models = sorted({m for emb in embs for m in me[emb]})
    composite: dict[str, dict[str, float]] = {}
    for t in tasks_all:
        stats = {}
        for emb in embs:
            vals = [me[emb].get(m, {}).get(t) for m in all_models]
            vals = [v for v in vals if v is not None
                    and not (isinstance(v, float) and (np.isnan(v) or v == 0))]
            if vals:
                stats[emb] = (float(np.mean(vals)), float(np.std(vals)) or 1.0)
        for m in all_models:
            zs = []
            for emb in embs:
                if emb not in stats:
                    continue
                v = me[emb].get(m, {}).get(t)
                if v is None or (isinstance(v, float) and (np.isnan(v) or v == 0)):
                    continue
                mu, sd = stats[emb]
                zs.append((v - mu) / sd)
            if zs:
                composite.setdefault(m, {})[t] = float(np.mean(zs))

    bench_keys = {
        "Arena CW":        "arena_cw",
        "EQ-Bench CW":     "eq_bench_cw",
        "Mazur CW v2":     "mazur_cw_v2",
        "Hivemind Div.":   "hivemind_diversity",
        "NovBench Util.":  "noveltybench_utility",
        "LiveIdeaBench":   "liveideabench",
    }
    test_key = {
        "DAT":      "dat",
        "CDAT":     "cdat",
        "CDAT-N":   "cdat_novelty",
        "CDAT-A":   "cdat_appropriateness",
        "PACE":     "pace",
    }

    benchmarks = []
    bench_data: dict[str, dict[str, tuple[float, float]]] = {}
    for blabel, bkey in bench_keys.items():
        ms_yg = [m for m, d in BMARKS.items()
                 if bkey in d and "arena_overall" in d and "mmlu_pro" in d]
        ys = np.array([BMARKS[m][bkey] for m in ms_yg], dtype=float)
        A = np.array([[BMARKS[m]["arena_overall"], BMARKS[m]["mmlu_pro"]]
                      for m in ms_yg], dtype=float)
        A1 = np.column_stack([np.ones(len(ys)), A])
        beta = np.linalg.lstsq(A1, ys, rcond=None)[0]
        yhat = A1 @ beta
        R2 = 1.0 - np.sum((ys - yhat) ** 2) / np.sum((ys - ys.mean()) ** 2)
        R = float(np.sqrt(max(0.0, R2)))
        # Sign R by the sign of the dominant proxy (Arena Overall) correlation.
        if np.corrcoef(A[:, 0], ys)[0, 1] < 0:
            R = -R
        benchmarks.append((blabel, R))

        resid_by_m = dict(zip(ms_yg, ys - yhat))
        y_by_m = dict(zip(ms_yg, ys))

        pts: dict[str, tuple[float, float]] = {}
        for tlabel in test_order:
            tk = test_key[tlabel]
            kept = [m for m in ms_yg if composite.get(m, {}).get(tk) is not None]
            if len(kept) < 5:
                continue
            xs = np.array([composite[m][tk] for m in kept])
            ys_k = np.array([y_by_m[m] for m in kept])
            rs_k = np.array([resid_by_m[m] for m in kept])
            v = float(np.corrcoef(xs, ys_k)[0, 1])
            s = float(np.corrcoef(xs, rs_k)[0, 1])
            pts[tlabel] = (v, s)
        bench_data[blabel] = pts

    v_grid = np.linspace(-1, 1, 400)

    fig, axes = plt.subplots(1, 6, figsize=(17.0, 3.8),
                              sharex=True, sharey=True)

    for ax, (bench_name, c) in zip(axes, benchmarks):
        root1mc2 = np.sqrt(1 - c**2)
        root1mv2 = np.sqrt(1 - v_grid**2)
        upper = v_grid * root1mc2 + abs(c) * root1mv2
        lower = v_grid * root1mc2 - abs(c) * root1mv2

        # Feasible region lens
        ax.fill_between(v_grid, lower, upper,
                         color="#d0d0d0", alpha=0.35, zorder=0,
                         label="feasible region")
        # Upper envelope (the ceiling) in solid; lower in dotted
        ax.plot(v_grid, upper, color="black", linewidth=1.2,
                linestyle="-", zorder=1)
        ax.plot(v_grid, lower, color="black", linewidth=0.7,
                linestyle=":", alpha=0.5, zorder=1)

        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)

        for test in test_order:
            pt = bench_data[bench_name].get(test)
            if pt is None:
                continue
            v, spec = pt
            ax.scatter(v, spec, marker="o", s=55,
                       c=[test_colors[test]],
                       edgecolor="black", linewidth=0.8,
                       zorder=3, alpha=0.95)

        ax.set_title(f"{bench_name}  ($R = {c:+.2f}$)", fontsize=11)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlabel(r"Validity  ($r$)", fontsize=10.5)
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-1.05, 1.05)

    axes[0].set_ylabel(r"Specificity  ($r \mid g$)", fontsize=10.5)

    # Test legend at bottom
    test_handles = [
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor=test_colors[t], markeredgecolor="black",
               markeredgewidth=0.8, markersize=9, label=t)
        for t in test_order
    ]
    envelope_handle = Line2D([], [], color="black", linewidth=1.2,
                              linestyle="-", label="theoretical ceiling")
    fig.legend(handles=[envelope_handle, *test_handles],
               loc="lower center", bbox_to_anchor=(0.5, 0.00),
               ncol=7, frameon=False, fontsize=10,
               handletextpad=0.4, columnspacing=1.4)

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    for out_dir in [FIGS_DIR, PAPER_FIGS_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_specificity_ceilings.pdf"
        plt.savefig(out)
        plt.savefig(out.with_suffix(".png"))
        print(f"Saved {out}")
    plt.close()


def fig_headline_combined():
    """Combined headline + per-benchmark ceilings figure.

    Top row (3 panels) is the construct-level scatter (Creative Writing,
    Divergent Thinking, Scientific Ideation) with the panel-averaged
    theoretical ceiling overlay. Bottom row (6 panels) is the per-benchmark
    feasible-region lens with observed test points. A single legend for
    test colours and the theoretical-ceiling line sits at the bottom.

    Saved as fig_headline.pdf, replacing the previous standalone version.
    """
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mtransforms

    test_colors = {
        "DAT":      C_DAT,
        "CDAT":     C_CDAT,
        "CDAT-N":   C_CNOV,
        "CDAT-A":   C_CAPP,
        "PACE":     C_PACE,
        "RAT":      "#8C5A3D",
    }
    test_order = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "PACE", "RAT"]
    test_key = {
        "DAT":      "dat",
        "CDAT":     "cdat",
        "CDAT-N":   "cdat_novelty",
        "CDAT-A":   "cdat_appropriateness",
        "PACE":     "pace",
    }

    # ---------- Top-row data (Overall block of Table 1, hardcoded) ----------
    cw_benchmarks = ["Arena CW", "EQ-Bench CW", "Mazur CW v2"]
    dt_benchmarks = ["Hivemind Div.", "NovBench Util."]
    si_benchmarks = ["LiveIdeaBench"]
    cw_data = {
        # (val_r, spec_r, val_sig, spec_sig) per (test, benchmark) using the
        # Overall block of Table~\ref{tab:correlations} (TIGER-Lab MMLU-Pro,
        # true semi-partial r(X, Y - Y_hat_g)).
        "DAT":      [(+0.47, +0.05, True,  False),
                     (+0.72, +0.41, True,  True),
                     (+0.60, +0.49, True,  False)],
        "CDAT":     [(-0.13, +0.22, False, False),
                     (-0.06, +0.03, False, False),
                     (+0.07, +0.43, False, False)],
        "CDAT-N":   [(-0.18, +0.20, False, False),
                     (-0.14, +0.08, False, False),
                     (+0.09, +0.40, False, False)],
        "CDAT-A":   [(+0.54, -0.05, True,  False),
                     (+0.48, +0.05, True,  False),
                     (+0.24, -0.28, False, False)],
        "PACE":     [(+0.71, +0.11, True,  False),
                     (+0.71, +0.21, True,  False),
                     (+0.76, +0.14, True,  False)],
        "RAT":      [(+0.76, -0.03, True,  False),
                     (+0.57, -0.04, True,  False),
                     (+0.50, +0.08, True,  False)],
    }
    dt_data = {
        "DAT":      [(+0.33, +0.01, False, False),
                     (+0.15, -0.21, False, False)],
        "CDAT":     [(+0.25, +0.10, False, False),
                     (+0.60, +0.60, False, False)],
        "CDAT-N":   [(+0.24, +0.08, False, False),
                     (+0.54, +0.45, True,  False)],
        "CDAT-A":   [(-0.39, -0.09, False, False),
                     (-0.67, -0.40, True,  False)],
        "PACE":     [(-0.05, +0.33, False, False),
                     (+0.18, -0.00, False, False)],
        "RAT":      [(-0.55, +0.05, True,  False),
                     (-0.30, -0.05, False, False)],
    }
    # LIB Average (5-facet mean from the live-page leaderboard, including
    # Clarity for v2 models). These mirror the Average panel in fig:si-headline.
    si_data = {
        "DAT":      [(+0.18, +0.22, False, False)],
        "CDAT":     [(+0.02, +0.29, False, False)],
        "CDAT-N":   [(-0.08, -0.05, False, False)],
        "CDAT-A":   [(+0.15, +0.12, False, False)],
        "PACE":     [(+0.23, +0.22, False, False)],
        "RAT":      [(+0.20, +0.10, False, False)],
    }
    # Composite-mean positions (rough, for offset-design intuition):
    #   CW:  DAT (+.60,+.32)  CDAT (-.04,+.23)  CDAT-N (-.08,+.23)
    #        CDAT-A (+.42,-.09)  PACE (+.73,+.15)  RAT (+.61,+.00)
    #   DT:  DAT (+.24,-.10)  CDAT (+.43,+.35)  CDAT-N (+.39,+.27)
    #        CDAT-A (-.53,-.25)  PACE (+.07,+.17)  RAT (-.43,+.00)
    #   SI:  DAT (+.18,+.22)  CDAT (+.02,+.29)  CDAT-N (-.08,-.05)
    #        CDAT-A (+.15,+.12)  PACE (+.23,+.22)  RAT (+.20,+.10)
    label_offsets = {
        "Creative Writing": {
            "DAT":      ( 0.045, +0.045, "left",   "bottom"),  # above-right (DAT high & isolated)
            "CDAT":     ( 0.045, +0.000, "left",   "center"),  # right (CDAT/CDAT-N stacked)
            "CDAT-N":   (-0.045, +0.000, "right",  "center"),  # left
            "CDAT-A":   ( 0.000, -0.060, "center", "top"),     # below (CDAT-A low + isolated)
            "PACE":     ( 0.045, +0.045, "left",   "bottom"),  # above-right
            "RAT":      ( 0.045, -0.045, "left",   "top"),     # below-right (separates from DAT)
        },
        "Divergent Thinking": {
            "DAT":      ( 0.000, -0.060, "center", "top"),     # below (DAT isolated low)
            "CDAT":     ( 0.045, +0.045, "left",   "bottom"),  # above-right (CDAT/CDAT-N stacked)
            "CDAT-N":   (-0.045, -0.045, "right",  "top"),     # below-left
            "CDAT-A":   ( 0.045, +0.000, "left",   "center"),  # right (CDAT-A bottom-left, far from cluster)
            "PACE":     ( 0.000, +0.060, "center", "bottom"),  # above (PACE in middle)
            "RAT":      (-0.045, +0.000, "right",  "center"),  # left (RAT left side)
        },
        "Scientific Ideation": {
            # Post-spec-pool layout: CDAT (+.02,+.34), DAT (+.22,+.28),
            # PACE (+.32,+.30) cluster horizontally in the upper band;
            # CDAT-A (+.20,+.17) and RAT (+.20,+.10) stack vertically below
            # the DAT/PACE column; CDAT-N (-.13,-.10) sits bottom-left.
            "DAT":      ( 0.000, +0.060, "center", "bottom"),  # directly above
            "CDAT":     (-0.060, +0.000, "right",  "center"),  # left
            "CDAT-N":   ( 0.060, +0.000, "left",   "center"),  # right
            "CDAT-A":   (-0.060, +0.000, "right",  "center"),  # left (away from RAT)
            "PACE":     ( 0.060, +0.000, "left",   "center"),  # right
            "RAT":      ( 0.000, -0.060, "center", "top"),     # directly below (away from CDAT-A)
        },
    }

    # ---------- Bottom-row data (recomputed per benchmark) ----------
    bench_keys = {
        "Arena CW":        "arena_cw",
        "EQ-Bench CW":     "eq_bench_cw",
        "Mazur CW v2":     "mazur_cw_v2",
        "Hivemind Div.":   "hivemind_diversity",
        "NovBench Util.":  "noveltybench_utility",
        "LiveIdeaBench":   "liveideabench",
    }
    me_path = RESULTS_DIR / "multi_embed_scores.json"
    with open(me_path) as f:
        me = json.load(f)
    with open(BENCH_PATH) as f:
        BMARKS = json.load(f)

    # RAT scores (zero-shot strict accuracy). Pilot + expansion summaries;
    # pilot wins on overlap. Keys are OpenRouter ids ("openai/gpt-4o");
    # we lookup against ms_yg (ormap form) via a converted map.
    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    rat_zs: dict[str, float] = {}
    for _path in [
        PROJECT_ROOT / "data/new_tests/rat/expansion_v1/summary.json",
        PROJECT_ROOT / "data/new_tests/rat/pilot_v1/summary.json",
    ]:
        if _path.exists():
            with open(_path) as f:
                _rat_summary = json.load(f)
            for _m, _s in _rat_summary.items():
                if _s.get("n_total", 0) > 0 and _s.get("n_errors", 0) < _s["n_total"]:
                    rat_zs[_m] = _s["zs_accuracy_strict"]

    def _ormap(s: str) -> str:
        return s.replace("/", "_").replace(".", "-").replace(":", "_")
    rat_orm = {_ormap(m): v for m, v in rat_zs.items()}

    embs = sorted(me.keys())
    tasks_all = ["dat", "cdat", "cdat_novelty", "cdat_appropriateness", "pace"]
    all_models = sorted({m for emb in embs for m in me[emb]})
    composite: dict[str, dict[str, float]] = {}
    for t in tasks_all:
        stats = {}
        for emb in embs:
            vals = [me[emb].get(m, {}).get(t) for m in all_models]
            vals = [v for v in vals if v is not None
                    and not (isinstance(v, float) and (np.isnan(v) or v == 0))]
            if vals:
                stats[emb] = (float(np.mean(vals)), float(np.std(vals)) or 1.0)
        for m in all_models:
            zs = []
            for emb in embs:
                if emb not in stats:
                    continue
                v = me[emb].get(m, {}).get(t)
                if v is None or (isinstance(v, float) and (np.isnan(v) or v == 0)):
                    continue
                mu, sd = stats[emb]
                zs.append((v - mu) / sd)
            if zs:
                composite.setdefault(m, {})[t] = float(np.mean(zs))

    from scipy.stats import pearsonr as _pearsonr
    bottom_panels = []  # (label, R, {test: (v, s)})
    bottom_panel_sig = {}  # label -> {test: (p_v < .05, p_s < .05)}
    for blabel, bkey in bench_keys.items():
        ms_yg = [m for m, d in BMARKS.items()
                 if bkey in d and "arena_overall" in d and "mmlu_pro" in d]
        ys = np.array([BMARKS[m][bkey] for m in ms_yg], dtype=float)
        A = np.array([[BMARKS[m]["arena_overall"], BMARKS[m]["mmlu_pro"]]
                      for m in ms_yg], dtype=float)
        A1 = np.column_stack([np.ones(len(ys)), A])
        beta = np.linalg.lstsq(A1, ys, rcond=None)[0]
        yhat = A1 @ beta
        R2 = 1.0 - np.sum((ys - yhat) ** 2) / np.sum((ys - ys.mean()) ** 2)
        R = float(np.sqrt(max(0.0, R2)))
        if np.corrcoef(A[:, 0], ys)[0, 1] < 0:
            R = -R
        resid_by_m = dict(zip(ms_yg, ys - yhat))
        y_by_m = dict(zip(ms_yg, ys))
        pts: dict[str, tuple[float, float]] = {}
        for tlabel in test_order:
            if tlabel == "RAT":
                # RAT — accuracy from rat_orm (ormap-keyed); no embedding.
                kept = [m for m in ms_yg if m in rat_orm]
                if len(kept) < 5:
                    continue
                xs = np.array([rat_orm[m] for m in kept])
            else:
                tk = test_key[tlabel]
                kept = [m for m in ms_yg if composite.get(m, {}).get(tk) is not None]
                if len(kept) < 5:
                    continue
                xs = np.array([composite[m][tk] for m in kept])
            ys_k = np.array([y_by_m[m] for m in kept])
            rs_k = np.array([resid_by_m[m] for m in kept])
            v_r, p_v = _pearsonr(xs, ys_k)
            s_r, p_s = _pearsonr(xs, rs_k)
            v = float(v_r); s = float(s_r)
            pts[tlabel] = (v, s)
            bottom_panel_sig.setdefault(blabel, {})[tlabel] = (
                float(p_v) < 0.05, float(p_s) < 0.05,
            )
        bottom_panels.append((blabel, R, pts))

    # Top-row panel-average ceilings need R values per construct
    panel_bench_keys = {
        "Creative Writing":    ["arena_cw", "eq_bench_cw", "mazur_cw_v2"],
        "Divergent Thinking":  ["hivemind_diversity", "noveltybench_utility"],
        "Scientific Ideation": ["liveideabench"],
    }
    panel_R = {}
    for panel, keys in panel_bench_keys.items():
        Rs = [_benchmark_signed_R(k, BMARKS) for k in keys]
        panel_R[panel] = [r for r in Rs if r is not None]

    # ---------- LAYOUT ----------
    fig = plt.figure(figsize=(20.0, 12.4))
    gs = GridSpec(
        2, 6, figure=fig,
        height_ratios=[1.30, 1.00],
        hspace=0.55, wspace=0.20,
        left=0.045, right=0.99, top=0.935, bottom=0.165,
    )
    ax_top_l = fig.add_subplot(gs[0, 0:2])
    ax_top_m = fig.add_subplot(gs[0, 2:4], sharex=ax_top_l, sharey=ax_top_l)
    ax_top_r = fig.add_subplot(gs[0, 4:6], sharex=ax_top_l, sharey=ax_top_l)
    axes_bot = [fig.add_subplot(gs[1, i]) for i in range(6)]
    for ax in axes_bot[1:]:
        ax.sharex(axes_bot[0]); ax.sharey(axes_bot[0])

    # Row-level subplot titles (above each row, centred on the figure).
    fig.text(0.5, 0.975, "(a) Prediction by construct",
             ha="center", va="bottom", fontsize=28.0, fontweight="bold")
    fig.text(0.5, 0.485, "(b) Prediction by benchmark",
             ha="center", va="bottom", fontsize=28.0, fontweight="bold")

    # ---------- TOP ROW ----------
    title_fs_top, axis_fs_top, tick_fs_top, annotate_fs_top = 25.0, 22.0, 18.0, 16.0
    s_ind, s_overall, s_star = 60, 290, 65
    overall_edge_lw, ind_sig_lw, ind_nosig_lw = 1.4, 1.0, 0.4
    star_off_pts = 8

    def draw_top(ax, data, benchmarks, title):
        ax.add_patch(Rectangle(
            (0, 0), 10, 10,
            facecolor="#7ec587", alpha=0.10,
            edgecolor="none", zorder=-1,
        ))
        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        v_grid = np.linspace(-1, 1, 400)
        R_list = panel_R.get(title, [])
        ceil_y = None
        if R_list:
            ceil_y = _panel_max_ceiling(R_list, v_grid)
            ax.plot(v_grid, ceil_y,
                    color="black", linewidth=1.6, linestyle="-",
                    alpha=0.9, zorder=1)
            # Mark the point on the ceiling that maximises v + r(X,Y|g).
            opt_idx = int(np.argmax(v_grid + ceil_y))
            v_opt, s_opt = v_grid[opt_idx], ceil_y[opt_idx]
            ax.scatter(v_opt, s_opt, marker="D", s=70,
                       facecolor="black", edgecolor="black",
                       linewidth=0.8, zorder=7)
            ax.annotate(
                f"$(v^*, s^*)=({v_opt:+.2f}, {s_opt:+.2f})$",
                xy=(v_opt, s_opt),
                xytext=(-12, -8), textcoords="offset points",
                fontsize=19.0, ha="right", va="top", color="black",
                zorder=8,
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=2.0, alpha=0.85),
            )
        star_trans = mtransforms.offset_copy(
            ax.transData, fig=fig,
            x=star_off_pts, y=star_off_pts, units="points",
        )
        for test in test_order:
            pts = data[test]
            color = test_colors[test]
            for bench_label, (val, spec, val_sig, spec_sig) in zip(benchmarks, pts):
                any_sig = val_sig or spec_sig
                both_pos_sig = (val_sig and spec_sig and val > 0 and spec > 0)
                ax.scatter(val, spec, marker="o", s=s_ind, c=[color],
                           edgecolor=("black" if any_sig else "white"),
                           linewidth=(ind_sig_lw if any_sig else ind_nosig_lw),
                           zorder=3, alpha=0.55)
                if both_pos_sig:
                    ax.scatter(val, spec, marker="*", s=s_star,
                               facecolor="#ffcc00", edgecolor="black",
                               linewidth=0.4, zorder=5, transform=star_trans)
            vals = [v for v, _, *_ in pts]
            specs = [s for _, s, *_ in pts]
            mx, my = float(np.mean(vals)), float(np.mean(specs))
            ax.scatter(mx, my, marker="o", s=s_overall, c=[color],
                       edgecolor="black", linewidth=overall_edge_lw, zorder=4)
            dx, dy, ha, va = label_offsets[title][test]
            ax.annotate(
                test, xy=(mx, my), xytext=(mx + dx, my + dy),
                ha=ha, va=va, fontsize=annotate_fs_top, fontweight="bold",
                color="black", zorder=6,
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=1.0, alpha=0.85),
            )
        ax.set_title(title, fontsize=title_fs_top)
        ax.tick_params(axis="both", labelsize=tick_fs_top)
        ax.set_xlabel(r"Validity  ($r$)", fontsize=axis_fs_top)

    # SI panel has only one benchmark (LiveIdeaBench), so its top-row
    # data must equal the bottom-row LIB cells exactly. Rebuild si_data
    # live from bottom_panels so the two panels can't drift apart.
    _lib_pts = next((pts for blabel, _R, pts in bottom_panels
                     if blabel == "LiveIdeaBench"), None)
    _lib_sig = bottom_panel_sig.get("LiveIdeaBench", {})
    if _lib_pts is not None:
        si_data = {
            t: [(v, s,
                 _lib_sig.get(t, (False, False))[0],
                 _lib_sig.get(t, (False, False))[1])]
            for t, (v, s) in _lib_pts.items()
        }

    draw_top(ax_top_l, cw_data, cw_benchmarks, "Creative Writing")
    draw_top(ax_top_m, dt_data, dt_benchmarks, "Divergent Thinking")
    draw_top(ax_top_r, si_data, si_benchmarks, "Scientific Ideation")
    ax_top_l.set_ylabel(r"Specificity  ($r \mid g$)", fontsize=axis_fs_top)

    all_vals, all_specs = [], []
    for data in (cw_data, dt_data, si_data):
        for pts in data.values():
            for v, s, *_ in pts:
                all_vals.append(v); all_specs.append(s)
            all_vals.append(float(np.mean([v for v, _, *_ in pts])))
            all_specs.append(float(np.mean([s for _, s, *_ in pts])))
    # Include the per-panel optima so the diamond markers aren't clipped.
    v_grid_full = np.linspace(-1, 1, 400)
    opt_xs, opt_ys = [], []
    for Rs in panel_R.values():
        if Rs:
            ceil = _panel_max_ceiling(Rs, v_grid_full)
            j = int(np.argmax(v_grid_full + ceil))
            opt_xs.append(v_grid_full[j])
            opt_ys.append(ceil[j])
    xpad = 0.12 * (max(all_vals) - min(all_vals))
    ypad = 0.10 * (max(all_specs) - min(all_specs))
    xlim_lo = min(all_vals) - xpad
    xlim_hi = max(max(all_vals), max(opt_xs) if opt_xs else 0) + xpad
    ymax_data = max(all_specs) + ypad
    ymax_top = max(ymax_data,
                   (max(opt_ys) + 0.06) if opt_ys else ymax_data)
    ymin_top = min(all_specs) - ypad
    ax_top_l.set_xlim(xlim_lo, xlim_hi)
    ax_top_l.set_ylim(ymin_top, ymax_top)
    for ax in (ax_top_m, ax_top_r):
        plt.setp(ax.get_yticklabels(), visible=False)

    # ---------- BOTTOM ROW ----------
    v_grid_full = np.linspace(-1, 1, 400)
    title_fs_bot = 21.0
    for i, (ax, (bench_name, c, pts)) in enumerate(zip(axes_bot, bottom_panels)):
        root1mc2 = np.sqrt(max(0.0, 1 - c**2))
        root1mv2 = np.sqrt(np.clip(1 - v_grid_full**2, 0, None))
        upper = v_grid_full * root1mc2 + abs(c) * root1mv2
        lower = v_grid_full * root1mc2 - abs(c) * root1mv2
        ax.fill_between(v_grid_full, lower, upper,
                         color="#d0d0d0", alpha=0.35, zorder=0)
        ax.plot(v_grid_full, upper, color="black", linewidth=1.6,
                linestyle="-", zorder=1)
        ax.plot(v_grid_full, lower, color="black", linewidth=0.7,
                linestyle=":", alpha=0.5, zorder=1)
        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":",
                    alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":",
                    alpha=0.7, zorder=0)
        # Mark the point on the upper ceiling that maximises v + s.
        opt_idx = int(np.argmax(v_grid_full + upper))
        v_opt, s_opt = v_grid_full[opt_idx], upper[opt_idx]
        ax.scatter(v_opt, s_opt, marker="D", s=55,
                   facecolor="black", edgecolor="black",
                   linewidth=0.8, zorder=7)
        for test in test_order:
            pt = pts.get(test)
            if pt is None:
                continue
            v, spec = pt
            ax.scatter(v, spec, marker="o", s=85, c=[test_colors[test]],
                       edgecolor="black", linewidth=0.9, zorder=3, alpha=0.95)
        ax.set_title(bench_name, fontsize=title_fs_bot)
        ax.tick_params(axis="both", labelsize=18.0)
        ax.set_xlabel(r"Validity  ($r$)", fontsize=22.0)
    axes_bot[0].set_xlim(-1.02, 1.02)
    axes_bot[0].set_ylim(-1.05, 1.05)
    axes_bot[0].set_ylabel(r"Specificity  ($r \mid g$)", fontsize=22.0)
    for ax in axes_bot[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    # ---------- LEGEND ----------
    test_handles = [
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor=test_colors[t], markeredgecolor="black",
               markeredgewidth=1.0, markersize=15, label=t)
        for t in test_order
    ]
    ceiling_handle = Line2D([], [], color="black", linewidth=1.4,
                             linestyle="-", label="theoretical ceiling")
    optimum_handle = Line2D([], [], marker="D", linestyle="none",
                             markerfacecolor="black", markeredgecolor="black",
                             markersize=10, label=r"$(v^*, s^*) = \arg\max\,(v + s)$")
    fig.legend(
        handles=[*test_handles, ceiling_handle, optimum_handle],
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=len(test_handles) + 2, frameon=False, fontsize=22,
        handletextpad=0.4, columnspacing=1.4,
    )

    for out_dir in [FIGS_DIR, PAPER_FIGS_DIR]:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "fig_headline.pdf"
        plt.savefig(out)
        plt.savefig(out.with_suffix(".png"))
        print(f"Saved {out}")
    plt.close()


def fig_qualitative_embedding():
    """Per-test 2D t-SNE projections of response words from three
    models across DAT, CDAT, and PACE.

    One panel per test. Within each panel, each model's response words
    are embedded under SBERT and jointly projected to 2D with t-SNE so
    the reader can see how the three models' word clouds arrange
    themselves on that test. Words are labelled inline. Same three
    model colours across all panels.
    """
    from sklearn.manifold import TSNE
    import json

    ROOT = Path(__file__).resolve().parents[3]
    RUN_DIR = ROOT / "data" / "dat_eval" / "run_v1"
    sys.path.insert(0, str(ROOT))
    from src.dat_eval.scripts.multi_embed_appendix import SBERTEmbedder

    print("Loading SBERT...", flush=True)
    sbert = SBERTEmbedder("all-mpnet-base-v2")

    models = [
        ("mistralai_mistral-7b-instruct-v0-1", "Mistral 7B",      "#D95F02"),
        ("anthropic_claude-3-haiku",            "Claude 3 Haiku",  "#7570B3"),
        ("openai_gpt-5",                        "GPT-5",           "#1B9E77"),
    ]
    tests = [
        ("DAT",  "dat_responses_t1-0.json"),
        ("CDAT", "cdat_responses_t1-5.json"),
        ("PACE", "pace_responses_t0-0.json"),
    ]

    def get_words(mkey: str, test: str, fname: str) -> list[str]:
        p = RUN_DIR / mkey / fname
        if not p.exists():
            return []
        d = json.load(open(p))
        if test == "DAT":
            for t in d:
                if t.get("words"):
                    return t["words"][:10]
            return []
        if test == "CDAT":
            return d.get("rock", {}).get("words", [])[:10]
        if test == "PACE":
            chains = d.get("rock", {}).get("chains", [])
            if not chains:
                return []
            return chains[0].get("chain", [])[:20]
        return []

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.0))

    for ai, (test, fname) in enumerate(tests):
        ax = axes[ai]
        # Collect (word, model_idx) for this test
        words, model_idx = [], []
        for mi, (mkey, _, _) in enumerate(models):
            for w in get_words(mkey, test, fname):
                words.append(w)
                model_idx.append(mi)
        if not words:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                     transform=ax.transAxes, color=C_GREY)
            ax.set_title(test, fontsize=12, weight="bold")
            continue

        X = np.vstack([sbert.encode(w) for w in words])
        # Per-test t-SNE with perplexity suited to the set size.
        perplexity = max(3, min(15, (len(words) - 1) // 3))
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                     random_state=0, learning_rate="auto")
        Y = tsne.fit_transform(X)

        # Draw points per model, coloured by model.
        for mi, (_, mlabel, mcolor) in enumerate(models):
            mask = [i for i, m in enumerate(model_idx) if m == mi]
            if not mask:
                continue
            pts = Y[mask]
            ax.scatter(pts[:, 0], pts[:, 1], s=80, color=mcolor,
                       alpha=0.8, edgecolor="white", linewidth=0.7,
                       zorder=3, label=mlabel if ai == 0 else None)

        # Label every word inline so the reader can read what each
        # cluster contains.
        for i, w in enumerate(words):
            mcolor = models[model_idx[i]][2]
            ax.annotate(w, xy=(Y[i, 0], Y[i, 1]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=8.0, color=mcolor,
                        zorder=4)

        ax.set_title(test, fontsize=12, weight="bold", pad=8)
        ax.set_xlabel("t-SNE 1", fontsize=9)
        if ai == 0:
            ax.set_ylabel("t-SNE 2", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=7,
                        labelleft=(ai == 0))
        # Expand limits so labels fit inside the panel.
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.set_xlim(xlim[0] - 8, xlim[1] + 30)
        ax.set_ylim(ylim[0] - 8, ylim[1] + 8)

    # Single legend across the whole figure.
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=mcolor, markersize=9, markeredgecolor="white",
                   markeredgewidth=0.6, label=mlabel)
        for _, mlabel, mcolor in models
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
               fontsize=10)

    fig.suptitle("Response words projected into SBERT embedding space (per-test t-SNE)",
                  fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])

    out = FIGS_DIR / "fig_qualitative_embedding.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig_qualitative_heatmaps():
    """3x3 grid of pairwise-cosine-distance heatmaps.

    Rows = three models spanning the capability range (Mistral 7B,
    Claude 3 Haiku, GPT-5). Columns = the three tests (DAT, CDAT, PACE).
    Each cell's heatmap shows the cosine distance between every pair of
    words in that test response under the test's canonical embedding
    (GloVe for DAT, SBERT for CDAT, FastText for PACE). Axes are labelled
    with the actual words.
    """
    from scipy.spatial.distance import cosine as cosine_distance
    import json

    ROOT = Path(__file__).resolve().parents[3]
    RUN_DIR = ROOT / "data" / "dat_eval" / "run_v1"
    sys.path.insert(0, str(ROOT))
    from src.dat_eval.scripts.multi_embed_appendix import (
        GloVeEmbedder, FastTextEmbedder, SBERTEmbedder,
    )

    print("Loading GloVe...", flush=True)
    glove = GloVeEmbedder()
    print("Loading FastText...", flush=True)
    fasttext = FastTextEmbedder()
    print("Loading SBERT...", flush=True)
    sbert = SBERTEmbedder("all-mpnet-base-v2")
    embedders = {"glove": glove, "fasttext": fasttext, "sbert": sbert}

    models = [
        ("mistralai_mistral-7b-instruct-v0-1", "Mistral 7B"),
        ("anthropic_claude-3-haiku",            "Claude 3 Haiku"),
        ("openai_gpt-5",                        "GPT-5"),
    ]
    tests = [
        ("DAT",  "glove",    "dat_responses_t1-0.json"),
        ("CDAT", "sbert",    "cdat_responses_t1-5.json"),
        ("PACE", "fasttext", "pace_responses_t0-0.json"),
    ]

    def get_words(model_key: str, test: str, fname: str) -> list[str]:
        p = RUN_DIR / model_key / fname
        if not p.exists():
            return []
        d = json.load(open(p))
        if test == "DAT":
            for t in d:
                if t.get("words"):
                    return t["words"][:10]
            return []
        if test == "CDAT":
            return d.get("rock", {}).get("words", [])[:10]
        if test == "PACE":
            chains = d.get("rock", {}).get("chains", [])
            if not chains:
                return []
            # Chain positions come as a list of strings in 'chain' field.
            return chains[0].get("chain", [])[:20]
        return []

    def pairwise_distance(words: list[str], embedder) -> tuple[np.ndarray, list[str]]:
        vecs, valid = [], []
        for w in words:
            v = embedder.encode(w)
            if np.linalg.norm(v) > 0:
                vecs.append(v)
                valid.append(w)
        n = len(vecs)
        if n < 2:
            return np.zeros((0, 0)), valid
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.linalg.norm(vecs[i]) == 0 or np.linalg.norm(vecs[j]) == 0:
                    M[i, j] = np.nan
                else:
                    M[i, j] = cosine_distance(vecs[i], vecs[j])
        return M, valid

    fig, axes = plt.subplots(3, 3, figsize=(11.5, 12.5))
    scores_by_cell: dict[tuple[str, str], float] = {}
    vmax = 0.0

    # First pass: build all matrices, find a shared vmax for the colour scale
    cells = {}
    for i, (mkey, mlabel) in enumerate(models):
        for j, (test, emb_key, fname) in enumerate(tests):
            words = get_words(mkey, test, fname)
            emb = embedders[emb_key]
            M, valid = pairwise_distance(words, emb)
            cells[(i, j)] = (M, valid)
            if M.size:
                # Mean off-diagonal as a uniform descriptive score (×100 to match DAT convention).
                triu = M[np.triu_indices(M.shape[0], k=1)]
                triu = triu[~np.isnan(triu)]
                if triu.size:
                    scores_by_cell[(mlabel, test)] = float(triu.mean()) * 100
                    vmax = max(vmax, float(np.nanmax(M)))

    vmax = max(vmax, 1.0)

    for i, (mkey, mlabel) in enumerate(models):
        for j, (test, emb_key, fname) in enumerate(tests):
            ax = axes[i, j]
            M, words = cells[(i, j)]
            if M.size == 0:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                         transform=ax.transAxes, color=C_GREY, fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            im = ax.imshow(M, vmin=0, vmax=vmax, cmap=CMAP_SEQ, aspect="equal")
            n = len(words)
            # Blank out the strictly upper triangle (matrix is symmetric) by
            # painting white rectangles on top. Matches the lower-triangle
            # style used in fig_benchmark_correlations.
            for a in range(n):
                for b in range(n):
                    if b > a:
                        ax.add_patch(plt.Rectangle(
                            (b - 0.5, a - 0.5), 1, 1,
                            facecolor="white", edgecolor="none", zorder=2))
            # Hide the axis spines so the triangle sits flush on the page.
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Label every word on both axes; shrink font as words grow.
            label_fs = 7.0 if n <= 10 else 5.5
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(words, rotation=60, ha="right", fontsize=label_fs)
            ax.set_yticklabels(words, fontsize=label_fs)
            ax.tick_params(axis="both", which="major", length=0)
            # Annotate each lower-triangle cell with its cosine distance.
            # Scale the font with n so 20x20 PACE heatmaps stay legible.
            ann_fs = 6.0 if n <= 10 else 4.2
            for a in range(n):
                for b in range(a + 1):
                    if a == b:
                        txt = "—"
                        colour = C_GREY
                    else:
                        v = M[a, b]
                        if np.isnan(v):
                            continue
                        txt = f"{v:.2f}".lstrip("0") if v < 1 else f"{v:.2f}"
                        # Pick ink colour for contrast against the Batlow fill.
                        colour = "white" if v < 0.45 else "black"
                    ax.text(b, a, txt, ha="center", va="center",
                            fontsize=ann_fs, color=colour, zorder=3)
            # Title: "Model x Test  score=NN.N" (plain text — LaTeX \textit isn't
            # supported by matplotlib outside math mode).
            score = scores_by_cell.get((mlabel, test))
            title = f"{mlabel} $\\times$ {test}"
            if score is not None:
                title += f"     score = {score:.1f}"
            ax.set_title(title, fontsize=10, pad=6)

    fig.subplots_adjust(hspace=0.6, wspace=0.35, right=0.9)

    # Shared colour bar in its own axes so subplots_adjust has control.
    cbar_ax = fig.add_axes([0.915, 0.28, 0.013, 0.44])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("cosine distance", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    out = FIGS_DIR / "fig_qualitative_heatmaps.pdf"
    plt.savefig(out)
    plt.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"Saved {out}")
    print("Per-cell mean pairwise cosine distance × 100:")
    for (mlabel, test), s in sorted(scores_by_cell.items()):
        print(f"  {mlabel:18s} {test:4s}: {s:5.1f}")


def main():
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    corr, scores, benchmarks = load_data()

    fig1_correlation_matrix(corr)
    fig2_combined_grid(scores, benchmarks)
    fig4_cdat_by_temperature(corr)
    fig_correlation_summary_heatmap(corr)
    fig_inter_metric_triangle(corr)
    fig_benchmark_correlations(benchmarks)
    fig_scatter_by_embedding(benchmarks)
    fig_validity_specificity(benchmarks)
    fig_headline_combined()

    print(f"\nAll figures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
