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
    ("cdat",                "CDAT (gated novelty)",   C_CAPP),
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


def fig_benchmark_correlations(benchmarks):
    """Two vertically stacked Pearson correlation triangles:

      (a) capability proxies + outcome benchmarks (6x6)
      (b) inter-test correlations among DAT, CDAT, PACE using composite
          z-scores across all 3 embeddings (3x3, scaled down)
    """
    from scipy.stats import pearsonr
    from matplotlib.gridspec import GridSpec

    # Panel (a): capability proxies + outcome benchmarks (unchanged logic)
    keys_a = ["arena_overall", "mmlu_pro", "arena_cw", "eq_bench_cw",
              "mazur_cw_v2", "hivemind_diversity", "noveltybench_utility"]
    labels_a = ["Arena Ovr", "MMLU-Pro", "Arena CW", "EQ-B. CW",
                "Mazur V2", "Hive. Div.", "NovB. Util."]
    na = len(keys_a)
    mat_a = np.full((na, na), np.nan)
    for i, ki in enumerate(keys_a):
        for j, kj in enumerate(keys_a):
            xs, ys = [], []
            for _, v in benchmarks.items():
                if ki in v and kj in v:
                    xs.append(v[ki])
                    ys.append(v[kj])
            if len(xs) >= 3:
                r, _ = pearsonr(xs, ys)
                mat_a[i, j] = r

    # Panel (b): inter-test correlations with composite (overall z-score)
    composite = load_composite_scores()
    tasks = ["dat", "cdat", "cdat_novelty", "cdat_appropriateness", "pace"]
    labels_b = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "PACE"]
    nb = len(tasks)
    mat_b = np.full((nb, nb), np.nan)
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            xs, ys = [], []
            for _, sc in composite.items():
                vi = sc.get(ti)
                vj = sc.get(tj)
                if vi is None or vj is None:
                    continue
                if isinstance(vi, float) and np.isnan(vi):
                    continue
                if isinstance(vj, float) and np.isnan(vj):
                    continue
                xs.append(vi); ys.append(vj)
            if len(xs) >= 3:
                r, _ = pearsonr(xs, ys)
                mat_b[i, j] = r

    def draw_triangle(ax, mat, labels):
        n = len(labels)
        display = np.where(np.isnan(mat), 0.0, mat)
        im = ax.imshow(display, vmin=-1, vmax=1, cmap=CMAP_SEQ, aspect="equal")
        for i in range(n):
            for j in range(n):
                if j > i:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                color="white", zorder=2))
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        for i in range(n):
            for j in range(n):
                if j > i:
                    continue
                v = mat[i, j]
                if np.isnan(v):
                    continue
                txt = "—" if i == j else f"{v:+.2f}"
                color = "white" if v < -0.2 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7.5, color=color, zorder=3)
        ax.tick_params(axis="both", which="major", length=0)
        return im

    # Two stacked panels. Panel (a) spans the full width; panel (b) is
    # half-width so its cells stay the same visual size as panel (a).
    fig = plt.figure(figsize=(3.6, 5.4))
    gs = GridSpec(2, 2, figure=fig,
                   height_ratios=[na, nb], width_ratios=[nb, na - nb],
                   hspace=0.55, wspace=0.0)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])

    im1 = draw_triangle(ax1, mat_a, labels_a)
    draw_triangle(ax2, mat_b, labels_b)

    ax1.set_title("(a) capability proxies + outcome benchmarks",
                   fontsize=8.5, loc="left", pad=4)
    ax2.set_title("(b) inter-test correlations\n(composite across embeddings)",
                   fontsize=8.5, loc="left", pad=4)

    cbar = fig.colorbar(im1, ax=[ax1, ax2], shrink=0.55, pad=0.04,
                         location="right")
    cbar.set_label("Pearson $r$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    out = FIGS_DIR / "fig_benchmark_correlations.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


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
        ("cdat", "CDAT", C_CNOV),
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


def fig_headline():
    """Two-panel headline scatter pulling the ``Overall'' (mean z-score
    across GloVe / FastText / SBERT) block from Table~1. Left panel =
    Creative Writing benchmarks (Arena CW, EQ-Bench CW, Mazur CW v2),
    right panel = Divergent Thinking benchmarks (Hivemind diversity,
    NoveltyBench utility). Colours encode tests; small translucent
    circles are per-benchmark cells and the large black-outlined circle
    per test is the within-panel benchmark average (``Overall'').

    Saved to both the report's figures directory and the ICCC paper's
    figures directory so the paper can reference the same file via a
    ``figure*`` (two-column) environment.
    """
    from matplotlib.lines import Line2D

    figsize = (8.4, 4.8)
    title_fs, axis_fs, tick_fs = 14.0, 10.5, 9.0
    leg_fs, leg_title_fs, annotate_fs = 12.0, 12.0, 8.5
    s_ind, s_overall, s_star = 38, 170, 40
    overall_edge_lw, ind_sig_lw, ind_nosig_lw = 1.3, 0.9, 0.4
    leg_ms_test, leg_ms_overall, leg_ms_star, leg_ms_outl = 12, 14, 13, 9
    star_off_pts = 6
    rect = [0, 0.09, 1, 1]
    out_dirs = [FIGS_DIR, PAPER_FIGS_DIR]

    # Colors encode tests. 6 well-separated categorical samples from Batlow;
    # upper bound capped at 0.82 so PACE stays saturated.
    test_samples = CMAP_SEQ(np.linspace(0.05, 0.82, 6))
    test_colors = {
        "DAT":      test_samples[0],
        "CDAT":     test_samples[1],
        "CDAT-N":   test_samples[2],
        "CDAT-A":   test_samples[3],
        "CDAT-N×A": test_samples[4],
        "PACE":     test_samples[5],
    }
    test_order = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "CDAT-N×A", "PACE"]

    cw_benchmarks = ["Arena CW", "EQ-Bench CW", "Mazur CW v2"]
    dt_benchmarks = ["Hivemind Div.", "NovBench Util."]

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
        "CDAT-N×A": [(+0.40, +0.30, True,  True),
                     (+0.39, +0.28, True,  False),
                     (+0.54, +0.43, True,  False)],
        "PACE":     [(+0.72, +0.05, True,  False),
                     (+0.70, +0.20, True,  False),
                     (+0.75, +0.18, True,  False)],
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
        "CDAT-N×A": [(-0.07, +0.14, False, False),
                     (+0.33, +0.25, False, False)],
        "PACE":     [(-0.05, +0.37, False, False),
                     (+0.18, -0.06, False, False)],
    }

    # Per-panel label placement for the black "Overall" composite points.
    # (dx, dy, horizontal-align, vertical-align) in data coordinates.
    label_offsets = {
        "Creative Writing": {
            "DAT":      (+0.030, +0.008, "left",   "center"),
            "CDAT":     ( 0.000, +0.042, "center", "bottom"),
            "CDAT-N":   ( 0.000, -0.042, "center", "top"),
            "CDAT-A":   ( 0.000, -0.042, "center", "top"),
            "CDAT-N×A": ( 0.000, +0.048, "center", "bottom"),
            "PACE":     (-0.030, +0.008, "right",  "center"),
        },
        "Divergent Thinking": {
            "DAT":      ( 0.000, -0.042, "center", "top"),
            "CDAT":     (+0.030, +0.008, "left",   "center"),
            "CDAT-N":   (-0.030, +0.008, "right",  "center"),
            "CDAT-A":   ( 0.000, +0.048, "center", "bottom"),
            "CDAT-N×A": ( 0.000, +0.048, "center", "bottom"),
            "PACE":     ( 0.000, -0.048, "center", "top"),
        },
    }

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True,
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
    _plot(ax_r, dt_data, dt_benchmarks, "Divergent Thinking")
    ax_l.set_ylabel(r"Specificity  ($r \mid g$)", fontsize=axis_fs)

    # Axis limits: include composite points too.
    all_vals, all_specs = [], []
    for data in (cw_data, dt_data):
        for pts in data.values():
            for v, s, *_ in pts:
                all_vals.append(v); all_specs.append(s)
            all_vals.append(float(np.mean([v for v, _, *_ in pts])))
            all_specs.append(float(np.mean([s for _, s, *_ in pts])))
    xpad = 0.12 * (max(all_vals) - min(all_vals))
    ypad = 0.10 * (max(all_specs) - min(all_specs))
    for ax in (ax_l, ax_r):
        ax.set_xlim(min(all_vals) - xpad, max(all_vals) + xpad)
        ax.set_ylim(min(all_specs) - ypad, max(all_specs) + ypad)

    # --- Single flat Test legend at the bottom. Indicator conventions
    # (Overall marker, both-axes star, one-axis outline) are explained
    # in the figure caption rather than in the figure itself. ---
    test_handles = [
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor=test_colors[t], markeredgecolor="black",
               markeredgewidth=1.0, markersize=leg_ms_test, label=t)
        for t in test_order
    ]
    leg_tests = fig.legend(
        handles=test_handles,
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

    test_samples = CMAP_SEQ(np.linspace(0.05, 0.82, 6))
    test_colors = {
        "DAT":      test_samples[0],
        "CDAT":     test_samples[1],
        "CDAT-N":   test_samples[2],
        "CDAT-A":   test_samples[3],
        "CDAT-N×A": test_samples[4],
        "PACE":     test_samples[5],
    }
    test_order = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "CDAT-N×A", "PACE"]

    # (benchmark label, c = r(Y, Arena Overall) computed from benchmarks.json)
    benchmarks = [
        ("Arena CW",        +0.98),
        ("EQ-Bench CW",     +0.83),
        ("Mazur CW v2",     +0.79),
        ("Hivemind Div.",   -0.67),
        ("NovBench Util.",  -0.27),
    ]

    # Overall block from Table 1: {benchmark: {test: (validity, specificity)}}.
    bench_data = {
        "Arena CW": {
            "DAT":      (+0.44, +0.08),
            "CDAT":     (-0.13, +0.28),
            "CDAT-N":   (-0.18, +0.23),
            "CDAT-A":   (+0.54, -0.12),
            "CDAT-N×A": (+0.40, +0.30),
            "PACE":     (+0.72, +0.05),
        },
        "EQ-Bench CW": {
            "DAT":      (+0.71, +0.50),
            "CDAT":     (-0.06, +0.13),
            "CDAT-N":   (-0.14, +0.15),
            "CDAT-A":   (+0.47, -0.02),
            "CDAT-N×A": (+0.39, +0.28),
            "PACE":     (+0.70, +0.20),
        },
        "Mazur CW v2": {
            "DAT":      (+0.59, +0.50),
            "CDAT":     (+0.07, +0.39),
            "CDAT-N":   (+0.09, +0.35),
            "CDAT-A":   (+0.24, -0.21),
            "CDAT-N×A": (+0.54, +0.43),
            "PACE":     (+0.75, +0.18),
        },
        "Hivemind Div.": {
            "DAT":      (+0.33, +0.26),
            "CDAT":     (+0.25, +0.19),
            "CDAT-N":   (+0.24, +0.17),
            "CDAT-A":   (-0.39, -0.16),
            "CDAT-N×A": (-0.07, +0.14),
            "PACE":     (-0.05, +0.37),
        },
        "NovBench Util.": {
            "DAT":      (+0.15, -0.26),
            "CDAT":     (+0.60, +0.57),
            "CDAT-N":   (+0.54, +0.46),
            "CDAT-A":   (-0.67, -0.40),
            "CDAT-N×A": (+0.33, +0.25),
            "PACE":     (+0.18, -0.06),
        },
    }

    v_grid = np.linspace(-1, 1, 400)

    fig, axes = plt.subplots(1, 5, figsize=(15.0, 3.8),
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
            v, spec = bench_data[bench_name][test]
            ax.scatter(v, spec, marker="o", s=55,
                       c=[test_colors[test]],
                       edgecolor="black", linewidth=0.8,
                       zorder=3, alpha=0.95)

        ax.set_title(f"{bench_name}  ($c = {c:+.2f}$)", fontsize=11)
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
    fig_headline()
    fig_specificity_ceilings()

    print(f"\nAll figures saved to {FIGS_DIR}")


if __name__ == "__main__":
    main()
