"""Transformational-creativity (tc) bar chart per source (plot_twist Sec.3).

Implements the paper's metrics exactly:

  Div(T)  = (1 / n(n-1)) * sum_{i!=j} [1 - cos(f(T_i), f(T_j))]     # set twist diversity
  tc(T)   = Div(T) * (1/n) sum_i S(T_i) * Coh(T_i)                  # final tc score

where f(T) is the twist's conceptual embedding (we embed the annotated `reveal`),
S is surprise and Coh is coherence (rubric judge, 1-5). Computed per source (each
LLM and the human gold set) and drawn as a bar chart.

Usage:
    python src/plot_twist/scripts/make_tc_barplot.py configs/plot_twist/tc.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Inter font (downloaded to resources/fonts/inter) for figure styling.
from matplotlib import font_manager as _fm
_inter_dir = Path(__file__).resolve().parents[3] / "resources" / "fonts" / "inter" / "extras" / "ttf"
for _ttf in sorted(_inter_dir.glob("Inter-*.ttf")):
    try:
        _fm.fontManager.addfont(str(_ttf))
    except Exception:
        pass
plt.rcParams.update({"font.family": "Inter", "mathtext.fontset": "stixsans"})

from src.utils import init_directory, load_config, save_config
from src.plot_twist.sets import twist_types, keep_story


def _div(emb: np.ndarray) -> float:
    """Mean pairwise cosine distance (emb rows L2-normalized): 1 - mean off-diagonal cos."""
    if len(emb) < 2:
        return float("nan")
    sims = emb @ emb.T
    n = len(emb)
    return float(1.0 - (sims.sum() - np.trace(sims)) / (n * (n - 1)))


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "annotations_json", "embed_model"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    records = json.loads(Path(cfg["annotations_json"]).read_text())
    field = cfg.get("embed_field", "reveal")

    # group records by source, keeping only those with a reveal + valid S,Coh
    def num(r, k):
        v = r.get("scores", {}).get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Human ceiling = only STRONG (genuine reinterpretation) twists; LLM = all.
    strong_only = cfg.get("human_strong_only", False)
    types = twist_types(cfg["manifest"]) if strong_only else {}

    groups: dict[str, list[dict]] = {}
    dropped = 0
    for r in records:
        if not r.get(field):
            continue
        if num(r, "surprise") is None or num(r, "coherence") is None:
            continue
        if not keep_story(r["id"], types, strong_only):
            dropped += 1
            continue
        groups.setdefault(r["source"], []).append(r)
    if strong_only:
        print(f"human ceiling = STRONG twists only (dropped {dropped} non-STRONG human stories)")

    # Drop sources with too few stories for a reliable diversity estimate (these are
    # mostly reasoning models whose thinking ate the token budget -> empty generations).
    min_stories = cfg.get("min_stories", 5)
    small = [s for s, rs in groups.items() if len(rs) < min_stories]
    for s in small:
        del groups[s]
    if small:
        print(f"dropped {len(small)} sources with <{min_stories} stories: "
              f"{[s.split('/')[-1] for s in small]}")

    # optional realism scores (4th dimension), keyed by story id
    realism = {}
    if cfg.get("realism_scores") and Path(cfg["realism_scores"]).exists():
        realism = json.loads(Path(cfg["realism_scores"]).read_text())

    # embed all reveals once
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["embed_model"])
    all_recs = [r for rs in groups.values() for r in rs]
    embs = model.encode([r[field] for r in all_recs], normalize_embeddings=True, show_progress_bar=False)
    emb_by_id = {id(r): e for r, e in zip(all_recs, np.asarray(embs, dtype=np.float32))}

    # per-source Div, facet means, tc
    rows = []
    for src, rs in groups.items():
        E = np.array([emb_by_id[id(r)] for r in rs])
        div = _div(E)
        sc = np.mean([num(r, "surprise") * num(r, "coherence") for r in rs])
        rv = [realism[r["id"]] for r in rs if r["id"] in realism]
        rows.append({
            "source": src, "n": len(rs), "div": div,
            "mean_surprise": float(np.mean([num(r, "surprise") for r in rs])),
            "mean_coherence": float(np.mean([num(r, "coherence") for r in rs])),
            "mean_realism": float(np.mean(rv)) if rv else float("nan"),
            "mean_SxCoh": float(sc), "tc": div * float(sc),
        })

    # Equal-weighted overall score: z-score each facet across sources (mean 0, SD 1),
    # then average with equal weight. Realism is the 4th equal-weighted dimension when
    # available. Same equal-weight-of-standardized-indicators construction as AGC mean_z.
    EQ_FACETS = ["mean_surprise", "mean_coherence", "div"]
    if realism and all(not np.isnan(d["mean_realism"]) for d in rows):
        EQ_FACETS.append("mean_realism")
        print(f"including realism as a 4th equal-weighted facet")
    _zs = {}
    for k in EQ_FACETS:
        v = np.array([d[k] for d in rows], dtype=float)
        _zs[k] = (float(v.mean()), float(v.std()) or 1.0)
    for d in rows:
        d["overall_eq"] = float(np.mean([(d[k] - _zs[k][0]) / _zs[k][1] for k in EQ_FACETS]))

    rows.sort(key=lambda d: -d["overall_eq"])
    print(f"{'source':<34}{'n':>4}{'Div':>8}{'mean(S*Coh)':>13}{'tc':>9}{'overall_eq':>11}")
    for d in rows:
        print(f"{d['source']:<34}{d['n']:>4}{d['div']:>8.3f}{d['mean_SxCoh']:>13.2f}{d['tc']:>9.3f}{d['overall_eq']:>11.3f}")
    (out / "tc.json").write_text(json.dumps(rows, indent=2))

    # --- bar chart (camera-ready) ---
    def short(s):
        return "Expert humans" if s == "human" else s.split("/")[-1]

    # Batlow palette (Crameri), matching the paper's Figure 1 role colours.
    BATLOW_BLUE = "#103D5F"    # LLM
    BATLOW_ORANGE = "#EE9D6B"  # human (highlight role)
    from matplotlib.patches import Patch

    # Colour LLM bars by model provider; humans keep the batlow-orange highlight.
    PROVIDERS = sorted({d["source"].split("/")[0] for d in rows if d["source"] != "human"})
    from cmcrameri import cm as _cmc  # batlowS = the CATEGORICAL batlow (distinct hues)

    def _adj(c):  # darken only the genuinely near-white batlowS tints so bars stay visible
        r, g, b = c[0], c[1], c[2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        if lum > 0.74:
            s = 0.62 / lum
            r, g, b = r * s, g * s, b * s
        return (r, g, b, 1.0)

    # Long-tail providers collapsed into a single grey "Other" bucket (bars + legend).
    OTHER_PROVIDERS = {"minimax", "morph", "z-ai", "deepcogito",
                       "nousresearch", "tencent", "ai21", "baidu"}
    OTHER_COLOR = "#8c8c8c"
    PROV_COLOR = {p: (OTHER_COLOR if p in OTHER_PROVIDERS else _adj(_cmc.batlowS(i)))
                  for i, p in enumerate(PROVIDERS)}
    HUMAN_COLOR = "#000000"  # Expert humans = black, to stand out from every provider
    BAR_EDGE = dict(edgecolor="#666666", linewidth=0.5)
    PROV_NAME = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google",
                 "meta-llama": "Meta", "deepseek": "DeepSeek", "qwen": "Qwen",
                 "mistralai": "Mistral", "amazon": "Amazon", "nvidia": "NVIDIA",
                 "ibm-granite": "IBM", "z-ai": "Z-AI", "moonshotai": "Moonshot"}

    def bar_color(src):
        return HUMAN_COLOR if src == "human" else PROV_COLOR[src.split("/")[0]]

    from matplotlib.lines import Line2D

    def _swatch(c, lab):
        return Line2D([0], [0], marker="o", linestyle="none", markersize=15,
                      markerfacecolor=c, markeredgecolor="none", label=lab)

    def provider_legend_handles():
        majors = [p for p in PROVIDERS if p not in OTHER_PROVIDERS]
        return [_swatch(HUMAN_COLOR, "Expert humans")] + \
               [_swatch(PROV_COLOR[p], PROV_NAME.get(p, p)) for p in majors] + \
               [_swatch(OTHER_COLOR, "Other")]

    # Horizontal leaderboard (sorted by tc; scales to many models). Human bar highlighted.
    n = len(rows)
    srt = sorted(rows, key=lambda d: -d["overall_eq"])
    labels = [short(d["source"]) for d in srt]
    vals = [d["overall_eq"] for d in srt]
    colors = [bar_color(d["source"]) for d in srt]

    fig, ax = plt.subplots(figsize=(11, max(7, 0.36 * n)))
    y = list(range(n))
    ax.barh(y, vals, color=colors, height=0.78, **BAR_EDGE)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.invert_yaxis()
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == "Expert humans":
            lbl.set_fontweight("bold")
            lbl.set_color("#000000")
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin
    ax.set_xlim(min(0, vmin) - span * 0.12, vmax + span * 0.12)
    ax.axvline(0, color="#999999", lw=0.9, zorder=0)
    for yi, v in zip(y, vals):
        if v >= 0:
            ax.text(v + span * 0.015, yi, f"{v:+.2f}", va="center", ha="left", fontsize=11)
        else:
            ax.text(v - span * 0.015, yi, f"{v:+.2f}", va="center", ha="right", fontsize=11)
    ax.set_xlabel("Overall score (mean z-score)", fontsize=18, labelpad=10)
    ax.tick_params(axis="x", labelsize=13)
    fig.legend(handles=provider_legend_handles(), loc="lower center", ncol=6, fontsize=12,
               frameon=True, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = out / "tc_by_model.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- temperature vs tc line plot (one line per model; human = dotted reference) ---
    import re
    from matplotlib.colors import LinearSegmentedColormap

    batlow = LinearSegmentedColormap.from_list("batlow3", [BATLOW_BLUE, "#426F52", BATLOW_ORANGE])
    model_order = [m for m in [
        "meta-llama/llama-3.2-3b-instruct", "meta-llama/llama-3.1-8b-instruct",
        "google/gemma-3-12b-it", "openai/gpt-4o-mini", "anthropic/claude-haiku-4.5",
        "openai/gpt-4.1", "google/gemini-2.5-pro", "anthropic/claude-opus-4.5",
    ] if m in groups]
    temps = [0.9, 1.0, 1.2]

    def temp_of(rid):
        m = re.search(r"__t(\d{2})__", rid)
        return int(m.group(1)) / 10.0 if m else None

    # tc per (model, temperature)
    bytemp: dict[tuple, list] = {}
    for src in model_order:
        for r in groups[src]:
            t = temp_of(r["id"])
            if t is not None:
                bytemp.setdefault((src, t), []).append(r)
    tc_mt = {}
    for (src, t), rs in bytemp.items():
        if len(rs) < 2:
            continue
        E = np.array([emb_by_id[id(r)] for r in rs])
        sc = np.mean([num(r, "surprise") * num(r, "coherence") for r in rs])
        tc_mt[(src, t)] = _div(E) * float(sc)
    human_tc = next((d["tc"] for d in rows if d["source"] == "human"), None)

    fig, ax = plt.subplots(figsize=(13, 8))
    cols = [batlow(x) for x in np.linspace(0.0, 0.62, len(model_order))]  # blue->green (avoid orange)
    for src, c in zip(model_order, cols):
        ys = [tc_mt.get((src, t)) for t in temps]
        ax.plot(temps, ys, marker="o", ms=10, lw=2.6, color=c, label=short(src))
    if human_tc is not None:
        ax.axhline(human_tc, ls=":", lw=3.2, color="#000000")
        ax.text(temps[-1], human_tc, f"Expert humans ({human_tc:.2f})  ",
                va="bottom", ha="right", fontsize=17, color="#000000")
    ax.set_xticks(temps)
    ax.set_xlabel("Sampling temperature", fontsize=24, labelpad=10)
    ax.set_ylabel("Transformational creativity  tc", fontsize=24, labelpad=12)
    ax.tick_params(labelsize=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=16,
              title="Model", title_fontsize=18, labelspacing=0.5)
    fig.tight_layout()
    p_temp = out / "tc_vs_temperature.png"
    fig.savefig(p_temp, dpi=220, bbox_inches="tight")
    plt.close(fig)
    (out / "tc_by_temp.json").write_text(
        json.dumps({f"{s}|{t}": v for (s, t), v in tc_mt.items()}, indent=2)
    )

    # --- composite: (a) Overall (top 20, LHS, vertical) | (b,c,d) facet top-10 stacked (RHS) ---
    colr = bar_color  # colour by provider; humans = batlow orange

    def bold_human(getters):
        for lbl in getters:
            if lbl.get_text() == "Expert humans":
                lbl.set_fontweight("bold")
                lbl.set_color("#000000")

    TOP_A, TOP_R = 20, 10
    topA = sorted(rows, key=lambda d: -d["overall_eq"])[:TOP_A]

    facet_specs = [("mean_surprise", "Surprise", "{:.2f}"),
                   ("mean_coherence", "Coherence", "{:.2f}"),
                   ("div", "Diversity", "{:.3f}")]
    if "mean_realism" in EQ_FACETS:
        facet_specs.append(("mean_realism", "Realistic", "{:.2f}"))
    nrhs = len(facet_specs)
    fig = plt.figure(figsize=(25, max(12.5, 3.3 * nrhs)))
    gs = fig.add_gridspec(nrhs, 2, width_ratios=[1.3, 1.0], hspace=0.38, wspace=0.42)
    axA = fig.add_subplot(gs[:, 0])
    rhs_axes = [fig.add_subplot(gs[i, 1]) for i in range(nrhs)]

    def despine(ax):
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#888888")
        ax.spines["bottom"].set_color("#888888")
        ax.tick_params(length=0)

    # (a) vertical tc bars, top 20
    barsA = axA.bar(range(len(topA)), [d["overall_eq"] for d in topA],
                    color=[colr(d["source"]) for d in topA], width=0.78, zorder=3, **BAR_EDGE)
    axA.set_xticks(range(len(topA)))
    axA.set_xticklabels([short(d["source"]) for d in topA], rotation=45, ha="right", fontsize=16)
    bold_human(axA.get_xticklabels())
    axA.tick_params(axis="y", labelsize=17)
    axA.set_ylabel("Overall score (mean $z$-score)", fontsize=23, labelpad=10)
    axA.set_title("(a) Overall transformational creativity", fontsize=26, pad=14, fontweight="bold")
    axA.margins(y=0.16)
    axA.axhline(0, color="#555555", lw=1.1, zorder=2)
    axA.grid(axis="y", color="#e2e2e2", lw=0.8, zorder=0)
    despine(axA)
    _vspan = max(d["overall_eq"] for d in topA) - min(d["overall_eq"] for d in topA)
    for b, d in zip(barsA, topA):
        v = d["overall_eq"]
        axA.text(b.get_x() + b.get_width() / 2, v + (_vspan * 0.02 if v >= 0 else -_vspan * 0.02),
                 f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=12, zorder=4)

    # (b..e) horizontal top-10 by each facet
    def hbar(ax, key, title, letter, fmt="{:.2f}"):
        srt = sorted(rows, key=lambda d: -d[key])[:TOP_R]
        vals = [d[key] for d in srt]
        cols = [colr(d["source"]) for d in srt]
        y = list(range(len(srt)))
        ax.barh(y, vals, color=cols, height=0.74, zorder=3, **BAR_EDGE)
        ax.set_yticks(y)
        ax.set_yticklabels([short(d["source"]) for d in srt], fontsize=15)
        bold_human(ax.get_yticklabels())
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=14)
        ax.set_title(f"({letter}) {title}", fontsize=22, fontweight="bold", pad=8)
        vmax = max(vals)
        ax.set_xlim(0, vmax * 1.20)
        ax.grid(axis="x", color="#e2e2e2", lw=0.8, zorder=0)
        despine(ax)
        for yi, v in zip(y, vals):
            ax.text(v + vmax * 0.012, yi, fmt.format(v), va="center", ha="left", fontsize=12.5, zorder=4)

    for ax, (key, title, fmt), letter in zip(rhs_axes, facet_specs, "bcdefg"):
        hbar(ax, key, title, letter, fmt=fmt)

    fig.tight_layout()
    # Legend placed BELOW the whole figure (negative figure-y) so it can never
    # collide with panel (a)'s rotated x-ticks; bbox_inches="tight" captures it.
    fig.legend(handles=provider_legend_handles(), loc="upper center", ncol=6, fontsize=18,
               frameon=True, bbox_to_anchor=(0.5, -0.07))
    p_comp = out / "tc_breakdown.png"
    fig.savefig(p_comp, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- scorecard: Overall kept EXACTLY as before -- tall vertical panel on the
    #     LEFT (horizontal bars, all ranked systems, value labels). The four facets
    #     go in a 2x2 grid of top-8 horizontal bars on the RIGHT. ---
    TOP_S = 24
    sc_rows = sorted(rows, key=lambda d: -d["overall_eq"])[:TOP_S]
    nrow = len(sc_rows)
    sc_colors = [colr(d["source"]) for d in sc_rows]
    y = list(range(nrow))
    TOP_F = 10

    def nm(s, trunc=None):  # drop noisy "-instruct" suffix; optionally truncate long names
        t = short(s)
        if t.endswith("-instruct"):
            t = t[:-len("-instruct")]
        if trunc and len(t) > trunc:
            t = t[:trunc - 1] + "…"
        return t

    # Fixed, near-square canvas so that at \textwidth the panels stay large and easy
    # to read (we deliberately do NOT use bbox_inches="tight", which would blow up the
    # width from overhanging labels and shrink everything on the page).
    fig = plt.figure(figsize=(22, max(21, 0.95 * nrow)))
    # Empty gap columns (1 = overall->facet, 3 = facet->facet) with wspace=0, so each gap
    # width is set independently via the column ratios. Gap col 3 (2.15) is a touch wider
    # than gap col 1 (1.85): slightly more room between (b)/(d) and (c)/(e), while the
    # (a)->(b) gap stays exactly as before.
    gs = fig.add_gridspec(2, 5, width_ratios=[1.55, 1.85, 0.85, 2.15, 0.85], wspace=0,
                          hspace=0.28, left=0.205, right=0.915, top=0.955, bottom=0.135)
    axO = fig.add_subplot(gs[:, 0])
    facet_axes = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 4]),
                  fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 4])]

    # Overall (LHS): horizontal bars, diverging axis, value labels -- unchanged.
    ovals = [d["overall_eq"] for d in sc_rows]
    axO.barh(y, ovals, color=sc_colors, height=0.78, zorder=3, **BAR_EDGE)
    axO.invert_yaxis()
    axO.set_title("(a) Overall ($z$)", fontsize=48, fontweight="bold", pad=14)
    axO.grid(axis="x", color="#e2e2e2", lw=0.9, zorder=0)
    despine(axO)
    axO.tick_params(axis="x", labelsize=28)
    axO.set_yticks(y)
    axO.set_yticklabels([nm(d["source"], trunc=20) for d in sc_rows], fontsize=28)
    bold_human(axO.get_yticklabels())
    axO.axvline(0, color="#555555", lw=1.2, zorder=2)
    vmn, vmx = min(ovals), max(ovals)
    sp = vmx - vmn
    axO.set_xlim(vmn - sp * 0.06, vmx + sp * 0.28)
    for yi, v in zip(y, ovals):
        ha, off = ("left", sp * 0.02) if v >= 0 else ("right", -sp * 0.02)
        axO.text(v + off, yi, f"{v:+.2f}", va="center", ha=ha, fontsize=25, zorder=4)

    # Facets (RHS 2x2): horizontal top-N, each panel re-ranked by its own facet.
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    for ax, (key, title, fmt), letter in zip(facet_axes, facet_specs, "bcde"):
        srt = sorted(rows, key=lambda d: -d[key])[:TOP_F]
        vals = [d[key] for d in srt]
        cols = [colr(d["source"]) for d in srt]
        yy = list(range(len(srt)))
        ax.barh(yy, vals, color=cols, height=0.76, zorder=3, **BAR_EDGE)
        ax.set_yticks(yy)
        ax.set_yticklabels([nm(d["source"], trunc=18) for d in srt], fontsize=26)
        bold_human(ax.get_yticklabels())
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=24)
        ax.set_title(f"({letter}) {title}", fontsize=42, fontweight="bold", pad=10)
        # Zoom the value axis to the data range (not 0) so small differences are visible;
        # exact values stay labelled on every bar.
        vmn, vmx = min(vals), max(vals)
        rng = (vmx - vmn) or max(vmx * 0.05, 1e-6)
        ax.set_xlim(vmn - rng * 0.18, vmx + rng * 0.34)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2))  # few, well-spaced ticks (no crowding)
        if letter in ("b", "d"):  # 2-decimal ticks for surprise and diversity
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="x", color="#e2e2e2", lw=0.9, zorder=0)
        despine(ax)
        for yi, v in zip(yy, vals):
            ax.text(v + rng * 0.04, yi, fmt.format(v), va="center", ha="left", fontsize=22, zorder=4)

    sc_handles = provider_legend_handles()
    sc_ncol = -(-len(sc_handles) // 2)  # 2 rows; fits within the canvas width after grouping
    fig.legend(handles=sc_handles, loc="lower center", ncol=sc_ncol,
               fontsize=29, frameon=False, handletextpad=0.35, columnspacing=1.0,
               labelspacing=0.8, bbox_to_anchor=(0.5, 0.006))
    p_score = out / "tc_scorecard.png"
    fig.savefig(p_score, dpi=320)
    plt.close(fig)

    print(f"\nsaved: {p}\n       {p_temp}\n       {p_comp}\n       {p_score}\n       {out/'tc.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
