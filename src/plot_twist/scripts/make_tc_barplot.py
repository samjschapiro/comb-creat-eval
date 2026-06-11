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

    # embed all reveals once
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["embed_model"])
    all_recs = [r for rs in groups.values() for r in rs]
    embs = model.encode([r[field] for r in all_recs], normalize_embeddings=True, show_progress_bar=False)
    emb_by_id = {id(r): e for r, e in zip(all_recs, np.asarray(embs, dtype=np.float32))}

    # per-source Div, mean(S*Coh), tc
    rows = []
    for src, rs in groups.items():
        E = np.array([emb_by_id[id(r)] for r in rs])
        div = _div(E)
        sc = np.mean([num(r, "surprise") * num(r, "coherence") for r in rs])
        rows.append({
            "source": src, "n": len(rs), "div": div,
            "mean_surprise": float(np.mean([num(r, "surprise") for r in rs])),
            "mean_coherence": float(np.mean([num(r, "coherence") for r in rs])),
            "mean_SxCoh": float(sc), "tc": div * float(sc),
        })

    # Equal-weighted overall score: z-score each facet across sources (mean 0, SD 1),
    # then average with equal weight (surprise, coherence, diversity each 1/3). This is
    # the same equal-weight-of-standardized-indicators construction as AGC-Bench mean_z.
    EQ_FACETS = ["mean_surprise", "mean_coherence", "div"]
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

    def _adj(c):  # darken any too-light batlowS tint (e.g. the pale pink) so bars stay visible
        r, g, b = c[0], c[1], c[2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        if lum > 0.66:
            s = 0.55 / lum
            r, g, b = r * s, g * s, b * s
        return (r, g, b, 1.0)

    PROV_COLOR = {p: _adj(_cmc.batlowS(i)) for i, p in enumerate(PROVIDERS)}
    HUMAN_COLOR = "#000000"  # Expert humans = black, to stand out from every provider
    BAR_EDGE = dict(edgecolor="#666666", linewidth=0.5)
    PROV_NAME = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google",
                 "meta-llama": "Meta", "deepseek": "DeepSeek", "qwen": "Qwen",
                 "mistralai": "Mistral", "amazon": "Amazon", "nvidia": "NVIDIA",
                 "ibm-granite": "IBM", "z-ai": "Z-AI", "moonshotai": "Moonshot"}

    def bar_color(src):
        return HUMAN_COLOR if src == "human" else PROV_COLOR[src.split("/")[0]]

    def provider_legend_handles():
        return [Patch(color=HUMAN_COLOR, label="Expert humans")] + \
               [Patch(color=PROV_COLOR[p], label=PROV_NAME.get(p, p)) for p in PROVIDERS]

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

    fig = plt.figure(figsize=(25, 12.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], hspace=0.38, wspace=0.42)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[2, 1])

    # (a) vertical tc bars, top 20
    barsA = axA.bar(range(len(topA)), [d["overall_eq"] for d in topA],
                    color=[colr(d["source"]) for d in topA], width=0.74, **BAR_EDGE)
    axA.set_xticks(range(len(topA)))
    axA.set_xticklabels([short(d["source"]) for d in topA], rotation=45, ha="right", fontsize=16)
    bold_human(axA.get_xticklabels())
    axA.tick_params(axis="y", labelsize=17)
    axA.set_ylabel("Overall score (mean z-score)", fontsize=23, labelpad=10)
    axA.set_title("(a) Overall score", fontsize=28, pad=12, fontweight="bold")
    axA.margins(y=0.15)
    axA.axhline(0, color="#999999", lw=0.9, zorder=0)
    _vspan = max(d["overall_eq"] for d in topA) - min(d["overall_eq"] for d in topA)
    for b, d in zip(barsA, topA):
        v = d["overall_eq"]
        axA.text(b.get_x() + b.get_width() / 2, v + (_vspan * 0.02 if v >= 0 else -_vspan * 0.02),
                 f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=12)

    # (b,c,d) horizontal top-10 by each facet
    def hbar(ax, key, title, letter, fmt="{:.2f}"):
        srt = sorted(rows, key=lambda d: -d[key])[:TOP_R]
        vals = [d[key] for d in srt]
        cols = [colr(d["source"]) for d in srt]
        y = list(range(len(srt)))
        ax.barh(y, vals, color=cols, height=0.72, **BAR_EDGE)
        ax.set_yticks(y)
        ax.set_yticklabels([short(d["source"]) for d in srt], fontsize=16)
        bold_human(ax.get_yticklabels())
        ax.invert_yaxis()
        ax.tick_params(axis="x", labelsize=15)
        ax.set_title(f"({letter}) {title}", fontsize=22, fontweight="bold", pad=6)
        vmax = max(vals)
        ax.set_xlim(0, vmax * 1.18)
        for yi, v in zip(y, vals):
            ax.text(v + vmax * 0.012, yi, fmt.format(v), va="center", ha="left", fontsize=13)

    hbar(axB, "mean_surprise", "Surprise", "b")
    hbar(axC, "mean_coherence", "Coherence", "c")
    hbar(axD, "div", "Diversity", "d", fmt="{:.3f}")

    fig.tight_layout()
    # Legend placed BELOW the whole figure (negative figure-y) so it can never
    # collide with panel (a)'s rotated x-ticks; bbox_inches="tight" captures it.
    fig.legend(handles=provider_legend_handles(), loc="upper center", ncol=6, fontsize=18,
               frameon=True, bbox_to_anchor=(0.5, -0.07))
    p_comp = out / "tc_breakdown.png"
    fig.savefig(p_comp, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved: {p}\n       {p_temp}\n       {p_comp}\n       {out/'tc.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
