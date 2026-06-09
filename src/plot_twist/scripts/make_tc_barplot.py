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

    rows.sort(key=lambda d: -d["tc"])
    print(f"{'source':<34}{'n':>4}{'Div':>8}{'mean(S*Coh)':>13}{'tc':>9}")
    for d in rows:
        print(f"{d['source']:<34}{d['n']:>4}{d['div']:>8.3f}{d['mean_SxCoh']:>13.2f}{d['tc']:>9.3f}")
    (out / "tc.json").write_text(json.dumps(rows, indent=2))

    # --- bar chart (camera-ready) ---
    def short(s):
        return "Expert humans" if s == "human" else s.split("/")[-1]

    # Batlow palette (Crameri), matching the paper's Figure 1 role colours.
    BATLOW_BLUE = "#103D5F"    # LLM
    BATLOW_ORANGE = "#EE9D6B"  # human gold (highlight role)
    labels = [short(d["source"]) for d in rows]
    vals = [d["tc"] for d in rows]
    colors = [BATLOW_ORANGE if d["source"] == "human" else BATLOW_BLUE for d in rows]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(range(len(rows)), vals, color=colors, width=0.72)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("Transformational creativity  tc", fontsize=24, labelpad=12)
    ax.margins(y=0.12)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=15)
    # legend distinguishing human vs LLM
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BATLOW_ORANGE, label="Expert humans"), Patch(color=BATLOW_BLUE, label="LLM")],
              fontsize=18, frameon=True, loc="upper right")
    fig.tight_layout()
    p = out / "tc_by_model.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
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
        ax.axhline(human_tc, ls=":", lw=3.2, color=BATLOW_ORANGE)
        ax.text(temps[-1], human_tc, f"Expert humans ({human_tc:.2f})  ",
                va="bottom", ha="right", fontsize=17, color="#B5651D")
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

    # --- composite figure: (a) overall score | (b,c,d) component breakdown ---
    order = sorted(rows, key=lambda d: -d["tc"])
    nb = len(order)

    def colr(src):
        return BATLOW_ORANGE if src == "human" else BATLOW_BLUE

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.18, 1.0], hspace=0.85, wspace=0.58)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    axD = fig.add_subplot(gs[2, 1])

    # (a) vertical tc bars (sorted by tc)
    labelsA = [short(d["source"]) for d in order]
    barsA = axA.bar(range(nb), [d["tc"] for d in order], color=[colr(d["source"]) for d in order], width=0.72)
    axA.set_xticks(range(nb))
    axA.set_xticklabels(labelsA, rotation=40, ha="right", fontsize=16)
    for lbl in axA.get_xticklabels():
        if lbl.get_text() == "Expert humans":
            lbl.set_fontweight("bold")
    axA.tick_params(axis="y", labelsize=17)
    axA.set_ylabel("Transformational creativity (tc)", fontsize=23, labelpad=12)
    axA.set_title("(a) Overall score", fontsize=27, pad=14, fontweight="bold")
    axA.margins(y=0.15)
    for b, d in zip(barsA, order):
        axA.text(b.get_x() + b.get_width() / 2, d["tc"] + 0.12, f"{d['tc']:.1f}",
                 ha="center", va="bottom", fontsize=14)
    from matplotlib.patches import Patch
    axA.legend(handles=[Patch(color=BATLOW_ORANGE, label="Expert humans"), Patch(color=BATLOW_BLUE, label="LLM")],
               fontsize=17, loc="upper right", framealpha=0.95)

    # (b,c,d) horizontal bars, each sorted by its OWN value (decreasing).
    # x-limit gives headroom so value labels never clip; labels offset off the bar tip.
    def hbar(ax, key, title, letter, fmt="{:.2f}"):
        srt = sorted(order, key=lambda d: -d[key])
        labels = [short(d["source"]) for d in srt]
        vals = [d[key] for d in srt]
        cols = [colr(d["source"]) for d in srt]
        y = list(range(nb))
        ax.barh(y, vals, color=cols, height=0.42)  # thin bars -> large gaps
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=19)
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == "Expert humans":
                lbl.set_fontweight("bold")
        ax.set_ylim(nb - 0.5, -0.5)  # invert + tight so the wide gaps fill the panel
        ax.tick_params(axis="x", labelsize=18)
        ax.set_title(f"({letter}) {title}", fontsize=23, pad=10, fontweight="bold")
        vmax = max(vals)
        ax.set_xlim(0, vmax * 1.26)
        for yi, v in zip(y, vals):
            ax.text(v + vmax * 0.015, yi, fmt.format(v), va="center", ha="left",
                    fontsize=15, clip_on=False)

    hbar(axB, "mean_surprise", "Surprise", "b")
    hbar(axC, "mean_coherence", "Coherence", "c")
    hbar(axD, "div", "Diversity", "d", fmt="{:.3f}")

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
