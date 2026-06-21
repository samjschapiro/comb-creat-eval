"""Combined 3-panel boxplot: (a) reasoning effort, (b) sampling temperature, (c) prompting
strategy vs the Overall composite. Each model x condition cell is summarized by four facets
(mean surprise/coherence/realism + reveal diversity) z-scored on the main analysis pool
(tc.json) -- the SAME scale as every other figure in the paper; the cell composite is their
equal-weight mean. Each dot is one (model x condition) cell.

The human reference is the BEST-8 human stories (top 8 by per-story surprise+coherence+realism),
projected onto the same z-frame. This is apples-to-apples with the dots: an 8-story cell is the
LLM's BEST config, so the human is shown at its best 8-story batch too -- both are best-selections.

Equal-weight composite (NOT tc=Div*mean(S*Coh)); see docs memory "plot-twist-headline-metric".

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/make_effort_temp_boxplots.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as cmc

from src.plot_twist.join import mean_pairwise_distance

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

CELLS = Path("data/plot_twist/thinking/downstream/analysis/thinking_cells.json")
PROMPT_CELLS = Path("data/plot_twist/prompt_methods/downstream/analysis/prompt_cells.json")
MAIN_TC = Path("data/plot_twist/tc/tc.json")  # main analysis pool (71 models + human): the z-frame
ANN = Path("data/plot_twist/annotations/annotations.json")
REAL = Path("data/plot_twist/realism/realism_scores.json")
OUT = Path("data/plot_twist/tc/downstream/temp")
FIG = Path("papers/pt2cb-iclr-2027/figures")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
BOX_COLS = [cmc.batlow(x) for x in (0.12, 0.5, 0.86)]
HUMAN_COL = "#B2182B"
FACETS_4 = ["mean_surprise", "mean_coherence", "div", "mean_realism"]

_ST = None


def _embedder():
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(EMBED_MODEL)
    return _ST


def _zparams(cells, facets=FACETS_4):
    return {k: (np.nanmean([c.get(k, np.nan) for c in cells]),
                np.nanstd([c.get(k, np.nan) for c in cells]) or 1.0)
            for k in facets}


def _overall(facets_dict, zs, facets=FACETS_4):
    """Equal-weight mean of z-scored facets for one cell, given pooled z-params zs."""
    return float(np.nanmean([(facets_dict[k] - zs[k][0]) / zs[k][1] for k in facets]))


def _score(a, k):
    v = a.get("scores", {}).get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def human_topN_facets(n=8):
    """Raw facet means for the BEST-8 human stories (top n by per-story
    surprise+coherence+realism; diversity is a set metric measured over those n)."""
    ann = json.loads(ANN.read_text())
    realism = json.loads(REAL.read_text())
    rows = []
    for a in (a for a in ann if a["source"] == "human"):
        s, c, r = _score(a, "surprise"), _score(a, "coherence"), realism.get(a["id"])
        if None in (s, c) or r is None:
            continue
        rows.append((s + c + r, a, s, c, r))
    rows.sort(key=lambda x: -x[0])
    top = rows[:n]
    E = _embedder().encode([a.get("reveal") or "" for _, a, *_ in top],
                           normalize_embeddings=True, show_progress_bar=False)
    return {"mean_surprise": float(np.mean([x[2] for x in top])),
            "mean_coherence": float(np.mean([x[3] for x in top])),
            "mean_realism": float(np.mean([x[4] for x in top])),
            "div": mean_pairwise_distance(np.array(E))}


def human_top8(zs, n=8):
    """Composite of the best-8 human stories, projected onto the z-frame zs."""
    return _overall(human_topN_facets(n), zs)


def effort_cells():
    cells = json.loads(CELLS.read_text())
    return cells, "level", ["low", "medium", "high"], ["low", "medium", "high"]


def strategy_cells():
    cells = json.loads(PROMPT_CELLS.read_text())
    return (cells, "method", ["baseline", "be_creative", "incontext_regen"],
            ["baseline", "be creative", "in-context"])


def temp_cells():
    ann = json.loads(ANN.read_text())
    realism = json.loads(REAL.read_text())
    emb = _embedder().encode([a.get("reveal") or "" for a in ann],
                             normalize_embeddings=True, show_progress_bar=False)
    emb_by_id = {a["id"]: emb[i] for i, a in enumerate(ann)}
    TEMPS = [0.9, 1.0, 1.2]

    def temp_of(rid):
        m = re.search(r"__t(\d{2})__", rid)
        return int(m.group(1)) / 10.0 if m else None

    cellmap = defaultdict(list)
    for a in ann:
        if a["source"] == "human":
            continue
        t = temp_of(a["id"])
        if t in TEMPS:
            cellmap[(a["source"], t)].append(a)
    rows = []
    for (model, t), rs in cellmap.items():
        sur = [_score(a, "surprise") for a in rs if _score(a, "surprise") is not None]
        coh = [_score(a, "coherence") for a in rs if _score(a, "coherence") is not None]
        rea = [realism[a["id"]] for a in rs if a["id"] in realism]
        E = np.array([emb_by_id[a["id"]] for a in rs])
        if len(sur) < 2 or len(E) < 2:
            continue
        rows.append({"model": model, "temp": t, "mean_surprise": float(np.mean(sur)),
                     "mean_coherence": float(np.mean(coh)),
                     "mean_realism": float(np.mean(rea)) if rea else np.nan,
                     "div": mean_pairwise_distance(E)})
    by_model = defaultdict(set)
    for r in rows:
        by_model[r["model"]].add(r["temp"])
    keep = {m for m, ts in by_model.items() if set(TEMPS) <= ts}
    rows = [r for r in rows if r["model"] in keep]
    return rows, "temp", TEMPS, [f"{t}" for t in TEMPS]


def panel(ax, levels, labels, by, xlabel, title, human):
    data = [by[lv] for lv in levels]
    bp = ax.boxplot(data, positions=range(len(levels)), widths=0.55, patch_artist=True, vert=True,
                    medianprops=dict(color="black", lw=1.6), whiskerprops=dict(color="#444"),
                    capprops=dict(color="#444"), flierprops=dict(marker="", alpha=0))
    for patch, col in zip(bp["boxes"], BOX_COLS):
        patch.set_facecolor((*col[:3], 0.35)); patch.set_edgecolor("#333"); patch.set_linewidth(0.9)
    rng = np.random.default_rng(0)
    for i, lv in enumerate(levels):
        ys = by[lv]
        jx = i + (rng.random(len(ys)) - 0.5) * 0.18
        ax.scatter(jx, ys, s=22, color=BOX_COLS[i], edgecolor="#333", linewidth=0.4,
                   alpha=0.45, zorder=3)
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel); ax.set_ylabel("Overall ($z$)")
    ax.axhline(0, color="#bbb", lw=0.8, ls=":", zorder=0)
    hl = ax.axhline(human, color=HUMAN_COL, lw=1.7, ls="--", zorder=4)
    ax.set_title(title, loc="left", fontweight="bold")
    return hl


def main():
    # z-frame = the MAIN analysis pool (tc.json), 4 facets -> the SAME scale as every other
    # figure. Human reference = best-8 human stories on this frame (apples-to-apples with the
    # best-config cells); all panel cells projected onto the frame.
    zs = _zparams(json.loads(MAIN_TC.read_text()))
    human = human_top8(zs)
    panels = [effort_cells(), temp_cells(), strategy_cells()]
    titles = ["(a)", "(b)", "(c)"]
    xlabels = ["Reasoning effort", "Sampling temperature", "Prompting strategy"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5), sharey=True)
    hl = None
    for ax, (cells, keyf, levels, labels), xl, ti in zip(axes, panels, xlabels, titles):
        by = {lv: [_overall(c, zs) for c in cells if c[keyf] == lv] for lv in levels}
        hl = panel(ax, levels, labels, by, xl, ti, human)
    for ax in axes[1:]:
        ax.set_ylabel("")
    axes[0].legend([hl], ["human (best 8)"], loc="lower left", fontsize=10, frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (OUT / "effort_temp_boxplots.pdf", FIG / "effort_temp_boxplots.pdf", OUT / "effort_temp_boxplots.png"):
        fig.savefig(d)
    plt.close(fig)
    print(f"saved -> {FIG/'effort_temp_boxplots.pdf'}  (human best-8 z = {human:+.2f})")


if __name__ == "__main__":
    main()
