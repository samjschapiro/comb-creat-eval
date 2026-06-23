"""Combined 2-panel boxplot: (a) reasoning effort, (b) prompting strategy vs the Overall
composite. Each model x condition cell is summarized by the realism-gated composite facets
(gated surprise/coherence + reveal diversity), z-scored on the main analysis pool (tc.json);
the cell composite is their equal-weight mean. z (not percentile) keeps the top of the pool
de-saturated -- humans sit visibly above the frontier cluster. Each dot is one
(model x condition) cell.

The human reference is the BEST-8 human stories (top 8 by per-story surprise+coherence+realism),
projected onto the same z-frame. This is apples-to-apples with the dots: an 8-story
cell is the LLM's BEST config, so the human is shown at its best 8-story batch too.

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

from src.plot_twist.join import mean_pairwise_distance, gated_means, REALISM_GATE

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
HUMAN_COL = "#000000"  # human reference line (black, matching the leaderboard's human bars)
# Headline composite facets: realism-GATED surprise/coherence + diversity (S/Coh count
# only when realism == 5). Same gate as make_tc_barplot / join.EQ_FACETS.
FACETS_4 = ["mean_surprise_g", "mean_coherence_g", "div"]

_ST = None


def _embedder():
    global _ST
    if _ST is None:
        from sentence_transformers import SentenceTransformer
        _ST = SentenceTransformer(EMBED_MODEL)
    return _ST


def _zparams(cells, facets=FACETS_4):
    """Pooled (mean, std) per composite facet -- the z-frame for the cells."""
    return {k: (np.nanmean([c.get(k, np.nan) for c in cells]),
                np.nanstd([c.get(k, np.nan) for c in cells]) or 1.0)
            for k in facets}


def _overall(facets_dict, zs, facets=FACETS_4):
    """Cell composite = equal-weight mean of z-scored facets, given pooled z-params zs.
    z (not percentile) so the top of the pool de-saturates: humans sit visibly above the
    frontier cluster rather than piling up at the percentile ceiling."""
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
    g = lambda s, r: s if r >= REALISM_GATE else 0.0  # realism gate
    return {"mean_surprise": float(np.mean([x[2] for x in top])),
            "mean_coherence": float(np.mean([x[3] for x in top])),
            "mean_realism": float(np.mean([x[4] for x in top])),
            "mean_surprise_g": float(np.mean([g(x[2], x[4]) for x in top])),
            "mean_coherence_g": float(np.mean([g(x[3], x[4]) for x in top])),
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
        rea = [realism[a["id"]] for a in rs if a["id"] in realism]
        E = np.array([emb_by_id[a["id"]] for a in rs])
        if len(sur) < 2 or len(E) < 2:
            continue
        gm = gated_means(rs, realism)
        rows.append({"model": model, "temp": t, **gm, "div": mean_pairwise_distance(E)})
    by_model = defaultdict(set)
    for r in rows:
        by_model[r["model"]].add(r["temp"])
    keep = {m for m, ts in by_model.items() if set(TEMPS) <= ts}
    rows = [r for r in rows if r["model"] in keep]
    return rows, "temp", TEMPS, [f"{t}" for t in TEMPS]


def _short(model):
    """Compact model label for an annotation (e.g. 'Sonnet-4.5')."""
    n = model.split("/")[-1]
    for pre in ("claude-", "gpt-", "deepseek-"):
        if n.startswith(pre):
            n = n[len(pre):]
    return n[:1].upper() + n[1:]


def panel(ax, levels, labels, by, xlabel, title, human, annotate=None):
    """`by[lv]` is a list of (model, z). `annotate` maps a level -> x-offset for the label
    of its top (max-z) cell -- the one tying the human line -- shown with its model name."""
    annotate = annotate or {}
    data = [[z for _, z in by[lv]] for lv in levels]
    bp = ax.boxplot(data, positions=range(len(levels)), widths=0.55, patch_artist=True, vert=True,
                    medianprops=dict(color="black", lw=1.6), whiskerprops=dict(color="#444"),
                    capprops=dict(color="#444"), flierprops=dict(marker="", alpha=0))
    for patch, col in zip(bp["boxes"], BOX_COLS):
        patch.set_facecolor((*col[:3], 0.35)); patch.set_edgecolor("#333"); patch.set_linewidth(0.9)
    rng = np.random.default_rng(0)
    for i, lv in enumerate(levels):
        ys = [z for _, z in by[lv]]
        jx = i + (rng.random(len(ys)) - 0.5) * 0.18
        ax.scatter(jx, ys, s=22, color=BOX_COLS[i], edgecolor="#333", linewidth=0.4,
                   alpha=0.45, zorder=3)
        if lv in annotate and ys:
            k = int(np.argmax(ys))
            xoff = annotate[lv]
            ha = "left" if xoff > 0 else ("right" if xoff < 0 else "center")
            ax.annotate(_short(by[lv][k][0]), xy=(jx[k], ys[k]), xytext=(i + xoff, ys[k] + 0.6),
                        ha=ha, va="bottom", fontsize=10, fontstyle="italic", zorder=7,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="#333", shrinkA=1, shrinkB=2))
    ax.set_xticks(range(len(levels))); ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel); ax.set_ylabel("Overall ($z$)")
    ax.axhline(0, color="#bbb", lw=0.8, ls=":", zorder=0)  # pool mean
    hl = ax.axhline(human, color=HUMAN_COL, lw=1.7, ls="--", zorder=4)
    ax.set_title(title, loc="left", fontweight="bold")
    return hl


def main():
    # z-frame = the MAIN analysis pool (tc.json) gated facets. Human reference = best-8 human
    # stories on this frame (apples-to-apples with the best-config cells); all cells projected
    # onto it. z keeps the top de-saturated (vs percentile, which piles the frontier at ~99).
    zs = _zparams(json.loads(MAIN_TC.read_text()))
    human = human_top8(zs)
    panels = [effort_cells(), strategy_cells()]
    titles = ["(a)", "(b)"]
    xlabels = ["Reasoning effort", "Prompting strategy"]
    # Vertically stacked (2x1) so the figure is narrow -> sits at ~0.4 text width as a
    # wrapfigure. Kept compact in height so it doesn't span more lines than the adjacent
    # text. Both panels keep the percentile y-axis (shared scale).
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.5), sharey=True)
    # Annotate the top (human-tying) cell in (a) low effort and (b) in-context regeneration.
    # low is shifted right so its label clears the "(a)" panel title.
    annotates = [{"low": 0.5}, {"incontext_regen": 0.0}]
    hl = None
    for ax, (cells, keyf, levels, labels), xl, ti, ann in zip(axes, panels, xlabels, titles, annotates):
        by = {lv: [(c["model"], _overall(c, zs)) for c in cells if c[keyf] == lv] for lv in levels}
        hl = panel(ax, levels, labels, by, xl, ti, human, annotate=ann)
    axes[0].set_ylim(top=3.1)  # headroom for the annotation labels (shared y-axis)
    axes[0].legend([hl], ["human"], loc="lower left", fontsize=10, frameon=False)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    # Paper copy is named for its figure number (Figure 5); data-dir copies keep the descriptive name.
    for d in (OUT / "effort_temp_boxplots.pdf", FIG / "fig5_effort_temp.pdf", OUT / "effort_temp_boxplots.png"):
        fig.savefig(d)
    plt.close(fig)
    print(f"saved -> {FIG/'fig5_effort_temp.pdf'}  (human best-8 z = {human:+.2f})")


if __name__ == "__main__":
    main()
