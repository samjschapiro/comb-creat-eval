"""Combined 2-panel boxplot: (a) reasoning effort and (b) sampling temperature vs the
within-model Overall composite. Both panels are HORIZONTAL (Overall z on the x-axis,
the intervention level on the y-axis), matching make_thinking_boxplot.py exactly, so the
two interventions sit side by side in one full-width figure.

Panel (a) reuses thinking_cells.json (tc_within per model x effort level). Panel (b) is
computed here: per (model, temperature) cell we average surprise/coherence/realism (rubric
+ realism scores) and diversity (mean pairwise distance of the cell's reveal embeddings),
z-score each facet across cells, and average -> Overall composite. Equal-weight composite
(NOT tc=Div*mean(S*Coh)); see docs memory "plot-twist-headline-metric".

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
ANN = Path("data/plot_twist/annotations/annotations.json")
REAL = Path("data/plot_twist/realism/realism_scores.json")
OUT = Path("data/plot_twist/tc/downstream/temp")
FIG = Path("papers/pt2cb-iclr-2027/figures")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
BOX_COLS = [cmc.batlow(x) for x in (0.12, 0.5, 0.86)]


def effort_data():
    cells = json.loads(CELLS.read_text())
    LEVELS = ["low", "medium", "high"]
    return LEVELS, ["low", "medium", "high"], {lv: [c["tc_within"] for c in cells if c["level"] == lv]
                                               for lv in LEVELS}


def temp_data():
    ann = json.loads(ANN.read_text())
    realism = json.loads(REAL.read_text())
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMBED_MODEL)
    emb = st.encode([a.get("reveal") or "" for a in ann], normalize_embeddings=True, show_progress_bar=False)
    emb_by_id = {a["id"]: emb[i] for i, a in enumerate(ann)}
    TEMPS = [0.9, 1.0, 1.2]

    def temp_of(rid):
        m = re.search(r"__t(\d{2})__", rid)
        return int(m.group(1)) / 10.0 if m else None

    def s(a, k):
        v = a.get("scores", {}).get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    cellmap = defaultdict(list)
    for a in ann:
        if a["source"] == "human":
            continue
        t = temp_of(a["id"])
        if t in TEMPS:
            cellmap[(a["source"], t)].append(a)
    rows = []
    for (model, t), rs in cellmap.items():
        sur = [s(a, "surprise") for a in rs if s(a, "surprise") is not None]
        coh = [s(a, "coherence") for a in rs if s(a, "coherence") is not None]
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
    FACETS = ["mean_surprise", "mean_coherence", "mean_realism", "div"]
    zs = {k: (np.nanmean([r[k] for r in rows]), np.nanstd([r[k] for r in rows]) or 1.0) for k in FACETS}
    for r in rows:
        r["overall"] = float(np.mean([(r[k] - zs[k][0]) / zs[k][1] for k in FACETS]))
    return TEMPS, [f"{t}" for t in TEMPS], {t: [r["overall"] for r in rows if r["temp"] == t] for t in TEMPS}


def panel(ax, levels, labels, by, xlabel, title):
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
    ax.set_xlabel(xlabel); ax.set_ylabel("Overall (within-model $z$)")
    ax.axhline(0, color="#bbb", lw=0.8, ls=":", zorder=0)
    ax.set_title(title, loc="left", fontweight="bold")


def strategy_data():
    """Prompting-strategy ablation (Exp 2): within-model Overall composite (tc_within) per
    method, from prompt_cells.json. baseline reused from the main run; be_creative and
    incontext_regen newly generated on the 5-model reasoning subset."""
    cells = json.loads(PROMPT_CELLS.read_text())
    levels = ["baseline", "be_creative", "incontext_regen"]
    labels = ["baseline", "be creative", "in-context"]
    by = {lv: [c["tc_within"] for c in cells if c["method"] == lv] for lv in levels}
    return levels, labels, by


def main():
    e_levels, e_labels, e_by = effort_data()
    t_levels, t_labels, t_by = temp_data()
    p_levels, p_labels, p_by = strategy_data()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5), sharey=True)
    panel(axes[0], e_levels, e_labels, e_by, "Reasoning effort", "(a)")
    panel(axes[1], t_levels, t_labels, t_by, "Sampling temperature", "(b)")
    panel(axes[2], p_levels, p_labels, p_by, "Prompting strategy", "(c)")
    for ax in axes[1:]:
        ax.set_ylabel("")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (OUT / "effort_temp_boxplots.pdf", FIG / "effort_temp_boxplots.pdf", OUT / "effort_temp_boxplots.png"):
        fig.savefig(d)
    plt.close(fig)
    print(f"saved -> {FIG/'effort_temp_boxplots.pdf'}")


if __name__ == "__main__":
    main()
