"""Transformational creativity vs sampling temperature (appendix).

Within-model design mirroring the reasoning-effort boxplot (Fig.~\ref{fig:thinking_boxplot}):
for every (model, temperature) cell we compute the four facet means -- surprise, coherence,
realism (rubric/realism scores) and diversity (mean pairwise distance of the cell's reveal
embeddings) -- z-score each facet across all cells, and average the four z-scores into an
Overall composite. We then box the composite at each temperature (0.9, 1.0, 1.2) with the
same model connected across temperatures. Uses the equal-weight composite (NOT the old
tc=Div*mean(S*Coh)); see docs memory "plot-twist-headline-metric".

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/make_tc_vs_temp.py
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
from scipy import stats

from src.plot_twist.join import mean_pairwise_distance, gated_means

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "font.size": 11, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 11,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

ANN = Path("data/plot_twist/annotations/annotations.json")
REAL = Path("data/plot_twist/realism/realism_scores.json")
OUT = Path("data/plot_twist/tc/downstream/temp")
FIG = Path("papers/pt2cb-iclr-2027/figures")
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
TEMPS = [0.9, 1.0, 1.2]


def temp_of(rid):
    m = re.search(r"__t(\d{2})__", rid)
    return int(m.group(1)) / 10.0 if m else None


def main():
    ann = json.loads(ANN.read_text())
    realism = json.loads(REAL.read_text())

    # embed every reveal once (L2-normalised), matching the main diversity metric
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(EMBED_MODEL)
    revs = [a.get("reveal") or "" for a in ann]
    emb = st.encode(revs, normalize_embeddings=True, show_progress_bar=False)
    emb_by_id = {a["id"]: emb[i] for i, a in enumerate(ann)}

    # group annotations into (model, temp) cells (LLMs only)
    cells = defaultdict(list)
    for a in ann:
        if a["source"] == "human":
            continue
        t = temp_of(a["id"])
        if t in TEMPS:
            cells[(a["source"], t)].append(a)

    def s(a, k):
        v = a.get("scores", {}).get(k)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    rows = []
    for (model, t), rs in cells.items():
        sur = [s(a, "surprise") for a in rs if s(a, "surprise") is not None]
        E = np.array([emb_by_id[a["id"]] for a in rs])
        if len(sur) < 2 or len(E) < 2:
            continue
        gm = gated_means(rs, realism)
        rows.append({"model": model, "temp": t, **gm, "div": mean_pairwise_distance(E)})
    # keep only models present at all three temps (paired within-model design)
    by_model = defaultdict(set)
    for r in rows:
        by_model[r["model"]].add(r["temp"])
    keep = {m for m, ts in by_model.items() if set(TEMPS) <= ts}
    rows = [r for r in rows if r["model"] in keep]

    # Headline composite facets: realism-GATED surprise/coherence + diversity.
    FACETS = ["mean_surprise_g", "mean_coherence_g", "div"]
    # z-score each facet across all cells, then average -> Overall composite
    zs = {k: (np.nanmean([r[k] for r in rows]), np.nanstd([r[k] for r in rows]) or 1.0) for k in FACETS}
    for r in rows:
        r["overall"] = float(np.mean([(r[k] - zs[k][0]) / zs[k][1] for k in FACETS]))
    print(f"{len(keep)} models x 3 temps = {len(rows)} cells")

    # per-temp distributions + per-model paired lines
    by_temp = {t: [r["overall"] for r in rows if r["temp"] == t] for t in TEMPS}
    models = sorted(keep)
    paired = {m: [next((r["overall"] for r in rows if r["model"] == m and r["temp"] == t), np.nan)
                  for t in TEMPS] for m in models}
    fried = stats.friedmanchisquare(*[[paired[m][i] for m in models] for i in range(3)])
    print(f"Friedman across temps: chi2={fried.statistic:.2f} p={fried.pvalue:.3f}")
    print("means:", {t: round(float(np.mean(by_temp[t])), 3) for t in TEMPS})

    fig, ax = plt.subplots(figsize=(3.7, 3.1))
    xpos = list(range(3))
    for m in models:
        ax.plot(xpos, paired[m], "-", color="0.7", lw=0.5, alpha=0.6, zorder=1)
    ax.boxplot([by_temp[t] for t in TEMPS], positions=xpos, widths=0.55,
               showfliers=False, medianprops=dict(color="#C0392B", lw=1.6),
               boxprops=dict(color="#333"), whiskerprops=dict(color="#333"),
               capprops=dict(color="#333"), zorder=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{t}" for t in TEMPS])
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Overall composite ($z$)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (OUT / "tc_vs_temperature.pdf", FIG / "tc_vs_temperature.pdf", OUT / "tc_vs_temperature.png"):
        fig.savefig(d)
    plt.close(fig)
    print(f"saved -> {FIG/'tc_vs_temperature.pdf'}")


if __name__ == "__main__":
    main()
