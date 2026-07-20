"""Per-pair complementary analysis: P(a model finds the analogy) vs. anchor distance.

Each data point is one of the 200 random analogy PAIRS. Its difficulty is estimated as the
fraction of the model suite that produced a valid analogy for it (a pooled success probability),
and plotted against the embedding distance between the two anchors. Answers: does the chance a
model can bridge two entities fall as the entities get more semantically distant?

    .venv_mlx/bin/python src/kg_creat/scripts/plot_analogy_difficulty.py data/kg_creat/scores_analogy_v2
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.plot_analogy import _node_distinct, _relations_match, _structures_disjoint

INK, MUTED, GRID, DOT, TREND = "#1f2933", "#66727f", "#e3e8ee", "#2563EB", "#EA580C"


def _valid(paths):
    p0, p1 = paths.get(0), paths.get(1)
    if not p0 or p1 is None:
        return False
    fact = [f for r in paths.values() for f in (r.get("factual") or [])]
    return (bool(p0.get("semantic_sat")) and fact and all(fact)
            and all(_node_distinct(r["triples"]) for r in paths.values() if r.get("triples"))
            and _relations_match(p0["triples"], p1["triples"])
            and _structures_disjoint(p0["triples"], p1["triples"]))


def main(scores_dir):
    embed = get_embedder()
    def cos(a, b):
        x, y = embed(a), embed(b)
        return 1 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    # per prompt_id: attempts (models with a parsed pair) and successes across the suite
    attempts = defaultdict(int)
    successes = defaultdict(int)
    labels = {}
    for md in Path(scores_dir).glob("*/path_scores.json"):
        recs = json.loads(md.read_text())
        byp = defaultdict(dict)
        for r in recs:
            if r["mode"] == "analogy":
                byp[r["prompt_id"]][r["path_idx"]] = r
        for pid, paths in byp.items():
            p0 = paths.get(0)
            if not p0 or paths.get(1) is None:
                continue
            attempts[pid] += 1
            successes[pid] += int(_valid(paths))
            labels[pid] = (p0["u_label"], p0["v_label"])

    xs, ys = [], []
    for pid, a in attempts.items():
        if a == 0:
            continue
        u, v = labels[pid]
        xs.append(cos(u, v))
        ys.append(successes[pid] / a)
    xs, ys = np.array(xs), np.array(ys)

    # correlations
    from scipy.stats import pearsonr, spearmanr
    pr, pp = pearsonr(xs, ys)
    sr, sp = spearmanr(xs, ys)
    print(f"n pairs = {len(xs)}")
    print(f"Pearson  r = {pr:+.3f}  (p = {pp:.1e})")
    print(f"Spearman r = {sr:+.3f}  (p = {sp:.1e})")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)
    j = (np.random.RandomState(0).rand(len(ys)) - 0.5) * 0.02
    ax.scatter(xs, ys + j, s=40, color=DOT, alpha=0.45, edgecolors="white", linewidths=0.5, zorder=3)
    # binned trend
    bins = np.linspace(xs.min(), xs.max(), 7)
    cx, cy = [], []
    for i in range(len(bins) - 1):
        m = (xs >= bins[i]) & (xs < bins[i + 1] if i < len(bins) - 2 else xs <= bins[i + 1])
        if m.sum() >= 3:
            cx.append((bins[i] + bins[i + 1]) / 2)
            cy.append(ys[m].mean())
    ax.plot(cx, cy, "-", color=TREND, lw=2.6, marker="D", markersize=7, zorder=5, label="binned mean")
    # linear fit
    b1, b0 = np.polyfit(xs, ys, 1)
    xf = np.array([xs.min(), xs.max()])
    ax.plot(xf, b0 + b1 * xf, "--", color=MUTED, lw=1.5, zorder=4, label=f"linear fit (slope {b1:+.2f})")

    ax.set_xlabel("Anchor distance  (1 - cosine between the two entities)", color=MUTED)
    ax.set_ylabel("P(a model finds a valid analogy)  — fraction of the 8-model suite", color=MUTED)
    ax.set_ylim(-0.05, 1.0)
    ax.set_title(f"Per-pair difficulty vs. anchor distance  (n={len(xs)} random pairs)\n"
                 f"Pearson r = {pr:+.2f} (p={pp:.0e}),  Spearman r = {sr:+.2f}",
                 color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    out = Path(scores_dir) / "analogy_difficulty_vs_distance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_analogy_v2")
