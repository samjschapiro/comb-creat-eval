"""Ranked per-model analogy success (exact-relation-match), with the distance effect.

For a model suite the success-vs-distance scatter gets unreadable, so we show a ranked
horizontal bar of overall valid-analogy rate per model, with the near- and far-tertile
rates overlaid as markers (near left-of far => success declines as pairs get more unrelated).

Analogy success = exact-relation-match (both structures share the identical relation sequence)
AND both structures node-distinct AND both factual AND judged a genuine role-corresponding analogy.

    .venv_mlx/bin/python src/kg_creat/scripts/plot_analogy_suite.py data/kg_creat/scores_analogy_v2
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

INK, MUTED, GRID = "#1f2933", "#66727f", "#e3e8ee"
BAR = "#93b4e6"          # recessive fill
NEAR, FAR = "#2563EB", "#EA580C"  # near / far tertile markers (CVD-safe blue/orange)


def _clean_name(n):
    n = n.replace("_", "/").split("/")[-1]
    return {"claude-haiku-4-5": "Claude-Haiku-4.5", "claude-sonnet-4-6": "Claude-Sonnet-4.6",
            "gemini-2-5-flash": "Gemini-2.5-Flash", "gemini-2-5-flash-lite": "Gemini-2.5-Flash-Lite",
            "llama-3-3-70b-instruct": "Llama-3.3-70B", "llama-3-1-8b-instruct": "Llama-3.1-8B",
            "gpt-4o-mini": "GPT-4o-mini", "gpt-4-1-mini": "GPT-4.1-mini"}.get(n, n)


def main(scores_dir):
    embed = get_embedder()
    def cos(a, b):
        x, y = embed(a), embed(b)
        return 1 - float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))

    rows = []
    for md in Path(scores_dir).glob("*/path_scores.json"):
        recs = json.loads(md.read_text())
        byp = defaultdict(dict)
        for r in recs:
            if r["mode"] == "analogy":
                byp[r["prompt_id"]][r["path_idx"]] = r
        succ, dist = [], []
        for _pid, paths in byp.items():
            p0, p1 = paths.get(0), paths.get(1)
            if not p0:
                continue
            fact = [f for r in paths.values() for f in (r.get("factual") or [])]
            struct = all(_node_distinct(r["triples"]) for r in paths.values() if r.get("triples"))
            rel = p1 is not None and _relations_match(p0.get("triples"), p1.get("triples"))
            disj = p1 is not None and _structures_disjoint(p0.get("triples"), p1.get("triples"))
            succ.append(bool(p0.get("semantic_sat")) and fact and all(fact) and struct and rel and disj)
            dist.append(cos(p0["u_label"], p0["v_label"]))
        n = len(succ)
        idx = np.argsort(dist); t = n // 3
        near = np.mean([succ[i] for i in idx[:t]]) if t else np.nan
        far = np.mean([succ[i] for i in idx[2 * t:]]) if t else np.nan
        rows.append((sum(succ) / n, _clean_name(md.parent.name), n, near, far))
    rows.sort()  # ascending so best ends up on top of horizontal bars

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ys = range(len(rows))
    ax.barh(list(ys), [r[0] for r in rows], color=BAR, height=0.6, zorder=2)
    for y, (rate, name, n, near, far) in zip(ys, rows):
        ax.scatter([near], [y], marker="o", s=55, color=NEAR, zorder=4, edgecolors="white", linewidths=1)
        ax.scatter([far], [y], marker="D", s=48, color=FAR, zorder=4, edgecolors="white", linewidths=1)
        ax.text(rate + 0.006, y, f"{rate*100:.0f}%  (n={n})", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[1] for r in rows], fontsize=10, color=INK)
    ax.set_xlabel("Valid analogies found  (exact-relation-match, factual, node-distinct)", color=MUTED)
    ax.set_xlim(0, max(r[0] for r in rows) + 0.08)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=NEAR, markersize=9, label="near-tertile (related pairs)"),
                       Line2D([0], [0], marker="D", color="w", markerfacecolor=FAR, markersize=9, label="far-tertile (unrelated pairs)")],
              frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Can models find valid analogies between arbitrary entities?  (n=200 random pairs each)",
                 color=INK, fontsize=12.5)
    fig.tight_layout()
    out = Path(scores_dir) / "analogy_suite.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_analogy_v2")
