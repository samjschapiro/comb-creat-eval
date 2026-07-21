"""Plot analogy success vs. endpoint distance (the analogy-specific result).

For analogy the endpoints are random, so *difficulty* is set by how distant the pairing is
(cos distance between the two entity labels), not by the model. We ask: can a model find a
VALID analogy between arbitrary entities, and how does that fall off with distance?

Analogy success (computed here, correctly across BOTH structures) =
  semantic_sat (judge: genuine role-corresponding parallel) AND both paths fully factual.

Marker shape encodes within- vs cross-domain; a binned success-rate line per model shows the
trend. Run in the 3.12 env (needs the local embedder):

    .venv_mlx/bin/python src/kg_creat/scripts/plot_analogy.py data/kg_creat/scores_domains_v1
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.kg_creat.embed import get_embedder
from src.kg_creat import regime_b as RB
from src.kg_creat.scoring import cosine_distance

MODEL_COLORS = ["#2563EB", "#EA580C", "#059669", "#7C3AED"]
INK, MUTED, GRID = "#1f2933", "#66727f", "#e3e8ee"


def _short(m):
    for tag in ("3B", "7B", "14B", "32B", "72B"):
        if tag in m:
            return f"Qwen2.5-{tag}" if "Qwen2" in m else f"{m.split('_')[0]}-{tag}"
    return m


def analogy_points(scores_dir, embed):
    """Per model -> list of (distance, success, cross_domain) over analogy prompts."""
    out = {}
    for md in sorted(Path(scores_dir).iterdir()):
        psj = md / "path_scores.json"
        if not psj.exists():
            continue
        recs = json.loads(psj.read_text())
        by_prompt = defaultdict(dict)
        for r in recs:
            if r["mode"] == "analogy":
                by_prompt[r["prompt_id"]][r["path_idx"]] = r
        pts = []
        for pid, paths in by_prompt.items():
            p0 = paths.get(0)
            if p0 is None:
                continue
            fact = []
            for r in paths.values():
                fact += (r.get("factual") or [])
            factual_ok = len(fact) > 0 and all(fact)
            # both structures must be node-distinct (no loop-backs)
            structural_ok = all(RB.node_distinct(r["triples"]) for r in paths.values() if r.get("triples"))
            p1 = paths.get(1)
            # structure-mapping floor: same relation sequence AND disjoint entities across structures
            relations_ok = p1 is not None and RB.relations_match(p0.get("triples"), p1.get("triples"))
            disjoint_ok = p1 is not None and RB.structures_disjoint(p0.get("triples"), p1.get("triples"))
            success = bool(p0.get("semantic_sat")) and factual_ok and structural_ok and relations_ok and disjoint_ok
            u, v = p0["u_label"], p0["v_label"]
            dist = cosine_distance(embed(u), embed(v))
            pts.append((dist, int(success), p0.get("cross_domain")))
        out[md.name] = pts
    return out


def main(scores_dir):
    embed = get_embedder()
    data = analogy_points(scores_dir, embed)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED)

    bins = [0.4, 0.55, 0.7, 0.8, 0.9, 1.01]
    centers = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    rng = np.random.RandomState(0)
    for mi, (model, pts) in enumerate(data.items()):
        c = MODEL_COLORS[mi % len(MODEL_COLORS)]
        for dist, succ, cross in pts:
            jitter = (rng.rand() - 0.5) * 0.06
            marker = "o" if cross else "s"  # o = cross-domain, s = within-domain
            ax.scatter([dist], [succ + jitter], s=45, color=c, alpha=0.55,
                       marker=marker, edgecolors="white", linewidths=0.6, zorder=3)
        # binned success-rate line
        xs, ys = [], []
        for i in range(len(bins) - 1):
            inb = [s for d, s, _ in pts if bins[i] <= d < bins[i + 1]]
            if inb:
                xs.append(centers[i])
                ys.append(sum(inb) / len(inb))
        ax.plot(xs, ys, "-", color=c, lw=2.4, marker="D", markersize=7, zorder=5,
                label=f"{_short(model)}  (n={len(pts)})")

    ax.set_xlabel("Endpoint distance  (1 - cosine;  higher = more unrelated pairing)", color=MUTED)
    ax.set_ylabel("Valid analogy found  (rate)", color=MUTED)
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title("Can models find valid analogies between arbitrary entities?\n"
                 "success vs. how unrelated the pairing is  (○ cross-domain, □ within-domain)",
                 color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    out = Path(scores_dir) / "analogy_success_vs_distance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_domains_v1")
