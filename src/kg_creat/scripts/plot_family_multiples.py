"""Rate of tau = 2 inventive multiples within each provider family, against the cross-family rate.

Reads `model_pair_matrix` in inventive_multiples.json (per model pair: the fraction of items on which
the two inventions are a tau-multiple, and the number of items compared, per task). For each provider
with at least two models in the pool, the within-family rate is the item-weighted mean over its model
pairs, pooled over blending and analogy; the cross-family rate pools every pair of models from different
providers. Writes the numbers to analysis/family_multiples.json (so the paper cites a file, not a chat
computation) and the figure to the multiples report's figure folder.

    .venv/bin/python -m src.kg_creat.scripts.plot_family_multiples
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT_JSON = Path("data/kg_creat/kombine_test30/analysis/family_multiples.json")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
NAME = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google", "x-ai": "xAI", "deepseek": "DeepSeek",
        "meta-llama": "Meta", "qwen": "Qwen", "z-ai": "Z.ai", "moonshotai": "Moonshot", "microsoft": "Microsoft"}
SAME, DIFF = "#486878", "#C9CDD1"          # the slate of Figure 6 and Table 4's row tint, so the multiples figures share one palette
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "font.size": 15,
                     "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black"})


def main():
    d = json.loads(SRC.read_text()); m = d["model_pair_matrix"]
    models, provs = m["models"], m["providers"]
    n = len(models)
    hits = defaultdict(float); items = defaultdict(float); cross_hits = cross_items = 0.0
    for task in ("blending", "analogy"):
        R, N = np.array(m[task]["multiple_rate"], float), np.array(m[task]["n_items"], float)
        for i in range(n):
            for j in range(i + 1, n):
                if not np.isfinite(R[i, j]) or N[i, j] == 0:
                    continue
                if provs[i] == provs[j]:
                    hits[provs[i]] += R[i, j] * N[i, j]; items[provs[i]] += N[i, j]
                else:
                    cross_hits += R[i, j] * N[i, j]; cross_items += N[i, j]
    n_models = defaultdict(int)
    for p in provs:
        n_models[p] += 1
    rows = [{"provider": p, "name": NAME.get(p, p), "n_models": n_models[p], "n_pair_items": int(items[p]),
             "n_multiples": int(round(hits[p])), "within_pct": 100.0 * hits[p] / items[p]} for p in items if items[p] > 0]
    rows.sort(key=lambda r: -r["within_pct"])
    cross_pct = 100.0 * cross_hits / cross_items
    OUT_JSON.write_text(json.dumps({"tau": 2, "cross_family_pct": cross_pct, "cross_family_pair_items": int(cross_items), "by_provider": rows}, indent=1))
    for r in rows:
        print(f"{r['name']:10s} {r['n_models']} models  {r['n_multiples']:3d} / {r['n_pair_items']:5d} = {r['within_pct']:.1f}%")
    print(f"cross-family {cross_pct:.2f}% over {int(cross_items)} pair-items")

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(len(rows)); v = [r["within_pct"] for r in rows]
    ax.bar(x, v, 0.62, color=SAME, zorder=3, label="Same")
    for xi, r in zip(x, rows):
        ax.text(xi, r["within_pct"] + 0.25, f"{r['within_pct']:.1f}%", ha="center", va="bottom", fontsize=12, color="black")
    ax.axhline(cross_pct, color="black", ls="--", lw=1.3, zorder=4, label=f"Different ({cross_pct:.1f}%)")
    # the provider mark sits just under the axis and the provider name hangs from it (as in the generic-space grid)
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from src.kg_creat.scripts.plot_multiples_matrix import brand_logos
    logos = brand_logos(); LOGO_KEY = {"meta-llama": "meta"}
    ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in rows], fontsize=12)
    ax.tick_params(axis="x", length=0, pad=24)
    for xi, r in zip(x, rows):
        img = logos.get(LOGO_KEY.get(r["provider"], r["provider"]))
        if img is not None:
            ab = AnnotationBbox(OffsetImage(img, zoom=0.040, alpha=0.95), (xi, -0.85), frameon=False,
                                box_alignment=(0.5, 0.5), annotation_clip=False, xycoords="data")
            ab.set_clip_on(False); ax.add_artist(ab)
    ax.set_ylabel("$\\tau=2$ multiples (% of pairs)", fontsize=13.5)
    ax.set_ylim(0, max(v) * 1.28); ax.set_yticks([0, 2, 4, 6, 8, 10]); ax.set_yticklabels([f"{t}%" for t in (0, 2, 4, 6, 8, 10)], fontsize=12)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6E6E6", zorder=0)
    ax.legend(frameon=False, fontsize=12, loc="upper right", handlelength=1.6)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_family_multiples.{ext}", dpi=220, bbox_inches="tight")
    print("saved", OUT / "fig_family_multiples.{png,pdf}")


if __name__ == "__main__":
    main()
