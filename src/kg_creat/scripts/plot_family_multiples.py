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
    """Concept level, opportunity-matched (analyze_inventive_multiples.concept_level): per provider, the share of its
    concepts with >= 1 tau = 2 multiple from a model of the same provider ("Same"), beside the share expected from the
    same number of models of other providers ("Different"), so providers with many models are not favoured."""
    d = json.loads(SRC.read_text())
    row = [r for r in d["concept_level"] if r["tau"] == 2][0]
    rows = []
    for key, v in row.items():
        if not key.startswith("provider:"):
            continue
        prov = key[len("provider:"):]
        rows.append({"provider": prov, "name": NAME.get(prov, prov), "n_concepts": v["n_concepts"],
                     "mean_same_candidates": v["mean_same_candidates"], "same_pct": v["same_provider_pct"],
                     "different_matched_pct": v["different_provider_matched_pct"], "different_raw_pct": v["different_provider_pct"]})
    rows.sort(key=lambda r: -r["same_pct"])
    overall = row["all"]
    OUT_JSON.write_text(json.dumps({"tau": 2, "unit": "concept", "overall_same_pct": overall["same_provider_pct"],
                                    "overall_different_matched_pct": overall["different_provider_matched_pct"], "by_provider": rows}, indent=1))
    for r in rows:
        print(f"{r['name']:10s} n={r['n_concepts']:4d} k={r['mean_same_candidates']:.1f}  same {r['same_pct']:.1f}%  different (matched) {r['different_matched_pct']:.1f}%")
    print(f"overall: same {overall['same_provider_pct']:.1f}%  different (matched) {overall['different_provider_matched_pct']:.1f}%")

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    x = np.arange(len(rows)); w = 0.62
    S = [r["same_pct"] for r in rows]; Dm = [r["different_matched_pct"] for r in rows]
    E = [s_ - d_ for s_, d_ in zip(S, Dm)]                  # excess over the opportunity-matched other-provider expectation
    rows.sort(key=lambda r: -(r["same_pct"] - r["different_matched_pct"])); S = [r["same_pct"] for r in rows]
    Dm = [r["different_matched_pct"] for r in rows]; E = [s_ - d_ for s_, d_ in zip(S, Dm)]
    ax.bar(x, E, w, color=SAME, zorder=3, label="Same provider")
    ymax = max(E) * 1.25; ymin = -5.0                        # a clean floor below the one negative bar
    for xi, e_ in zip(x, E):
        ax.text(xi, e_ + ymax * 0.012 if e_ >= 0 else e_ - ymax * 0.012, f"{e_:+.1f}", ha="center",
                va="bottom" if e_ >= 0 else "top", fontsize=11, color="black")
    ax.axhline(0, color="black", ls="--", lw=1.3, zorder=4, label="Other providers (expected)")
    # the provider mark sits just under the axis and the provider name hangs from it (as in the generic-space grid)
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from src.kg_creat.scripts.plot_multiples_matrix import brand_logos
    logos = brand_logos(); LOGO_KEY = {"meta-llama": "meta"}
    ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in rows])
    for xi, r in zip(x, rows):
        img = logos.get(LOGO_KEY.get(r["provider"], r["provider"]))
        if img is not None:
            ab = AnnotationBbox(OffsetImage(img, zoom=0.040, alpha=0.95), (xi, ymin - (ymax - ymin) * 0.09), frameon=False,
                                box_alignment=(0.5, 0.5), annotation_clip=False, xycoords="data")
            ab.set_clip_on(False); ax.add_artist(ab)
    ax.set_ylabel("Concepts with a sibling multiple\n(points beyond expectation)", fontsize=12)
    ax.set_ylim(ymin, ymax); ticks = list(range(int(ymin), int(ymax) + 1, 5))
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:+d}" if t else "0" for t in ticks], fontsize=12)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6E6E6", zorder=0)
    ax.tick_params(axis="x", length=0, pad=24)                # set last: later axis calls rebuild the ticks and drop label styling
    for lab in ax.get_xticklabels():
        lab.set_fontsize(10.5); lab.set_rotation(30); lab.set_ha("right"); lab.set_rotation_mode("anchor")
    ax.legend(frameon=False, fontsize=12.5, loc="upper right", handlelength=1.6)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_family_multiples.{ext}", dpi=220, bbox_inches="tight")
    print("saved", OUT / "fig_family_multiples.{png,pdf}")


if __name__ == "__main__":
    main()
