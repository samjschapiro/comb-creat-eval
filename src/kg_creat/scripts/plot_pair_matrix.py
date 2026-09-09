"""Which models invent the same thing as which: a model x model heatmap, averaged over all items.

Reads the `model_pair_matrix` block written by analyze_inventive_multiples.py. For each pair of
models and each task, the cell is the MEAN NUMBER OF SHARED PROPERTIES over the anchor pairs both
models answered (30 per task), under the paper's criterion (theta, one-to-one, name and anchor echo
excluded). Rows and columns are grouped by provider, and within a provider ordered by the model's
row mean, so the hubs sit at the top of each block. Both panels share one colour scale, so the
blending / analogy contrast is honest.

  (a) blending    (b) analogy

    .venv/bin/python -m src.kg_creat.scripts.plot_pair_matrix
    .venv/bin/python -m src.kg_creat.scripts.plot_pair_matrix --cell multiple_rate   # share of items on
                                                                                     # which the pair is a multiple
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.kg_creat.scripts.plot_radar import BRAND, _load_logos, _provider

SRC = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
PROV_ORDER = ["anthropic", "openai", "google", "x-ai", "deepseek", "qwen", "z-ai", "meta-llama",
              "microsoft", "moonshotai"]
PROV_LABEL = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google", "x-ai": "xAI",
              "deepseek": "DeepSeek", "qwen": "Qwen", "z-ai": "Z-AI", "meta-llama": "Meta",
              "microsoft": "Microsoft", "moonshotai": "Moonshot"}

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 16, "axes.titlesize": 20, "axes.labelsize": 17})


def _brand(prov):
    return BRAND.get({"meta-llama": "meta", "moonshotai": "moonshot"}.get(prov, prov), "#8A94A3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["mean_shared", "multiple_rate"], default="mean_shared")
    a = ap.parse_args()
    d = json.loads(SRC.read_text())
    if "model_pair_matrix" not in d:
        raise SystemExit(f"{SRC} has no `model_pair_matrix` block -- re-run analyze_inventive_multiples.py")
    pm = d["model_pair_matrix"]
    models, provs = pm["models"], pm["providers"]
    unknown = sorted({p for p in provs if p not in PROV_ORDER})
    if unknown:
        raise ValueError(f"FATAL: providers with no place in PROV_ORDER: {unknown}")
    logos = _load_logos()

    mats = {t: np.array(pm[t][a.cell], float) for t in ("blending", "analogy")}
    # one order for both panels: provider blocks, then by blending row mean within the block
    row_mean = np.nanmean(mats["blending"], axis=1)
    order = sorted(range(len(models)), key=lambda i: (PROV_ORDER.index(provs[i]), -row_mean[i], models[i]))
    prov_o = [provs[i] for i in order]
    n = len(order)
    vmax = float(np.nanmax(np.concatenate([np.nan_to_num(mats[t][np.ix_(order, order)]) for t in mats])))
    label = {"mean_shared": "mean shared properties per item",
             "multiple_rate": r"share of items on which the pair is a $\tau = 2$ multiple"}[a.cell]

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 8.6), gridspec_kw={"wspace": 0.10})
    cmap = plt.get_cmap("Blues").copy(); cmap.set_bad("white")
    for ax, (task, letter) in zip(axes, (("blending", "a"), ("analogy", "b"))):
        Mx = mats[task][np.ix_(order, order)]
        Mx = np.ma.masked_invalid(Mx)
        np.fill_diagonal(Mx, np.nan); Mx = np.ma.masked_invalid(Mx)
        im = ax.imshow(Mx, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
        cuts = [i for i in range(1, n) if prov_o[i] != prov_o[i - 1]]
        for c in cuts:
            ax.axhline(c - .5, color="white", lw=2.4); ax.axvline(c - .5, color="white", lw=2.4)
        # provider colour strips outside the frame, with the logo (or the name) on the blocks
        for i, p in enumerate(prov_o):
            ax.add_patch(plt.Rectangle((i - .5, n - .5 + 0.25), 1, 0.7, color=_brand(p), clip_on=False))
            ax.add_patch(plt.Rectangle((-.5 - 0.95, i - .5), 0.7, 1, color=_brand(p), clip_on=False))
        starts = [0] + cuts + [n]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            p = prov_o[s0]; mid = (s0 + s1 - 1) / 2
            img = logos.get({"meta-llama": "meta"}.get(p, p))
            if img is not None:
                ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.06), (mid, n + 1.35), frameon=False,
                                             box_alignment=(0.5, 0.5), annotation_clip=False))
                ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.06), (-2.0, mid), frameon=False,
                                             box_alignment=(0.5, 0.5), annotation_clip=False))
            elif s1 - s0 >= 1:
                ax.text(mid, n + 1.35, PROV_LABEL[p][:2], ha="center", va="center", fontsize=12, color=_brand(p))
                ax.text(-2.0, mid, PROV_LABEL[p][:2], ha="center", va="center", fontsize=12, color=_brand(p))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        overall = float(np.nanmean(mats[task][np.ix_(order, order)][~np.eye(n, dtype=bool)]))
        ax.set_title(f"({letter}) {task.capitalize()}", pad=40)
        ax.text(n - .5, -1.4, f"mean {overall:.2f}", ha="right", va="bottom", fontsize=14, color="#333333")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label(label, fontsize=16); cb.ax.tick_params(labelsize=14)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = "fig_pair_matrix" + ("" if a.cell == "mean_shared" else "_rate")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {stem} -> {OUT}  ({n} models, cell = {a.cell}, vmax = {vmax:.2f})")


if __name__ == "__main__":
    main()
