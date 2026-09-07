"""Camera-ready figures for the artifact-homogeneity analysis.

  fig_hivemind_heatmaps   2 rows of lower-triangular model x model similarity heatmaps.
                          Row 1 = BASE combinatorial artifact (association bridge / analogy mapping /
                          blend structure); Row 2 = EMERGENT invention (analogy h / blend Delta;
                          association has none). Shared colour scale -> brightness is comparable.
  fig_hivemind_mechanism  EMERGENT view: A anchor->invention RSA; B anchor distance vs convergence.

    .venv/bin/python -m src.kg_creat.scripts.plot_hivemind
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333", "text.color": "#222222",
    "axes.labelcolor": "#222222", "font.size": 15,
    "axes.titlesize": 17, "axes.labelsize": 16, "legend.fontsize": 13,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
})

C = {"baseline": "#5AA07A", "analogy": "#3B6EA5", "blending": "#C2703D"}
MK = {"analogy": "o", "blending": "s"}
LABEL = {"baseline": "Association", "analogy": "Analogy", "blending": "Blending"}

D = json.loads(Path("data/kg_creat/kombine_test30/analysis/invention_homogeneity.json").read_text())
OUT = Path("docs/reports/2026-08-31_kg_creat_invention_homogeneity/figures")
OUT.mkdir(parents=True, exist_ok=True)
order = D["base"]["blending"]["plot"]["model_order"]
n = len(order)


def fit(x, y):
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(min(x), max(x), 50)
    return xs, a + b * xs


def lower(M):
    return np.where(np.tril(np.ones_like(M), k=-1) == 1, M, np.nan)


# shared colour scale over all off-diagonal similarities in both views
allv = []
for view, modes in (("base", ["baseline", "analogy", "blending"]), ("emergent", ["analogy", "blending"])):
    for m in modes:
        M = np.array(D[view][m]["plot"]["sim_matrix"])
        allv += [M[i, j] for i in range(n) for j in range(i) if not np.isnan(M[i, j])]
vmin, vmax = float(np.min(allv)), float(np.max(allv))
cmap = plt.cm.magma_r.copy(); cmap.set_bad("#FBFBFB")

# ===================== Figure 1: 2-row heatmap grid =====================
fig1, axes = plt.subplots(2, 3, figsize=(15.6, 11.0))
GRID = [[("base", "baseline"), ("base", "analogy"), ("base", "blending")],
        [None, ("emergent", "analogy"), ("emergent", "blending")]]
TITLE = {("base", "baseline"): "Association", ("base", "analogy"): "Analogy",
         ("base", "blending"): "Blending", ("emergent", "analogy"): "Analogy",
         ("emergent", "blending"): "Blending"}
im = None
for ri in range(2):
    for ci in range(3):
        ax = axes[ri][ci]
        cell = GRID[ri][ci]
        if cell is None:                                      # association has no emergent artifact
            ax.text(0.5, 0.5, "no emergent artifact\n(association produces\nno invention)",
                    ha="center", va="center", fontsize=15, color="#999999", style="italic",
                    transform=ax.transAxes)
            ax.axis("off"); continue
        view, mode = cell
        R = D[view][mode]
        M = np.array(R["plot"]["sim_matrix"], dtype=float)
        im = ax.imshow(lower(M), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{TITLE[cell]}   (mean {R['hivemind_index']:.2f})",
                     fontsize=15, fontweight="bold", color=C[mode], pad=7)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        show_y = (ci == 0) or (ri == 1 and ci == 1)           # leftmost heatmap of each row
        ax.set_yticklabels(order if show_y else [""] * n, fontsize=10)
        show_x = (ri == 1) or (ri == 0 and ci == 0)           # bottom-most heatmap of each column
        ax.set_xticklabels(order if show_x else [""] * n, rotation=90, fontsize=10)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
# row labels
fig1.text(0.028, 0.72, "BASE\ncombinatorial\nartifact", fontsize=17, fontweight="bold",
          rotation=90, va="center", ha="center", color="#444444")
fig1.text(0.028, 0.29, "EMERGENT\ninvention", fontsize=17, fontweight="bold",
          rotation=90, va="center", ha="center", color="#444444")
fig1.subplots_adjust(left=0.145, right=0.9, top=0.93, bottom=0.11, hspace=0.28, wspace=0.1)
cax = fig1.add_axes([0.915, 0.11, 0.012, 0.82])
fig1.colorbar(im, cax=cax).set_label("pairwise artifact similarity (cosine)", fontsize=15)
fig1.suptitle("Inter-model artifact similarity  (lower triangle; each cell = one model pair)",
              fontsize=19, y=0.985)
for ext in ("png", "pdf"):
    fig1.savefig(OUT / f"fig_hivemind_heatmaps.{ext}", dpi=300, bbox_inches="tight")

print("saved fig_hivemind_heatmaps ->", OUT)
