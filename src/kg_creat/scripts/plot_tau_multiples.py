"""The tau-inventive-multiples table as a figure: rate of multiples at tau = 1, 2, 3.

Three panels from `tau_curve` in inventive_multiples.json, one row, text width:
  (a) % of all inventions that belong to at least one tau-multiple;
  (b) % of invention pairs that are tau-multiples, blends vs analogies (log scale; the blend/analogy
      ratio printed above each tau), in the paper's task colours;
  (c) the same split by whether the two models share a provider family (same/different ratio above).
Bars carry their value; a zero rate is drawn as no bar with "0" written at the axis.

    .venv/bin/python -m src.kg_creat.scripts.plot_tau_multiples
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
BLEND, ANALOGY = "#3F6F8F", "#9A7D2E"          # the paper's task colours (facet-correlation figure: TASK_COL)
SAME, DIFF = "#4D4D4D", "#BDBDBD"              # model family: neutral greys
INV = "#103D5F"                                 # the paper's batlowBlue
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"], "font.size": 16})
YMIN = 0.03                                      # log-axis floor; a zero rate is written just above it


def bars(ax, taus, left, right, left_lab, right_lab, cl, cr, ratio_key=None, log=True):
    x = np.arange(len(taus)); w = 0.36
    for off, vals, lab, c in ((-w / 2, left, left_lab, cl), (w / 2, right, right_lab, cr)):
        v = np.array(vals, float); pos = x + off
        ax.bar(pos[v > 0], v[v > 0], w, color=c, label=lab, edgecolor="none", zorder=3)
        for xi, vi in zip(pos, v):
            y = (vi * 1.15) if vi > 0 else YMIN * 1.15
            lab = "0%" if vi == 0 else (f"{vi:.2f}%" if vi < 1 else f"{vi:.1f}%")
            ax.text(xi, y, lab, ha="center", va="bottom", fontsize=12.5, color="#222222")
    if ratio_key:
        for xi, (l, r) in enumerate(zip(left, right)):
            if r > 0:
                top = max(l, r) * (3.2 if log else 1) + (0 if log else 6)
                ax.text(xi, top, f"{l / r:.1f}$\\times$", ha="center", va="bottom", fontsize=15, fontweight="bold", color="#222222")
    ax.set_xticks(x); ax.set_xticklabels([f"$\\tau={t}$" for t in taus], fontsize=15)
    if log:
        ax.set_yscale("log"); ax.set_ylim(YMIN, 300)
        ax.set_yticks([0.1, 1, 10, 100]); ax.set_yticklabels(["0.1%", "1%", "10%", "100%"], fontsize=14)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6E6E6", zorder=0)
    ax.legend(frameon=False, fontsize=14, loc="upper right", handlelength=1.2, borderaxespad=0.2)


def main():
    d = json.loads(SRC.read_text())
    rows = {r["tau"]: r for r in d["tau_curve"]}
    taus = [1, 2, 3]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.05, 1.15, 1.15]})
    ax = axes[0]; v = [rows[t]["inventions_pct"] for t in taus]; x = np.arange(3)
    ax.bar(x, v, 0.55, color=INV, zorder=3)
    for xi, vi in zip(x, v): ax.text(xi, vi + 1.5, f"{vi:.1f}%", ha="center", va="bottom", fontsize=12.5, color="#222222")
    ax.set_xticks(x); ax.set_xticklabels([f"$\\tau={t}$" for t in taus], fontsize=15); ax.set_ylim(0, 80)
    ax.set_yticks([0, 20, 40, 60, 80]); ax.set_yticklabels([f"{t}%" for t in (0, 20, 40, 60, 80)], fontsize=14)
    ax.set_title("(a) Inventions in a multiple", loc="left", fontsize=17)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#E6E6E6", zorder=0)
    bars(axes[1], taus, [rows[t]["blending_pct"] for t in taus], [rows[t]["analogy_pct"] for t in taus], "Blend", "Analogy", BLEND, ANALOGY, ratio_key=True)
    axes[1].set_title("(b) % of pairs, by task", loc="left", fontsize=17); axes[1].set_ylabel("$\\tau$-multiples (% of pairs)", fontsize=15)
    bars(axes[2], taus, [rows[t]["same_provider_pct"] for t in taus], [rows[t]["cross_provider_pct"] for t in taus], "Same family", "Different", SAME, DIFF, ratio_key=True)
    axes[2].set_title("(c) % of pairs, by model family", loc="left", fontsize=17)
    fig.tight_layout(w_pad=1.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_tau_multiples.{ext}", dpi=220, bbox_inches="tight")
    print("saved", OUT / "fig_tau_multiples.{png,pdf}")
    for t in taus:
        r = rows[t]; print(f"tau={t}: blend {r['blending_pct']:.2f} analogy {r['analogy_pct']:.2f} same {r['same_provider_pct']:.2f} diff {r['cross_provider_pct']:.2f} inv {r['inventions_pct']:.1f}")


if __name__ == "__main__":
    main()
