"""Appendix companions to Figure 1 (tc_over_time), reusing its EXACT palette/styling.

(1) facets_over_time.pdf  -- 1x4 panels: surprise, coherence, realism, diversity
    (each facet's percentile vs model release date), same colours as Fig 1.
(2) tc_per_org_over_time.pdf -- small-multiples: headline TC (mean of the realism-gated
    composite-facet percentiles) vs release date, ONE panel per major org (the named
    providers in the Fig 1 legend) + an "Other" bucket; 4-column grid.

No API: release dates are read from the cached model_created.json that tc_over_time.py
wrote. Reads the same tc.json.

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/make_over_time_appendix.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from src.plot_twist.scripts.tc_over_time import (
    OTHER_COLOR, OTHER_PROVIDERS, PROV_NAME, provider_colors,
)

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "font.size": 9, "axes.labelsize": 11, "axes.titlesize": 11,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "legend.fontsize": 8, "legend.frameon": False, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

TC_JSON = Path("data/plot_twist/tc/tc.json")
DATES = Path("data/plot_twist/tc/downstream/over_time/model_created.json")
OUT = Path("data/plot_twist/tc/downstream/over_time_appendix")
FIG = Path("papers/pt2cb-iclr-2027/figures")

# user order: surprise, coherence, realism, diversity
FACETS = [("mean_surprise", "Surprise"), ("mean_coherence", "Coherence"),
          ("mean_realism", "Realism"), ("div", "Diversity")]
# Per-org headline TC uses the realism-GATED composite (S/Coh count only when realism==5),
# matching tc_over_time / the leaderboard. (The facets_over_time decomposition below is shown
# UNGATED on purpose -- see its PANELS.)
COMPOSITE_FACETS = ["mean_surprise_g", "mean_coherence_g", "div"]


def load():
    tc = json.loads(TC_JSON.read_text())
    created = json.loads(DATES.read_text())
    # Pool covers the displayed raw facets AND the gated composite facets (S/Coh count
    # only when fully realistic). The headline composite `tc` is the mean of the GATED
    # facet percentiles -- realism enters as the gate, not as an additive facet.
    POOL_KEYS = [k for k, _ in FACETS] + COMPOSITE_FACETS
    pool = {k: np.array([d[k] for d in tc], dtype=float) for k in POOL_KEYS}

    def pctrank(k, x):
        v = pool[k]
        return 100.0 * (np.sum(v < x) + 0.5 * np.sum(v == x)) / len(v)

    pts = []
    for d in tc:
        if d["source"] == "human":
            continue
        ts = created.get(d["source"])
        if not ts:
            continue
        pc = {k: pctrank(k, d[k]) for k in POOL_KEYS}
        pts.append({"model": d["source"], "provider": d["source"].split("/")[0],
                    "date": datetime.fromtimestamp(ts, timezone.utc),
                    "pc": pc, "tc": float(np.mean([pc[k] for k in COMPOSITE_FACETS]))})
    pts.sort(key=lambda p: p["date"])
    human = next((d for d in tc if d["source"] == "human"), None)
    hpc = {k: pctrank(k, human[k]) for k in POOL_KEYS} if human else None
    htc = float(np.mean([hpc[k] for k in COMPOSITE_FACETS])) if hpc else None
    providers = sorted({d["source"].split("/")[0] for d in tc if d["source"] != "human"})
    return pts, hpc, htc, provider_colors(providers)


def _date_axis(ax, interval=8):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    for lab in ax.get_xticklabels():
        lab.set_rotation(30); lab.set_ha("right")


def facets_over_time(pts, hpc, prov_color):
    # Row 1: the four individual facets. Row 2: the composite built up cumulatively
    # (Surprise -> +Coherence -> +Diversity -> Overall), each panel the mean of its facet
    # percentiles. UNGATED throughout -- raw surprise/coherence (no realism gate); realism
    # appears as its own facet (row 1) and enters only the 4-facet Overall.
    PANELS = [
        ("Surprise", ["mean_surprise"]),
        ("Coherence", ["mean_coherence"]),
        ("Realism", ["mean_realism"]),
        ("Diversity", ["div"]),
        ("Surprise", ["mean_surprise"]),
        ("Surprise +\nCoherence", ["mean_surprise", "mean_coherence"]),
        ("Surprise +\nCoherence +\nDiversity", ["mean_surprise", "mean_coherence", "div"]),
        ("Overall", ["mean_surprise", "mean_coherence", "mean_realism", "div"]),
    ]
    # Large fonts for readability: smaller figure footprint (closer to 1:1 at \textwidth)
    # plus big point sizes so nothing is tiny on the page.
    big = {"axes.titlesize": 21, "axes.labelsize": 21, "xtick.labelsize": 16,
           "ytick.labelsize": 16, "legend.fontsize": 16, "font.size": 17}
    with mpl.rc_context(big):
        fig, axes = plt.subplots(2, 4, figsize=(10, 6.2), sharex=True, sharey=True)
        axes = axes.ravel()
        dates = [p["date"] for p in pts]
        for ax, (name, keys) in zip(axes, PANELS):
            y = np.array([float(np.mean([p["pc"][k] for k in keys])) for p in pts])
            for p_, yy in zip(pts, y):
                hi = p_["provider"] not in OTHER_PROVIDERS
                ax.scatter(p_["date"], yy, s=34 if hi else 20, color=prov_color.get(p_["provider"], OTHER_COLOR),
                           edgecolor="#333" if hi else "none", linewidth=0.4,
                           alpha=0.95 if hi else 0.55, zorder=3 if hi else 2)
            ax.plot(dates, np.maximum.accumulate(y), color="#000", lw=1.8, drawstyle="steps-post", zorder=4)
            if hpc is not None:
                ax.axhline(float(np.mean([hpc[k] for k in keys])), color="#000", ls=":", lw=1.6, zorder=1)
            ax.set_title(name); ax.set_ylim(-3, 103); ax.set_yticks([0, 25, 50, 75, 100])
            _date_axis(ax, interval=12)
        for r in (0, 4):
            axes[r].set_ylabel("Percentile")
        present = {p["provider"] for p in pts}
        named = [pr for pr in PROV_NAME if pr not in OTHER_PROVIDERS and pr in present]
        handles = [Patch(color=prov_color[pr], label=PROV_NAME[pr]) for pr in named]
        if present & OTHER_PROVIDERS:
            handles.append(Patch(color=OTHER_COLOR, label="Other"))
        handles.append(plt.Line2D([0], [0], color="#000", lw=1.8, label="Frontier"))
        handles.append(plt.Line2D([0], [0], color="#000", ls=":", lw=1.6, label="Expert-human ceiling"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.07),
                   ncol=5, fontsize=15, handlelength=1.2, columnspacing=1.2, handletextpad=0.4)
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        OUT.mkdir(parents=True, exist_ok=True)
        for d in (OUT / "facets_over_time.pdf", FIG / "facets_over_time.pdf", OUT / "facets_over_time.png"):
            fig.savefig(d)
        plt.close(fig)


def tc_per_org_over_time(pts, htc, prov_color):
    MIN_MODELS = 4  # drop sparse orgs
    present = {p["provider"] for p in pts}
    named = [pr for pr in PROV_NAME if pr not in OTHER_PROVIDERS and pr in present]
    named = [pr for pr in named if sum(1 for p in pts if p["provider"] == pr) >= MIN_MODELS]
    named.sort(key=lambda pr: -sum(1 for p in pts if p["provider"] == pr))  # busiest first
    groups = [(PROV_NAME[pr], [p for p in pts if p["provider"] == pr], prov_color[pr]) for pr in named]
    other = [p for p in pts if p["provider"] in OTHER_PROVIDERS]
    if len(other) >= MIN_MODELS:
        groups.append(("Other", other, OTHER_COLOR))

    big = {"axes.titlesize": 21, "axes.labelsize": 21, "xtick.labelsize": 16,
           "ytick.labelsize": 16, "font.size": 17}
    with mpl.rc_context(big):
        ncol = 4
        nrow = int(np.ceil(len(groups) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.1 * nrow), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        xmin, xmax = min(p["date"] for p in pts), max(p["date"] for p in pts)
        xpad = (xmax - xmin) * 0.05  # so markers at the extremes aren't clipped by the spine
        for ax, (name, grp, col) in zip(axes, groups):
            grp = sorted(grp, key=lambda p: p["date"])
            gx = [p["date"] for p in grp]; gy = np.array([p["tc"] for p in grp])
            if htc is not None:
                ax.axhline(htc, color="#000", ls=":", lw=1.6, zorder=1)
            # running-best staircase within the org (matches the frontier style elsewhere),
            # held flat out to the right x-limit so it always reaches the panel edge.
            cm = np.maximum.accumulate(gy)
            ax.plot(list(gx) + [xmax + xpad], list(cm) + [cm[-1]],
                    color="#000", lw=1.8, drawstyle="steps-post", zorder=2)
            ax.scatter(gx, gy, s=42, color=col, edgecolor="#333", linewidth=0.4, zorder=3)
            # annotate the staircase leader (model holding the top of the running best:
            # max TC, latest among ties); place the label in the emptiest corner (white
            # space) with a thin leader line back to the point.
            mx = float(cm[-1])
            best = max((p for p in grp if p["tc"] == mx), key=lambda p: p["date"])
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            def frac(d, v):  # data -> axes fraction
                return (mdates.date2num(d) - x0) / (x1 - x0), (v - y0) / (y1 - y0)
            # obstacles to avoid: data points, the staircase vertices, and the ceiling line
            obst = [frac(p["date"], p["tc"]) for p in grp]
            obst += [frac(d, c) for d, c in zip(gx, cm)]
            if htc is not None:
                fyc = (htc - y0) / (y1 - y0)
                obst += [(t, fyc) for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
            # candidate label anchors (stay below the ceiling band; left/centre/right)
            cands = [(cxx, cyy, ha, va)
                     for cxx, ha in ((0.04, "left"), (0.5, "center"), (0.96, "right"))
                     for cyy, va in ((0.06, "bottom"), (0.32, "center"), (0.56, "center"), (0.78, "top"))]
            cx, cy, ha, va = max(cands, key=lambda c: min(
                ((c[0] - px) ** 2 + (c[1] - py) ** 2) ** 0.5 for px, py in obst))
            ax.annotate(best["model"].split("/")[-1], xy=(best["date"], best["tc"]),
                        xytext=(cx, cy), textcoords="axes fraction", ha=ha, va=va,
                        fontsize=11, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="0.45", shrinkA=0, shrinkB=3))
            ax.set_title(f"{name} ($n{{=}}{len(grp)}$)")
            ax.set_ylim(-3, 103); ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_xlim(xmin - xpad, xmax + xpad); _date_axis(ax, interval=12)
        for ax in axes[len(groups):]:
            ax.set_visible(False)
        for r in range(nrow):
            axes[r * ncol].set_ylabel("TC percentile")
        fig.tight_layout()
        for d in (OUT / "tc_per_org_over_time.pdf", FIG / "tc_per_org_over_time.pdf", OUT / "tc_per_org_over_time.png"):
            fig.savefig(d)
        plt.close(fig)


def main():
    pts, hpc, htc, prov_color = load()
    print(f"{len(pts)} dated models; human TC pctl = {htc:.1f}")
    facets_over_time(pts, hpc, prov_color)
    tc_per_org_over_time(pts, htc, prov_color)
    print(f"saved facets_over_time.pdf + tc_per_org_over_time.pdf -> {FIG}")


if __name__ == "__main__":
    main()
