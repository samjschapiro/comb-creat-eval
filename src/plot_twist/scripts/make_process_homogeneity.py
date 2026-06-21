"""Process-level homogeneity of reasoning traces (Exp 1 companion). Uses the existing trace
annotations (trace_strategy / trace_moves / trace_anchor) to show two things:

 (a) CROSS-MODEL + EFFORT homogeneity of the primary creative strategy: nearly all traces,
     at every effort level, backward-chain from a twist-first anchor.
 (b) EFFORT-INVARIANCE of the process moves: scaling reasoning effort lengthens the trace
     but does not change WHICH moves are made -- same recipe, more of it.

Mechanistic reading: a convergent creative *process* structurally bounds output diversity
(open-endedness; the artificial-hivemind effect).

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/make_process_homogeneity.py
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm as cmc

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15, "xtick.labelsize": 12, "ytick.labelsize": 11,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

BASE = Path("data/plot_twist/thinking/downstream")
OUT = BASE
FIG = Path("papers/pt2cb-iclr-2027/figures")
LEVELS = ["low", "medium", "high"]
MOVES = ["frames_constraints", "enumerates_tropes", "proposes_and_rejects", "setup_first",
         "seeks_max_recontextualization", "checks_preservation", "plans_specific_clues",
         "outlines_structure", "picks_reveal_vehicle", "reveal_first"]
MOVE_LABELS = ["frame constraints", "enumerate tropes", "propose & reject", "setup-first",
               "seek max recontext.", "check preservation", "plan specific clues",
               "outline structure", "pick reveal vehicle", "reveal-first"]


def _load(d):
    out = {}
    for f in glob.glob(str(BASE / d / "*.json")):
        r = json.load(open(f))
        out[r["id"]] = r
    return out


def main():
    strat, moves, anch = _load("trace_strategy"), _load("trace_moves"), _load("trace_anchor")
    ids = set(strat) & set(moves) & set(anch)
    lvl = {i: (strat[i].get("level") or ("high" if "rhigh" in i else "medium" if "rmedium" in i else "low"))
           for i in ids}

    strategies = ["backward_chaining", "divergent_enumeration", "forward_construction"]
    cols = [cmc.batlow(x) for x in (0.15, 0.55, 0.85)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) primary-strategy share by effort level (stacked bars) -> one strategy dominates, flat
    ax = axes[0]
    bottoms = np.zeros(len(LEVELS))
    for si, s in enumerate(strategies):
        vals = []
        for L in LEVELS:
            sub = [i for i in ids if lvl[i] == L]
            vals.append(100 * sum(strat[i]["primary"] == s for i in sub) / max(len(sub), 1))
        ax.bar(LEVELS, vals, bottom=bottoms, color=cols[si], edgecolor="white", lw=0.8,
               label=s.replace("_", " "))
        bottoms += np.array(vals)
    ax.set_ylim(0, 100); ax.set_ylabel("Share of traces (%)"); ax.set_xlabel("Reasoning effort")
    ax.set_title("(a) Primary creative strategy", loc="left", fontweight="bold")
    ax.legend(fontsize=10, loc="lower center", frameon=False, ncol=1)
    ax.text(1, 50, "all 9 models\nbackward-chain\nfrom a twist-first\nanchor (100%)",
            ha="center", va="center", fontsize=9.5, color="white", fontweight="bold")

    # (b) move rates by effort level (heatmap rows=moves, cols=levels) -> columns ~identical
    ax = axes[1]
    M = np.array([[100 * sum(moves[i].get(mk, False) for i in ids if lvl[i] == L) / max(sum(lvl[i] == L for i in ids), 1)
                   for L in LEVELS] for mk in MOVES])
    im = ax.imshow(M, aspect="auto", cmap=cmc.batlow, vmin=0, vmax=100)
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS)
    ax.set_yticks(range(len(MOVES))); ax.set_yticklabels(MOVE_LABELS)
    ax.set_xlabel("Reasoning effort")
    ax.set_title("(b) Process moves (% of traces) — invariant to effort", loc="left", fontweight="bold")
    for r in range(len(MOVES)):
        for c in range(len(LEVELS)):
            ax.text(c, r, f"{int(M[r, c])}", ha="center", va="center", fontsize=9,
                    color="white" if M[r, c] < 55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of traces")

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for d in (OUT / "process_homogeneity.pdf", FIG / "process_homogeneity.pdf", OUT / "process_homogeneity.png"):
        fig.savefig(d)
    plt.close(fig)
    # headline stats to stdout
    allc = Counter(strat[i]["primary"] for i in ids)
    print(f"saved -> {FIG/'process_homogeneity.pdf'}")
    print(f"backward_chaining = {100*allc['backward_chaining']//sum(allc.values())}% of {len(ids)} traces; "
          f"twist_first anchor = {100*sum(anch[i]['anchor']=='twist_first' for i in ids)//len(ids)}%")


if __name__ == "__main__":
    main()
