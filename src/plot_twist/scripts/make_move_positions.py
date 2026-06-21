"""Median within-trace position of common reasoning moves (Option A: ordered
lollipop / IQR strip).

For each move we locate every occurrence in every reasoning trace, normalise its
character offset to [0,1] (0 = start of the trace, 1 = end), and plot the median
(dot) and inter-quartile range (bar) across all occurrences. Moves are sorted by
median position, so the divergent "fix the twist" moves (warm) rise to the top and
the convergent "secure the coherence" moves (cool) sink to the bottom -- the
twist-first -> retrofit structure, made visual. The consistency move is split into
*promise coherence* (early, a restated goal) and *verify it coheres* (late, an
actual check) to show coherence is promised up front but only acted on at the end.

Reads the thinking-trace stories directly (no API). Writes PDF+PNG to the output
dir and a copy of the PDF into the paper's figures/ folder.

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/make_move_positions.py
"""

from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "font.size": 10, "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10.5,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

STORIES = "data/plot_twist/thinking/stories/*/*.json"
OUT_DIR = Path("data/plot_twist/thinking/downstream/move_positions")
FIG_DIR = Path("papers/pt2cb-iclr-2027/figures")

# Phase colours, matching tables/tab_moves.tex (moveSurprise / moveCoherence / moveFraming).
WARM = "#CB6A4F"   # surprise / divergent  (terracotta)
COOL = "#2C7A86"   # coherence / convergent (teal)
GREY = "#7D7468"   # framing               (warm grey)

# (label, side, regex).  side -> colour: S=surprise, C=coherence, N=framing
MOVES = [
    ("frame the task",           "N", r"(2[,.]?000|3[,.]?000 word|no title|choose (your|my) own|roughly 2)"),
    ("promise coherence",        "N", r"(must be consistent|should (be|feel) (consistent|prepared)|fair play|consistent with everything|prepared,? not arbitrary|feel (earned|inevitable|prepared))"),
    ("list potential twists",         "S", r"(classic twist|trope|unreliable narrator|narrator is (actually )?dead|turns out to be a (ghost|robot|simulation)|been done|overdone|common twist)"),
    ("plan setting",          "S", r"(a (man|woman|girl|boy|story) (named|who|about)|protagonist|the situation|setting:|main character)"),
    ("restate recontextualization goal", "S", r"(reinterpret|recontextualiz|reframe|the whole story|changes the meaning)"),
    ("propose, reject, & finalize twist", "S", r"(too (obvious|predictable|cliche|clich|on-the-nose|simple|supernatural)|let me think of something|another (idea|angle|option)|what if|maybe (the|she|he))"),
    ("plan clues to plant",      "C", r"(foreshadow|plant (a )?clue|clue:|hint(s)? (at|that)|set up (the|a) (detail|hint))"),
    ("choose a reveal event",  "C", r"(a letter|the ledger|a diary|the mirror|a photograph|a recording|a document|newspaper|obituary|medical (record|chart))"),
    ("outline full plot",        "C", r"(act (one|i|1)\b|section 1|opening scene|beats|paragraph|POV:|structure:|outline)"),
    ("verify it coheres",        "C", r"(snap(s)? into place|holds together|all the (clues|pieces)|re-?read|in hindsight (it|the)|still (works|makes sense|holds)|goes back and)"),
]
COLOR = {"S": WARM, "C": COOL, "N": GREY}
SIDE = {lab: side for lab, side, _ in MOVES}
SPANS_DIR = Path("data/plot_twist/thinking/downstream/move_spans")  # LLM span-extraction cache


def _span_positions():
    """From the LLM span cache: ({level: {label: [pos]}}, pooled {label: [pos]})."""
    by_level, pooled = defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for f in glob.glob(str(SPANS_DIR / "*.json")):
        if f.endswith("move_spans_stats.json"):
            continue
        o = json.load(open(f))
        if not isinstance(o, dict):
            continue
        for lab, c in (o.get("steps") or {}).items():
            if c.get("present") and c.get("pos") is not None:
                by_level[o.get("level")][lab].append(c["pos"])
                pooled[lab].append(c["pos"])
    return by_level, pooled


def _rows(posmap):
    """label->[pos]  ->  rows [{label, side, med, q1, q3, ntr}] in MOVES order."""
    rows = []
    for lab in [l for l, _, _ in MOVES]:
        ps = posmap.get(lab) or []
        if not ps:
            continue
        a = np.array(ps)
        rows.append({"label": lab, "side": SIDE[lab], "med": float(np.median(a)),
                     "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75)), "ntr": len(a)})
    return rows


def compute_rows(traces):
    """Per-move median/IQR position and #traces containing it, over `traces`."""
    rows = []
    for label, side, pat in MOVES:
        pos, ntr = [], 0
        rx = re.compile(pat, re.I)
        for r in traces:
            t = r["reasoning_trace"]; L = len(t)
            ms = [m.start() / L for m in rx.finditer(t)]
            if ms:
                ntr += 1; pos += ms
        if not pos:
            continue
        pos = np.array(pos)
        rows.append({"label": label, "side": side, "med": float(np.median(pos)),
                     "q1": float(np.percentile(pos, 25)), "q3": float(np.percentile(pos, 75)),
                     "ntr": ntr})
    return rows


def _lollipop(ax, rows, ys, cut, last_s, first_c, show_n=True):
    """Draw the median-dot + IQR-bar lollipops for `rows` at y-positions `ys`."""
    ax.axvspan(last_s, first_c, color="0.92", zorder=0)
    ax.axvline(cut, ls="--", lw=0.9, color="0.55", zorder=1)
    for y, r in zip(ys, rows):
        if r is None:
            continue
        c = COLOR[r["side"]]
        ax.plot([r["q1"], r["q3"]], [y, y], "-", color=c, lw=4, alpha=0.35,
                solid_capstyle="round", zorder=2)
        ax.plot(r["med"], y, "o", color=c, ms=7, zorder=3)
        if show_n:
            ax.text(1.005, y, f"{r['ntr']}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7.5, color="0.5")


def _shade(hexc, f):
    """Lighten `hexc` toward white by fraction f (f=0 keeps it; f=1 -> white)."""
    r, g, b = mpl.colors.to_rgb(hexc)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _level_of(r):
    return (r.get("reasoning_level")
            or ("high" if "rhigh" in r["id"] else "medium" if "rmedium" in r["id"] else "low"))


def figure_by_effort():
    """SINGLE-panel forest plot from the LLM SPAN positions (move_spans cache): low/medium/high
    overlaid per move, encoded by increasingly dark shades of the move's phase colour
    (light=low -> dark=high) at small vertical offsets. The three shades clustering at the same
    position shows the divergent->convergent structure is invariant to reasoning effort.
    Shared y-order = pooled span-median sort."""
    by_level, pooled = _span_positions()
    order_rows = sorted(_rows(pooled), key=lambda d: d["med"])
    labels = [r["label"] for r in order_rows]
    sides = {r["label"]: r["side"] for r in order_rows}
    n = len(labels)
    ys = list(range(n, 0, -1))
    last_s = max((r["med"] for r in order_rows if r["side"] == "S"), default=0.33)
    first_c = min((r["med"] for r in order_rows if r["side"] == "C"), default=0.46)
    cut = (last_s + first_c) / 2

    # (level, vertical offset within the move row, lighten fraction): low lightest -> high darkest
    LV = [("low", 0.26, 0.60), ("medium", 0.0, 0.30), ("high", -0.26, 0.0)]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.axvspan(last_s, first_c, color="0.92", zorder=0)
    ax.axvline(cut, ls="--", lw=0.9, color="0.55", zorder=1)
    for lv, dy, f in LV:
        by = {r["label"]: r for r in _rows(by_level.get(lv, {}))}
        for y, lab in zip(ys, labels):
            r = by.get(lab)
            if r is None:
                continue
            c = _shade(COLOR[sides[lab]], f)
            ax.plot([r["q1"], r["q3"]], [y + dy, y + dy], "-", color=c, lw=2.6, alpha=0.7,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["med"], y + dy, "o", color=c, ms=4.6, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.4)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    for tick, lab in zip(ax.get_yticklabels(), labels):
        tick.set_color(COLOR[sides[lab]] if sides[lab] != "N" else "0.35")
    ax.set_ylim(0.4, n + 0.6); ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xlabel("position in reasoning trace")
    # effort-shade legend (neutral grey, light -> dark = low -> high)
    handles = [Line2D([0], [0], marker="o", ls="", ms=6, color=_shade("#333333", f), label=lv)
               for lv, dy, f in LV]
    ax.legend(handles=handles, title="reasoning effort", loc="upper right", frameon=False,
              fontsize=8.5, title_fontsize=8.5, handletextpad=0.3, labelspacing=0.3)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"move_positions_by_effort.{ext}")
        fig.savefig(FIG_DIR / f"move_positions_by_effort.{ext}")
    print(f"saved: {FIG_DIR/'move_positions_by_effort.pdf'} (+ .png)")


def main() -> None:
    traces = [json.load(open(f)) for f in glob.glob(STORIES)]
    traces = [r for r in traces if (r.get("reasoning_trace") or "").strip()]

    rows = compute_rows(traces)
    rows.sort(key=lambda d: d["med"])  # earliest at top

    fig, ax = plt.subplots(figsize=(4.0, 4.3))
    n = len(rows)
    ys = list(range(n, 0, -1))  # top row highest y

    # handoff band between the last surprise move and the first coherence move
    last_s = max((r["med"] for r in rows if r["side"] == "S"), default=0.33)
    first_c = min((r["med"] for r in rows if r["side"] == "C"), default=0.46)
    cut = (last_s + first_c) / 2
    ax.axvspan(last_s, first_c, color="0.92", zorder=0)
    ax.axvline(cut, ls="--", lw=0.9, color="0.55", zorder=1)

    for y, r in zip(ys, rows):
        c = COLOR[r["side"]]
        ax.plot([r["q1"], r["q3"]], [y, y], "-", color=c, lw=4, alpha=0.35,
                solid_capstyle="round", zorder=2)
        ax.plot(r["med"], y, "o", color=c, ms=7, zorder=3)
        ax.text(1.005, y, f"{r['ntr']}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7.5, color="0.5")

    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows])
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(COLOR[r["side"]] if r["side"] != "N" else "0.35")
    ax.set_ylim(0.4, n + 0.6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("position in reasoning trace")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # phase annotations
    ax.text(last_s - 0.02, n + 0.45, "fix the twist", ha="right", va="bottom",
            fontsize=9.5, style="italic", color=WARM)
    ax.text(first_c + 0.02, n + 0.45, "retrofit the plot", ha="left", va="bottom",
            fontsize=9.5, style="italic", color=COOL)

    handles = [Line2D([0], [0], marker="o", ls="", color=WARM, label="surprise / divergent"),
               Line2D([0], [0], marker="o", ls="", color=COOL, label="coherence / convergent"),
               Line2D([0], [0], marker="o", ls="", color=GREY, label="framing")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5,
              handletextpad=0.3, borderaxespad=0.4)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"move_positions.{ext}")
    fig.savefig(FIG_DIR / "move_positions.pdf")
    # also dump the per-move stats so the examples table can show median/IQR/n
    import json as _json
    (OUT_DIR / "move_positions_stats.json").write_text(_json.dumps(rows, indent=1))
    print(f"saved: {OUT_DIR/'move_positions.pdf'} and {FIG_DIR/'move_positions.pdf'}")
    print(f"rows (top->bottom): " + " | ".join(f"{r['label']}={r['med']:.2f}" for r in rows))

    # by-effort forest plot, now built from the LLM SPAN positions (move_spans cache)
    figure_by_effort()


if __name__ == "__main__":
    main()
