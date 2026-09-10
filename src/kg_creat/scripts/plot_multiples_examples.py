"""Inventive multiples as the reader should meet them: two inventions side by side, matched properties
joined, everything else greyed.

Each panel is one tau-multiple from the `multiples` block of inventive_multiples.json: the anchors,
the two models (logo + coined name), then the properties. Matched properties sit on the same row
and are joined by a line carrying their cosine; unmatched properties follow below in grey, so the
reader sees both what the two models agreed on and how much they did not. The coined names are
shown because they are the point of Finding #3d: often different for the same invention.

Panels are chosen by an explicit list (below), not sampled, so the caption must say so. They are
picked as tau = 3 pairs on four different items -- three cross-provider, one same-family -- with
different coined names throughout.

    .venv/bin/python -m src.kg_creat.scripts.plot_multiples_examples
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from src.kg_creat.scripts.plot_radar import BRAND, DISPLAY, _load_logos, _provider

SRC = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")

# (task, u, v, model a, model b): which multiples to show, in panel order. Each must be a tau-multiple
# in the JSON or the script fails.
PANELS = [   # all tau = 3 (the deepest agreement in the benchmark; no analogy pair reaches it), four items
    ("blending", "Opera", "Documentary film", "anthropic_claude-fable-5-1", "google_gemini-3-7-flash"),
    ("blending", "The immune system", "Black holes", "anthropic_claude-opus-5", "google_gemini-3-7-flash"),
    ("blending", "Photosynthesis", "Bread", "anthropic_claude-fable-5", "openai_gpt-5-6-sol"),
    ("blending", "The Roman Empire", "Crystals", "anthropic_claude-opus-4-5", "anthropic_claude-opus-4-6"),
]
WRAP = 30           # characters per line inside a property box
COLS = 2            # panels per row; 2 x 2 keeps the type legible at text width
MATCH_COL = "#2F6B8E"
GREY = "#9AA3AD"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 14})


def _disp(model_key):
    return DISPLAY.get(model_key, model_key.split("_", 1)[-1])


def find(mults, task, u, v, ma, mb):
    for m in mults:
        if (m["task"], m["u"], m["v"]) != (task, u, v):
            continue
        ms = {m["a"]["model"], m["b"]["model"]}
        if ms == {ma, mb}:
            if m["a"]["model"] != ma:                  # present in the order the panel list gives
                m = dict(m, a=m["b"], b=m["a"], matches=[{"a": x["b"], "b": x["a"], "cos": x["cos"]} for x in m["matches"]])
            return m
    raise SystemExit(f"FATAL: ({task}, {u}, {v}) {ma} x {mb} is not a tau-multiple in {SRC}")


def draw_panel(ax, m, letter, logos):
    a, b = m["a"], m["b"]
    matched_a = {x["a"]: x for x in m["matches"]}
    matched_b = {x["b"] for x in m["matches"]}
    rows = [(x["a"], x["b"], x["cos"]) for x in sorted(m["matches"], key=lambda x: -x["cos"])]
    rest_a = [i for i in range(len(a["properties"])) if i not in matched_a]
    rest_b = [j for j in range(len(b["properties"])) if j not in matched_b]
    n_rows = len(rows) + max(len(rest_a), len(rest_b))
    xa, xb = 0.24, 0.76                                    # column centres (axes fraction)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # the operator is the notation: u + v for a blend, u :: v for an analogy
    op = "+" if m["task"] == "blending" else "::"
    ax.set_title(f"({letter}) {m['u']} {op} {m['v']}", fontsize=20, loc="left", pad=8)

    # header: logo, model, coined name
    y_head = 0.93
    for x, inv in ((xa, a), (xb, b)):
        prov = _provider(inv["model"])
        img = logos.get(prov)
        x0 = x - 0.21                                      # left edge of the header block
        if img is not None:
            ax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.045), (x0 + 0.02, y_head + 0.03), frameon=False,
                                         box_alignment=(0.5, 0.5)))
        ax.text(x0 + 0.055, y_head + 0.03, _disp(inv["model"]), fontsize=13, color="#555555", va="center", ha="left")
        ax.text(x0, y_head - 0.02, f"“{inv['name']}”", fontsize=16.5, fontweight="bold",
                color=BRAND.get(prov, "#333333"), va="top", ha="left")
    ax.plot([0.02, 0.98], [y_head - 0.075, y_head - 0.075], color="#DDDDDD", lw=0.8)

    # property rows: a fixed pitch, so panels with fewer properties simply end higher
    y0, step = y_head - 0.13, 0.128
    def box(x, y, text, matched):
        t = "\n".join(textwrap.wrap(text, WRAP))
        ax.text(x, y, t, ha="center", va="center", fontsize=13.5,
                color="#111111" if matched else GREY,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#EAF1F7" if matched else "#F6F6F6",
                          edgecolor=MATCH_COL if matched else "#E0E0E0", linewidth=1.2 if matched else 0.8))
    y = y0
    for i, j, c in rows:
        box(xa, y, a["properties"][i], True); box(xb, y, b["properties"][j], True)
        ax.plot([xa + 0.17, xb - 0.17], [y, y], color=MATCH_COL, lw=1.6, zorder=0)
        ax.text(0.5, y + 0.012, f"{c:.2f}", ha="center", va="bottom", fontsize=11, color=MATCH_COL)
        y -= step
    for k in range(max(len(rest_a), len(rest_b))):
        if k < len(rest_a):
            box(xa, y, a["properties"][rest_a[k]], False)
        if k < len(rest_b):
            box(xb, y, b["properties"][rest_b[k]], False)
        y -= step


def main():
    d = json.loads(SRC.read_text())
    if "multiples" not in d:
        raise SystemExit(f"{SRC} has no `multiples` block -- re-run analyze_inventive_multiples.py")
    logos = _load_logos()
    picks = [find(d["multiples"], *spec) for spec in PANELS]
    n = len(picks)
    rows_max = max(len(m["matches"]) + max(len(m["a"]["properties"]) - len(m["matches"]),
                                          len(m["b"]["properties"]) - len(m["matches"])) for m in picks)
    nrow = -(-n // COLS)
    fig, axes = plt.subplots(nrow, COLS, figsize=(6.6 * COLS, (1.1 + 0.86 * rows_max) * nrow))
    axes = axes.ravel()
    for ax, m, letter in zip(axes, picks, "abcdefgh"):
        draw_panel(ax, m, letter, logos)
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_multiples_examples.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig_multiples_examples -> {OUT}  ({n} panels)")


if __name__ == "__main__":
    main()
