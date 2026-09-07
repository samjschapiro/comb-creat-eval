"""Model x property matrix for one inventive multiple: what the models agree on once the coined name
is set aside.

Columns are the models that answered the item (cluster members first, then the rest); rows are the
consensus (relation, object) slots from analyze_inventive_multiples.py, sorted by how many models
assert them. A filled cell means that model asserted that property of its invention -- asserted or not,
nothing further encoded. The coined names run along the top and the provider mark sits under each
column, so the figure shows in one frame that the NAMES diverge while some properties recur, that the
recurrence runs past the edge of the cluster, and that most of what a model says about its invention is
still its own (about a third of a member's triples land in any shared slot).

    .venv/bin/python -m src.kg_creat.scripts.plot_multiples_matrix
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Affine2D
from svgpath2mpl import parse_path

from src.kg_creat.scripts.plot_radar import (BRAND, DISPLAY, LOGO_DIR, LOGO_SLUG, _rasterize,
                                             _provider as _radar_prov)

SRC = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
# Democracy + Banking is the paper's worked example already, and Opera + Documentary film pulls all 21
# models (no contrast block). Hinduism + Gravity is the clearest remaining case: 9 of 21 models across
# 4 provider families, and the shared slots are unmistakably cross-domain.
ITEM = ("blending", "Hinduism", "Gravity")
N_SLOTS = 8
# The right-hand block is a SELECTION, not the full set: the models that answered the same item and
# share the least of the cluster's structure. It is there to show what the cluster is being contrasted
# against, so the caption has to say it is chosen, not sampled.
N_OUT = 6
FILL = "#3F6F8F"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "text.color": "#222222"})


def _disp(m):
    return DISPLAY.get(m, m.split("_", 1)[-1])


def brand_logos():
    """Provider marks rasterised in the provider's brand colour -- the same hues the invention
    landscape colours its markers with, so the two figures read as one system."""
    out = {}
    for prov, slug in LOGO_SLUG.items():
        f = LOGO_DIR / f"{slug}.svg"
        if not f.exists():
            continue
        d = re.search(r'\sd="([^"]+)"', f.read_text()).group(1)
        path = parse_path(d)
        polys = path.to_polygons()
        verts = np.concatenate(polys) if polys else path.vertices
        (x0, y0), (x1, y1) = verts.min(0), verts.max(0)
        s = 1.0 / max(x1 - x0, y1 - y0)
        t = Affine2D().translate(-(x0 + x1) / 2, -(y0 + y1) / 2).scale(s, -s)
        out[prov] = _rasterize(path.transformed(t), color=BRAND.get(prov, "#333333"))
    return out


def main():
    data = json.loads(SRC.read_text())
    task, u, v = ITEM
    cl = max((c for c in data["clusters"] if (c["task"], c["u"], c["v"]) == (task, u, v)),
             key=lambda c: c["size"])
    members = [m["model"] for m in cl["members"]]
    outs = [o["model"] for o in cl["outsiders"]]
    name = {m["model"]: m["name"] for m in cl["members"]}
    name.update({o["model"]: o["name"] for o in cl["outsiders"]})
    slots = cl["consensus"][:N_SLOTS]
    # inside each block, models that assert most of the shown slots come first: the eye should read
    # density, not model names. The outsider block keeps only the N_OUT least-overlapping models.
    depth = {m: sum(1 for r in slots if m in r["assertions"]) for m in members + outs}
    n_item, n_out_all = len(members) + len(outs), len(outs)
    outs = sorted(outs, key=lambda m: (depth[m], m))[:N_OUT]
    order = sorted(members, key=lambda m: (-depth[m], m)) + sorted(outs, key=lambda m: (-depth[m], m))
    n_c, n_m = len(order), len(slots)

    fig, ax = plt.subplots(figsize=(13.2, 0.66 * n_m + 1.9))
    ax.set_xlim(-0.5, n_c - 0.5); ax.set_ylim(n_m - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    for j in range(n_c):                                   # column ground: the cluster block is tinted
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, n_m, facecolor="#EEF2F7" if j < len(members)
                                   else "#F7F7F5", edgecolor="none", zorder=0))
    for i, r in enumerate(slots):
        for j, m in enumerate(order):
            if m in r["assertions"]:                       # asserted / not: no further encoding
                ax.add_patch(plt.Rectangle((j - 0.38, i - 0.38), 0.76, 0.76, facecolor=to_rgba(FILL, .9),
                                           edgecolor="white", linewidth=0.8, zorder=3))
            else:
                ax.add_patch(plt.Rectangle((j - 0.34, i - 0.34), 0.68, 0.68, facecolor="none",
                                           edgecolor="#D8DCE3", linewidth=0.7, zorder=2))
        ax.text(-0.85, i, r["gloss"], ha="right", va="center", fontsize=16)
        # the count is over EVERY model that answered the item, not just the columns drawn
        ax.text(n_c - 0.15, i, f'{r["models"]}/{n_item}', ha="left", va="center", fontsize=14,
                color="#5C6472", family="monospace")

    logos = brand_logos()                                  # provider mark under each column
    for j, m in enumerate(order):
        img = logos.get(_radar_prov(m))
        if img is not None:
            # y grows downwards here (the axis is inverted), so this sits a clear half-cell BELOW the
            # bottom row rather than crowding it
            ab = AnnotationBbox(OffsetImage(img, zoom=0.075, alpha=0.95), (j, n_m + 0.02),
                                frameon=False, box_alignment=(0.5, 0.5), annotation_clip=False)
            ab.set_clip_on(False); ab.set_zorder(6)        # it sits outside the axes; default clips it
            ax.add_artist(ab)

    # group brackets sit above the matrix; the coined names are deliberately absent -- the figure is
    # about which PROPERTIES recur, and eighteen rotated names crowded that out
    y = -0.95
    for x0, x1, col, lab, bold in (
            (-0.42, len(members) - 0.58, FILL, f"inventive multiple ×{len(members)}", "bold"),
            (len(members) - 0.42, n_c - 0.58, "#8A94A3",
             f"same anchors, another invention ({len(outs)} of {n_out_all})", "normal")):
        ax.plot([x0, x1], [y, y], color=col, lw=1.6, clip_on=False)
        for x in (x0, x1):
            ax.plot([x, x], [y, y + 0.16], color=col, lw=1.6, clip_on=False)
        ax.text((x0 + x1) / 2, y - 0.18, lab, ha="center", va="bottom", fontsize=15, color=col,
                fontweight=bold, clip_on=False)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_multiples_matrix.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved fig_multiples_matrix -> {OUT}  ({n_m} slots x {n_c} models)")


if __name__ == "__main__":
    main()
