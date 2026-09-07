"""Camera-ready model-profile grid for the top-N Kombine models.

A 3-row grid, 2 columns per task (6 columns total):
  Row 1: a per-task RANKING of the models by composite score (z-composite shown in parentheses),
         spanning that task's two columns.
  Row 2: two dimension charts per task (utility, surprise), directly beneath the task.
  Row 3: up to two dimension charts per task (originality, and emergent where it exists).
Dimension bars use RAW scores (all non-negative), with a shared y-scale per dimension across tasks.
Model colors come from the batlow scientific colormap. Font is Nimbus Roman to match the paper.

    python src/kg_creat/scripts/plot_profiles.py data/kg_creat/kombine_v2/scores/composite.json \\
        papers/kg_creat-iclr/media/profiles_grid
"""

import argparse
import colorsys
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Affine2D
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from svgpath2mpl import parse_path

# Provider maps and display names live in plot_radar and are imported, not copied: this file used to
# carry its own stale copies, so new providers were silently drawn grey and unlabelled.
from src.kg_creat.scripts.plot_radar import BRAND, DISPLAY, LOGO_SLUG, _provider  # noqa: E402

LOGO_DIR = Path(__file__).resolve().parents[3] / "assets" / "logos"


def _shades(base_hex, k):
    """k lightness-varied shades of a brand hue, darkest first (for the higher-ranked model)."""
    r, g, b = to_rgb(base_hex)
    h, lgt, s = colorsys.rgb_to_hls(r, g, b)
    offs = [0.0] if k == 1 else np.linspace(-0.14, 0.16, k)
    return [colorsys.hls_to_rgb(h, min(0.88, max(0.22, lgt + o)), s) for o in offs]


def _rasterize(path, color="#333333", px=300):
    """Render a normalized (unit, origin-centered) Path to a transparent RGBA image. Rasterizing
    sidesteps the PDF backend's mishandling of compound SVG paths inside offset boxes."""
    fig = Figure(figsize=(1, 1), dpi=px)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.55, 0.55)
    ax.add_patch(PathPatch(path, fc=color, ec="none"))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _load_logos():
    """Each provider's single-path SVG -> a transparent RGBA logo image, normalized to a unit box."""
    out = {}
    for prov, slug in LOGO_SLUG.items():
        f = LOGO_DIR / f"{slug}.svg"
        if not f.exists():
            continue
        d = re.search(r'\sd="([^"]+)"', f.read_text()).group(1)
        p = parse_path(d)
        # normalize by the flattened glyph extent (curve points, not bezier control points)
        polys = p.to_polygons()
        verts = np.concatenate(polys) if polys else p.vertices
        (x0, y0), (x1, y1) = verts.min(0), verts.max(0)
        s = 1.0 / max(x1 - x0, y1 - y0)
        t = Affine2D().translate(-(x0 + x1) / 2, -(y0 + y1) / 2).scale(s, -s)
        out[prov] = _rasterize(p.transformed(t))
    return out

# (task label, internal key, [row2 dims], [row3 dims])
TASKS = [
    ("Association", "association", ["utility", "surprise"], ["originality"]),
    ("Analogy", "analogy", ["utility", "surprise"], ["originality", "emergent"]),
    ("Blending", "blending", ["utility", "surprise"], ["originality", "emergent"]),
]
DIM_LABEL = {"utility": "Utility", "surprise": "Surprise", "originality": "Originality",
             "emergent": "Emergent"}


def _bars(ax, vals, colors, logos, title, show_y):
    x = np.arange(len(vals))
    ax.bar(x, vals, 0.72, color=colors, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([])
    ax.grid(axis="y", color="#DDDDDD", lw=0.6, alpha=0.8, zorder=0)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    if show_y:
        ax.set_yticklabels(["0", "25", "50", "75", "100"])
        ax.tick_params(axis="y", labelsize=9, color="#888888")
        ax.set_ylabel("% of max", fontsize=11)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=15, pad=8)
    # provider logo beneath each bar (in place of x tick labels)
    for i, img in enumerate(logos):
        if img is None:
            continue
        ab = AnnotationBbox(OffsetImage(img, zoom=0.062), ((i + 0.5) / len(vals), -0.03),
                            xycoords="axes fraction", frameon=False,
                            box_alignment=(0.5, 1.0), pad=0, annotation_clip=False)
        ax.add_artist(ab)


def main(composite_path, out_stem, top_n):
    plt.rcParams.update({"font.family": "Nimbus Roman", "mathtext.fontset": "custom",
                         "mathtext.rm": "Nimbus Roman", "axes.linewidth": 0.8, "font.size": 12})
    c = json.loads(Path(composite_path).read_text())
    models, top = c["ranking"], c["ranking"][:top_n]
    # color bars by provider brand hue; models sharing a provider get distinct shades.
    prov_models = {}
    for m in top:
        prov_models.setdefault(_provider(m), []).append(m)
    cmap = {}
    for prov, ms in prov_models.items():
        for m, col in zip(ms, _shades(BRAND.get(prov, "#777777"), len(ms))):
            cmap[m] = col

    # "emergent" is a PANEL, not a composite dimension: the composite stores the emergent block as
    # em_originality / em_utility / em_integration, so the panel is their mean. Reading it as a plain
    # key silently returned None for every model and drew an empty panel.
    EMERGENT_PARTS = ("em_originality", "em_utility", "em_integration")

    def _dim(m, task, d):
        block = c["per_model"][m]["raw"].get(task, {})
        if d != "emergent":
            return block.get(d)
        vals = [block.get(k) for k in EMERGENT_PARTS]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return sum(vals) / len(vals) if vals else None

    raw = {(t[1], d): {m: _dim(m, t[1], d) for m in models}
           for t in TASKS for d in t[2] + t[3]}
    empty = [k for k, v in raw.items() if all(x is None for x in v.values())]
    if empty:   # an empty panel is a silent lie about the data, so refuse to draw one
        raise ValueError(f"FATAL: no model has a value for {empty} -- the panel would be blank")
    comp = {t[1]: {m: c["per_model"][m]["per_task"][t[1]] for m in models} for t in TASKS}
    # percentile of each model's raw score among ALL evaluated models, per (task, dimension)
    def pctl(v, arr):
        return 100.0 * (sum(a < v for a in arr) + 0.5 * sum(a == v for a in arr)) / len(arr)
    pct = {}
    for (key, d), vals in raw.items():
        allv = [x for x in vals.values() if x is not None]
        pct[(key, d)] = {m: (pctl(vals[m], allv) if vals[m] is not None else None) for m in models}

    logos = _load_logos()
    fig = plt.figure(figsize=(15.0, 7.4))
    subfigs = fig.subfigures(1, 3, wspace=0.04)
    for ti, (label, key, r2, r3) in enumerate(TASKS):
        sf = subfigs[ti]
        letter = "abc"[ti]
        sf.suptitle(f"({letter}) {label}", fontsize=26, y=1.0)
        # 4 sub-columns: each dim panel spans 2; a lone bottom panel spans the middle two (centered).
        gs = sf.add_gridspec(2, 4, hspace=0.5, wspace=0.85)
        for row, dims in ((0, r2), (1, r3)):
            for j, d in enumerate(dims):
                cs = slice(2 * j, 2 * j + 2) if len(dims) == 2 else slice(1, 3)
                n = (0 if row == 0 else len(r2)) + j + 1
                ax = sf.add_subplot(gs[row, cs])
                show_y = (ti == 0 and j == 0)  # leftmost panel of each row (shared 0-100 scale)
                pairs = sorted(((100.0 * raw[(key, d)][m] if raw[(key, d)][m] is not None else 0.0, m) for m in top), key=lambda t: -t[0])
                _bars(ax, [p[0] for p in pairs], [cmap[m] for _, m in pairs],
                      [logos.get(_provider(m)) for _, m in pairs],
                      f"({letter}.{n}) {DIM_LABEL[d]}", show_y)

    handles = [Patch(color=cmap[m], label=DISPLAY.get(m, m)) for m in top]
    fig.legend(handles=handles, loc="lower center", ncol=len(top), frameon=False, fontsize=15,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.6, handlelength=1.2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", bbox_inches="tight", dpi=300)
    print(f"Wrote {out_stem}.pdf / .png  (top {top_n}: {', '.join(DISPLAY.get(m, m) for m in top)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("composite_path")
    ap.add_argument("out_stem")
    ap.add_argument("--top_n", type=int, default=5)
    a = ap.parse_args()
    main(a.composite_path, a.out_stem, a.top_n)
