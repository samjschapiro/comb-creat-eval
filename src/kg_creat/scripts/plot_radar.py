"""Camera-ready radar (model-profile) plots for the top-N Kombine models -- ONE radar per task.

Each task gets its own radar whose axes are that task's scoring dimensions (association: utility,
surprise, originality; analogy and blending add the emergent-creativity dimensions -- emergent
integration and emergent utility). A model's value on each axis is its gated score for that task as a
% of its maximum (stationary; pool-independent). All three radars share the 0-100 radial scale.
Lines are coloured by provider brand (distinct shades within a provider); the legend carries provider
logos. Font is Nimbus Roman to match the paper.

    python src/kg_creat/scripts/plot_radar.py data/kg_creat/kombine_test30/scores/composite.json \\
        papers/kg_creat-iclr/media/radar_profiles
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
from matplotlib.colors import to_rgb
from matplotlib.patches import PathPatch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.transforms import Affine2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from svgpath2mpl import parse_path

LOGO_DIR = Path(__file__).resolve().parents[3] / "assets" / "logos"
# names/brands/provider mapping live in one place now -- see src/kg_creat/model_names.py
from src.kg_creat.model_names import LOGO_SLUG, BRAND, DISPLAY, _provider  # noqa: E402,F401
# slug -> logo file stem; providers without an SVG (deepseek, z-ai) are still listed here so provider
# DETECTION works (they simply get no logo). Order also drives the landscape legend order.

# (task label, internal key, dimensions on that task's radar). Emergent creativity is split into its
# separate dimensions -- emergent utility and integration quality (paper 05_benchmark.tex).
TASKS = [
    ("Association", "association", ["utility", "surprise", "originality"]),
    ("Analogy", "analogy", ["utility", "surprise", "originality", "em_integration", "em_utility"]),
    ("Blending", "blending", ["utility", "surprise", "originality", "em_integration", "em_utility"]),
]
DIM_LABEL = {"utility": "Utility", "surprise": "Surprise", "originality": "Originality",
             "em_utility": "Emergent\nutility", "em_integration": "Emergent\nintegration"}


def _shades(base_hex, k):
    """k lightness-varied shades of a brand hue, darkest first (for the higher-ranked model)."""
    r, g, b = to_rgb(base_hex)
    h, lgt, s = colorsys.rgb_to_hls(r, g, b)
    offs = [0.0] if k == 1 else np.linspace(-0.10, 0.18, k)
    return [colorsys.hls_to_rgb(h, min(0.86, max(0.20, lgt + o)), s) for o in offs]


def _rasterize(path, color="#333333", px=300):
    fig = Figure(figsize=(1, 1), dpi=px)
    FigureCanvasAgg(fig)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off(); ax.patch.set_alpha(0)
    ax.set_xlim(-0.55, 0.55); ax.set_ylim(-0.55, 0.55)
    ax.add_patch(PathPatch(path, fc=color, ec="none"))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _load_logos():
    out = {}
    for prov, slug in LOGO_SLUG.items():
        f = LOGO_DIR / f"{slug}.svg"
        if not f.exists():
            continue
        d = re.search(r'\sd="([^"]+)"', f.read_text()).group(1)
        p = parse_path(d)
        polys = p.to_polygons()
        verts = np.concatenate(polys) if polys else p.vertices
        (x0, y0), (x1, y1) = verts.min(0), verts.max(0)
        s = 1.0 / max(x1 - x0, y1 - y0)
        t = Affine2D().translate(-(x0 + x1) / 2, -(y0 + y1) / 2).scale(s, -s)
        out[prov] = _rasterize(p.transformed(t))
    return out


def main(composite_path, out_stem, top_n):
    plt.rcParams.update({"font.family": "Nimbus Roman", "mathtext.fontset": "custom",
                         "mathtext.rm": "Nimbus Roman", "axes.linewidth": 0.8, "font.size": 13})
    c = json.loads(Path(composite_path).read_text())
    models, top = c["ranking"], c["ranking"][:top_n]

    # brand colour per model: group the plotted models by provider, shade within provider (darkest =
    # higher rank).
    prov_models = {}
    for m in top:
        prov_models.setdefault(_provider(m), []).append(m)
    cmap = {}
    for prov, ms in prov_models.items():
        for m, col in zip(ms, _shades(BRAND.get(prov, "#777777"), len(ms))):
            cmap[m] = col
    logos = _load_logos()

    panels = []
    for label, key, dims in TASKS:
        raw = {m: c["per_model"][m]["raw"].get(key, {}) for m in models}
        pct = {d: np.array([100.0 * raw[m].get(d, np.nan) for m in models], float) for d in dims}
        panels.append((label, dims, pct))
    ymin, ymax, yticks = 0.0, 100.0, [0, 25, 50, 75, 100]

    fig, axes = plt.subplots(1, len(TASKS), figsize=(14.5, 5.6), subplot_kw=dict(polar=True))
    for ax, (label, dims, pct) in zip(axes, panels):
        N = len(dims)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(ymin, ymax)
        for i, m in enumerate(top):
            v = [pct[d][models.index(m)] for d in dims]
            v += v[:1]
            ax.plot(angles, v, color=cmap[m], lw=2.2, zorder=3)
            ax.fill(angles, v, color=cmap[m], alpha=0.07, zorder=1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([DIM_LABEL[d] for d in dims], fontsize=13)
        ax.tick_params(axis="x", pad=8)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{t}" for t in yticks], fontsize=9, color="#888888")
        ax.grid(color="#BBBBBB", lw=0.6, alpha=0.8)
        ax.spines["polar"].set_color("#BBBBBB")
        ax.set_title(label, fontsize=30, pad=24)

    # branded legend: [logo] [model name in brand colour], evenly spaced across the bottom.
    lax = fig.add_axes([0.03, -0.02, 0.94, 0.1]); lax.axis("off")
    lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    for i, m in enumerate(top):
        xc = (i + 0.5) / len(top)
        img = logos.get(_provider(m))
        name = DISPLAY.get(m, m)
        tx = xc - 0.008 * len(name) / 2       # left edge of the (centered) name
        if img is not None:
            lax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.052), (tx - 0.022, 0.5),
                                          frameon=False, box_alignment=(0.5, 0.5)))
        lax.text(tx, 0.5, name, color=cmap[m], fontsize=16, va="center", ha="left", fontweight="bold")

    fig.tight_layout(rect=(0, 0.08, 1, 1))
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
