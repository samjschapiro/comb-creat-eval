"""Transformational-creativity RADAR small-multiples (alternative to the scorecard).

Each panel is a 4-axis radar (Surprise, Coherence, Diversity, Realism), z-scored
across the full pool so the pool average is the 0-ring. The expert-human profile is
overlaid faintly on every panel as a reference, so a model's failing facet shows up
as a vertex pulled *inward* from the human silhouette.

The 2x3 grid is organised by the two failure modes from Sec. 4 (one per row):
  row 1  Under-diversification        humans (ref) | Claude-Opus-4.6 | Claude-Opus-4.5
  row 2  Breaking the world model     Gemma-4-31B | DeepSeek-V3.2-Exp | Qwen3-Next-80B

Colours match the scorecard (batlowS per provider; Other = grey; humans = black).

Usage:
    .venv/bin/python src/plot_twist/scripts/make_tc_radar.py \
        data/plot_twist/tc/tc.json papers/pt2cb-iclr-2027/figures
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ---- facets (clockwise from top) -------------------------------------------------
FACETS = [("mean_surprise", "Surprise"),
          ("mean_coherence", "Coherence"),
          ("div", "Diversity"),
          ("mean_realism", "Realistic")]

# ---- panels: (source, display name); None = expert-human reference panel ----------
ROWS = [
    ("Mode collapse (high quality, low diversity)",
     [("human", "Expert humans"),
      ("anthropic/claude-opus-4.6", "Claude-Opus-4.6"),
      ("anthropic/claude-opus-4.5", "Claude-Opus-4.5")]),
    ("Breaking the world model (surprising & coherent; but unrealistic)",
     [("google/gemma-4-31b-it", "Gemma-4-31B"),
      ("deepseek/deepseek-v3.2-exp", "DeepSeek-V3.2-Exp"),
      ("qwen/qwen3-next-80b-a3b-instruct", "Qwen3-Next-80B")]),
]

HUMAN_COLOR = "#000000"
OTHER_COLOR = "#8c8c8c"
OTHER_PROVIDERS = {"minimax", "morph", "z-ai", "deepcogito",
                   "nousresearch", "tencent", "ai21", "baidu"}


def provider_colors(rows):
    """Replicate the scorecard's per-provider batlowS palette exactly."""
    from cmcrameri import cm as _cmc

    def _adj(c):
        r, g, b = c[0], c[1], c[2]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        if lum > 0.74:
            s = 0.62 / lum
            r, g, b = r * s, g * s, b * s
        return (r, g, b, 1.0)

    provs = sorted({d["source"].split("/")[0] for d in rows if d["source"] != "human"})
    return {p: (OTHER_COLOR if p in OTHER_PROVIDERS else _adj(_cmc.batlowS(i)))
            for i, p in enumerate(provs)}


def color_of(src, pcol):
    return HUMAN_COLOR if src == "human" else pcol[src.split("/")[0]]


def main(tc_json: str, out_dir: str) -> None:
    rows = json.loads(Path(tc_json).read_text())
    by_src = {d["source"]: d for d in rows}
    pcol = provider_colors(rows)

    # Per-facet PERCENTILE RANK across the full 72-model pool: radius = % of the pool
    # scoring below this value on that facet. Rings are pool quartiles (25/50/75); the
    # 50-ring is the pool median. Chosen for maximum visual separation of the systems.
    pool = {key: np.array([d[key] for d in rows], dtype=float) for key, _ in FACETS}
    # Overall pctl (panel titles) = percentile of the gated headline composite, matching
    # the leaderboard; the spokes stay the RAW facets (incl. realism) as the breakdown.
    pool["overall_eq"] = np.array([d["overall_eq"] for d in rows], dtype=float)

    def pctrank(key, x):
        v = pool[key]
        return 100.0 * (np.sum(v < x) + 0.5 * np.sum(v == x)) / len(v)

    def zvec(src):  # name kept; now returns the per-facet percentile-rank vector
        d = by_src[src]
        return np.array([pctrank(key, d[key]) for key, _ in FACETS])

    human_z = zvec("human")

    # radial geometry
    n = len(FACETS)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang_closed = np.concatenate([angles, angles[:1]])
    # Radial range tracks the actual data (shown models span z in [-1.2, 1.9]); a
    # tight range avoids a large empty core that would crush near-rim differences
    # (e.g. the diversity gap between top models and the human reference).
    # Percentile radius (0-100); small buffer so vertices avoid the dead centre / rim.
    RMIN, RMAX = -4.0, 104.0
    RTICKS = [25, 50, 75]

    def to_r(z):  # clamp into the drawable band
        return np.clip(z, RMIN, RMAX)

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 13.6),
                             subplot_kw=dict(polar=True))
    fig.subplots_adjust(left=0.06, right=0.95, top=0.78, bottom=0.06,
                        hspace=0.90, wspace=1.40)

    for ri, (row_label, panels) in enumerate(ROWS):
        for ci, (src, disp) in enumerate(panels):
            ax = axes[ri, ci]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_ylim(RMIN, RMAX)
            ax.set_xticks(angles)
            ax.set_xticklabels([lab for _, lab in FACETS], fontsize=23)
            ax.tick_params(axis="x", pad=14)  # push facet labels clear of the rim/polygon
            # Push ONLY the left/right labels (Realistic, Coherence) further outward,
            # horizontally, so they clear the wide left/right polygon vertices. The
            # top/bottom labels (Surprise, Diversity) are left as-is.
            from matplotlib.transforms import ScaledTranslation
            _lbls = ax.get_xticklabels()  # order: Surprise, Coherence, Diversity, Realistic
            _lbls[1].set_transform(_lbls[1].get_transform()
                                   + ScaledTranslation(0.05, 0, fig.dpi_scale_trans))
            _lbls[3].set_transform(_lbls[3].get_transform()
                                   + ScaledTranslation(-0.05, 0, fig.dpi_scale_trans))
            ax.set_yticks(RTICKS)
            ax.set_yticklabels([f"{t}" for t in RTICKS], fontsize=18, color="#8f8f8f")
            ax.set_rlabel_position(45)  # tuck radial labels into the empty Surprise/Coherence gap
            for lbl in ax.get_yticklabels():  # white backing so they read over polygon edges
                lbl.set_bbox(dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.85))
            ax.grid(color="#d9d9d9", lw=0.9)
            ax.spines["polar"].set_color("#cccccc")
            # emphasise the pool-median ring (50th percentile)
            ax.plot(np.linspace(0, 2 * np.pi, 200), [50] * 200,
                    color="#b0b0b0", lw=1.2, ls=(0, (4, 3)), zorder=1)

            # human reference overlay (skip on the human panel itself)
            if src != "human":
                hr = np.concatenate([to_r(human_z), to_r(human_z[:1])])
                ax.plot(ang_closed, hr, color="#000000", lw=1.6,
                        ls=(0, (5, 3)), alpha=0.55, zorder=3)

            col = color_of(src, pcol)
            zr = np.concatenate([to_r(zvec(src)), to_r(zvec(src)[:1])])
            ax.fill(ang_closed, zr, color=col, alpha=0.22, zorder=4)
            ax.plot(ang_closed, zr, color=col, lw=4.0, zorder=5)
            ax.scatter(angles, to_r(zvec(src)), color=col, s=80, zorder=6,
                       edgecolors="white", linewidths=1.3)

            ov = pctrank("overall_eq", by_src[src]["overall_eq"])  # gated-composite percentile
            wt = "bold" if src == "human" else "normal"
            # The human panel is named by its own header (a); blank first line keeps
            # its radar vertically aligned with the two-line model titles beside it.
            t1 = "" if src == "human" else disp
            ax.set_title(f"{t1}\nOverall pctl $={ov:.0f}$", fontsize=31,
                         fontweight=wt, pad=14)

    # Group headers above each row, each on ONE line. The bold "(x) Title" and the UNBOLD
    # parenthetical are drawn as two adjacent runs on the same baseline. A single font size
    # -- the largest that fits every header on one line within its span -- keeps them uniform.
    fig.canvas.draw()
    _rend = fig.canvas.get_renderer()
    _fig_w_px = fig.get_size_inches()[0] * fig.dpi

    def _measure(s, fs, weight):
        t = fig.text(0.5, 0.5, s, fontsize=fs, fontweight=weight)
        w = t.get_window_extent(_rend).width / _fig_w_px
        t.remove()
        return w

    def _parts(letter, label):
        title, rest = (label.split(" (", 1) + [""])[:2]
        return f"({letter}) {title}", ("(" + rest if rest else "")

    # one (x_center, y, title, paren, avail) spec per header
    _specs = []
    for ri, (row_label, _panels) in enumerate(ROWS):
        p = [axes[ri, c].get_position() for c in range(3)]
        ytop = p[0].y1 + 0.135  # baseline; sits well above the two-line panel titles below
        if ri == 0:
            ta, pa = _parts("a", "Expert humans")
            _specs.append(((p[0].x0 + p[0].x1) / 2, ytop, ta, pa, p[1].x0 - p[0].x0))
            tb, pb = _parts("b", row_label)
            cb = (p[1].x0 + p[2].x1) / 2  # centre over cols 1-2; allow symmetric overhang
            avb = 2 * min(0.965 - cb, cb - (p[0].x1 + 0.01))
            _specs.append((cb, ytop, tb, pb, avb))
        else:
            tt, pp = _parts("cd"[ri - 1], row_label)
            _specs.append(((p[0].x0 + p[2].x1) / 2, ytop, tt, pp, p[2].x1 - p[0].x0))

    _GAP = 0.013  # gap between bold title and the (same-size) unbold parenthetical (fig fraction)

    # One uniform font size: bold title + same-size unbold parenthetical on ONE line.
    # (b) has no parenthetical, so the longest full-width row (c) sets the size.
    def _row_w(title, paren, fs):
        w = _measure(title, fs, "bold")
        return w + _GAP + _measure(paren, fs, "normal") if paren else w

    HEADER_FS = 40
    while HEADER_FS > 12 and not all(_row_w(t, pn, HEADER_FS) <= av * 0.97
                                     for _, _, t, pn, av in _specs):
        HEADER_FS -= 1

    for x, y, title, paren, _av in _specs:
        tw = _measure(title, HEADER_FS, "bold")
        pw = (_GAP + _measure(paren, HEADER_FS, "normal")) if paren else 0
        left = x - (tw + pw) / 2
        fig.text(left, y, title, ha="left", va="baseline", fontsize=HEADER_FS,
                 fontweight="bold", color="#222222")
        if paren:
            fig.text(left + tw + _GAP, y, paren, ha="left", va="baseline",
                     fontsize=HEADER_FS, fontweight="normal", color="#222222")

    # (No legend: the caption explains solid = system profile, dashed = expert-human
    # reference, dotted = pool-average 0-ring.)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 400), ("pdf", 400)):
        p = out / f"tc_radar.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
        print(f"saved: {p}")
    plt.close(fig)


if __name__ == "__main__":
    tc = sys.argv[1] if len(sys.argv) > 1 else "data/plot_twist/tc/tc.json"
    od = sys.argv[2] if len(sys.argv) > 2 else "papers/pt2cb-iclr-2027/figures"
    main(tc, od)
