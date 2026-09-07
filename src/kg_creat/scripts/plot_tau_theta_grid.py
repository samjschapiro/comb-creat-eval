"""How the two headline ratios of Findings #3 depend on BOTH free parameters.

tau-inventive multiples have exactly two knobs: tau (how many properties must be re-used) and theta
(the cosine at which two "relation object" texts count as the same property). tau is reported as a
curve; theta is a number with no recorded derivation. This plots both ratios over the full (tau,
theta) grid, so a reader can see whether the claims survive the choice or depend on it.

  left    overall tau-inventive multiple rate      (the headline of Finding #3a)
  middle  blending rate / analogy rate             (Finding #3b)
  right   same-provider rate / cross-provider rate (Finding #3c)

A cell is masked where its denominator rate is 0, since the ratio is then undefined rather than large.

    .venv_mlx/bin/python -m src.kg_creat.scripts.plot_tau_theta_grid
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator

import src.kg_creat.scripts.analyze_inventive_multiples as M
from src.kg_creat.embed import get_embedder

THETAS = [0.45, 0.48, 0.50, 0.53, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70]
TAUS = [1, 2, 3, 4]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.labelcolor": "#111111", "text.color": "#111111",
    "figure.dpi": 200,
})


def build():
    d = np.load(M.NPZ, allow_pickle=True)
    names, tk, us, vs, mo = d["names"], d["tasks"], d["u"], d["v"], d["models"]
    _, _, struct, _ = M.load_records()
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    slot_vec, SLOTS = {}, {}
    for i in range(len(names)):
        st = M.slot_texts(str(tk[i]), struct.get((str(tk[i]), str(us[i]), str(vs[i]), str(mo[i])), []))
        for t, _ in st:
            if t not in slot_vec:
                slot_vec[t] = un(np.asarray(embed(t), float))
        SLOTS[i] = st
    SMAT = {i: (np.vstack([slot_vec[t] for t, _ in SLOTS[i]]) if SLOTS[i] else np.zeros((0, 384)))
            for i in range(len(names))}
    groups = defaultdict(list)
    for i in range(len(names)):
        groups[(str(tk[i]), str(us[i]), str(vs[i]))].append(i)
    meta = [(task, a, b) for (task, _, _), idx in groups.items()
            for a, b in itertools.combinations(idx, 2)]
    task = np.array([t for t, _, _ in meta])
    same = np.array([M._provider(mo[a]) == M._provider(mo[b]) for _, a, b in meta])
    # shared count at every theta. The greedy matching depends on theta, so it is recomputed rather
    # than derived from one pass -- correctness over speed; the matrices are at most 7x7.
    shared = {th: np.array([M.shared_properties(SMAT[a], SMAT[b], tau=th) for _, a, b in meta])
              for th in THETAS}
    return task, same, shared, len(meta)


def grids(task, same, shared):
    """(overall rate %, blend/analogy, same/cross) as theta x tau matrices.
    Ratio cells are NaN where the denominator rate is 0 -- undefined, not large."""
    bl, an = task == "blending", task == "analogy"
    ov = np.full((len(THETAS), len(TAUS)), np.nan)
    ba = np.full((len(THETAS), len(TAUS)), np.nan)
    sc = np.full((len(THETAS), len(TAUS)), np.nan)
    for i, th in enumerate(THETAS):
        for j, tau in enumerate(TAUS):
            hit = shared[th] >= tau
            r = hit.mean()
            if r > 0:
                ov[i, j] = 100.0 * r
            r_bl, r_an = hit[bl].mean(), hit[an].mean()
            r_s, r_c = hit[same].mean(), hit[~same].mean()
            if r_an > 0:
                ba[i, j] = r_bl / r_an
            if r_c > 0:
                sc[i, j] = r_s / r_c
    return ov, ba, sc


def panel(ax, G, title, cmap, pct=False):
    finite = G[np.isfinite(G)]
    norm = LogNorm(vmin=max(finite.min(), 1e-3), vmax=finite.max())
    Gm = np.ma.masked_invalid(G)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad("#EDEDED")
    im = ax.imshow(Gm, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(TAUS)), [str(t) for t in TAUS])
    ax.set_yticks(range(len(THETAS)), [f"{t:.2f}" for t in THETAS])
    ax.set_xlabel(r"$\tau$  (shared properties required)")
    ax.set_title(title, fontsize=10.5, pad=8)
    for i in range(len(THETAS)):
        for j in range(len(TAUS)):
            v = G[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=7.5, color="#999999")
                continue
            # label colour follows cell luminance so it reads on both ends of the ramp
            rgba = cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt = (f"{v:.2f}" if v < 1 else f"{v:.1f}") + ("%" if pct else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.0,
                    color="white" if lum < 0.55 else "#111111")
    # mark the value the paper currently quotes
    if 0.58 in THETAS and 2 in TAUS:
        ax.add_patch(plt.Rectangle((TAUS.index(2) - 0.5, THETAS.index(0.58) - 0.5), 1, 1,
                                   fill=False, edgecolor="#111111", lw=1.8))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    task, same, shared, n = build()
    ov, ba, sc = grids(task, same, shared)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    im0 = panel(axes[0], ov, "Overall multiple rate", "Greens", pct=True)
    im1 = panel(axes[1], ba, "Blending / analogy", "PuBu")
    im2 = panel(axes[2], sc, "Same-provider / cross-provider", "YlOrBr")
    axes[0].set_ylabel(r"$\theta$  (property-match cosine)")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    for ax, im in ((axes[0], im0), (axes[1], im1), (axes[2], im2)):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7.5)
        # plain decimals, not 6x10^0 -- these are rates and ratios a reader reads off directly
        cb.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
        cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        # bind is_rate now: a lambda closing over `ax` would resolve it after the loop had finished,
        # and every colourbar would be formatted as the LAST panel's kind.
        cb.ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _, is_rate=(ax is axes[0]):
                (f"{v:g}%" if is_rate else f"{v:g}$\\times$") if v >= 0.01 else ""))
        cb.set_label("rate (log scale)" if ax is axes[0] else "ratio (log scale)", fontsize=8)
    fig.suptitle(r"Findings #3 over the full $(\tau, \theta)$ grid", fontsize=11.5, y=1.0)
    fig.text(0.5, -0.045,
             f"Left: the share of all {n:,} co-response model pairs that are tau-inventive multiples. "
             "Middle and right: ratios of that rate between subsets. "
             r"The boxed cell is the reported setting ($\tau=2$, $\theta=0.58$)." "\n"
             "Grey cells are undefined: the denominator rate is 0, not large. Colour is on a log scale.",
             ha="center", fontsize=7.8, color="#555555", linespacing=1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_tau_theta_ratios.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (out.parent / "tau_theta_grid.json").write_text(json.dumps(
        {"thetas": THETAS, "taus": TAUS, "n_pairs": n,
         "overall_pct": [[None if not np.isfinite(v) else v for v in r] for r in ov],
         "blend_over_analogy": [[None if not np.isfinite(v) else v for v in r] for r in ba],
         "same_over_cross": [[None if not np.isfinite(v) else v for v in r] for r in sc]}, indent=2))
    print(f"wrote {out}/fig_tau_theta_ratios.{{pdf,png}}")
    print(f"n pairs {n:,}   overall rate {np.nanmin(ov):.2f}%-{np.nanmax(ov):.2f}%   "
          f"blend/analogy {np.nanmin(ba):.1f}-{np.nanmax(ba):.1f}   "
          f"same/cross {np.nanmin(sc):.1f}-{np.nanmax(sc):.1f}")


if __name__ == "__main__":
    main()
