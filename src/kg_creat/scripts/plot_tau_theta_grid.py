"""Joint density of tau-inventive multiples over (tau, theta).

tau-inventive multiples have two knobs: tau (how many properties must be re-used) and theta (the
cosine at which two "relation object" texts count as the same property). Rather than pick a cell and
report a rate, this shows the DENSITY of the underlying evidence over the whole plane.

Each co-response pair of inventions is matched greedily and one-to-one with NO threshold, giving its
matched cosines sorted descending, s_1 >= s_2 >= ... >= s_m. The pair then contributes one point at
(tau = i, theta = s_i) for each i: "this pair's i-th best shared property was matched at similarity
s_i". Because the matched sequence is descending, a pair is a tau-inventive multiple at threshold
theta exactly when its point at tau lies at or above theta -- so the mass ABOVE a horizontal line in
column tau is the multiple count at that (tau, theta), and the density is the object those rates are
integrals of.

  left    joint density over all co-response pairs
  middle  blending density / analogy density        (Finding #3b)
  right   same-provider density / cross-provider    (Finding #3c)

Ratio panels are per-subset normalised (each subset's density integrates to 1) so the ratio is a
relative risk, and are masked where either subset has too little mass for the bin to mean anything.

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
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, LogLocator

import src.kg_creat.scripts.analyze_inventive_multiples as M
from src.kg_creat.embed import get_embedder

TAU_MAX = 6                                  # inventions carry up to 7 properties, but no PAIR
                                             # ever shares 7, so the column would be empty
THETA_EDGES = np.arange(0.20, 1.0001, 0.025)  # cosine bins
MIN_MASS = 30                                # bins with fewer raw pairs than this are not shown as a ratio

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.linewidth": 0.9, "axes.edgecolor": "#333333",
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.labelcolor": "#111111", "text.color": "#111111",
    "figure.dpi": 200,
})


def matched_cosines(A, B):
    """The greedy one-to-one matching of two inventions' properties, UNTHRESHOLDED, sorted descending.

    Same assignment rule as analyze_inventive_multiples.shared_properties, with the threshold removed
    so the whole sequence is available: shared(theta) is then just how many entries are >= theta.
    """
    if not len(A) or not len(B):
        return []
    Mx = A @ B.T
    used, out = set(), []
    for ai in np.argsort(-Mx.max(axis=1)):
        cand = [(Mx[ai, j], j) for j in range(Mx.shape[1]) if j not in used]
        if not cand:
            break
        s, bj = max(cand)
        out.append(float(s))
        used.add(bj)
    return sorted(out, reverse=True)


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

    taus, thetas, task, same = [], [], [], []
    n_pairs = 0
    for (t, _, _), idx in groups.items():
        for a, b in itertools.combinations(idx, 2):
            n_pairs += 1
            sp = _provider_same(mo, a, b)
            for i, s in enumerate(matched_cosines(SMAT[a], SMAT[b]), start=1):
                if i > TAU_MAX:
                    break
                taus.append(i); thetas.append(s); task.append(t); same.append(sp)
    return (np.array(taus), np.array(thetas), np.array(task), np.array(same), n_pairs)


def _provider_same(mo, a, b):
    return M._provider(mo[a]) == M._provider(mo[b])


def hist(taus, thetas, sel=None):
    """Counts on the (tau, theta) grid; tau is discrete so it gets one column per integer."""
    t, th = (taus, thetas) if sel is None else (taus[sel], thetas[sel])
    H, _, _ = np.histogram2d(t, th, bins=[np.arange(0.5, TAU_MAX + 1.5), THETA_EDGES])
    return H.T  # theta on rows, tau on columns


def _axes_common(ax, title):
    ax.set_xticks(range(TAU_MAX), [str(t) for t in range(1, TAU_MAX + 1)])
    ax.set_xlabel(r"$\tau$  (shared-property depth)", fontsize=17)
    ax.set_title(title, fontsize=18, pad=10)
    yt = [i for i, e in enumerate(THETA_EDGES[:-1]) if round(e, 3) in (0.20, 0.35, 0.50, 0.65, 0.80, 0.95)]
    ax.set_yticks(yt, [f"{THETA_EDGES[i]:.2f}" for i in yt])
    ax.tick_params(labelsize=14)
    # the reported theta, as a reference line rather than a boxed cell
    y058 = float(np.searchsorted(THETA_EDGES, 0.58) - 1)
    ax.axhline(y058, color="#111111", lw=1.2, ls="--", alpha=0.85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    taus, thetas, task, same, n_pairs = build()
    H_all = hist(taus, thetas)
    H_bl, H_an = hist(taus, thetas, task == "blending"), hist(taus, thetas, task == "analogy")
    H_sp, H_cp = hist(taus, thetas, same), hist(taus, thetas, ~same)

    def rel(Hx, Hy):
        """log2 relative density, masked where either side has too little raw mass to be read."""
        px, py = Hx / max(Hx.sum(), 1), Hy / max(Hy.sum(), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            R = np.log2(px / py)
        return np.ma.masked_where((Hx + Hy < MIN_MASS) | ~np.isfinite(R), R)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))

    cmap0 = plt.get_cmap("magma_r").copy(); cmap0.set_bad("#F2F2F2")
    Hm = np.ma.masked_where(H_all <= 0, H_all)
    im0 = axes[0].pcolormesh(np.arange(TAU_MAX + 1) - 0.5, np.arange(len(THETA_EDGES)) - 0.5, Hm,
                             cmap=cmap0, norm=LogNorm(vmin=1, vmax=Hm.max()), shading="auto")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cb0.set_label("pairs per bin (log)", fontsize=14)
    cb0.ax.tick_params(labelsize=13)
    cb0.ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0), numticks=10))
    cb0.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    lim = 3.0
    for ax, R, title, cmap in ((axes[1], rel(H_bl, H_an), "Blending / analogy", "PuOr_r"),
                               (axes[2], rel(H_sp, H_cp), "Same-provider / cross-provider", "BrBG_r")):
        cm = plt.get_cmap(cmap).copy(); cm.set_bad("#F2F2F2")
        im = ax.pcolormesh(np.arange(TAU_MAX + 1) - 0.5, np.arange(len(THETA_EDGES)) - 0.5, R,
                           cmap=cm, norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), shading="auto")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("relative density", fontsize=14)
        cb.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cb.ax.set_yticklabels([r"$\frac{1}{8}\times$", r"$\frac{1}{4}\times$", r"$\frac{1}{2}\times$",
                               r"$1\times$", r"$2\times$", r"$4\times$", r"$8\times$"], fontsize=14)

    for ax, t in zip(axes, ("(a) Joint density of shared properties",
                            "(b) Blending / analogy", "(c) Same-provider / cross-provider")):
        _axes_common(ax, t)
    axes[0].set_ylabel(r"$\theta$  (cosine of the matched property)", fontsize=17)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_tau_theta_density.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (out.parent / "tau_theta_density.json").write_text(json.dumps(
        {"theta_edges": THETA_EDGES.tolist(), "tau_max": TAU_MAX, "n_pairs": n_pairs,
         "counts_all": H_all.tolist(), "counts_blending": H_bl.tolist(),
         "counts_analogy": H_an.tolist(), "counts_same_provider": H_sp.tolist(),
         "counts_cross_provider": H_cp.tolist()}, indent=1))
    print(f"wrote {out}/fig_tau_theta_density.{{pdf,png}}")
    print(f"{n_pairs:,} pairs -> {len(taus):,} matched-property points; "
          f"max bin {int(H_all.max()):,}")


if __name__ == "__main__":
    main()
