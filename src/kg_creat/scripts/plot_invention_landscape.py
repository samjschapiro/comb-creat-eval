"""Invention landscape: for two chosen anchor pairs, scatter every model's invention in a similarity map of its
sentence embedding, coloured by provider, and draw the INVENTIVE MULTIPLES -- the clusters of models
that invented the same entity through the same abstraction -- as shaded regions labelled with the
shared invention and how many models reached it. Every marker keeps its true projected position: the
region outlines the members where they fall, and each member carries a ring in the cluster's colour,
which is what settles membership when a non-member projects inside the outline. Each panel carries one
dominant multiple beside a smaller competing abstraction, which is the hivemind effect at item level.

The clusters are NOT recomputed here -- they are read from the analysis output, so the figure and the
reported rates cannot disagree. Run analyze_inventive_multiples.py first.

    .venv/bin/python -m src.kg_creat.scripts.plot_invention_landscape
"""
import itertools
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from pathlib import Path
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull

from src.kg_creat.scripts.plot_radar import BRAND, _provider, LOGO_SLUG, _load_logos

MARK = {"blending": "s", "analogy": "o"}   # blend = square, analogy = circle
# one hue per cluster within a panel (task is already carried by the marker shape); two clusters of
# the same task often sit next to each other, so a per-task colour would merge them into one blob.
HULL = ["#3F6F8F", "#9A7D2E", "#7A5C9E", "#2F7D6E", "#A8476A", "#B4692F"]
N_SINGLETON = 3                            # unclustered inventions labelled per panel (context only)

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "text.color": "#222222", "axes.edgecolor": "#666666"})

NPZ = Path("data/kg_creat/kombine_test30/analysis/invention_vectors.npz")
CLUST = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")   # the multiples report
# (u, v) for the two panels, both tasks shown per panel. Labels must match the anchor strings in the
# response files exactly, since clusters are matched to a panel by (u, v).
PANELS = [("Hinduism", "Gravity"),
          ("The immune system", "Black holes")]
TITLE_CASE = {"The immune system": "The Immune System"}   # display only; matching uses the pool string
PROV_LABEL = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "x-ai": "xAI",
              "deepseek": "DeepSeek", "qwen": "Qwen", "z-ai": "Z-AI", "meta": "Meta",
              "microsoft": "Microsoft", "moonshotai": "Moonshot"}   # last two have no logo asset


def layout(X, members, pull=0.55, seed=0):
    """2D similarity map of the panel's inventions.

    Positions come from metric MDS on the SAME quantity the multiple criterion uses -- cosine distance
    between invention embeddings -- rather than a linear projection, so "close on the page" means
    "close by the measure that defines a multiple". Distances between two models in the same multiple
    are additionally scaled by `pull`, so a cluster reads as one group instead of a chain: a component
    is transitive (a--b, b--c) and its ends can be far apart, which a faithful projection spreads
    across the panel. The warp only ever moves points of an already-identified multiple TOWARDS each
    other; every other distance is the measured one. Returns (Z, normalized stress).
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    D = np.clip(1.0 - Xn @ Xn.T, 0.0, None)
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    for g in members:
        for a, b in itertools.combinations(g, 2):
            D[a, b] = D[b, a] = D[a, b] * pull
    m = MDS(n_components=2, metric=True, dissimilarity="precomputed", n_init=8, max_iter=800,
            normalized_stress=True, random_state=seed)
    return m.fit_transform(D), float(m.stress_)


def capsule(P, pad, n=28):
    """A rounded region enclosing the cluster's points: the convex hull of a disc of radius `pad`
    drawn around each point. Handles 2 points and collinear points, which a bare hull cannot."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([np.cos(th), np.sin(th)]) * pad
    S = np.vstack([p + ring for p in P])
    return S[ConvexHull(S).vertices]


def cluster_label(members):
    """The cluster's shared invention: the most common coined name, lower-cased."""
    return Counter(str(m["name"]).lower() for m in members).most_common(1)[0][0]


def main():
    d = np.load(NPZ, allow_pickle=True)
    vecs, models, tasks, us, vs, names = d["vecs"], d["models"], d["tasks"], d["u"], d["v"], d["names"]
    provs_present = []
    for m in models:
        p = _provider(m)
        if p and p not in provs_present:
            provs_present.append(p)
    unlabelled = [p for p in provs_present if p not in PROV_LABEL]
    if unlabelled:   # otherwise their points are drawn but silently absent from the legend
        raise ValueError(f"FATAL: providers in the data with no PROV_LABEL entry: {unlabelled}")
    provs_present = [p for p in PROV_LABEL if p in provs_present]  # stable order
    logos = _load_logos()

    fig = plt.figure(figsize=(16.4, 6.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.42, 6.0, 6.0], wspace=0.20)

    # ---- legend down the left-hand side: providers, then the two tasks, then the cluster mark ----
    lax = fig.add_subplot(gs[0, 0]); lax.axis("off"); lax.set_xlim(0, 1); lax.set_ylim(0, 1)
    rows = ([("prov", p, PROV_LABEL[p]) for p in provs_present] +
            [("gap", None, ""), ("task", "blending", "blend"), ("task", "analogy", "analogy"),
             ("gap", None, ""), ("clust", None, "inventive\nmultiple")])
    y, step = 0.94, 0.062
    for kind, key, lab in rows:
        if kind == "gap":
            y -= step * 0.75
            continue
        if kind == "prov":
            lax.scatter([0.07], [y], s=150, color=BRAND[key], edgecolors="white", linewidths=0.7,
                        zorder=3)
            img = logos.get(key)
            if img is not None:
                lax.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.041), (0.24, y),
                                              frameon=False, box_alignment=(0.5, 0.5)))
            tx = 0.38
        elif kind == "clust":
            # drawn as markers, not patches: this axes is tall and narrow, so a Circle would render
            # as an ellipse. Member dots sit on a ring, scaled per axis to look round.
            lax.scatter([0.13], [y], s=760, facecolor="#8A8A8A", alpha=0.16, edgecolors="#8A8A8A",
                        linewidths=1.1, zorder=2)
            th = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 5
            lax.scatter(0.13 + 0.055 * np.cos(th), y + 0.021 * np.sin(th), s=22, color="#555555",
                        zorder=3)
            tx = 0.30
        else:
            lax.scatter([0.10], [y], s=165, marker=MARK[key], color="#555555", edgecolors="white",
                        linewidths=0.7, zorder=3)
            tx = 0.30
        lax.text(tx, y, lab, fontsize=14, va="center")
        y -= step

    # ---- 2D PCA landscape (position = similarity); MARKER SIZE = originality (within-task kNN
    #      embedding distance). Bigger marker = more original invention.
    def knn_orig(X, k=5):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        D = 1.0 - Xn @ Xn.T
        np.fill_diagonal(D, np.inf)
        D.sort(axis=1)
        return D[:, :min(k, D.shape[1] - 1)].mean(axis=1)

    clusters = json.loads(CLUST.read_text())["clusters"]   # structural inventive multiples, size >= 2

    util_all, integ_all = d["util"], d["integ"]
    coords = []
    for u, v in PANELS:
        sel = (us == u) & (vs == v)
        X = np.vstack(vecs[sel]); tk = tasks[sel]; mm = models[sel]
        groups = []                                         # multiples, as positions within this panel
        for c in clusters:
            if (c["u"], c["v"]) != (u, v):
                continue
            g = [int(x[0]) for m in c["members"]
                 for x in [np.where((tk == c["task"]) & (mm == m["model"]))[0]] if len(x)]
            if len(g) >= 2:
                groups.append(g)
        Z, stress = layout(X, groups)
        print(f"  ({u}, {v}): MDS normalized stress {stress:.3f}")
        rho = np.zeros(len(X))                              # originality WITHIN task (matches benchmark)
        for t in ("blending", "analogy"):
            idx = np.where(tk == t)[0]
            if len(idx) >= 3:
                rho[idx] = knn_orig(X[idx])
        coords.append((Z, models[sel], names[sel], tk, rho, util_all[sel], integ_all[sel]))

    # marker size = COMPOSITE emergent creativity = mean(originality, utility J^utl, integration J^qua).
    r_lo, r_hi = np.nanmin(np.concatenate([c[4] for c in coords])), np.nanmax(np.concatenate([c[4] for c in coords]))
    def composite(rho, util, integ):
        on = (rho - r_lo) / (r_hi - r_lo + 1e-9)            # originality normalised to [0,1]
        cc = np.nanmean(np.vstack([on, util, integ]), axis=0)
        return np.where(np.isnan(cc), 0.0, cc)
    comps = [composite(c[4], c[5], c[6]) for c in coords]
    c_lo = min(x.min() for x in comps); c_hi = max(x.max() for x in comps)
    GAMMA = 2.6                                        # >1 accentuates size differences
    # markers stay small: the figure's subject is which inventions are the SAME, so the multiple-graph
    # must stay visible underneath them (size is secondary information).
    msize = lambda cc: 16 + 300 * (np.clip((cc - c_lo) / (c_hi - c_lo + 1e-9), 0.0, None) ** GAMMA)

    labels = {}                                        # per-axes annotations, to fit the limits around
    for pi, ((u, v), (Z, mm, nm, tk, rho, ut, ig)) in enumerate(zip(PANELS, coords)):
        ax = fig.add_subplot(gs[0, pi + 1])      # column 0 is the legend
        comp = composite(rho, ut, ig)
        cols = np.array([BRAND.get(_provider(m), "#777777") for m in mm])
        cen = Z.mean(0)
        sx = Z[:, 0].max() - Z[:, 0].min(); sy = Z[:, 1].max() - Z[:, 1].min()

        # ---- inventive multiples (drawn first, behind the markers) ----
        # Every marker keeps its true projected position. A cluster is outlined by a rounded region
        # enclosing its members, and each member also gets a ring in the cluster's colour -- the ring
        # is what settles membership, since a non-member can project inside the outline.
        Zp = Z
        pad = 0.052 * max(sx, sy)
        panel_clusters, ring = [], {}
        for ci, c in enumerate(sorted((c for c in clusters if (c["u"], c["v"]) == (u, v)),
                                      key=lambda c: -c["size"])):
            pos = [np.where((tk == c["task"]) & (mm == m["model"]))[0] for m in c["members"]]
            idx = [int(x[0]) for x in pos if len(x)]
            if len(idx) < 2:
                continue                                    # a member whose invention is not in this panel
            P = Z[idx]
            col = HULL[len(panel_clusters) % len(HULL)]
            # clusters of the same panel often interleave in the projection, so later (smaller) ones
            # get a tighter, dashed outline: nested rather than merged into one blob.
            k = len(panel_clusters)
            # NB: patch-level alpha would fade the edge too, so the fill carries its own alpha
            ax.add_patch(plt.Polygon(capsule(P, pad * max(0.62, 1.0 - 0.14 * k)), closed=True,
                                     facecolor=to_rgba(col, 0.10), edgecolor=to_rgba(col, 0.85),
                                     linewidth=1.5, linestyle="-" if k == 0 else (0, (5, 2.5)),
                                     zorder=1 + 0.01 * k))
            for j in idx:
                ring[j] = col
            medoid = P[np.argmin(np.hypot(*(P - P.mean(0)).T))]     # anchor the label on a real point
            panel_clusters.append((medoid, pad, f"{cluster_label(c['members'])} \u00d7{len(idx)}", col, idx))

        for t in ("blending", "analogy"):                  # one scatter per task (marker is per-call)
            msk = tk == t
            if msk.any():
                where = np.where(msk)[0]
                ax.scatter(Z[msk, 0], Z[msk, 1], s=msize(comp[msk]), c=cols[msk], marker=MARK[t],
                           edgecolors=[ring.get(j, "white") for j in where],
                           linewidths=[1.8 if j in ring else 1.0 for j in where], alpha=0.95, zorder=3)

        # ---- labels: every cluster, then a few outer singletons for context. All labels sit on a
        #      common ring outside the cloud, fanned by angle so they never overlap.
        clustered = {j for *_, idx in panel_clusters for j in idx}
        dvec = Zp - cen
        rr = np.hypot(dvec[:, 0], dvec[:, 1]) + 1e-9
        ang = np.arctan2(dvec[:, 1], dvec[:, 0])
        MIN_GAP = 0.55                                     # min angular separation between labels (rad)
        targets = []                                       # (angle, xy, text, colour, weight)
        for cpt, rad, lab, col, idx in sorted(panel_clusters, key=lambda c: -len(c[4])):
            a = np.arctan2(cpt[1] - cen[1], cpt[0] - cen[0])
            if all(min(abs(a - b), 2 * np.pi - abs(a - b)) > MIN_GAP for b, *_ in targets):
                targets.append((a, cpt, lab, col, "bold"))
        n_sing = 0
        for i in sorted((i for i in range(len(Z)) if i not in clustered), key=lambda i: -comp[i]):
            if n_sing >= N_SINGLETON or rr[i] < np.percentile(rr, 42):
                continue
            if all(min(abs(ang[i] - b), 2 * np.pi - abs(ang[i] - b)) > MIN_GAP for b, *_ in targets):
                targets.append((ang[i], Zp[i], str(nm[i]), "#6A6A6A", "normal"))
                n_sing += 1

        Rx, Ry = 0.72 * sx, 0.66 * sy                      # ring sits outside every marker
        for a, xy, lab, col, weight in targets:
            ca, sa = np.cos(a), np.sin(a)
            lx, ly = cen[0] + Rx * ca, cen[1] + Ry * sa
            ha = "left" if ca > 0.15 else "right" if ca < -0.15 else "center"
            va = "bottom" if sa > 0.15 else "top" if sa < -0.15 else "center"
            labels.setdefault(ax, []).append(ax.annotate(
                lab, xy=xy, xytext=(lx, ly), textcoords="data",
                fontsize=13.5 if weight == "bold" else 12.5, color=col if weight == "bold" else "#5A5A5A",
                fontweight=weight, zorder=5, ha=ha, va=va,
                arrowprops=dict(arrowstyle="-", color="#BBBBBB", lw=0.5)))
        ax.set_title(f"({TITLE_CASE.get(u, u)},  {TITLE_CASE.get(v, v)})", fontsize=18,
                     fontweight="bold", pad=10)
        ax.set_xlabel("Similarity map (MDS) dimension one", fontsize=13.5)
        ax.set_ylabel("Dimension two", fontsize=13.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(cen[0] - 1.42 * sx, cen[0] + 1.12 * sx)   # starting room; widened to fit labels below
        ax.set_ylim(cen[1] - 0.94 * sy, cen[1] + 0.94 * sy)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Label text is only measurable once drawn, so widen each panel around its own labels afterwards: a
    # long label on the left would otherwise run into (or through) the y-axis. This has to ITERATE --
    # widening the limits zooms out, so the same text then covers more data units and needs more room
    # again -- so repeat until the limits stop moving, then verify no label crosses the axes frame.
    for _ in range(12):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        moved = False
        for ax, anns in labels.items():
            x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
            inv = ax.transData.inverted()
            mx, my = 0.045 * (x1 - x0), 0.045 * (y1 - y0)
            nx0, nx1, ny0, ny1 = x0, x1, y0, y1
            for an in anns:
                bb = an.get_window_extent(renderer=rend)
                (bx0, by0), (bx1, by1) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
                nx0, nx1 = min(nx0, bx0 - mx), max(nx1, bx1 + mx)
                ny0, ny1 = min(ny0, by0 - my), max(ny1, by1 + my)
            if max(abs(nx0 - x0), abs(nx1 - x1)) > 0.002 * (x1 - x0) or \
               max(abs(ny0 - y0), abs(ny1 - y1)) > 0.002 * (y1 - y0):
                ax.set_xlim(nx0, nx1); ax.set_ylim(ny0, ny1)
                moved = True
        if not moved:
            break
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    for ax, anns in labels.items():                    # loud check: a label over the frame is a bug
        fr = ax.get_window_extent(renderer=rend)
        for an in anns:
            bb = an.get_window_extent(renderer=rend)
            if bb.x0 < fr.x0 or bb.x1 > fr.x1 or bb.y0 < fr.y0 or bb.y1 > fr.y1:
                raise RuntimeError(f"FATAL: label {an.get_text()!r} falls outside its panel frame")

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_invention_landscape.{ext}", dpi=300, bbox_inches="tight")
    print("saved fig_invention_landscape ->", OUT)


if __name__ == "__main__":
    main()
