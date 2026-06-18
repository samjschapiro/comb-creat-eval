"""Cluster twist reveals: K-selection diagnostics + per-cluster stats + 2D visualization.

Maps the *content* space of plot-twist reveals (the recurring motifs), separately from the
*mechanism* taxonomy. Reveal embeddings (mpnet) are clustered with K-Means; K is justified by
elbow (inertia) + silhouette diagnostics rather than picked by hand. Each cluster is
characterized by size, population mix, mean realism/surprise/coherence, dominant mechanism
codes, and keyword signature, and the clusters are projected to 2D with UMAP.

Usage:
    python src/plot_twist/scripts/cluster_reveals.py configs/plot_twist/cluster_reveals.yaml --overwrite
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils import init_directory, load_config, save_config
from src.plot_twist.sets import twist_types

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

_STOP = set((
    "the a an and or but of to in on at for with as by from is are was were be been being this "
    "that these those it its he she they them his her their you your we our not no into has had "
    "have who whom which what when where why how all out up off her his actor reveals revealing "
    "twist reinterpret reinterprets reinterpreting forcing forces reader earlier story events "
    "actually really truly entire whole leading led discover discovers discovered learns learned "
    "realizes realized himself herself themselves years ago time own life death dead about more "
    "most one two also just only even being rather than making interpret connection suggesting "
    "revealed not".split()
))


def _keywords(reveals, n=7):
    c = collections.Counter()
    for r in reveals:
        for w in re.findall(r"[a-zA-Z]+", r.lower()):
            if len(w) > 3 and w not in _STOP:
                c[w] += 1
    return ", ".join(w for w, _ in c.most_common(n))


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "classified_json"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)
    seed = cfg.get("seed", 0)

    cls = [r for r in json.loads(Path(cfg["classified_json"]).read_text()) if r.get("reveal")]
    realism = json.loads(Path(cfg["realism_scores"]).read_text()) if cfg.get("realism_scores") else {}
    types = twist_types(cfg["manifest"]) if cfg.get("manifest") else {}

    # embed reveals once (cache by content hash count -> just cache the matrix)
    cache = out / "embeddings.npy"
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.get("embed_model", "sentence-transformers/all-mpnet-base-v2"))
    E = np.asarray(model.encode([r["reveal"] for r in cls], normalize_embeddings=True, show_progress_bar=False))
    np.save(cache, E)
    print(f"embedded {len(cls)} reveals")

    # --- K-selection diagnostics: elbow (inertia) + silhouette ---
    lo, hi = cfg.get("k_range", [2, 18])
    Ks = list(range(lo, hi + 1))
    inertias, sils = [], []
    for K in Ks:
        km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(E)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(E, km.labels_, metric="cosine"))
    best_sil_K = Ks[int(np.argmax(sils))]
    print("\nK-selection diagnostics:")
    print(f"  {'K':>3}{'inertia':>12}{'silhouette':>13}")
    for K, inr, s in zip(Ks, inertias, sils):
        print(f"  {K:>3}{inr:>12.1f}{s:>13.4f}{'  <- max silhouette' if K==best_sil_K else ''}")

    fig, ax1 = plt.subplots(figsize=(5.2, 3.4))
    ax1.plot(Ks, inertias, "o-", color="#103D5F", label="inertia (elbow)")
    ax1.set_xlabel("number of clusters $K$"); ax1.set_ylabel("inertia", color="#103D5F")
    ax2 = ax1.twinx(); ax2.spines["top"].set_visible(False)
    ax2.plot(Ks, sils, "s--", color="#C24E00", label="silhouette")
    ax2.set_ylabel("silhouette (cosine)", color="#C24E00")
    ax1.axvline(cfg.get("k", 10), color="#999", ls=":", lw=1)
    fig.tight_layout(); fig.savefig(out / "k_selection.pdf"); fig.savefig(out / "k_selection.png"); plt.close(fig)

    # --- cluster at the chosen K + per-cluster stats ---
    K = cfg.get("k", 10)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(E)
    labs = km.labels_

    def num(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError, KeyError):
            return np.nan

    rows = []
    for k in range(K):
        idx = np.where(labs == k)[0]
        recs = [cls[i] for i in idx]
        nh = sum(r["source"] == "human" for r in recs)
        rv = [realism[r["id"]] for r in recs if r["id"] in realism]
        S = [num(r, "S") for r in recs]; Coh = [num(r, "Coh") for r in recs]
        codes = collections.Counter(r["code"] for r in recs)
        cent = E[idx].mean(0); order = idx[np.argsort(-(E[idx] @ cent))]
        rows.append({
            "cluster": k, "n": len(idx), "n_human": nh, "n_llm": len(idx) - nh,
            "realism": float(np.nanmean(rv)) if rv else float("nan"),
            "surprise": float(np.nanmean(S)), "coherence": float(np.nanmean(Coh)),
            "top_codes": dict(codes.most_common(4)),
            "keywords": _keywords([r["reveal"] for r in recs]),
            "exemplars": [cls[i]["reveal"] for i in order[:3]],
        })
    (out / "clusters.json").write_text(json.dumps(rows, indent=2))

    # curated labels per cluster (config keys may be int or str) -> "c{k}: label";
    # fall back to top-2 keywords. Used by BOTH the UMAP and the 3D plot for consistency.
    raw_labels = {int(k): v for k, v in (cfg.get("cluster_labels", {}) or {}).items()}
    clabel = lambda k: f"c{k}: " + raw_labels.get(k, " ".join(rows[k]["keywords"].split(", ")[:2]))

    print(f"\n=== per-cluster stats (K={K}) ===")
    print(f"  {'c':>2}{'n':>5}{'hum':>4}{'realism':>9}{'surprise':>10}{'coher':>7}  keywords")
    for d in rows:
        print(f"  {d['cluster']:>2}{d['n']:>5}{d['n_human']:>4}{d['realism']:>9.2f}"
              f"{d['surprise']:>10.2f}{d['coherence']:>7.2f}  {d['keywords']}")

    # --- 2D UMAP visualization ---
    import umap
    from scipy.spatial import ConvexHull
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(E)
    # distinct qualitative palette (tab10 = 10 maximally separated hues); shared with the 3D plot
    cols = [plt.get_cmap("tab10")(i) for i in range(K)]
    diffuse = set(cfg.get("diffuse_clusters", []))   # faint grey background, no hull/label
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    for k in range(K):
        P = xy[labs == k]
        if k in diffuse:
            ax.scatter(P[:, 0], P[:, 1], s=7, color="#cfcfcf", alpha=0.35, linewidths=0, zorder=0)
            continue
        # shaded "blob" behind the points: outlier-trimmed convex hull, filled in the cluster colour.
        # Tight trim (keep the inner hull_pct%) so the blobs overlap less and read clearly.
        c = P.mean(0); d = np.linalg.norm(P - c, axis=1)
        Pin = P[d <= np.percentile(d, cfg.get("hull_pct", 70))]
        if len(Pin) >= 3:
            poly = Pin[ConvexHull(Pin).vertices]
            ax.fill(poly[:, 0], poly[:, 1], color=cols[k], alpha=0.16, zorder=1)
            ax.fill(poly[:, 0], poly[:, 1], facecolor="none", edgecolor=cols[k], lw=1.2, alpha=0.7, zorder=1)
        ax.scatter(P[:, 0], P[:, 1], s=13, color=cols[k], alpha=0.8, linewidths=0, zorder=2)
    for k in range(K):  # labels for the genuine (non-diffuse) clusters only
        if k in diffuse:
            continue
        c = xy[labs == k].mean(0)
        ax.text(c[0], c[1], clabel(k), fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=cols[k], lw=1.2, alpha=0.92))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Twist-reveal content clusters (K-Means, K={K}; UMAP projection)", fontsize=12)
    fig.tight_layout(); fig.savefig(out / "clusters_umap.pdf"); fig.savefig(out / "clusters_umap.png"); plt.close(fig)

    # --- 3D visualization in (surprise, coherence, realism) score space ---
    sx = np.array([num(r, "S") for r in cls]); cy3 = np.array([num(r, "Coh") for r in cls])
    rz = np.array([realism.get(r["id"], np.nan) for r in cls])
    keep = ~(np.isnan(sx) | np.isnan(cy3) | np.isnan(rz))
    rng = np.random.default_rng(seed)
    jit = lambda v: v + (rng.random(len(v)) - 0.5) * 0.45   # spread the discrete 1-5 scores
    fig = plt.figure(figsize=(8.4, 7.6)); ax3 = fig.add_subplot(111, projection="3d")
    for k in range(K):
        m = keep & (labs == k)
        ax3.scatter(jit(sx[m]), jit(cy3[m]), jit(rz[m]), s=12, color=cols[k], alpha=0.5,
                    linewidths=0, depthshade=True, label=clabel(k))
    # cluster centroids in score space (large diamonds) with the cluster number INSIDE
    def _txtcol(c):  # contrasting label colour for the diamond fill
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        return "white" if lum < 0.55 else "black"
    for k in range(K):
        m = keep & (labs == k)
        if m.sum():
            cxm, cym, czm = sx[m].mean(), cy3[m].mean(), rz[m].mean()
            ax3.scatter(cxm, cym, czm, s=320, color=cols[k],
                        edgecolor="black", linewidth=1.0, marker="D", depthshade=False, zorder=10)
            ax3.text(cxm, cym, czm, str(k), fontsize=9, fontweight="bold",
                     ha="center", va="center", color=_txtcol(cols[k]), zorder=11)
    ax3.set_xlabel("surprise", labelpad=14, fontsize=16)
    ax3.set_ylabel("coherence", labelpad=14, fontsize=16)
    ax3.set_zlabel("realism", labelpad=14, fontsize=16)
    ax3.tick_params(labelsize=13, pad=2)
    ax3.set_xlim(1, 5); ax3.set_ylim(1, 5); ax3.set_zlim(1, 5)
    ax3.set_title("Twist clusters in (surprise, coherence, realism) space",
                  fontsize=18, pad=14)
    # legend BELOW the plot, two columns, larger markers -- fits the long curated labels
    leg = ax3.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=2, fontsize=11,
                     markerscale=2.2, handletextpad=0.4, columnspacing=1.2, frameon=False)
    ax3.view_init(elev=18, azim=-60)
    # shrink the 3D box so the (rotated) z-axis label sits inside the canvas and is not clipped
    ax3.set_box_aspect(None, zoom=0.82)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.96, bottom=0.12)
    fig.savefig(out / "clusters_3d_scores.pdf", pad_inches=0.35, bbox_inches="tight")
    fig.savefig(out / "clusters_3d_scores.png", dpi=200, pad_inches=0.35, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved: {out/'k_selection.pdf'}\n       {out/'clusters_umap.pdf'}"
          f"\n       {out/'clusters_3d_scores.pdf'}\n       {out/'clusters.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
