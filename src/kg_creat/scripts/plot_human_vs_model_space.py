"""Where human and model answers sit in meaning space, and how original they are (qualitative view).

Each answer (artifact) is the mean embedding of its non-anchor concepts and relations -- the space the
benchmark's originality is measured in (MiniLM, as score.py). Two figures, written to
data/kg_creat/human_study/figures/ and copied to scratch/human_vs_model_space/:

  space_all_items.png   per task, every item pooled after subtracting the item's mean answer (so the map
                        shows how answers spread WITHIN an item, not which item they answer); PCA and t-SNE
  space_pca_all_vs_passed.png   PCA only, all answers (top row) vs. answers that passed utility (bottom row)
  space_pca_utility.png         PCA, all answers; filled = passed utility, hollow = failed; size = originality
  space_showcase.png    one item per task (the item with the most human answers), PCA within the item;
                        point size = originality, filled = passed utility, hollow = failed; the three most
                        original PASSED human and model answers are labelled (failed answers are often
                        'original' only because they are odd)

Humans = kept sessions; models = all 35 models pooled. Originality from the shared human + model pool
(data/kg_creat/human_study/combined_scores). Two stages because the embedder and sklearn live in
different environments:

    .venv_mlx/bin/python -m src.kg_creat.scripts.plot_human_vs_model_space --stage embed
    .venv/bin/python -m src.kg_creat.scripts.plot_human_vs_model_space --stage plot
"""
import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np

SCORES = "data/kg_creat/human_study/combined_scores"
OUT = Path("data/kg_creat/human_study/figures")
CACHE = OUT / "answer_embeddings.npz"
META = OUT / "answer_meta.json"
MOCK = Path("scratch/human_vs_model_space")
TASKS = [("association", "baseline"), ("analogy", "analogy"), ("blending", "blending")]
HUMAN, MODEL = "#d95f02", "#1b6ca8"


def answers():
    """One record per answer: group, task, item, passed, originality, element surfaces, short label."""
    from src.kg_creat.scripts.score import _artifact_elements
    out = []
    for f in sorted(glob.glob(f"{SCORES}/*/path_scores.json")):
        ent = os.path.basename(os.path.dirname(f))
        if ent == "human_flagged":
            continue
        group = "human" if ent == "human_kept" else "model"
        recs = json.load(open(f))
        idx = {(r["prompt_id"], r.get("temperature"), r.get("sample_idx"), r["path_idx"]): r for r in recs}
        for r in recs:
            mode, trip = r.get("mode"), r.get("triples") or []
            if not trip:
                continue
            if mode == "analogy":
                if "pair_sat" not in r or r["path_idx"] % 2:
                    continue
                partner = idx.get((r["prompt_id"], r.get("temperature"), r.get("sample_idx"), r["path_idx"] + 1))
                paths = [trip, partner["triples"] if partner else []]
                passed = r.get("pair_sat") is True
                label = " → ".join(t[2] for t in trip[:2]) + " | " + " → ".join(t[2] for t in paths[1][:2])
            else:
                paths, passed = [trip], r.get("sat") is True
                label = (" → ".join([trip[0][0]] + [t[2] for t in trip]) if mode == "baseline"
                         else trip[0][0] + ": " + "; ".join(f"{t[1]} {t[2]}" for t in trip[:2]))
            els = sorted(s for _k, s in _artifact_elements(paths, r.get("u_label"), r.get("v_label")))
            if not els or r.get("originality") is None:
                continue
            out.append({"group": group, "entity": ent, "task": dict((m, t) for t, m in TASKS)[mode],
                        "item": r["prompt_id"], "u": r.get("u_label"), "v": r.get("v_label"),
                        "passed": passed, "originality": r["originality"], "elements": els, "label": label[:70]})
    return out


def stage_embed():
    from src.kg_creat.embed import get_embedder
    embed = get_embedder()
    recs = answers()
    vecs = []
    for a in recs:
        V = np.stack([embed(s) for s in a["elements"]])
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        v = V.mean(axis=0)
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=np.stack(vecs).astype(np.float32))
    json.dump([{k: v for k, v in a.items() if k != "elements"} for a in recs], open(META, "w"))
    print(f"embedded {len(recs)} answers -> {CACHE}")


def stage_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    # Nimbus Roman (the URW Times the paper's LaTeX uses) for every piece of text; fail loudly if it is missing
    from matplotlib import font_manager
    if "Nimbus Roman" not in {f.name for f in font_manager.fontManager.ttflist}:
        raise SystemExit("FATAL: Nimbus Roman is not installed")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"], "mathtext.fontset": "custom",
                         "mathtext.rm": "Nimbus Roman", "mathtext.it": "Nimbus Roman:italic",
                         "mathtext.bf": "Nimbus Roman:bold"})
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
                         "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13})
    X = np.load(CACHE)["X"]
    meta = json.load(open(META))
    grp = np.array([m["group"] for m in meta])
    task = np.array([m["task"] for m in meta])
    item = np.array([m["item"] for m in meta])
    orig = np.array([m["originality"] for m in meta])
    passed = np.array([m["passed"] for m in meta])

    # (1) all items per task, item-centred
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    letters = iter("abcdef")
    for col, (t, _m) in enumerate(TASKS):
        sel = task == t
        Xt = X[sel].copy()
        for it in np.unique(item[sel]):
            rows = item[sel] == it
            Xt[rows] -= Xt[rows].mean(axis=0)
        emb = {"PCA": PCA(n_components=2, random_state=0).fit_transform(Xt),
               "t-SNE": TSNE(n_components=2, init="pca", perplexity=40, random_state=0).fit_transform(Xt)}
        for row, (name, Z) in enumerate(emb.items()):
            ax = axes[row, col]
            g, o = grp[sel], orig[sel]
            for who, c, z in (("model", MODEL, 1), ("human", HUMAN, 2)):
                k = g == who
                ax.scatter(Z[k, 0], Z[k, 1], s=6 + 60 * (o[k] - o.min()) / (np.ptp(o) + 1e-9), c=c,
                           alpha=0.25 if who == "model" else 0.75, linewidths=0, zorder=z,
                           label=f"{who}s (n = {k.sum()})")
            ax.set_title(f"({next(letters)}) {t}, {name}")
            ax.set_xlabel(f"{name} 1"); ax.set_ylabel(f"{name} 2")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0 and col == 0:
                ax.legend(loc="upper right", markerscale=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "space_all_items.png", dpi=180)
    plt.close(fig)

    # (1b) PCA only: all answers (top) vs. answers that passed utility (bottom), item-centred within each subset
    fig, axes = plt.subplots(2, 3, figsize=(18, 11.5))
    letters = iter("abcdef")
    for row, (rowname, keep) in enumerate((("all answers", np.ones(len(meta), bool)), ("passed only", passed))):
        for col, (t, _m) in enumerate(TASKS):
            sel = (task == t) & keep
            Xt = X[sel].copy()
            for it in np.unique(item[sel]):
                rows = item[sel] == it
                Xt[rows] -= Xt[rows].mean(axis=0)
            Z = PCA(n_components=2, random_state=0).fit_transform(Xt)
            ax = axes[row, col]
            g, o = grp[sel], orig[sel]
            for who, c, z in (("model", MODEL, 1), ("human", HUMAN, 2)):
                k = g == who
                ax.scatter(Z[k, 0], Z[k, 1], s=6 + 60 * (o[k] - o.min()) / (np.ptp(o) + 1e-9), c=c,
                           alpha=0.25 if who == "model" else 0.75, linewidths=0, zorder=z,
                           label=f"{who}s (n = {k.sum()})")
            ax.set_title(f"({next(letters)}) {t}, {rowname}")
            ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_xticks([]); ax.set_yticks([])
            ax.legend(loc="upper right", markerscale=2, frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "space_pca_all_vs_passed.png", dpi=180)
    plt.close(fig)

    # (1c) PCA, all answers, one row: filled = passed utility, hollow = failed. Sized for a full-width paper figure:
    # a smaller canvas with large type, and one horizontal legend under the panels.
    from matplotlib.lines import Line2D
    big = {"font.size": 20, "axes.titlesize": 24, "axes.labelsize": 22, "legend.fontsize": 20}
    with plt.rc_context(big):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
        for ax, letter, (t, _m) in zip(axes, "abc", TASKS):
            sel = task == t
            Xt = X[sel].copy()
            for it in np.unique(item[sel]):
                rows = item[sel] == it
                Xt[rows] -= Xt[rows].mean(axis=0)
            Z = PCA(n_components=2, random_state=0).fit_transform(Xt)
            g, o, ok = grp[sel], orig[sel], passed[sel]
            size = 5 + 45 * (o - o.min()) / (np.ptp(o) + 1e-9)
            for who, c, z, a in (("model", MODEL, 1, 0.22), ("human", HUMAN, 2, 0.85)):
                for good in (False, True):
                    k = (g == who) & (ok == good)
                    ax.scatter(Z[k, 0], Z[k, 1], s=size[k], zorder=z + (0.5 if good else 0), alpha=a,
                               facecolors=c if good else "none", edgecolors=c, linewidths=0.6 if who == "model" else 1.1)
            nh, nm = int((g == "human").sum()), int((g == "model").sum())
            ax.set_title(f"({letter}) {t}")
            ax.text(0.03, 0.03, f"humans n = {nh} ({100 * ok[g == 'human'].mean():.0f}% passed)\n"
                                f"LLMs n = {nm} ({100 * ok[g == 'model'].mean():.0f}% passed)",
                    transform=ax.transAxes, fontsize=16, va="bottom", zorder=5,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))
            ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_xticks([]); ax.set_yticks([])
        fig.legend(handles=[Line2D([], [], marker="o", ls="", color=HUMAN, ms=13, label="human, passed"),
                            Line2D([], [], marker="o", ls="", mfc="none", mew=1.8, color=HUMAN, ms=13, label="human, failed"),
                            Line2D([], [], marker="o", ls="", color=MODEL, ms=13, label="LLM, passed"),
                            Line2D([], [], marker="o", ls="", mfc="none", mew=1.8, color=MODEL, ms=13, label="LLM, failed")],
                   loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01), columnspacing=2.2, handletextpad=0.4)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(OUT / "space_pca_utility.png", dpi=300)
        plt.close(fig)

    # (2) one showcase item per task: the item with the most human answers
    import textwrap
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(21, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1], hspace=0.05, wspace=0.12,
                          left=0.03, right=0.99, top=0.95, bottom=0.01)
    for col, (letter, (t, _m)) in enumerate(zip("abc", TASKS)):
        ax, tx = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        sel_t = task == t
        hum_counts = {it: int(((item == it) & (grp == "human")).sum()) for it in np.unique(item[sel_t])}
        best = max(hum_counts, key=hum_counts.get)
        sel = item == best
        Z = PCA(n_components=2, random_state=0).fit_transform(X[sel])
        g, o, p = grp[sel], orig[sel], passed[sel]
        size = 20 + 260 * (o - o.min()) / (np.ptp(o) + 1e-9)
        for who, c, z in (("model", MODEL, 1), ("human", HUMAN, 2)):
            for ok in (True, False):
                k = (g == who) & (p == ok)
                ax.scatter(Z[k, 0], Z[k, 1], s=size[k], zorder=z,
                           facecolors=c if ok else "none", edgecolors=c, linewidths=1.2,
                           alpha=0.45 if who == "model" else 0.9)
        labels = [meta[i] for i in np.where(sel)[0]]
        lines = []
        for who, tag, c in (("human", "H", HUMAN), ("model", "M", MODEL)):
            k = np.where((g == who) & p)[0]          # most original among answers that PASSED utility
            for n, j in enumerate(k[np.argsort(-o[k])][:3], 1):
                ax.annotate(f"{tag}{n}", (Z[j, 0], Z[j, 1]), fontsize=13, fontweight="bold", color=c,
                            xytext=(5, 5), textcoords="offset points", zorder=3)
                lines.append((c, f"{tag}{n} ({o[j]:.2f}): " + textwrap.shorten(labels[j]["label"], 56, placeholder="…")))
        u, v = labels[0]["u"], labels[0]["v"]
        sym = {"association": "→", "analogy": "::", "blending": "+"}[t]
        ax.set_title(f"({letter}) {t}: {u} {sym} {v}")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.set_xticks([]); ax.set_yticks([])
        tx.axis("off")
        for n, (c, line) in enumerate(lines):
            tx.text(0.0, 0.95 - n * 0.155, line, color=c, fontsize=12, transform=tx.transAxes, va="top")
        if col == 0:
            ax.legend(handles=[Line2D([], [], marker="o", ls="", color=HUMAN, ms=10, label="human, passed"),
                               Line2D([], [], marker="o", ls="", mfc="none", color=HUMAN, ms=10, label="human, failed"),
                               Line2D([], [], marker="o", ls="", color=MODEL, ms=10, label="model, passed"),
                               Line2D([], [], marker="o", ls="", mfc="none", color=MODEL, ms=10, label="model, failed")],
                      loc="lower left", frameon=True, framealpha=0.9)
    fig.savefig(OUT / "space_showcase.png", dpi=180)
    plt.close(fig)

    MOCK.mkdir(parents=True, exist_ok=True)
    for f in ("space_all_items.png", "space_pca_all_vs_passed.png", "space_pca_utility.png", "space_showcase.png"):
        shutil.copy(OUT / f, MOCK / f)
    print(f"wrote space_all_items.png, space_pca_all_vs_passed.png, space_pca_utility.png, space_showcase.png to {OUT} (copies in {MOCK})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["embed", "plot"], required=True)
    a = ap.parse_args()
    stage_embed() if a.stage == "embed" else stage_plot()
