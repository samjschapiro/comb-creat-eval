"""Does the distance between the two anchors predict how often models converge on the same invention?

Downstream of analyze_inventive_multiples.py: reads its `per_item` block (every convergence rate per
(task, anchor pair)) and asks whether anchor distance predicts it. The unit is the anchor pair
(n = 30 per task); pairs of models are never the unit, since every model sits in every item.

DISTANCE MEASURES (item-level). The Wikidata graph the anchors were drawn from covers only 6 of the
30 pairs with a path, so a graph geodesic is not available. Used instead:
  label_cos        cosine distance between the anchor labels (MiniLM), the measure the analysis used;
  desc_cos         cosine distance between "label: Wikidata description" strings -- the same encoder
                   on richer text, so "Banking" is not judged on one word;
  min_log_sitelinks, mean_log_sitelinks
                   prominence covariates (Wikipedia sitelinks from pool_wikidata.json): well-known
                   anchors may simply give models more to agree about, which is a confound for
                   distance, not a distance.

OUTCOMES (per task, per item): rate_tau2_pct (the headline), rate_tau1_pct, per_property_mean (the
size-free re-use rate), largest_component.

TESTS: Spearman rho with a permutation p (10,000 shuffles of the outcome across items), leave-one-out
(how many of the 30 deletions keep p < 0.05, and the worst p), tercile means, and a partial Spearman
of distance controlling for mean prominence. Nothing here is corrected for the number of cells; read
it as a screen, and the prose should quote the headline cell only.

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_anchor_distance
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

from src.kg_creat.embed import get_embedder

SRC = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
POOL = Path("data/kg_creat/pool_wikidata.json")
OUT = Path("data/kg_creat/kombine_test30/analysis/anchor_distance.json")
FIG = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
N_PERM = 10000
OUTCOMES = ["rate_tau2_pct", "rate_tau1_pct", "per_property_mean", "largest_component"]
DISTANCES = ["label_cos", "desc_cos", "min_log_sitelinks", "mean_log_sitelinks"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
                     "font.size": 12})


def perm_spearman(x, y, rng):
    rho = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(N_PERM)])
    return float(rho), float((np.sum(np.abs(null) >= abs(rho)) + 1) / (N_PERM + 1))


def partial_spearman(x, y, z):
    """Spearman of x and y with z partialled out (ranks, then residualise both on z)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    d = json.loads(SRC.read_text())
    if "per_item" not in d:
        raise SystemExit(f"{SRC} has no `per_item` block -- re-run analyze_inventive_multiples.py")
    pool = {r["label"]: r for r in json.loads(POOL.read_text())["rows"]}
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)

    items = sorted({(r["u"], r["v"]) for r in d["per_item"]})
    for u, v in items:
        for a in (u, v):
            if a not in pool:
                raise SystemExit(f"FATAL: anchor {a!r} missing from {POOL}")
    desc = lambda a: f"{a}: {pool[a]['description']}" if pool[a].get("description") else a
    E_lab = {a: un(np.asarray(embed(a), float)) for a in {x for it in items for x in it}}
    E_desc = {a: un(np.asarray(embed(desc(a)), float)) for a in E_lab}
    dist = {}
    for u, v in items:
        sl = [np.log1p(pool[a]["sitelinks"] or 0) for a in (u, v)]
        dist[(u, v)] = {"label_cos": 1.0 - float(E_lab[u] @ E_lab[v]),
                        "desc_cos": 1.0 - float(E_desc[u] @ E_desc[v]),
                        "min_log_sitelinks": float(min(sl)), "mean_log_sitelinks": float(np.mean(sl)),
                        "u_description": pool[u].get("description"), "v_description": pool[v].get("description")}
    print(f"{len(items)} anchor pairs; label_cos vs desc_cos Spearman "
          f"{spearmanr([dist[i]['label_cos'] for i in items], [dist[i]['desc_cos'] for i in items]).statistic:+.2f}")

    rng = np.random.default_rng(0)
    results = {}
    for task in ("blending", "analogy"):
        rows = {(r["u"], r["v"]): r for r in d["per_item"] if r["task"] == task}
        its = [i for i in items if i in rows]
        prom = np.array([dist[i]["mean_log_sitelinks"] for i in its])
        results[task] = {"n_items": len(its), "cells": {}}
        print(f"\n{task.upper()}  (n = {len(its)} anchor pairs)")
        print(f"  {'outcome':18s}{'distance':20s}{'rho':>7}{'perm p':>9}{'LOO<.05':>9}{'worst p':>9}"
              f"{'terciles (near -> far)':>28}{'partial|prom':>14}")
        for oc in OUTCOMES:
            y = np.array([rows[i][oc] for i in its], float)
            for dm in DISTANCES:
                x = np.array([dist[i][dm] for i in its])
                rho, p = perm_spearman(x, y, rng)
                loo = [spearmanr(np.delete(x, k), np.delete(y, k)).pvalue for k in range(len(x))]
                o = np.argsort(x); t = len(x) // 3
                terc = [float(np.mean(y[o[a:b]])) for a, b in ((0, t), (t, 2 * t), (2 * t, len(x)))]
                part = partial_spearman(x, y, prom) if dm in ("label_cos", "desc_cos") else float("nan")
                results[task]["cells"][f"{oc}|{dm}"] = {
                    "outcome": oc, "distance": dm, "spearman_rho": rho, "perm_p": p,
                    "loo_n_sig": int(sum(1 for q in loo if q < 0.05)), "loo_max_p": float(max(loo)),
                    "tercile_means": terc, "partial_rho_given_prominence": part}
                fmt = (lambda v: f"{v:.2f}") if oc != "largest_component" else (lambda v: f"{v:.1f}")
                print(f"  {oc:18s}{dm:20s}{rho:+7.2f}{p:9.3f}{sum(1 for q in loo if q < 0.05):6d}/{len(x):<2d}"
                      f"{max(loo):9.3f}   {' -> '.join(fmt(v) for v in terc):>25}"
                      f"{('' if np.isnan(part) else f'{part:+.2f}'):>14}")

    # ---- figure: headline rate and per-property re-use against the two embedding distances ----
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2))
    for ci, dm in enumerate(("label_cos", "desc_cos")):
        for ri, oc in enumerate(("rate_tau2_pct", "per_property_mean")):
            ax = axes[ri, ci]
            for task, col, mk in (("blending", "#2F6B8E", "s"), ("analogy", "#C8702A", "o")):
                rows = {(r["u"], r["v"]): r for r in d["per_item"] if r["task"] == task}
                its = [i for i in items if i in rows]
                x = [dist[i][dm] for i in its]; y = [rows[i][oc] for i in its]
                c = results[task]["cells"][f"{oc}|{dm}"]
                ax.scatter(x, y, s=42, marker=mk, color=col, alpha=0.85, edgecolors="white", linewidths=0.6,
                           label=f"{task}  $\\rho$ = {c['spearman_rho']:+.2f}, p = {c['perm_p']:.2f}")
            ax.set_xlabel({"label_cos": "anchor distance (label cosine)",
                           "desc_cos": "anchor distance (label + description cosine)"}[dm])
            ax.set_ylabel({"rate_tau2_pct": r"$\tau$ = 2 multiple rate (% of pairs)",
                           "per_property_mean": "per-property re-use"}[oc])
            ax.legend(frameon=False, fontsize=10.5, loc="upper left")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig_anchor_distance.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    OUT.write_text(json.dumps({"n_perm": N_PERM, "distances": {f"{u} | {v}": dist[(u, v)] for u, v in items},
                               "results": results}, indent=1))
    print(f"\nwrote {OUT} and {FIG}/fig_anchor_distance.{{png,pdf}}")


if __name__ == "__main__":
    main()
