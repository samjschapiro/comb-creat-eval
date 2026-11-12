"""DRAT correlates as a function of the weight between DRAT-N (novelty) and DRAT-A (appropriateness).

composite(w) = w * z(DRAT-N) + (1 - w) * z(DRAT-A), standardised within population, for w in [0, 1].
DRAT-N: mean over sets of the ungated GloVe distance among all valid words (`glove_dist_all`).
DRAT-A: mean over sets of the share of typed words that survive the GloVe relevance gate
(`glove_n_surv / n_typed`), a share rather than a count so it does not track how many words were typed.
For each criterion: Spearman rho with the composite at every w, with a percentile bootstrap band
(resampling people or models). Reads saved tables only; nothing is rescored. No length adjustment.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.drat_human import save, set_camera_ready_style
from src.utils import init_directory, load_config, save_config


def drat_facets(sets_csv):
    s = pd.read_csv(sets_csv)
    s = s[s["parsed"] == True]  # noqa: E712 (column is boolean)
    n_zero = int((s["n_typed"] == 0).sum())
    s = s[s["n_typed"] > 0].assign(share=lambda d: d["glove_n_surv"] / d["n_typed"])
    g = s.groupby(["population", "respondent"])
    return g.agg(drat_N=("glove_dist_all", "mean"), drat_A=("share", "mean")).dropna(), n_zero


def norm(m):
    """Model-name key used by benchmarks.json and the Kombine scores (as in neurips_camera_ready.py)."""
    return str(m).replace("/", "_").replace(".", "-").replace(":", "_").lower()


def criterion_table(units_csv, sources, criteria):
    """Unit x criterion table. A criterion is a column of the unit table, or comes from an external
    source ('benchmarks' or 'kombine') under the key `key`; missing or non-numeric entries are NaN."""
    t = pd.read_csv(units_csv, index_col="unit")
    for col, spec in criteria.items():
        if "source" in spec:
            src = sources[spec["source"]]
            vals = [src.get(norm(m), {}).get(spec["key"]) for m in t.index]
            t[col] = pd.to_numeric(pd.Series(vals, index=t.index), errors="coerce") * spec.get("sign", 1)
        elif col not in t.columns:
            raise ValueError(f"FATAL: criterion '{col}' not in {units_csv} and no source given")
    return t[list(criteria)]


def curves(facets, crit, weights, n_boot, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for col in crit.columns:
        d = facets.join(crit[col], how="inner").dropna()
        zN, zA, y = [((d[c] - d[c].mean()) / d[c].std()).to_numpy() for c in ("drat_N", "drat_A")] + [d[col].to_numpy()]
        comp = np.outer(weights, zN) + np.outer(1 - weights, zA)  # w x units
        rho = [stats.spearmanr(c, y)[0] for c in comp]
        boot = np.empty((n_boot, len(weights)))
        for b in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            boot[b] = [stats.spearmanr(c[i], y[i])[0] for c in comp]
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
        rows += [{"criterion": col, "n": len(d), "w_N": round(float(w), 2), "rho": r, "lo": l, "hi": h}
                 for w, r, l, h in zip(weights, rho, lo, hi)]
    return pd.DataFrame(rows)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("drat_sets", "human_units", "llm_units", "benchmarks", "kombine_composite", "panels", "ncols",
                "weight_step", "n_bootstrap", "seed", "output_dir"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()

    facets, n_zero = drat_facets(config["drat_sets"])
    sources = {"benchmarks": {norm(m): v for m, v in json.loads(Path(config["benchmarks"]).read_text()).items()},
               "kombine": {norm(m): v for m, v in
                           json.loads(Path(config["kombine_composite"]).read_text())["per_model"].items()}}
    weights = np.round(np.arange(0, 1 + 1e-9, config["weight_step"]), 3)
    n_boot = 100 if debug else config["n_bootstrap"]

    results, tables = {"n_sets_dropped_no_words": n_zero}, []
    for panel in config["panels"]:
        pop = panel["population"]
        crit = criterion_table(config[f"{pop}_units"], sources, panel["criteria"])
        f = facets.loc[pop]
        c = curves(f, crit, weights, n_boot, config["seed"]).assign(panel=panel["title"])
        tables.append(c)
        rN_A = stats.spearmanr(f["drat_N"], f["drat_A"])[0]
        results[panel["title"]] = {"n_units": len(f), "rho_N_A": round(float(rN_A), 3), "best_w": {
            k: {"w_N": float(g.loc[g["rho"].abs().idxmax(), "w_N"]), "rho": round(float(g.loc[g["rho"].abs().idxmax(), "rho"]), 3)}
            for k, g in c.groupby("criterion")}}
    curves_df = pd.concat(tables)
    curves_df.to_csv(out / "curves.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    set_camera_ready_style()
    ncols = config["ncols"]
    nrows = -(-len(config["panels"]) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 7.0 * nrows), sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.75})
    axes = axes.ravel()
    for ax in axes[len(config["panels"]):]:
        ax.set_visible(False)
    for ax, panel in zip(axes, config["panels"]):
        c = curves_df[curves_df["panel"] == panel["title"]]
        ax.axhline(0, color="#999999", lw=0.8, zorder=1)
        for col, spec in panel["criteria"].items():
            g = c[c["criterion"] == col]
            ax.fill_between(g["w_N"], g["lo"], g["hi"], color=spec["color"], alpha=0.12, lw=0, zorder=1)
            ax.plot(g["w_N"], g["rho"], color=spec["color"], lw=2, zorder=3,
                    label=f"{spec['label']} (n = {g['n'].iloc[0]})")
        ax.set_title(panel["title"])
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel("Weight on DRAT-N\n(0 = DRAT-A only, 1 = DRAT-N only)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False, fontsize=11)
    for r in range(nrows):
        axes[r * ncols].set_ylabel("Spearman ρ with criterion")
    save(fig, out / "figures" / "fig_drat_weight_curves")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
