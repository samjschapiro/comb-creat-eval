"""Relevance-gate comparison for the human DRAT: max (any anchor) vs mean vs min (every anchor).

Re-scores every set in GloVe under each gate, with each gate's threshold calibrated the package's
way (tau = 90th percentile of the same aggregate over the noun pool), using the exact word forms
the package looked up (`glove_form`). The max gate must reproduce the package's `glove_score`;
the script fails if it does not. Then compares, per gate: scorable sets, survivors, person-level
split-half reliability and correlations with the DAT, RAT, Raven's, SCTT length and openness.
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
from src.new_tests.drat_human import (RELATIONS, load_glove_subset, load_tables, mean_pairwise_distance,
                                      save, set_camera_ready_style, spearman_brown)
from src.utils import init_directory, load_config, save_config

GATES = ["max", "mean", "min"]
GATE_LABELS = {"max": "any anchor (max)", "mean": "average (mean)", "min": "every anchor (min)"}
GATE_COLORS = {"max": "#0072B2", "mean": "#009E73", "min": "#D55E00"}
CRITERIA = {"dat_composite": "DAT", "rat_correct": "RAT", "raven_standard_score": "Raven's",
            "sctt9_words_mean": "SCTT length", "bfi_openness": "Openness"}


def aggregate(sims, gate):
    return {"max": sims.max(axis=-1), "mean": sims.mean(axis=-1), "min": sims.min(axis=-1)}[gate]


def score_sets(s, r, vecs, pool, q, n_min):
    """One row per set and gate: tau, n_valid, n_surv, score (NaN if < n_min survivors)."""
    # Duplicates within a set still carry a glove_form but are excluded from scoring by the package.
    w = r[(r["test"] == "drat_word") & r["glove_form"].notna() & (r["duplicate"].astype(str) != "True")]
    words_by_set = w.groupby(["pid", "item_id"])["glove_form"].apply(list)
    pool_m = np.vstack([vecs[p] for p in pool])
    rows, tau_cache = [], {}
    for _, row in s.iterrows():
        anchors = row["anchors"].split("|")
        if row["anchors"] not in tau_cache:
            a = np.vstack([vecs[x] for x in anchors])
            pool_sims = pool_m @ a.T
            tau_cache[row["anchors"]] = (a, {g: float(np.quantile(aggregate(pool_sims, g), q)) for g in GATES})
        a, taus = tau_cache[row["anchors"]]
        words = words_by_set.get((row["pid"], row["set_id"]), [])
        wv = [vecs[x] for x in words]
        sims = np.vstack(wv) @ a.T if wv else np.zeros((0, len(anchors)))
        for g in GATES:
            keep = aggregate(sims, g) > taus[g] if len(wv) else np.array([], dtype=bool)
            surv = [v for v, k in zip(wv, keep) if k]
            rows.append({"pid": row["pid"], "set_id": row["set_id"], "relation": row["relation"], "k": row["k"],
                         "presentation_index": row["presentation_index"], "gate": g, "tau": taus[g],
                         "n_valid": len(wv), "n_surv": len(surv),
                         "score": mean_pairwise_distance(surv) if len(surv) >= n_min else np.nan,
                         "package_glove_score": row["glove_score"], "package_glove_n_surv": row["glove_n_surv"]})
    return pd.DataFrame(rows)


def check_reproduction(d):
    m = d[d["gate"] == "max"]
    same_surv = (m["n_surv"] == m["package_glove_n_surv"]).mean()
    both = m[m["score"].notna() | m["package_glove_score"].notna()]
    max_diff = (both["score"] - both["package_glove_score"]).abs().max()
    nan_mismatch = int((both["score"].isna() != both["package_glove_score"].isna()).sum())
    out = {"share_sets_same_n_surv": round(float(same_surv), 4), "max_abs_score_diff": float(max_diff),
           "scorable_mismatches": nan_mismatch}
    if same_surv < 0.99 or nan_mismatch > 0.01 * len(m) or max_diff > 0.5:
        raise ValueError(f"FATAL: max gate does not reproduce the package's GloVe scores: {out}")
    return out


def person_level(d, p, min_sets):
    rows = {}
    for g in GATES:
        dg = d[d["gate"] == g]
        sc = dg.groupby("pid")["score"].agg(["mean", "count"])
        rows[f"score_{g}"] = sc["mean"].where(sc["count"] >= min_sets)
        rows[f"surv_{g}"] = dg.groupby("pid")["n_surv"].mean()
        rows[f"zero_{g}"] = dg.assign(z=dg["score"].fillna(0)).groupby("pid")["z"].mean()
    t = pd.DataFrame(rows)
    return t.join(p.set_index("pid")[list(CRITERIA) + ["drat_n_typed_mean"]])


def split_half(d, g):
    dg = d[d["gate"] == g].assign(half=lambda x: x["presentation_index"] % 2)
    h = dg.groupby(["pid", "half"])["score"].mean().unstack().dropna()
    return {"n": int(len(h)), "spearman_brown": round(spearman_brown(h[0].corr(h[1])), 3)}


def gate_summary(d, t):
    out = {}
    for g in GATES:
        dg = d[d["gate"] == g]
        cors = {}
        for crit, lab in CRITERIA.items():
            for kind in ("score", "zero", "surv"):
                x = t[[f"{kind}_{g}", crit]].dropna()
                rho, pval = stats.spearmanr(x.iloc[:, 0], x.iloc[:, 1])
                cors[f"{kind}~{lab}"] = {"n": int(len(x)), "rho": round(rho, 3), "p": float(pval)}
        out[g] = {
            "pct_sets_scorable": round(100 * dg["score"].notna().mean(), 1),
            "pct_scorable_by_cell": {f"{rel}_k{k}": round(100 * v, 1) for (rel, k), v in
                                     dg.groupby(["relation", "k"])["score"].apply(lambda x: x.notna().mean()).items()},
            "mean_survivors": round(dg["n_surv"].mean(), 2),
            "n_persons_scored": int(t[f"score_{g}"].notna().sum()),
            "split_half": split_half(d, g),
            "score_vs_words_typed": round(stats.spearmanr(*t[[f"score_{g}", "drat_n_typed_mean"]].dropna().values.T)[0], 3),
            "correlations": cors,
        }
    return out


def plot_gates(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    cells = [f"{rel}_k{k}" for rel in RELATIONS for k in (2, 3)]
    x = np.arange(len(cells))
    for i, g in enumerate(GATES):
        axes[0].bar(x + (i - 1) * 0.27, [summary[g]["pct_scorable_by_cell"][c] for c in cells], width=0.27,
                    color=GATE_COLORS[g], label=GATE_LABELS[g])
    axes[0].set_xticks(x, [c.replace("_k", "\nk=").replace("categorical", "categ.").replace("constitutive", "const.")
                            .replace("unrelated", "unrel.") for c in cells], fontsize=12)
    axes[0].set_ylabel("% of sets scorable")
    axes[0].set_title("(a) Scorable sets")
    crits = list(CRITERIA.values())
    x = np.arange(len(crits))
    for i, g in enumerate(GATES):
        vals = [summary[g]["correlations"][f"score~{c}"]["rho"] for c in crits]
        axes[1].bar(x + (i - 1) * 0.27, vals, width=0.27, color=GATE_COLORS[g])
    axes[1].axhline(0, color="#333333", lw=0.8)
    axes[1].set_xticks(x, crits, fontsize=13)
    axes[1].set_ylabel(r"Spearman $\rho$ with DRAT score")
    axes[1].set_title("(b) Correlations")
    fig.tight_layout(w_pad=2.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
    save(fig, path)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "output_dir", "sample", "apply_exclude", "glove_path", "q", "n_min", "min_sets_scored"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()

    p, s, r = load_tables(config["upstream_dir"], config["sample"], config["apply_exclude"])
    pool = (Path(config["upstream_dir"]) / "stimuli" / "drat_noun_pool_v1.txt").read_text().split()
    anchors = {a for x in s["anchors"] for a in x.split("|")}
    forms = set(r.loc[r["test"] == "drat_word", "glove_form"].dropna())
    vecs = load_glove_subset(config["glove_path"], set(pool) | anchors | forms)
    missing = (set(pool) | anchors | forms) - set(vecs)
    if missing:
        raise ValueError(f"FATAL: {len(missing)} words have no GloVe vector, e.g. {sorted(missing)[:10]}")

    d = score_sets(s, r, vecs, pool, config["q"], config["n_min"])
    d.to_csv(out / "set_scores_by_gate.csv", index=False)
    res = {"n_participants": int(s["pid"].nunique()), "n_sets": int(len(s)), "reproduction": check_reproduction(d)}
    t = person_level(d, p, config["min_sets_scored"])
    t.to_csv(out / "person_scores_by_gate.csv")
    res["gates"] = gate_summary(d, t)
    res["max_vs_min_person_score_rho"] = round(t[["score_max", "score_min"]].corr("spearman").iloc[0, 1], 3)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))

    set_camera_ready_style()
    plot_gates(res["gates"], out / "figures" / "fig_gate_comparison")
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
