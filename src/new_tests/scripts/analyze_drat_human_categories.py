"""DRAT performance as a function of anchor category (relation type x k), main run.

Asks (1) how each DRAT facet differs across the four relation types and two anchor counts,
(2) whether relation type matters beyond graded anchor similarity (likelihood-ratio tests on
ML-fitted crossed mixed models), (3) whether the pattern holds in all three embedding models,
and (4) whether the category changes what the DRAT correlates with at the person level.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.drat_human import (RELATIONS, crossed_mixed, heatmap, load_tables, save,
                                      set_camera_ready_style)
from src.utils import init_directory, load_config, save_config

OUTCOMES = {
    "n_words": "words typed",
    "glove_n_surv": "survivors (GloVe)",
    "n_allanchor": "survivors related to every anchor",
    "scorable_f": "set scorable (0/1)",
    "composite_score": "DRAT score (scorable sets)",
    "glove_dist_all": "ungated distance",
    "difficulty": "rated difficulty",
}
REL_COLORS = {"categorical": "#0072B2", "constitutive": "#009E73", "causal": "#CC79A7", "unrelated": "#D55E00"}
CRITERIA = {"dat_composite": "DAT", "rat_correct": "RAT", "raven_standard_score": "Raven's"}


def add_allanchor_counts(s, r):
    """Per set: GloVe survivors whose MIN similarity to the anchors also clears tau."""
    w = r[(r["test"] == "drat_word") & r["glove_survives"].notna()].copy()
    w["hit"] = (w["glove_survives"].astype(str) == "True") & (w["glove_u_min"] > w["glove_tau"])
    counts = w.groupby(["pid", "item_id"])["hit"].sum().rename("n_allanchor")
    s = s.merge(counts, left_on=["pid", "set_id"], right_index=True, how="left")
    return s.assign(n_allanchor=s["n_allanchor"].fillna(0), scorable_f=s["scorable"].astype(float))


def lrt(s, outcome, reduced, full):
    f0, _ = crossed_mixed(s, outcome, reduced, reml=False)
    f1, _ = crossed_mixed(s, outcome, full, reml=False)
    stat = 2 * (f1.llf - f0.llf)
    df = len(f1.fe_params) - len(f0.fe_params)
    return {"chi2": round(stat, 2), "df": int(df), "p": float(stats.chi2.sf(stat, df))}


def category_tests(s, outcome):
    rel = "C(relation)"
    return {
        "k_given_relation": lrt(s, outcome, rel, f"{rel} + C(k)"),
        "relation_given_k": lrt(s, outcome, "C(k)", f"{rel} + C(k)"),
        "relation_x_k": lrt(s, outcome, f"{rel} + C(k)", f"{rel} * C(k)"),
        "cosine_given_k": lrt(s, outcome, "C(k)", "anchor_cosine * C(k)"),
        "relation_beyond_cosine": lrt(s, outcome, "anchor_cosine * C(k)", f"anchor_cosine * C(k) + {rel} * C(k)"),
    }


def pairwise_relations(s, outcome):
    """Relation contrasts averaged over k from relation + k (REML), Holm-adjusted."""
    fit, _ = crossed_mixed(s, outcome, "C(relation, Treatment('categorical')) + C(k)")
    names = list(fit.fe_params.index)
    coef = {r: f"C(relation, Treatment('categorical'))[T.{r}]" for r in RELATIONS[1:]}
    rows = []
    for a, b in itertools.combinations(RELATIONS, 2):
        c = np.zeros(len(names))
        if b in coef:
            c[names.index(coef[b])] += 1
        if a in coef:
            c[names.index(coef[a])] -= 1
        tt = fit.t_test(c.reshape(1, -1))
        rows.append({"outcome": outcome, "contrast": f"{b} - {a}", "diff": float(tt.effect[0]),
                     "se": float(tt.sd[0, 0]), "p": float(tt.pvalue)})
    out = pd.DataFrame(rows)
    out["p_holm"] = multipletests(out["p"], method="holm")[1]
    return out


def embedding_robustness(s):
    rows = []
    for model in ("glove", "fasttext", "sbert"):
        for facet in ("n_surv", "score"):
            col = f"{model}_{facet}"
            m = s.groupby(["relation", "k"])[col].mean().unstack().reindex(RELATIONS)
            test = lrt(s, col, "C(k)", "C(relation) + C(k)")
            for rel in RELATIONS:
                rows.append({"model": model, "facet": facet, "relation": rel, "k2": m.loc[rel, 2],
                             "k3": m.loc[rel, 3], "relation_lrt_p": test["p"]})
    return pd.DataFrame(rows).round(4)


def category_validity(s, p):
    """Person-level facet per relation type (both k pooled) vs DAT / RAT / Raven's."""
    pp = p.set_index("pid")
    rows = []
    for rel in RELATIONS:
        g = s[s["relation"] == rel].groupby("pid")
        facets = {"ungated distance": g["glove_dist_all"].mean(), "survivors": g["glove_n_surv"].mean(),
                  "DRAT score": g["composite_score"].mean()}
        for fname, f in facets.items():
            for crit, label in CRITERIA.items():
                d = pd.concat([f, pp[crit]], axis=1).dropna()
                rho, pval = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
                rows.append({"relation": rel, "facet": fname, "criterion": label, "n": len(d), "rho": rho, "p": pval})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.round(4)


def plot_distance(s, path):
    """Per anchor set (both k): facet vs anchor similarity, coloured by relation type."""
    items = s.groupby(["anchors", "relation", "k"]).agg(
        cos=("anchor_cosine", "first"), surv=("glove_n_surv", "mean"), diff=("difficulty", "mean"),
        share=("n_allanchor", "sum"), surv_sum=("glove_n_surv", "sum")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    panels = [("surv", "(a) Relevance", "Survivors per set"), ("diff", "(b) Rated difficulty", "Difficulty (1–7)")]
    for ax, (col, title, ylabel) in zip(axes, panels):
        for rel in RELATIONS:
            for k, marker in ((2, "o"), (3, "^")):
                d = items[(items["relation"] == rel) & (items["k"] == k)]
                ax.scatter(d["cos"], d[col], s=34, marker=marker, color=REL_COLORS[rel], alpha=0.75,
                           edgecolor="none")
        b = np.polyfit(items["cos"], items[col], 1)
        xs = np.linspace(items["cos"].min(), items["cos"].max(), 50)
        ax.plot(xs, np.polyval(b, xs), color="#333333", lw=1.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Anchor similarity (GloVe cosine)")
    handles = [plt.Line2D([], [], ls="", marker="s", ms=10, color=c, label=rel) for rel, c in REL_COLORS.items()]
    handles += [plt.Line2D([], [], ls="", marker=m, ms=9, color="#555555", label=f"$k = {k}$")
                for k, m in ((2, "o"), (3, "^"))]
    fig.tight_layout(w_pad=2.0)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, frameon=False,
               handletextpad=0.2, columnspacing=1.4)
    save(fig, path)


def plot_category_validity(cv, path):
    d = cv[cv["facet"] == "ungated distance"]
    mat = d.pivot(index="relation", columns="criterion", values="rho").loc[RELATIONS, list(CRITERIA.values())]
    sig = d.pivot(index="relation", columns="criterion", values="p_bh").loc[RELATIONS, list(CRITERIA.values())]
    stars = [["*" if v < 0.05 else "" for v in row] for row in sig.values]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = heatmap(ax, mat.values, stars)
    ax.set_xticks(range(mat.shape[1]), mat.columns)
    ax.set_yticks(range(mat.shape[0]), RELATIONS)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label(r"Spearman $\rho$")
    cbar.outline.set_visible(False)
    save(fig, path)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "output_dir", "sample", "apply_exclude"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()

    p, s, r = load_tables(config["upstream_dir"], config["sample"], config["apply_exclude"])
    s = add_allanchor_counts(s, r)

    cells = s.groupby(["relation", "k"])[list(OUTCOMES) + ["anchor_cosine"]].mean().reindex(RELATIONS, level=0)
    cells["allanchor_share"] = (s.groupby(["relation", "k"])["n_allanchor"].sum()
                                / s.groupby(["relation", "k"])["glove_n_surv"].sum()).reindex(cells.index)
    cells.round(3).to_csv(out / "cell_means.csv")

    res = {"n_participants": int(s["pid"].nunique()), "n_sets": int(len(s)),
           "n_anchor_sets": int(s["anchors"].nunique()), "tests": {}}
    for outcome in OUTCOMES:
        res["tests"][outcome] = category_tests(s, outcome)
    pd.concat([pairwise_relations(s, o) for o in OUTCOMES]).round(4).to_csv(out / "pairwise_relations.csv", index=False)
    embedding_robustness(s).to_csv(out / "embedding_robustness.csv", index=False)
    cv = category_validity(s, p)
    cv.to_csv(out / "category_validity.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))

    set_camera_ready_style()
    plot_distance(s, out / "figures" / "fig_anchor_similarity")
    plot_category_validity(cv, out / "figures" / "fig_category_validity")
    print(cells.round(3).to_string())
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
