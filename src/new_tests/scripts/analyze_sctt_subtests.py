"""SCTT creativity broken down by subtest (research questions, hypotheses, experiments).

Downstream of score_sctt_judging.py and analyze_drat_human_psychometrics.py: correlates each
subtest's person score with the DRAT facets, the other tests and the person factors, raw and with
that subtest's own response length partialled out (Spearman, BH within each panel), and reports
each subtest's reliability (alpha over its items in the common nine).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.drat_human import cronbach_alpha, heatmap, save, set_camera_ready_style
from src.utils import init_directory, load_config, save_config

SUBTESTS = {"all": "SCTT (all)", "rq": "Research questions", "hyp": "Hypotheses", "exp": "Experiments"}
TASK = {"rq": "research question", "hyp": "hypothesis", "exp": "experiment"}
COLUMNS = {
    "drat_score": "DRAT score", "drat_relevance": "DRAT relevance", "drat_ungated": "DRAT ungated",
    "dat": "DAT", "rat": "RAT", "raven_standard_score": "Raven's", "bfi_openness": "Openness",
    "english_native": "Native Eng.", "female": "Female",
}
COMMON_NINE_EXCLUDED = {"sctt_rq_1", "sctt_hyp_1", "sctt_exp_1"}


def subtest_tables(resp):
    """Person x subtest creativity and length (mean over the common-nine items of that subtest)."""
    resp = resp[~resp["item_id"].isin(COMMON_NINE_EXCLUDED)]
    item = resp.groupby(["pid", "item_id", "task"]).agg(score=("score", "mean"), words=("n_words", "mean")).reset_index()
    creat, length, alpha = {}, {}, {}
    for key, task in TASK.items():
        d = item[item["task"] == task]
        creat[key] = d.groupby("pid")["score"].mean()
        length[key] = d.groupby("pid")["words"].mean()
        alpha[key] = round(cronbach_alpha(d.pivot(index="pid", columns="item_id", values="score")), 3)
    creat["all"] = item.groupby("pid")["score"].mean()
    length["all"] = item.groupby("pid")["words"].mean()
    alpha["all"] = round(cronbach_alpha(item.pivot(index="pid", columns="item_id", values="score")), 3)
    return pd.DataFrame(creat), pd.DataFrame(length), alpha


def partial_spearman(x, y, z):
    d = pd.concat([x, y, z], axis=1).dropna().rank()
    d.columns = ["x", "y", "z"]
    rx = d["x"] - np.polyval(np.polyfit(d["z"], d["x"], 1), d["z"])
    ry = d["y"] - np.polyval(np.polyfit(d["z"], d["y"], 1), d["z"])
    rho, p = stats.pearsonr(rx, ry)
    return len(d), rho, p


def correlation_table(creat, length, persons):
    rows = []
    for sub in SUBTESTS:
        for col in COLUMNS:
            d = pd.concat([creat[sub], persons[col]], axis=1).dropna()
            rho, p = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
            n_p, rho_p, p_p = partial_spearman(creat[sub], persons[col], length[sub])
            rows.append({"subtest": sub, "measure": col, "n": len(d), "rho": rho, "p": p,
                         "n_partial": n_p, "rho_length_partialled": rho_p, "p_length_partialled": p_p})
    t = pd.DataFrame(rows)
    t["p_bh"] = multipletests(t["p"], method="fdr_bh")[1]
    t["p_length_partialled_bh"] = multipletests(t["p_length_partialled"], method="fdr_bh")[1]
    return t.round(4)


def plot(t, path):
    fig, axes = plt.subplots(1, 2, figsize=(19, 4.4), sharey=True)
    panels = [("rho", "p_bh", "(a) Raw"), ("rho_length_partialled", "p_length_partialled_bh", "(b) Length held constant")]
    for ax, (val, sig, title) in zip(axes, panels):
        mat = t.pivot(index="subtest", columns="measure", values=val).loc[list(SUBTESTS), list(COLUMNS)]
        ps = t.pivot(index="subtest", columns="measure", values=sig).loc[list(SUBTESTS), list(COLUMNS)]
        stars = [["*" if v < 0.05 else "" for v in row] for row in ps.values]
        im = heatmap(ax, mat.values, stars)
        ax.set_xticks(range(len(COLUMNS)), list(COLUMNS.values()), rotation=35, ha="right")
        ax.set_yticks(range(len(SUBTESTS)), list(SUBTESTS.values()))
        ax.axhline(0.5, color="white", lw=4)
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label(r"Spearman $\rho$")
    cbar.outline.set_visible(False)
    save(fig, path)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("sctt_scores_dir", "psychometrics_dir", "output_dir"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()

    resp = pd.read_csv(Path(config["sctt_scores_dir"]) / "response_scores.csv")
    persons = pd.read_csv(Path(config["psychometrics_dir"]) / "person_measures.csv", index_col="pid")
    creat, length, alpha = subtest_tables(resp)
    t = correlation_table(creat, length, persons)
    t.to_csv(out / "subtest_correlations.csv", index=False)
    res = {"n_persons": int(len(creat)), "alpha": alpha,
           "creativity_vs_own_length": {s: round(stats.spearmanr(creat[s], length[s])[0], 3) for s in SUBTESTS},
           "subtest_intercorrelations": creat[list(TASK)].corr("spearman").round(3).to_dict()}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    set_camera_ready_style()
    plot(t, out / "figures" / "fig_sctt_subtests")
    print(json.dumps(res, indent=2))
    print(t.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
