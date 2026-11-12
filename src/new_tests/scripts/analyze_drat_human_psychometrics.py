"""Psychometric properties of the human DRAT next to the DAT, RAT and SCTT (main run).

For each person-level measure: distribution, internal-consistency reliability (bootstrap CI over
participants), correlations with the other tests (raw and disattenuated), and correlates
(Raven's 2, Big Five, demographics, block order, fatigue), zero-order and in one regression.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.drat_human import (cronbach_alpha, heatmap, load_glove_subset, load_tables,
                                      mean_pairwise_distance, save, set_camera_ready_style,
                                      spearman_brown)
from src.utils import init_directory, load_config, save_config

MEASURES = {
    "drat_score": "DRAT score",
    "drat_relevance": "DRAT relevance",
    "drat_ungated": "DRAT ungated dist.",
    "dat": "DAT",
    "rat": "RAT",
    "sctt_length": "SCTT length",
}
BLOCK = {"drat_score": "B", "drat_relevance": "B", "drat_ungated": "B", "dat": "A", "rat": "A",
         "sctt_length": "B"}
FACTORS = {
    "raven_standard_score": "Raven's",
    "bfi_openness": "Openness",
    "bfi_conscientiousness": "Consc.",
    "bfi_extraversion": "Extrav.",
    "bfi_agreeableness": "Agree.",
    "bfi_neuroticism": "Neurot.",
    "age_years": "Age",
    "female": "Female",
    "english_native": "Native Eng.",
    "done_first": "Block first",
    "block_b_duration_min": "Block B min.",
}


# ---------------------------------------------------------------- item matrices (person x item)

def drat_cells(s, col):
    """Person x (relation, k) matrix: every participant has exactly one set per cell."""
    return s.pivot_table(index="pid", columns=["relation", "k"], values=col)


def rat_items(r):
    d = r[r["test"] == "rat"].assign(correct=lambda x: (x["correct"].astype(str) == "True").astype(float))
    return d.pivot_table(index="pid", columns="item_id", values="correct")


def sctt_items(r):
    d = r[(r["test"] == "sctt") & (r["in_9_item_form"].astype(str) == "True")]
    return d.pivot_table(index="pid", columns="item_id", values="n_words", aggfunc="mean")


def dat_halves(r, glove_path):
    """DAT score from odd vs even boxes (GloVe), plus the score over all valid words."""
    d = r[r["test"] == "dat_word"].dropna(subset=["response_clean"]).copy()
    d["w"] = d["response_clean"].str.replace("-", "", regex=False).str.strip()
    d = d[~d["w"].str.contains(" ")]
    vecs = load_glove_subset(glove_path, d["w"].unique())
    d = d[d["w"].isin(vecs)].drop_duplicates(["pid", "w"])
    rows = {}
    for pid, g in d.groupby("pid"):
        v = [vecs[w] for w in g["w"]]
        odd = [vecs[w] for w, i in zip(g["w"], g["response_index"]) if i % 2 == 1]
        even = [vecs[w] for w, i in zip(g["w"], g["response_index"]) if i % 2 == 0]
        rows[pid] = {"odd": mean_pairwise_distance(odd), "even": mean_pairwise_distance(even),
                     "all": mean_pairwise_distance(v), "n_valid": len(v)}
    return pd.DataFrame(rows).T, int(d["w"].nunique()), len(vecs)


# ---------------------------------------------------------------- reliability

def split_half_r(a, b):
    ok = a.notna() & b.notna()
    return spearman_brown(np.corrcoef(a[ok], b[ok])[0, 1])


def drat_score_halves(s, pids):
    """DRAT score from odd vs even presentation positions (scorable sets only)."""
    h = s[s["pid"].isin(pids)].assign(half=lambda d: d["presentation_index"] % 2)
    h = h.groupby(["pid", "half"])["composite_score"].mean().unstack()
    return h.reindex(pids)


def reliabilities(s, r, dat_split, n_boot, seed, sctt_judged=None):
    """Point estimate + percentile bootstrap CI (resampling participants) for each measure."""
    mats = {
        "drat_relevance": drat_cells(s, "glove_n_surv"),
        "drat_ungated": drat_cells(s, "glove_dist_all"),
        "rat": rat_items(r),
        "sctt_length": sctt_items(r),
    }
    if sctt_judged is not None:
        mats["sctt_creativity"] = sctt_judged
    estimators = {k: (lambda m: (lambda idx: cronbach_alpha(m.loc[idx])))(m) for k, m in mats.items()}
    pids_s = np.array(sorted(s["pid"].unique()))
    halves = drat_score_halves(s, pids_s)
    estimators["drat_score"] = lambda idx: split_half_r(halves.loc[idx, 0], halves.loc[idx, 1])
    estimators["dat"] = lambda idx: split_half_r(dat_split.loc[idx, "odd"], dat_split.loc[idx, "even"])
    method = {"drat_relevance": "alpha over 8 relation x k cells", "drat_ungated": "alpha over 8 cells",
              "rat": "KR-20 over 30 items", "sctt_length": "alpha over 9 scenarios",
              "sctt_creativity": "alpha over 9 scenarios (LLM-judge 1-5 creativity)",
              "drat_score": "split-half (odd/even set), Spearman-Brown",
              "dat": "split-half (odd/even box), Spearman-Brown"}
    population = {"drat_relevance": mats["drat_relevance"].dropna().index,
                  "drat_ungated": mats["drat_ungated"].dropna().index,
                  "rat": mats["rat"].dropna().index, "sctt_length": mats["sctt_length"].dropna().index,
                  **({"sctt_creativity": mats["sctt_creativity"].dropna().index} if sctt_judged is not None else {}),
                  "drat_score": halves.dropna().index, "dat": dat_split.dropna(subset=["odd", "even"]).index}
    rng = np.random.default_rng(seed)
    out = {}
    for key, est in estimators.items():
        idx = np.array(population[key])
        boots = [est(rng.choice(idx, len(idx), replace=True)) for _ in range(n_boot)]
        out[key] = {"reliability": round(float(est(idx)), 3), "ci95": [round(float(q), 3) for q in np.nanpercentile(boots, [2.5, 97.5])],
                    "n": int(len(idx)), "method": method[key]}
    return out


# ---------------------------------------------------------------- person table

def load_sctt_judged(scores_dir):
    """LLM-judge SCTT creativity: person score + person x item matrix over the nine common items."""
    scores_dir = Path(scores_dir)
    person = pd.read_csv(scores_dir / "person_scores.csv", index_col="pid")["sctt_creativity"]
    resp = pd.read_csv(scores_dir / "response_scores.csv")
    items = resp.pivot_table(index="pid", columns="item_id", values="score", aggfunc="mean")
    items = items[[c for c in items.columns if c not in ("sctt_rq_1", "sctt_hyp_1", "sctt_exp_1")]]
    return person, items


def person_table(p, s, r, min_sets):
    g = s.groupby("pid")
    t = pd.DataFrame({
        "drat_score": g["composite_score"].mean(),
        "drat_scorable": g["scorable"].sum(),
        "drat_relevance": g["glove_n_surv"].mean(),
        "drat_ungated": g["glove_dist_all"].mean(),
    })
    t.loc[t["drat_scorable"] < min_sets, "drat_score"] = np.nan
    pp = p.set_index("pid")
    t["dat"] = pp["dat_composite"]
    t["rat"] = pp["rat_correct"]
    t["sctt_length"] = sctt_items(r).mean(axis=1)
    for col in ("raven_standard_score", "bfi_openness", "bfi_conscientiousness", "bfi_extraversion",
                "bfi_agreeableness", "bfi_neuroticism", "age_years", "block_b_duration_min", "block_order"):
        t[col] = pp[col]
    t["female"] = pp["gender"].map({"Female": 1.0, "Male": 0.0})
    t["english_native"] = pp["english_native"].map({"Yes": 1.0, "No": 0.0})
    return t


def distribution(t, key):
    x = t[key].dropna()
    floor_ceiling = {"rat": (0, 30)}.get(key)
    d = {"n": int(len(x)), "mean": round(x.mean(), 2), "sd": round(x.std(), 2), "min": round(x.min(), 2),
         "max": round(x.max(), 2), "skew": round(stats.skew(x), 2),
         "shapiro_p": float(stats.shapiro(x).pvalue)}
    if floor_ceiling:
        d["pct_floor"] = round(100 * (x == floor_ceiling[0]).mean(), 1)
        d["pct_ceiling"] = round(100 * (x == floor_ceiling[1]).mean(), 1)
    return d


def inter_test(t, rel):
    keys = list(MEASURES) + ["raven_standard_score"]
    rows = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            d = t[[a, b]].dropna()
            rho, p = stats.spearmanr(d[a], d[b])
            r = d[a].corr(d[b])
            ra, rb = rel.get(a, {}).get("reliability"), rel.get(b, {}).get("reliability")
            dis = r / np.sqrt(ra * rb) if ra and rb and a != "raven_standard_score" else np.nan
            rows.append({"a": a, "b": b, "n": len(d), "spearman": rho, "p": p, "pearson": r,
                         "disattenuated": dis})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.round(4)


def correlates(t):
    rows = []
    for m in MEASURES:
        done_first = (t["block_order"] == ("BA" if BLOCK[m] == "B" else "AB")).astype(float)
        done_first[t["block_order"] == "B"] = np.nan
        tt = t.assign(done_first=done_first)
        for f in FACTORS:
            if f == m:
                continue
            d = tt[[m, f]].dropna()
            rho, p = stats.spearmanr(d[m], d[f])
            rows.append({"measure": m, "factor": f, "n": len(d), "rho": rho, "p": p})
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out.round(4)


def regressions(t):
    """z(measure) ~ z(Raven's) + z(openness) + female + native English + done-first (HC3 SEs)."""
    rows = []
    for m in MEASURES:
        d = t.assign(done_first=(t["block_order"] == ("BA" if BLOCK[m] == "B" else "AB")).astype(float))
        d = d[[m, "raven_standard_score", "bfi_openness", "female", "english_native", "done_first"]].dropna()
        for c in (m, "raven_standard_score", "bfi_openness"):
            d[c] = (d[c] - d[c].mean()) / d[c].std()
        fit = smf.ols(f"{m} ~ raven_standard_score + bfi_openness + female + english_native + done_first",
                      d).fit(cov_type="HC3")
        for term in fit.params.index[1:]:
            rows.append({"measure": m, "term": term, "n": int(fit.nobs), "b": fit.params[term],
                         "se": fit.bse[term], "p": fit.pvalues[term], "r2": fit.rsquared})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- figures

COLORS = {"drat_score": "#0072B2", "drat_relevance": "#56B4E9", "drat_ungated": "#009E73",
          "dat": "#E69F00", "rat": "#D55E00", "sctt_length": "#CC79A7",
          "sctt_creativity": "#882255"}


def plot_reliability(rel, path):
    keys = list(MEASURES)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    vals = [rel[k]["reliability"] for k in keys]
    lo = [rel[k]["reliability"] - rel[k]["ci95"][0] for k in keys]
    hi = [rel[k]["ci95"][1] - rel[k]["reliability"] for k in keys]
    ax.bar(range(len(keys)), vals, yerr=[lo, hi], capsize=4, color=[COLORS[k] for k in keys], width=0.65,
           error_kw={"lw": 1.2})
    ax.axhline(0.7, color="#666666", lw=1, ls="--")
    ax.set_xticks(range(len(keys)), [MEASURES[k].replace(" ", "\n", 1) for k in keys])
    ax.set_ylabel("Reliability")
    ax.set_ylim(0, 1)
    save(fig, path)


def plot_correlates(t, cor, inter, path):
    """Rows: measures. Columns: the other tests, then the factors. * = BH p < .05 within its table."""
    test_cols = [k for k in MEASURES if not k.startswith("drat_")]
    cols = test_cols + list(FACTORS)
    mat = np.full((len(MEASURES), len(cols)), np.nan)
    stars = [[""] * len(cols) for _ in MEASURES]
    for i, m in enumerate(MEASURES):
        for j, c in enumerate(cols):
            if c in test_cols:
                if c == m:
                    continue
                row = inter[((inter["a"] == m) & (inter["b"] == c)) | ((inter["a"] == c) & (inter["b"] == m))].iloc[0]
                mat[i, j], sig = row["spearman"], row["p_bh"] < 0.05
            else:
                row = cor[(cor["measure"] == m) & (cor["factor"] == c)].iloc[0]
                mat[i, j], sig = row["rho"], row["p_bh"] < 0.05
            stars[i][j] = "*" if sig else ""
    fig, ax = plt.subplots(figsize=(17, 5.2))
    im = heatmap(ax, mat, stars)
    ax.set_xticks(range(len(cols)), [MEASURES.get(c, FACTORS.get(c)) for c in cols], rotation=35, ha="right")
    ax.set_yticks(range(len(MEASURES)), list(MEASURES.values()))
    ax.axvline(len(test_cols) - 0.5, color="white", lw=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label(r"Spearman $\rho$")
    cbar.outline.set_visible(False)
    save(fig, path)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "output_dir", "sample", "apply_exclude", "min_sets_scored", "glove_path", "n_boot", "seed",
                "sctt_scores_dir"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    if not Path(config["glove_path"]).exists():
        raise FileNotFoundError(config["glove_path"])
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()

    p, s, r = load_tables(config["upstream_dir"], config["sample"], config["apply_exclude"])
    t = person_table(p, s, r, config["min_sets_scored"])
    sctt_judged = None
    if config["sctt_scores_dir"] is not None:  # null = no judged SCTT scores yet
        sctt_person, sctt_judged = load_sctt_judged(config["sctt_scores_dir"])
        t["sctt_creativity"] = sctt_person
        MEASURES["sctt_creativity"] = "SCTT creativity"
        BLOCK["sctt_creativity"] = "B"
    dat_split, n_dat_types, n_found = dat_halves(r, config["glove_path"])
    res = {"n_participants": int(len(t)),
           "dat_check": {"word_types": n_dat_types, "in_glove": n_found,
                         "r_all_words_vs_package_dat_glove": round(dat_split["all"].astype(float).corr(
                             p.set_index("pid")["dat_glove"]), 3)}}
    dat_split = dat_split.astype(float)

    res["distribution"] = {k: distribution(t, k) for k in MEASURES}
    res["reliability"] = reliabilities(s, r, dat_split, config["n_boot"], config["seed"], sctt_judged)
    inter = inter_test(t, res["reliability"])
    cor = correlates(t)
    reg = regressions(t)
    t.to_csv(out / "person_measures.csv")
    inter.to_csv(out / "inter_test.csv", index=False)
    cor.to_csv(out / "correlates.csv", index=False)
    reg.to_csv(out / "regressions.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))

    set_camera_ready_style()
    plot_reliability(res["reliability"], out / "figures" / "fig_reliability")
    plot_correlates(t, cor, inter, out / "figures" / "fig_correlates")
    print(json.dumps(res, indent=2, default=float))
    print(inter.to_string())
    print(cor[cor["p"] < 0.1].to_string())
    print(reg.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
