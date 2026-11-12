"""SCTT response diversity in humans vs LLMs: within-set, across-call (LLMs) and population level.

Humans: the 240 analysed participants of the 10 Nov 2026 package, nine common scenarios, 3 responses
each. LLMs: the 40-model human-battery run (same nine scenarios and wording), 3 independent calls per
scenario, each returning 3 responses. Scores are raw (no length adjustment). See
src/new_tests/sctt_diversity.py for the measures.
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
from src.new_tests.drat_human import load_tables, save, set_camera_ready_style
from src.new_tests.sctt_diversity import (bootstrap_between_unit, embed, item_alpha_and_split_half,
                                          load_human_sets, load_llm_sets, population_distance,
                                          score_pairs, within_and_across_pairs)
from src.utils import init_directory, load_config, save_config

REQUIRED = ("upstream_dir", "human_measures", "llm_responses_dir", "llm_model_table", "benchmarks", "items",
            "embedding_model", "classifier_threshold", "device", "n_bootstrap", "seed", "output_dir")
HUMAN_CRITERIA = {"drat_score": "DRAT score", "drat_ungated": "DRAT ungated", "drat_relevance": "DRAT relevance",
                  "dat": "DAT", "rat": "RAT", "sctt_creativity": "SCTT creativity", "ability": "General ability",
                  "bfi_openness": "BFI openness"}
LLM_CRITERIA = {k: v for k, v in HUMAN_CRITERIA.items() if k != "bfi_openness"} | {
    # the paper's output-diversity metrics first (NoveltyBench utility; Hivemind intra-similarity, higher = LESS
    # diverse), then the secondary ones with smaller coverage
    "noveltybench_utility": "NoveltyBench utility", "hivemind_intra_sim": "Hivemind intra-sim",
    "noveltybench_distinct": "NoveltyBench distinct", "hivemind_diversity": "Hivemind diversity"}
COLORS = {"human": "#2a78d6", "llm_within": "#eb6834", "llm_across": "#1baf7a"}  # dataviz reference slots 1-3


def score_population(df, label, config):
    vecs = embed(df["text"], config["embedding_model"], config["device"])
    pairs = score_pairs(within_and_across_pairs(df), df, vecs, config["device"], config["classifier_threshold"])
    item_level = pairs.groupby(["unit", "item_id", "kind"])[["sbert_dist", "distinct"]].mean().reset_index()
    pop_unit, pop_item = population_distance(df, vecs)
    ci = bootstrap_between_unit(df, vecs, config["n_bootstrap"], config["seed"])
    print(f"[{label}] {df['unit'].nunique()} units, {len(df)} responses, {len(pairs)} pairs scored")
    return pairs, item_level, pop_unit, pop_item.assign(population=label), ci


def unit_table(item_level, pop_unit):
    wide = item_level.pivot_table(index="unit", columns="kind", values=["sbert_dist", "distinct"], aggfunc="mean")
    wide.columns = [f"{kind}_{metric}" for metric, kind in wide.columns]
    wide["population_dist"] = pop_unit.groupby("unit")["population_dist"].mean()
    return wide


def reliability(item_level, pop_unit, kinds):
    out = {}
    for kind in kinds:
        for metric in ("sbert_dist", "distinct"):
            w = item_level[item_level["kind"] == kind].pivot(index="unit", columns="item_id", values=metric)
            out[f"{kind}_{metric}"] = item_alpha_and_split_half(w)
    out["population_dist"] = item_alpha_and_split_half(pop_unit.pivot(index="unit", columns="item_id",
                                                                       values="population_dist"))
    return out


def correlates(units, criteria, label):
    rows = []
    measures = [c for c in ("within_sbert_dist", "within_distinct", "across_sbert_dist", "across_distinct",
                            "population_dist") if c in units.columns]
    for measure in measures:
        for key, name in criteria.items():
            d = units[[measure, key]].dropna()
            rho, p = stats.spearmanr(d[measure], d[key])
            rows.append({"population": label, "measure": measure, "criterion": name, "n": len(d), "rho": rho, "p": p})
    t = pd.DataFrame(rows)
    ok = t["p"].notna()  # a measure with no variance (e.g. every pair distinct) has p = NaN; left out of BH
    t.loc[ok, "p_bh"] = multipletests(t.loc[ok, "p"], method="fdr_bh")[1]
    return t.round(4)


def human_vs_llm(h, l):
    out = {}
    for m in ("within_sbert_dist", "within_distinct", "population_dist"):
        u, p = stats.mannwhitneyu(h[m].dropna(), l[m].dropna(), alternative="two-sided")
        pct = [float(stats.percentileofscore(h[m].dropna(), v)) for v in l[m].dropna()]
        out[m] = {"human_mean": round(h[m].mean(), 4), "human_sd": round(h[m].std(), 4),
                  "llm_mean": round(l[m].mean(), 4), "llm_sd": round(l[m].std(), 4),
                  "mann_whitney_u": float(u), "p": float(p),
                  "rank_biserial": round(1 - 2 * u / (h[m].notna().sum() * l[m].notna().sum()), 3),
                  "llm_median_percentile_in_human_distribution": round(float(np.median(pct)), 1)}
    return out


def within_vs_across(l):
    out = {}
    for metric in ("sbert_dist", "distinct"):
        d = l[[f"within_{metric}", f"across_{metric}"]].dropna()
        w, p = stats.wilcoxon(d[f"within_{metric}"], d[f"across_{metric}"])
        rho, prho = stats.spearmanr(d[f"within_{metric}"], d[f"across_{metric}"])
        out[metric] = {"n": len(d), "within_mean": round(d.iloc[:, 0].mean(), 4),
                       "across_mean": round(d.iloc[:, 1].mean(), 4),
                       "share_models_across_below_within": round(float((d.iloc[:, 1] < d.iloc[:, 0]).mean()), 3),
                       "wilcoxon_p": float(p), "rho_within_across": round(rho, 3), "p_rho": float(prho)}
    return out


def plot(h, l, pop_items, path):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"wspace": 0.32, "width_ratios": [1.1, 1, 1.25]})
    rng = np.random.default_rng(0)

    ax = axes[0]
    groups = [("Humans\nwithin set", h["within_sbert_dist"], COLORS["human"]),
              ("LLMs\nwithin call", l["within_sbert_dist"], COLORS["llm_within"]),
              ("LLMs\nacross calls", l["across_sbert_dist"], COLORS["llm_across"])]
    for x, (name, vals, c) in enumerate(groups):
        v = vals.dropna().to_numpy()
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=22 if len(v) > 100 else 46, color=c, alpha=0.55,
                   edgecolors="white", linewidths=0.6, zorder=2)
        ax.hlines(np.median(v), x - 0.3, x + 0.3, color="#222222", linewidth=2, zorder=3)
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylabel("Mean SBERT distance between responses")
    ax.set_title("(a) Diversity of an individual's responses")

    ax = axes[1]
    ax.scatter(l["within_sbert_dist"], l["across_sbert_dist"], s=50, color=COLORS["llm_within"],
               edgecolors="white", linewidths=0.8, zorder=2)
    lo = np.nanmin(l[["within_sbert_dist", "across_sbert_dist"]].to_numpy()) - 0.02
    hi = np.nanmax(l[["within_sbert_dist", "across_sbert_dist"]].to_numpy()) + 0.02
    ax.plot([lo, hi], [lo, hi], color="#888888", linewidth=1, linestyle="--", zorder=1)
    mid = lo + 0.75 * (hi - lo)
    ax.text(mid, mid - 0.015, "y = x", color="#555555", va="top", ha="left", fontsize=12)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Within one call")
    ax.set_ylabel("Across separate calls")
    ax.set_title("(b) LLMs: within vs across calls")

    ax = axes[2]
    wide = pop_items.pivot(index="item_id", columns="population", values="mean_between_unit_dist")
    wide = wide.loc[wide["human"].sort_values().index]
    y = np.arange(len(wide))
    ax.hlines(y, wide["llm"], wide["human"], color="#cccccc", linewidth=2, zorder=1)
    ax.scatter(wide["human"], y, s=60, color=COLORS["human"], label="Humans (between people)", zorder=2,
               edgecolors="white", linewidths=0.8)
    ax.scatter(wide["llm"], y, s=60, color=COLORS["llm_within"], label="LLMs (between models)", zorder=2,
               edgecolors="white", linewidths=0.8)
    ax.set_yticks(y, [i.replace("sctt_", "").replace("_", " ") for i in wide.index])
    ax.set_xlabel("Mean SBERT distance between individuals")
    ax.set_title("(c) Population spread per scenario")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=12)
    save(fig, path)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in REQUIRED:
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()
    items = list(config["items"])

    p, _, _ = load_tables(config["upstream_dir"], "main", apply_exclude=True)
    hum_df, n_empty = load_human_sets(Path(config["upstream_dir"]) / "responses.csv", set(p["pid"]), items)
    lm = pd.read_csv(config["llm_model_table"], index_col="model")
    llm_df, call_log = load_llm_sets(config["llm_responses_dir"], items, set(lm.index), n_responses=3)
    if debug:
        hum_df = hum_df[hum_df["unit"].isin(sorted(hum_df["unit"].unique())[:20])].reset_index(drop=True)
        llm_df = llm_df[llm_df["unit"].isin(sorted(llm_df["unit"].unique())[:5])].reset_index(drop=True)
    call_log.to_csv(out / "llm_call_log.csv", index=False)
    n_dropped = int((call_log["action"] == "dropped").sum())
    n_truncated = int((call_log["action"] != "dropped").sum())
    print(f"humans: {hum_df['unit'].nunique()} people, {n_empty} empty responses dropped; "
          f"LLMs: {llm_df['unit'].nunique()} models, {n_dropped} calls dropped, {n_truncated} truncated to 3 "
          f"(see llm_call_log.csv)")

    h_pairs, h_items, h_pop, h_pop_items, h_ci = score_population(hum_df, "human", config)
    l_pairs, l_items, l_pop, l_pop_items, l_ci = score_population(llm_df, "llm", config)
    for name, t in (("human_pairs", h_pairs), ("llm_pairs", l_pairs), ("human_item_level", h_items),
                    ("llm_item_level", l_items)):
        t.to_csv(out / f"{name}.csv", index=False)

    hm = pd.read_csv(config["human_measures"], index_col="pid").rename(columns={"raven_standard_score": "ability"})
    bench = json.load(open(config["benchmarks"]))
    for k in ("noveltybench_utility", "hivemind_intra_sim", "noveltybench_distinct", "hivemind_diversity"):
        vals = [bench.get(m.replace("/", "_").replace(".", "-"), {}).get(k) for m in lm.index]
        lm[k] = np.array([np.nan if v is None else v for v in vals], dtype=float)  # json null = not run
    h_units = unit_table(h_items, h_pop).join(hm[list(HUMAN_CRITERIA)])
    l_units = unit_table(l_items, l_pop).join(lm[list(LLM_CRITERIA)])
    h_units.to_csv(out / "human_units.csv")
    l_units.to_csv(out / "llm_units.csv")

    corr = pd.concat([correlates(h_units, HUMAN_CRITERIA, "human"), correlates(l_units, LLM_CRITERIA, "llm")])
    corr.to_csv(out / "correlates.csv", index=False)
    pop_items = pd.concat([h_pop_items, l_pop_items])
    pop_items.to_csv(out / "population_by_item.csv", index=False)

    res = {
        "n_human": int(h_units.shape[0]), "n_llm": int(l_units.shape[0]),
        "human_empty_responses_dropped": n_empty, "llm_calls_dropped": n_dropped, "llm_calls_truncated": n_truncated,
        "llm_calls_used": int(llm_df.groupby(["unit", "item_id", "call"]).ngroups),
        "embedding_model": config["embedding_model"],
        "reliability": {"human": reliability(h_items, h_pop, ["within"]),
                        "llm": reliability(l_items, l_pop, ["within", "across"])},
        "human_vs_llm": human_vs_llm(h_units, l_units),
        "llm_within_vs_across": within_vs_across(l_units),
        "population_between_unit_dist": {
            "human": {"mean": round(h_pop_items["mean_between_unit_dist"].mean(), 4), "ci95": h_ci},
            "llm": {"mean": round(l_pop_items["mean_between_unit_dist"].mean(), 4), "ci95": l_ci}},
        "human_within_vs_population_rho": round(h_units[["within_sbert_dist", "population_dist"]]
                                                .corr("spearman").iloc[0, 1], 3),
    }
    (out / "results.json").write_text(json.dumps(res, indent=2))
    set_camera_ready_style()
    plot(h_units, l_units, pop_items, out / "figures" / "fig_sctt_diversity")
    print(json.dumps(res, indent=2))
    sig = corr[corr["p"] < 0.05]
    print(sig.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
