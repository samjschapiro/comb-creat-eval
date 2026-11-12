"""Merge the LLM judge's SCTT ratings with the hidden key and build person-level creativity scores.

Downstream of prep_sctt_judging.py: reads <upstream_dir>/ratings/*.jsonl and key/uid_key.csv.
Fails if any uid is unrated or rated twice. Reports judge consistency on the hidden repeats and
writes per-response and per-person scores (mean over the 3 responses per item, then over items;
the nine common items are the primary score, all administered items a sensitivity score).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.drat_human import cronbach_alpha
from src.utils import init_directory, load_config, save_config

DROPPED_IN_V4 = {"sctt_rq_1", "sctt_hyp_1", "sctt_exp_1"}


def load_ratings(upstream):
    rows = []
    for f in sorted((upstream / "ratings").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    ratings = pd.DataFrame(rows)
    if ratings["uid"].duplicated().any():
        raise ValueError(f"FATAL: duplicated uids {ratings.loc[ratings['uid'].duplicated(), 'uid'].tolist()[:10]}")
    if not ratings["score"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("FATAL: scores outside 1-5")
    return ratings


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "output_dir"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    upstream = Path(config["upstream_dir"])
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)

    key = pd.read_csv(upstream / "key" / "uid_key.csv")
    ratings = load_ratings(upstream)
    missing = set(key["uid"]) - set(ratings["uid"])
    extra = set(ratings["uid"]) - set(key["uid"])
    if missing or extra:
        raise ValueError(f"FATAL: {len(missing)} unrated uids, {len(extra)} unknown uids")
    d = key.merge(ratings[["uid", "score"]], on="uid")
    d["n_words"] = d["response"].str.split().str.len()

    # Judge consistency: each hidden repeat against the original rating of the same response.
    orig = d[~d["is_repeat"]].set_index(["pid", "item_id", "response_index"])["score"]
    rep = d[d["is_repeat"]].set_index(["pid", "item_id", "response_index"])["score"]
    pair = pd.concat([orig.loc[rep.index].rename("first"), rep.rename("second")], axis=1)
    res = {
        "n_rated": int(len(d)),
        "n_unique": int((~d["is_repeat"]).sum()),
        "score_distribution": d.loc[~d["is_repeat"], "score"].value_counts().sort_index().to_dict(),
        "mean_by_task": d[~d["is_repeat"]].groupby("task")["score"].mean().round(3).to_dict(),
        "mean_by_item": d[~d["is_repeat"]].groupby("item_id")["score"].mean().round(3).to_dict(),
        "judge_repeat": {"n": int(len(pair)), "exact_agreement": round((pair["first"] == pair["second"]).mean(), 3),
                         "within_one": round(((pair["first"] - pair["second"]).abs() <= 1).mean(), 3),
                         "pearson": round(pair["first"].corr(pair["second"]), 3)},
    }
    d = d[~d["is_repeat"]]
    res["score_vs_words_spearman"] = round(stats.spearmanr(d["score"], d["n_words"])[0], 3)
    d.drop(columns=["is_repeat"]).to_csv(out / "response_scores.csv", index=False)

    item_means = d.pivot_table(index="pid", columns="item_id", values="score", aggfunc="mean")
    common = [c for c in item_means.columns if c not in DROPPED_IN_V4]
    person = pd.DataFrame({
        "sctt_creativity": item_means[common].mean(axis=1),
        "sctt_creativity_all_items": item_means.mean(axis=1),
        "sctt_creativity_max": d[d["item_id"].isin(common)].groupby(["pid", "item_id"])["score"].max().groupby("pid").mean(),
    })
    for task, prefix in (("research question", "rq"), ("hypothesis", "hyp"), ("experiment", "exp")):
        cols = [c for c in common if c.startswith(f"sctt_{prefix}_")]
        person[f"sctt_creativity_{prefix}"] = item_means[cols].mean(axis=1)
    person.to_csv(out / "person_scores.csv")
    res["alpha_9_items"] = round(cronbach_alpha(item_means[common]), 3)
    res["n_persons_9_items"] = int(item_means[common].dropna().shape[0])
    sub = person[["sctt_creativity_rq", "sctt_creativity_hyp", "sctt_creativity_exp"]].corr().round(3)
    res["subtest_correlations"] = sub.to_dict()
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
