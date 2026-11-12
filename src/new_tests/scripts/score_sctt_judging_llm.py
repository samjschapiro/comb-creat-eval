"""Merge the judges' ratings of the LLMs' SCTT answers with the hidden key; check the judges' scale.

Downstream of prep_sctt_judging_llm.py. Fails on any unrated or doubly rated uid. Reports judge
consistency on hidden repeats, and agreement between these ratings and the same judges' earlier
ratings of the hidden human calibration answers (the scale check). Per-model creativity is the mean
over a model's answers per item, then over the nine items common to all human participants.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils import init_directory, load_config, save_config

DROPPED_IN_V4 = {"sctt_rq_1", "sctt_hyp_1", "sctt_exp_1"}


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "output_dir"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    up = Path(config["upstream_dir"])
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)

    key = pd.read_csv(up / "key" / "uid_key.csv")
    ratings = pd.DataFrame([json.loads(l) for f in sorted((up / "ratings").glob("*.jsonl"))
                            for l in f.read_text().splitlines() if l.strip()])
    if ratings["uid"].duplicated().any():
        raise ValueError("FATAL: duplicated uids in ratings")
    if set(key["uid"]) != set(ratings["uid"]):
        raise ValueError(f"FATAL: {len(set(key['uid']) - set(ratings['uid']))} unrated, "
                         f"{len(set(ratings['uid']) - set(key['uid']))} unknown uids")
    if not ratings["score"].isin([1, 2, 3, 4, 5]).all():
        raise ValueError("FATAL: scores outside 1-5")
    d = key.merge(ratings[["uid", "score"]], on="uid")

    k = ["source", "model", "item_id", "sample", "response_index"]
    first = d[~d["is_repeat"]].set_index(k)["score"]
    rep = d[d["is_repeat"]].set_index(k)["score"]
    pair = pd.concat([first.loc[rep.index].rename("a"), rep.rename("b")], axis=1)
    h = d[(d["source"] == "human") & ~d["is_repeat"]]
    res = {
        "n_rated": int(len(d)),
        "judge_repeat": {"n": int(len(pair)), "exact": round((pair.a == pair.b).mean(), 3), "pearson": round(pair.a.corr(pair.b), 3)},
        "human_calibration": {"n": int(len(h)), "mean_now": round(h["score"].mean(), 3),
                              "mean_human_run": round(h["human_judge_score"].mean(), 3),
                              "pearson": round(h["score"].corr(h["human_judge_score"]), 3),
                              "spearman": round(stats.spearmanr(h["score"], h["human_judge_score"])[0], 3),
                              "exact": round((h["score"] == h["human_judge_score"]).mean(), 3)},
    }
    llm = d[(d["source"] == "llm") & ~d["is_repeat"]].copy()
    llm["n_words"] = llm["response"].str.split().str.len()
    llm.drop(columns=["is_repeat", "human_judge_score"]).to_csv(out / "response_scores.csv", index=False)
    res["llm_score_distribution"] = llm["score"].value_counts().sort_index().to_dict()
    res["llm_mean_by_item"] = llm.groupby("item_id")["score"].mean().round(3).to_dict()
    res["llm_score_vs_words_spearman"] = round(stats.spearmanr(llm["score"], llm["n_words"])[0], 3)

    item = llm.groupby(["model", "item_id"]).agg(score=("score", "mean"), words=("n_words", "mean")).reset_index()
    common = item[~item["item_id"].isin(DROPPED_IN_V4)]
    models = pd.DataFrame({"sctt_creativity": common.groupby("model")["score"].mean(),
                           "sctt_creativity_all_items": item.groupby("model")["score"].mean(),
                           "sctt_length": common.groupby("model")["words"].mean()})
    for prefix in ("rq", "hyp", "exp"):
        models[f"sctt_creativity_{prefix}"] = common[common["item_id"].str.startswith(f"sctt_{prefix}_")].groupby("model")["score"].mean()
    models.to_csv(out / "model_scores.csv")
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(models.round(2).sort_values("sctt_creativity").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
