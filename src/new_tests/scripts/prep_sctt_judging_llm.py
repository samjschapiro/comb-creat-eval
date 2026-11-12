"""Build blinded per-item SCTT rating batches of LLM answers, with hidden human calibration answers.

Downstream of run_human_battery_llm.py. Each model answer (3 per call) becomes one response; a
random sample of already-rated human answers per item is mixed in, under the same opaque ids, so the
judges' scale for the LLM batch can be checked against their earlier human ratings. A small fraction
of hidden repeats measures judge consistency. The key (uid -> model or human, item, sample) stays
in key/ and is not shown to the judges.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.human_battery import SCTT_ITEMS, parse_list
from src.utils import init_directory, load_config, save_config

TASK = {"rq": "research question", "hyp": "hypothesis", "exp": "experiment"}


def llm_answers(run_dir):
    rows = []
    for f in sorted((Path(run_dir) / "responses").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            if rec["test"] != "sctt" or not rec["content"]:
                continue
            answers = parse_list(rec["content"]) or []
            for i, a in enumerate(answers[:3], start=1):
                rows.append({"source": "llm", "model": rec["model"], "item_id": rec["item_id"],
                             "sample": rec["sample"], "response_index": i, "response": " ".join(a.split())})
    return pd.DataFrame(rows)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("llm_run_dir", "human_scores_dir", "reference_dir", "output_dir", "human_per_item",
                "repeat_fraction", "seed"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    for sub in ("batches", "key", "ratings"):
        (out / sub).mkdir()

    llm = llm_answers(config["llm_run_dir"])
    human = pd.read_csv(Path(config["human_scores_dir"]) / "response_scores.csv")
    rng = np.random.default_rng(config["seed"])
    keys = []
    for item in SCTT_ITEMS:
        a = llm[llm["item_id"] == item]
        h = human[human["item_id"] == item].sample(config["human_per_item"], random_state=config["seed"])
        h = pd.DataFrame({"source": "human", "model": h["pid"], "item_id": item, "sample": 0,
                          "response_index": h["response_index"], "response": h["response"].map(lambda t: " ".join(str(t).split())),
                          "human_judge_score": h["score"]})
        g = pd.concat([a, h], ignore_index=True)
        rep = g.sample(frac=config["repeat_fraction"], random_state=config["seed"]).assign(is_repeat=True)
        g = pd.concat([g.assign(is_repeat=False), rep]).sample(frac=1, random_state=config["seed"])
        g["uid"] = [f"{item}_{i:04d}" for i in rng.permutation(len(g))]
        g = g.sort_values("uid")
        keys.append(g)
        header = (f"ITEM: {item}\nTASK: {TASK[item.split('_')[1]]}\nPROMPT SHOWN TO PARTICIPANTS: {SCTT_ITEMS[item]}\n"
                  f"N RESPONSES: {len(g)}\n")
        (out / "batches" / f"{item}.txt").write_text(
            header + "\n" + "\n".join(f"[{u}] {t}" for u, t in zip(g["uid"], g["response"])) + "\n")
    key = pd.concat(keys)
    key.to_csv(out / "key" / "uid_key.csv", index=False)
    examples = pd.read_csv(Path(config["reference_dir"]) / "example_responses.csv")
    examples[["item", "task", "prompt", "response", "jrt"]].to_csv(out / "calibration_examples.csv", index=False)
    print(f"{len(key)} rows ({int((key.source == 'llm').sum())} LLM, {int((key.source == 'human').sum())} human "
          f"calibration, {int(key.is_repeat.sum())} hidden repeats); {llm['model'].nunique()} models")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
