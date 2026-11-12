"""Build blinded per-item rating batches of the human SCTT responses for an LLM judge.

Each item's responses are shuffled and given opaque ids (no participant id, no other measures),
with a small fraction of hidden repeats so judge consistency can be measured. The id -> response
key stays in key/ and is not shown to the judge.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.human_battery import SCTT_ITEMS as ITEM_TEXT
from src.utils import init_directory, load_config, save_config



def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "reference_dir", "output_dir", "sample", "apply_exclude", "repeat_fraction", "seed"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    for sub in ("batches", "key", "ratings"):
        (out / sub).mkdir()

    r = pd.read_csv(Path(config["upstream_dir"]) / "responses.csv", low_memory=False)
    r = r[(r["sample"] == config["sample"]) & (r["test"] == "sctt")].copy()
    if config["apply_exclude"]:  # the study team's participant-level exclusions (packages from 2026-11-10 on)
        if "exclude" not in r.columns:
            raise ValueError("FATAL: apply_exclude is set but responses.csv has no `exclude` column")
        r = r[~r["exclude"].astype(bool)]
    missing = set(r["item_id"]) - set(ITEM_TEXT)
    if missing:
        raise ValueError(f"FATAL: no item wording for {missing}")
    n_blank = int(r["response"].fillna("").str.strip().eq("").sum())
    r = r[r["response"].fillna("").str.strip().ne("")]

    rng = np.random.default_rng(config["seed"])
    rows, n_repeats = [], 0
    for item, g in r.groupby("item_id"):
        g = g[["pid", "item_id", "task", "response_index", "response"]].reset_index(drop=True)
        rep = g.sample(frac=config["repeat_fraction"], random_state=config["seed"]).assign(is_repeat=True)
        n_repeats += len(rep)
        g = pd.concat([g.assign(is_repeat=False), rep]).sample(frac=1, random_state=config["seed"])
        g["uid"] = [f"{item}_{i:04d}" for i in rng.permutation(len(g))]
        g = g.sort_values("uid")
        rows.append(g)
        batch = g[["uid", "response"]]
        header = f"ITEM: {item}\nTASK: {g['task'].iloc[0]}\nPROMPT SHOWN TO PARTICIPANTS: {ITEM_TEXT[item]}\nN RESPONSES: {len(batch)}\n"
        (out / "batches" / f"{item}.txt").write_text(
            header + "\n" + "\n".join(f"[{u}] {t.strip()}" for u, t in zip(batch["uid"], batch["response"])) + "\n")
    key = pd.concat(rows)
    key.to_csv(out / "key" / "uid_key.csv", index=False)

    examples = pd.read_csv(Path(config["reference_dir"]) / "example_responses.csv")
    examples[["item", "task", "prompt", "response", "jrt"]].to_csv(out / "calibration_examples.csv", index=False)
    print(f"{len(key)} rows to rate ({len(key) - n_repeats} unique + {n_repeats} hidden repeats) "
          f"over {key['item_id'].nunique()} items; {n_blank} blank responses dropped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
