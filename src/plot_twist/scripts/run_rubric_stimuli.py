"""Score a stimuli JSON with the fixed-rubric judge (downstream of run_generate).

Generic counterpart to run_rubric_gold.py: instead of reading raw .txt + the
manifest, it reads a JSON list of records (each with at least `id` and `story`)
and scores every record with non-empty story text. Arbitrary metadata fields
(e.g. model, condition, prompt_id) are passed through to scores.csv so downstream
analysis can group by condition (twist vs predictable vs random) and by model.

Usage:
    python src/plot_twist/scripts/run_rubric_stimuli.py configs/plot_twist/rubric_contrast.yaml --overwrite [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from pathlib import Path

from src.utils import init_directory, load_config, save_config
from src.plot_twist.rubric_judge import RubricConfig, score_stories, save_scores


def _geomean(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else math.sqrt(a * b)


def _fmt(x: float | None) -> str:
    return "  NA" if x is None else f"{x:4.1f}"


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg_dict = load_config(config_path)
    for field_name in ("output_dir", "judge_models", "stimuli_file"):
        if field_name not in cfg_dict:
            raise ValueError(f"FATAL: '{field_name}' required in config")

    # --overwrite wipes; otherwise RESUME (don't wipe the per-story score cache),
    # so a re-run never re-spends on stories already judged.
    if overwrite:
        out = init_directory(cfg_dict["output_dir"], overwrite=True)
    else:
        out = Path(cfg_dict["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
    save_config(cfg_dict, out)

    stimuli_path = Path(cfg_dict["stimuli_file"])
    if not stimuli_path.exists():
        raise FileNotFoundError(f"stimuli_file not found: {stimuli_path}")
    records = json.loads(stimuli_path.read_text())
    # keep only records with a non-empty story
    records = [r for r in records if r.get("story")]
    if debug:
        records = records[:6]
    if not records:
        raise ValueError(f"FATAL: no records with non-empty 'story' in {stimuli_path}")

    meta_fields = cfg_dict.get("meta_fields", [])
    meta = {r["id"]: {k: r.get(k) for k in meta_fields} for r in records}
    stimuli = [{"id": r["id"], "story": r["story"]} for r in records]

    cfg = RubricConfig(
        judge_models=cfg_dict["judge_models"],
        temperature=cfg_dict.get("temperature", 0.0),
        max_tokens=cfg_dict.get("max_tokens", 400),
        concurrency=cfg_dict.get("concurrency", 16),
    )

    print(f"scoring {len(stimuli)} stories x {len(cfg.judge_models)} judges (rubric {cfg.rubric_version})")
    print(f"judges: {', '.join(cfg.judge_models)}\n")

    scores = asyncio.run(score_stories(cfg, stimuli, cache_dir=out / "scores"))

    n = len(scores)
    print("judge reliability (parsed replies / stories):")
    dead = []
    for m in cfg.judge_models:
        ok = sum(1 for s in scores if s.by_judge.get(m) is not None)
        flag = "  <-- ALL FAILED" if ok == 0 else ("  <-- some failures" if ok < n else "")
        print(f"  {m:<34} {ok}/{n}{flag}")
        if ok == 0:
            dead.append(m)
    print()
    if dead:
        raise RuntimeError(
            f"FATAL: judge(s) failed on every story: {dead}. Fix before trusting scores."
        )

    save_scores(scores, cfg, out)

    csv_path = out / "scores.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            list(meta_fields)
            + ["slug", "twist_present", "surprise", "coherence", "prose_quality", "overall", "geomean_surp_coh"]
        )
        for s in scores:
            m = meta.get(s.story_id, {})
            w.writerow(
                [m.get(k) for k in meta_fields]
                + [
                    s.story_id,
                    s.twist_present,
                    s.surprise,
                    s.coherence,
                    s.prose_quality,
                    s.overall,
                    _geomean(s.surprise, s.coherence),
                ]
            )

    print(f"saved: {out / 'rubric_scores.json'}\n       {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
