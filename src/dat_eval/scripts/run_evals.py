"""Run DAT, CDAT, and PACE evaluations against LLMs via OpenRouter.

Step 1 of the pipeline: queries each model for all three tasks and saves
raw responses.

Usage:
    uv run python src/dat_eval/scripts/run_evals.py configs/dat_eval/run_evals.yaml
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from tqdm import tqdm

from src.utils import load_config, init_directory, save_config
from src.dat_eval.llm import call_llm, extract_words_from_response, model_id_to_key
from src.dat_eval.dat import DAT_PROMPT
from src.dat_eval.cdat import cdat_prompt, DEFAULT_CUES
from src.dat_eval.pace import (
    pace_stage1_prompt,
    pace_stage2_prompt,
    DEFAULT_SEEDS,
)


def run_dat(model_id: str, n_trials: int, temperature: float) -> list[dict]:
    """Run DAT evaluation: generate n_trials sets of 10 divergent words."""
    results = []
    for trial in range(n_trials):
        try:
            raw = call_llm(
                messages=[{"role": "user", "content": DAT_PROMPT}],
                model=model_id,
                temperature=temperature,
            )
            words = extract_words_from_response(raw, expected_count=10)
            results.append({
                "trial": trial,
                "raw_response": raw,
                "words": words,
                "api_error": None,
            })
        except Exception as e:
            results.append({
                "trial": trial,
                "raw_response": None,
                "words": [],
                "api_error": f"{type(e).__name__}: {e}",
            })
            traceback.print_exc()
            time.sleep(1)
    return results


def run_cdat(
    model_id: str, cues: list[str], temperature: float
) -> dict[str, dict]:
    """Run CDAT evaluation: generate 10 associated-but-diverse words per cue."""
    results = {}
    for cue in cues:
        try:
            raw = call_llm(
                messages=[{"role": "user", "content": cdat_prompt(cue)}],
                model=model_id,
                temperature=temperature,
            )
            words = extract_words_from_response(raw, expected_count=10)
            results[cue] = {
                "raw_response": raw,
                "words": words,
                "api_error": None,
            }
        except Exception as e:
            results[cue] = {
                "raw_response": None,
                "words": [],
                "api_error": f"{type(e).__name__}: {e}",
            }
            traceback.print_exc()
            time.sleep(1)
    return results


def run_pace(
    model_id: str, seeds: list[str], temperature: float
) -> dict[str, dict]:
    """Run PACE evaluation: 3 association chains of 20 words per seed."""
    results = {}

    for seed in seeds:
        # Stage 1: get 3 first-associations
        try:
            raw1 = call_llm(
                messages=[{"role": "user", "content": pace_stage1_prompt(seed)}],
                model=model_id,
                temperature=temperature,
            )
            # Parse stage 1 response
            stage1_data = _parse_pace_stage1(raw1)
        except Exception as e:
            results[seed] = {
                "stage1_raw": None,
                "stage1_error": f"{type(e).__name__}: {e}",
                "chains": [],
            }
            traceback.print_exc()
            time.sleep(1)
            continue

        # Stage 2: build 3 chains
        chains = []
        for assoc in stage1_data:
            try:
                raw2 = call_llm(
                    messages=[{
                        "role": "user",
                        "content": pace_stage2_prompt(
                            seed, assoc["word"], assoc.get("reason", "")
                        ),
                    }],
                    model=model_id,
                    temperature=temperature,
                )
                chain = _parse_pace_stage2(raw2, seed, assoc["word"])
                chains.append({
                    "first_association": assoc,
                    "raw_response": raw2,
                    "chain": chain,
                    "api_error": None,
                })
            except Exception as e:
                chains.append({
                    "first_association": assoc,
                    "raw_response": None,
                    "chain": [seed, assoc["word"]],
                    "api_error": f"{type(e).__name__}: {e}",
                })
                traceback.print_exc()
                time.sleep(1)

        results[seed] = {
            "stage1_raw": raw1,
            "stage1_parsed": stage1_data,
            "chains": chains,
        }

    return results


def _parse_pace_stage1(raw: str) -> list[dict]:
    """Parse PACE stage 1 response into list of {word, reason} dicts."""
    import re

    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "results" in data and isinstance(data["results"], list):
                return [
                    {"word": r.get("word", ""), "reason": r.get("reason", "")}
                    for r in data["results"][:3]
                ]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: extract words from response
    words = extract_words_from_response(raw, expected_count=3)
    return [{"word": w, "reason": ""} for w in words[:3]]


def _parse_pace_stage2(raw: str, seed: str, second_word: str) -> list[str]:
    """Parse PACE stage 2 response into a word chain."""
    import re

    chain = [seed, second_word]

    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if "results" in data and isinstance(data["results"], list):
                for entry in data["results"]:
                    word = entry.get("word", "").strip().lower()
                    if word and word not in chain:
                        chain.append(word)
                    elif word:
                        chain.append(word)  # Allow duplicates in chain
                return chain[:20]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback
    words = extract_words_from_response(raw, expected_count=20)
    chain.extend(words)
    return chain[:20]


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    models = config["models"]
    temperature = config.get("temperature", 0.0)
    dat_trials = config.get("dat_trials", 5)
    cdat_cues = config.get("cdat_cues", DEFAULT_CUES)
    pace_seeds = config.get("pace_seeds", DEFAULT_SEEDS)
    evals_to_run = config.get("evals", ["dat", "cdat", "pace"])

    if debug:
        models = models[:1]
        dat_trials = 1
        cdat_cues = cdat_cues[:3]
        pace_seeds = pace_seeds[:2]

    print(f"Models: {len(models)}")
    print(f"Evals: {evals_to_run}")
    print(f"DAT trials: {dat_trials}")
    print(f"CDAT cues: {len(cdat_cues)}")
    print(f"PACE seeds: {len(pace_seeds)}")
    print(f"Temperature: {temperature}")

    for model_id in models:
        model_key = model_id_to_key(model_id)
        model_dir = output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{model_id}")
        print(f"{'='*60}")

        if "dat" in evals_to_run:
            print(f"  Running DAT ({dat_trials} trials)...")
            dat_results = run_dat(model_id, dat_trials, temperature)
            with open(model_dir / "dat_responses.json", "w") as f:
                json.dump(dat_results, f, indent=2)
            n_words = [len(r["words"]) for r in dat_results if r["words"]]
            print(f"  DAT done: {len(n_words)}/{dat_trials} successful, avg words: {sum(n_words)/max(len(n_words),1):.1f}")

        if "cdat" in evals_to_run:
            print(f"  Running CDAT ({len(cdat_cues)} cues)...")
            cdat_results = run_cdat(model_id, cdat_cues, temperature)
            with open(model_dir / "cdat_responses.json", "w") as f:
                json.dump(cdat_results, f, indent=2)
            n_ok = sum(1 for r in cdat_results.values() if r["words"])
            print(f"  CDAT done: {n_ok}/{len(cdat_cues)} successful")

        if "pace" in evals_to_run:
            n_calls = len(pace_seeds) * 4  # 1 stage1 + 3 stage2 per seed
            print(f"  Running PACE ({len(pace_seeds)} seeds, ~{n_calls} API calls)...")
            pace_results = run_pace(model_id, pace_seeds, temperature)
            with open(model_dir / "pace_responses.json", "w") as f:
                json.dump(pace_results, f, indent=2)
            n_ok = sum(1 for r in pace_results.values() if r.get("chains"))
            print(f"  PACE done: {n_ok}/{len(pace_seeds)} seeds with chains")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
