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


def run_dat(
    model_id: str,
    n_trials: int,
    temperature: float,
    base_seed: int = 1000,
    top_p: float | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """Run DAT evaluation: generate n_trials sets of 10 divergent words.

    Each trial uses a different seed (base_seed + trial) to break the model's
    prior on "first token" behavior — temperature alone is often insufficient
    because models have very peaked distributions for the leading token.
    """
    results = []
    for trial in range(n_trials):
        seed = base_seed + trial
        try:
            raw = call_llm(
                messages=[{"role": "user", "content": DAT_PROMPT}],
                model=model_id,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                top_k=top_k,
            )
            words = extract_words_from_response(raw, expected_count=10)
            results.append({
                "trial": trial,
                "seed": seed,
                "raw_response": raw,
                "words": words,
                "api_error": None,
            })
        except Exception as e:
            results.append({
                "trial": trial,
                "seed": seed,
                "raw_response": None,
                "words": [],
                "api_error": f"{type(e).__name__}: {e}",
            })
            traceback.print_exc()
            time.sleep(1)
    return results


def run_cdat(
    model_id: str,
    cues: list[str],
    temperature: float,
    base_seed: int = 2000,
    top_p: float | None = None,
    top_k: int | None = None,
) -> dict[str, dict]:
    """Run CDAT evaluation: generate 10 associated-but-diverse words per cue.

    Each cue uses a unique seed (base_seed + cue index) to ensure variance
    isn't lost to deterministic sampling priors.
    """
    results = {}
    for i, cue in enumerate(cues):
        seed = base_seed + i
        try:
            raw = call_llm(
                messages=[{"role": "user", "content": cdat_prompt(cue)}],
                model=model_id,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                top_k=top_k,
            )
            words = extract_words_from_response(raw, expected_count=10)
            results[cue] = {
                "seed": seed,
                "raw_response": raw,
                "words": words,
                "api_error": None,
            }
        except Exception as e:
            results[cue] = {
                "seed": seed,
                "raw_response": None,
                "words": [],
                "api_error": f"{type(e).__name__}: {e}",
            }
            traceback.print_exc()
            time.sleep(1)
    return results


def run_pace(
    model_id: str,
    seeds: list[str],
    temperature: float,
    top_p: float | None = None,
    top_k: int | None = None,
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
                top_p=top_p,
                top_k=top_k,
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
                    top_p=top_p,
                    top_k=top_k,
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


def _temp_suffix(temp: float) -> str:
    """Return a filename-safe suffix for a temperature value (e.g. 0.9 -> 't0-9')."""
    return f"t{str(temp).replace('.', '-')}"


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    output_dir = Path(config["output_dir"])
    if overwrite and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    models = config["models"]

    # Per-eval temperature lists. Each eval can use one or more temperatures.
    # Falls back to top-level "temperature" if the per-eval list is missing.
    default_temp = config.get("temperature", 0.0)
    dat_temps = config.get("dat_temperatures", [default_temp])
    cdat_temps = config.get("cdat_temperatures", [default_temp])
    pace_temps = config.get("pace_temperatures", [default_temp])

    dat_trials = config.get("dat_trials", 5)  # trials PER temperature
    cdat_cues = config.get("cdat_cues", DEFAULT_CUES)
    pace_seeds = config.get("pace_seeds", DEFAULT_SEEDS)
    evals_to_run = config.get("evals", ["dat", "cdat", "pace"])
    # Sampling controls. For DAT/CDAT we want maximum diversity, so disable
    # nucleus and top-k filtering by default (top_p=1.0, top_k=0).
    dat_top_p = config.get("dat_top_p", 1.0)
    dat_top_k = config.get("dat_top_k", 0)
    cdat_top_p = config.get("cdat_top_p", 1.0)
    cdat_top_k = config.get("cdat_top_k", 0)
    # PACE: leave at provider defaults (paper doesn't specify)
    pace_top_p = config.get("pace_top_p", None)
    pace_top_k = config.get("pace_top_k", None)

    if debug:
        models = models[:1]
        dat_trials = 1
        cdat_cues = cdat_cues[:3]
        pace_seeds = pace_seeds[:2]

    print(f"Models: {len(models)}")
    print(f"Evals: {evals_to_run}")
    print(f"DAT: {dat_trials} trials per temp at temps {dat_temps}")
    print(f"CDAT: {len(cdat_cues)} cues at temps {cdat_temps}")
    print(f"PACE: {len(pace_seeds)} seeds at temps {pace_temps}")

    for model_id in models:
        model_key = model_id_to_key(model_id)
        model_dir = output_dir / model_key
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"{model_id}")
        print(f"{'='*60}")

        if "dat" in evals_to_run:
            for temp in dat_temps:
                fname = f"dat_responses_{_temp_suffix(temp)}.json"
                dat_path = model_dir / fname
                if dat_path.exists():
                    print(f"  DAT temp={temp} already done, skipping")
                    continue
                print(f"  Running DAT temp={temp} top_p={dat_top_p} top_k={dat_top_k} ({dat_trials} trials)...")
                dat_results = run_dat(model_id, dat_trials, temp, top_p=dat_top_p, top_k=dat_top_k)
                with open(dat_path, "w") as f:
                    json.dump(dat_results, f, indent=2)
                n_words = [len(r["words"]) for r in dat_results if r["words"]]
                print(f"    Done: {len(n_words)}/{dat_trials} successful, avg words: {sum(n_words)/max(len(n_words),1):.1f}")

        if "cdat" in evals_to_run:
            for temp in cdat_temps:
                fname = f"cdat_responses_{_temp_suffix(temp)}.json"
                cdat_path = model_dir / fname
                if cdat_path.exists():
                    print(f"  CDAT temp={temp} already done, skipping")
                    continue
                print(f"  Running CDAT temp={temp} top_p={cdat_top_p} top_k={cdat_top_k} ({len(cdat_cues)} cues)...")
                cdat_results = run_cdat(model_id, cdat_cues, temp, top_p=cdat_top_p, top_k=cdat_top_k)
                with open(cdat_path, "w") as f:
                    json.dump(cdat_results, f, indent=2)
                n_ok = sum(1 for r in cdat_results.values() if r["words"])
                print(f"    Done: {n_ok}/{len(cdat_cues)} successful")

        if "pace" in evals_to_run:
            for temp in pace_temps:
                fname = f"pace_responses_{_temp_suffix(temp)}.json"
                pace_path = model_dir / fname
                # Backward compat: also accept the old un-suffixed pace_responses.json
                # if temp is the canonical PACE temp (0.0).
                old_pace = model_dir / "pace_responses.json"
                if pace_path.exists():
                    print(f"  PACE temp={temp} already done, skipping")
                    continue
                if temp == 0.0 and old_pace.exists():
                    print(f"  PACE temp=0.0 already done (legacy filename), renaming")
                    old_pace.rename(pace_path)
                    continue
                n_calls = len(pace_seeds) * 4
                print(f"  Running PACE temp={temp} ({len(pace_seeds)} seeds, ~{n_calls} API calls)...")
                pace_results = run_pace(model_id, pace_seeds, temp, top_p=pace_top_p, top_k=pace_top_k)
                with open(pace_path, "w") as f:
                    json.dump(pace_results, f, indent=2)
                n_ok = sum(1 for r in pace_results.values() if r.get("chains"))
                print(f"    Done: {n_ok}/{len(pace_seeds)} seeds with chains")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
