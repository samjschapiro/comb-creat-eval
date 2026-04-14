"""Run the word-circle construction eval against LLMs via OpenRouter.

For each model × seed, we sample N_TRIALS independent circle attempts at
temperature > 0. Each circle is an ordered list of n_words meant to close
back to the seed. No explicit rule-following — the task is structural.

Reuses the async / budget-cap / resume infrastructure from run_c_pace.py.

Usage:
    uv run python src/comb_eval/scripts/run_circle.py configs/comb_eval/run_circle.yaml
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, init_directory, save_config
from src.comb_eval.llm import call_llm_async, get_async_client, model_id_to_key
from src.comb_eval.circle import (
    DEFAULT_SEEDS,
    circle_prompt,
    parse_circle_response,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "safety"))
from cost_tracker import PRICING


# Per-call token estimates for circle construction.
# Input: seed + rule block is short (~250 tokens).
# Output: n_words entries in JSON with reasons (~40 tokens each) → ~500 for n=8.
EST_TOKENS = (250, 500)


REASONING_MODELS = {
    "qwen/qwq-32b",
    "deepseek/deepseek-r1",
    "openai/o3", "openai/o3-mini", "openai/o4-mini",
    "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano",
    "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
}


def max_tokens_for(model_id: str, base: int, mult: int) -> int:
    return base * mult if model_id in REASONING_MODELS else base


def estimate_model_cost(
    model_id: str,
    n_seeds: int,
    n_trials: int,
) -> float:
    pricing = PRICING.get(model_id)
    if pricing is None:
        return float("inf")
    in_p, out_p = pricing
    n_calls = n_seeds * n_trials
    in_t = n_calls * EST_TOKENS[0]
    out_t = n_calls * EST_TOKENS[1]
    return (in_t * in_p + out_t * out_p) / 1_000_000


async def _safe_call(async_client, sem, model, prompt, temperature,
                     max_tokens, reasoning):
    async with sem:
        try:
            raw = await call_llm_async(
                async_client,
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            if raw is None:
                return None, "RuntimeError: provider returned null content"
            return raw, None
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"


async def run_model(
    async_client,
    sem,
    model_id: str,
    seeds: list[str],
    n_words: int,
    n_trials: int,
    temperature: float,
    max_tokens_base: int,
    reasoning_multiplier: int,
    reasoning: dict | None,
    output_dir: Path,
) -> dict:
    """Fire all seed×trial calls concurrently, save responses.json."""
    model_key = model_id_to_key(model_id)
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    responses_path = model_dir / "circle_responses.json"

    if responses_path.exists():
        print(f"  {model_id}: already done, skipping")
        return {"model_id": model_id, "skipped": True}

    max_tokens = max_tokens_for(model_id, max_tokens_base, reasoning_multiplier)

    print(f"  {model_id}: {len(seeds)} seeds × {n_trials} trials "
          f"(n_words={n_words}, max_tokens={max_tokens})")

    t0 = time.time()

    tasks = []
    keys = []  # (seed, trial_idx)
    for seed in seeds:
        for trial in range(n_trials):
            tasks.append(_safe_call(
                async_client, sem, model_id,
                circle_prompt(seed, n_words),
                temperature, max_tokens, reasoning,
            ))
            keys.append((seed, trial))

    raws = await asyncio.gather(*tasks)

    # Parse and bucket by seed
    by_seed: dict[str, list[dict]] = {s: [] for s in seeds}
    n_parsed = 0
    n_api_err = 0
    for (seed, trial), (raw, err) in zip(keys, raws):
        if err:
            n_api_err += 1
        words = parse_circle_response(raw, seed=seed, n_words=n_words) if raw else []
        if words:
            n_parsed += 1
        by_seed[seed].append({
            "trial": trial,
            "raw_response": raw,
            "api_error": err,
            "words": words,
        })

    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s — parsed {n_parsed}/{len(tasks)} "
          f"api_errors={n_api_err}")

    output = {
        "model_id": model_id,
        "seeds": seeds,
        "n_words": n_words,
        "n_trials": n_trials,
        "results": [
            {"seed": s, "trials": by_seed[s]}
            for s in seeds
        ],
    }
    with open(responses_path, "w") as f:
        json.dump(output, f, indent=2)

    summary = {
        "model_id": model_id,
        "model_key": model_key,
        "n_seeds": len(seeds),
        "n_trials": n_trials,
        "n_parsed": n_parsed,
        "n_api_err": n_api_err,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


async def main(config_path: str, overwrite: bool = False, debug: bool = False,
               resume: bool = False):
    config = load_config(config_path)

    if resume:
        output_dir = Path(config["output_dir"])
        if not output_dir.exists():
            raise FileNotFoundError(
                f"--resume requires existing output_dir: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"RESUMING run in {output_dir}")
    else:
        output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    seeds = config.get("seeds", DEFAULT_SEEDS)
    if "n_seeds" in config:
        seeds = seeds[: config["n_seeds"]]
    n_words = config.get("n_words", 8)
    n_trials = config.get("n_trials", 3)
    temperature = config.get("temperature", 0.7)
    concurrency = config.get("concurrency", 20)
    budget_usd = config.get("budget_usd", 0.0)
    reasoning_cfg = config.get("reasoning", {"effort": "low", "exclude": True})
    reasoning_multiplier = config.get("reasoning_max_tokens_multiplier", 4)
    max_tokens_base = config.get("max_tokens", 1024)
    models = config["models"]

    if debug:
        seeds = seeds[:3]
        models = models[:1]

    print(f"Models: {len(models)}")
    print(f"Seeds: {len(seeds)}  n_words: {n_words}  n_trials: {n_trials}")
    print(f"temperature: {temperature}  concurrency: {concurrency}")
    if budget_usd > 0:
        print(f"Budget cap: ${budget_usd:.2f}")
    else:
        print("Budget cap: NONE")

    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    cumulative = 0.0
    summaries = []

    for model_id in models:
        model_dir = output_dir / model_id_to_key(model_id)
        already = (model_dir / "circle_responses.json").exists()
        if not already:
            cost = estimate_model_cost(model_id, len(seeds), n_trials)
            if budget_usd > 0 and (cumulative + cost) > budget_usd:
                print(f"\n{'='*60}")
                print("BUDGET CAP REACHED")
                print(f"  Cumulative (est): ${cumulative:.2f}")
                print(f"  Cap:              ${budget_usd:.2f}")
                print(f"  Next model:       ${cost:.2f}  ({model_id})")
                print(f"{'='*60}")
                break
            cumulative += cost
            print(f"\n{model_id}   (est ${cost:.2f}, cumulative ${cumulative:.2f})")
        else:
            print(f"\n{model_id}   (already done)")

        s = await run_model(
            async_client, sem,
            model_id=model_id, seeds=seeds,
            n_words=n_words, n_trials=n_trials,
            temperature=temperature,
            max_tokens_base=max_tokens_base,
            reasoning_multiplier=reasoning_multiplier,
            reasoning=reasoning_cfg,
            output_dir=output_dir,
        )
        summaries.append(s)

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.config_path, args.overwrite, args.debug, args.resume))
