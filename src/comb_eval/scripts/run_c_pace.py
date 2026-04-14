"""Run Constrained PACE (C-PACE) against LLMs via OpenRouter.

Two-stage flow (follows dat_eval PACE):
    Stage 1: seed -> 3 first-associations (one call per seed)
    Stage 2: (seed, first_assoc, level) -> 20-word chain with lexical
             constraints from that level (one call per combination)

Constraints are sampled once per (seed, level) from a seeded RNG and held
fixed across all models, so every model faces the same task.

Features ported from dat_eval's runner:
- Async with a shared semaphore (all stage-1 calls fire in parallel,
  then all stage-2 calls fire in parallel).
- Reasoning config forwarded via OpenRouter's unified reasoning API;
  reasoning models get base * multiplier max_tokens budget.
- Budget cap via scripts/safety/cost_tracker.PRICING.
- Resume-safe: skips models with an existing c_pace_responses.json.

Usage:
    uv run python src/comb_eval/scripts/run_c_pace.py configs/comb_eval/run_c_pace.yaml
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, init_directory, save_config
from src.comb_eval.llm import call_llm_async, get_async_client, model_id_to_key
from src.comb_eval.c_pace import (
    Constraints,
    DEFAULT_SEEDS,
    FastTextEmbeddings,
    build_semantic_anchor_fn,
    c_pace_stage1_prompt,
    c_pace_stage2_prompt,
    generate_constraints,
    generate_semantic_constraints,
    parse_stage1,
    parse_stage2_chain,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "safety"))
from cost_tracker import PRICING


# Per-call token estimates. Stage 1 is short; stage 2 is a 20-word chain with
# reasons, which can get chatty especially on reasoning models.
STAGE1_TOKENS = (200, 300)   # input, output
STAGE2_TOKENS = (300, 600)


REASONING_MODELS = {
    "qwen/qwq-32b",
    "deepseek/deepseek-r1",
    "openai/o3",
    "openai/o3-mini",
    "openai/o4-mini",
    "openai/gpt-5",
    "openai/gpt-5-mini",
    "openai/gpt-5-nano",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
}


def max_tokens_for(model_id: str, base: int, reasoning_multiplier: int) -> int:
    if model_id in REASONING_MODELS:
        return base * reasoning_multiplier
    return base


def estimate_model_cost(
    model_id: str,
    n_seeds: int,
    n_first_assocs: int,
    n_levels: int,
) -> float:
    pricing = PRICING.get(model_id)
    if pricing is None:
        return float("inf")
    in_price, out_price = pricing
    n_stage1 = n_seeds
    n_stage2 = n_seeds * n_first_assocs * n_levels
    in_tok = n_stage1 * STAGE1_TOKENS[0] + n_stage2 * STAGE2_TOKENS[0]
    out_tok = n_stage1 * STAGE1_TOKENS[1] + n_stage2 * STAGE2_TOKENS[1]
    return (in_tok * in_price + out_tok * out_price) / 1_000_000


async def _safe_call(
    async_client,
    sem: asyncio.Semaphore,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    reasoning: dict | None,
) -> tuple[str | None, str | None]:
    """Returns (raw_response, error_str). One always None."""
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
    sem: asyncio.Semaphore,
    model_id: str,
    seeds: list[str],
    levels: list[int],
    constraints_map: dict,
    n_first_assocs: int,
    temperature: float,
    stage1_max_tokens_base: int,
    stage2_max_tokens_base: int,
    reasoning_multiplier: int,
    reasoning: dict | None,
    output_dir: Path,
) -> dict:
    """Run stage 1 then stage 2 for one model; write c_pace_responses.json."""
    model_key = model_id_to_key(model_id)
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    responses_path = model_dir / "c_pace_responses.json"

    if responses_path.exists():
        print(f"  {model_id}: already done, skipping")
        return {"model_id": model_id, "skipped": True}

    stage1_max = max_tokens_for(model_id, stage1_max_tokens_base, reasoning_multiplier)
    stage2_max = max_tokens_for(model_id, stage2_max_tokens_base, reasoning_multiplier)

    print(
        f"  {model_id}: {len(seeds)} seeds × {n_first_assocs} first-assocs × "
        f"{len(levels)} levels (stage1_max={stage1_max}, stage2_max={stage2_max})"
    )

    # --- stage 1: concurrent over seeds
    t0 = time.time()
    stage1_tasks = [
        _safe_call(
            async_client, sem, model_id,
            c_pace_stage1_prompt(seed),
            temperature, stage1_max, reasoning,
        )
        for seed in seeds
    ]
    stage1_raw = await asyncio.gather(*stage1_tasks)

    stage1_by_seed: dict[str, dict] = {}
    for seed, (raw, err) in zip(seeds, stage1_raw):
        parsed = parse_stage1(raw)[:n_first_assocs] if raw else []
        stage1_by_seed[seed] = {
            "raw": raw,
            "error": err,
            "parsed": parsed,
        }

    n_stage1_ok = sum(1 for v in stage1_by_seed.values() if len(v["parsed"]) >= 1)
    print(f"    stage1: {n_stage1_ok}/{len(seeds)} seeds had ≥1 parsed association "
          f"({time.time()-t0:.1f}s)")

    # --- stage 2: concurrent over (seed, first_assoc_idx, level)
    # Skip any seed that couldn't produce a first association.
    t0 = time.time()
    stage2_tasks = []
    stage2_keys = []  # parallel list of (seed, first_assoc_idx, level)

    for seed in seeds:
        parsed = stage1_by_seed[seed]["parsed"]
        if not parsed:
            continue
        for fa_idx, fa in enumerate(parsed):
            for level in levels:
                constraints = constraints_map[(seed, level)]
                prompt = c_pace_stage2_prompt(
                    seed=seed,
                    second_word=fa["word"],
                    reason=fa["reason"],
                    constraints=constraints,
                )
                stage2_tasks.append(_safe_call(
                    async_client, sem, model_id,
                    prompt, temperature, stage2_max, reasoning,
                ))
                stage2_keys.append((seed, fa_idx, level))

    stage2_raw = await asyncio.gather(*stage2_tasks) if stage2_tasks else []

    # Bucket stage-2 back by seed
    stage2_by_seed: dict[str, list] = {seed: [] for seed in seeds}
    for (seed, fa_idx, level), (raw, err) in zip(stage2_keys, stage2_raw):
        fa = stage1_by_seed[seed]["parsed"][fa_idx]
        chain = parse_stage2_chain(raw, seed=seed, second_word=fa["word"])
        constraints = constraints_map[(seed, level)]
        stage2_by_seed[seed].append({
            "first_assoc_idx": fa_idx,
            "first_assoc_word": fa["word"],
            "level": level,
            "constraints": constraints.to_dict(),
            "raw_response": raw,
            "api_error": err,
            "chain": chain,
        })

    n_stage2_ok = sum(
        1 for rec_list in stage2_by_seed.values() for r in rec_list
        if r["api_error"] is None and len(r["chain"]) >= 2
    )
    print(f"    stage2: {n_stage2_ok}/{len(stage2_tasks)} chains parsed "
          f"({time.time()-t0:.1f}s)")

    # Serialize
    output = {
        "model_id": model_id,
        "seeds": seeds,
        "levels": levels,
        "n_first_assocs": n_first_assocs,
        "constraints_map": {
            f"{seed}|{level}": constraints_map[(seed, level)].to_dict()
            for seed in seeds for level in levels
        },
        "results": [
            {
                "seed": seed,
                "stage1": stage1_by_seed[seed],
                "stage2": stage2_by_seed[seed],
            }
            for seed in seeds
        ],
    }
    with open(responses_path, "w") as f:
        json.dump(output, f, indent=2)

    summary = {
        "model_id": model_id,
        "model_key": model_key,
        "n_seeds": len(seeds),
        "n_stage1_ok": n_stage1_ok,
        "n_stage2_attempted": len(stage2_tasks),
        "n_stage2_parsed": n_stage2_ok,
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
                f"--resume requires an existing output_dir: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"RESUMING run in existing {output_dir} (models with "
              f"c_pace_responses.json will be skipped)")
    else:
        output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    models = config["models"]
    seeds = config.get("seeds", DEFAULT_SEEDS)
    if "n_seeds" in config:
        seeds = seeds[: config["n_seeds"]]
    levels = config.get("levels", [1, 2, 3, 4])
    n_first_assocs = config.get("n_first_assocs", 3)
    temperature = config.get("temperature", 0.0)
    concurrency = config.get("concurrency", 20)
    budget_usd = config.get("budget_usd", 0.0)
    reasoning_cfg = config.get("reasoning", {"effort": "low", "exclude": True})
    reasoning_multiplier = config.get("reasoning_max_tokens_multiplier", 4)
    stage1_max_tokens = config.get("stage1_max_tokens", 400)
    stage2_max_tokens = config.get("stage2_max_tokens", 1200)
    constraint_seed = config.get("constraint_seed", 42)
    constraint_type = config.get("constraint_type", "lexical")
    semantic_threshold = config.get("semantic_threshold", 0.4)
    semantic_seed_rejection = config.get("semantic_seed_rejection_threshold", 0.4)
    fasttext_path = config.get("fasttext_path", "resources/crawl-300d-2M.vec")

    if debug:
        seeds = seeds[:3]
        models = models[:1]

    # Pre-generate constraints per (seed, level) so they're fixed across all models.
    rng = random.Random(constraint_seed)
    constraints_map: dict[tuple[str, int], Constraints] = {}

    if constraint_type == "lexical":
        for seed in seeds:
            for level in levels:
                constraints_map[(seed, level)] = generate_constraints(level, rng)
    elif constraint_type == "semantic":
        print(f"Loading FastText for semantic constraint generation ({fasttext_path})...")
        embeddings = FastTextEmbeddings(fasttext_path)
        for seed in seeds:
            anchor_draw = build_semantic_anchor_fn(
                seed, embeddings, rng,
                seed_neighborhood_threshold=semantic_seed_rejection,
            )
            for level in levels:
                constraints_map[(seed, level)] = generate_semantic_constraints(
                    level, rng, anchor_draw, threshold=semantic_threshold,
                )
    else:
        raise ValueError(f"Unknown constraint_type: {constraint_type}")

    print(f"Models: {len(models)}")
    print(f"Seeds: {len(seeds)}  levels: {levels}  first-assocs: {n_first_assocs}")
    print(f"temperature: {temperature}  concurrency: {concurrency}")
    if budget_usd > 0:
        print(f"Budget cap: ${budget_usd:.2f}")
    else:
        print("Budget cap: NONE")

    # Print a sample of constraints so we can see what the task looks like
    print(f"\nconstraint_type = {constraint_type}")
    print("Sample constraints (first 6):")
    for k, v in list(constraints_map.items())[:6]:
        if v.type == "lexical":
            inc, exc = v.include_letters, v.exclude_letters
        else:
            inc, exc = v.include_anchors, v.exclude_anchors
        print(f"  seed={k[0]} level={k[1]}: include={inc} exclude={exc}")

    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    cumulative_cost = 0.0
    all_summaries: list[dict] = []

    for model_id in models:
        model_dir = output_dir / model_id_to_key(model_id)
        already_done = (model_dir / "c_pace_responses.json").exists()

        if not already_done:
            cost_est = estimate_model_cost(
                model_id, len(seeds), n_first_assocs, len(levels)
            )
            if budget_usd > 0 and (cumulative_cost + cost_est) > budget_usd:
                remaining = budget_usd - cumulative_cost
                print(f"\n{'='*60}")
                print(f"BUDGET CAP REACHED")
                print(f"  Cumulative (est): ${cumulative_cost:.2f}")
                print(f"  Cap:              ${budget_usd:.2f}")
                print(f"  Next model:       ${cost_est:.2f}  ({model_id})")
                print(f"  Remaining:        ${remaining:.2f}")
                print(f"{'='*60}")
                break
            cumulative_cost += cost_est
            print(f"\n{model_id}   (est ${cost_est:.2f}, cumulative ${cumulative_cost:.2f})")
        else:
            print(f"\n{model_id}   (already done)")

        summary = await run_model(
            async_client, sem,
            model_id=model_id,
            seeds=seeds,
            levels=levels,
            constraints_map=constraints_map,
            n_first_assocs=n_first_assocs,
            temperature=temperature,
            stage1_max_tokens_base=stage1_max_tokens,
            stage2_max_tokens_base=stage2_max_tokens,
            reasoning_multiplier=reasoning_multiplier,
            reasoning=reasoning_cfg,
            output_dir=output_dir,
        )
        all_summaries.append(summary)

    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume into an existing output_dir; models with "
                             "c_pace_responses.json are skipped.")
    args = parser.parse_args()
    asyncio.run(main(args.config_path, args.overwrite, args.debug, args.resume))
