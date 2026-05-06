"""Smoke test for the DRAT pipeline.

Runs DRAT on a small hand-crafted bank against one cheap model, verifying
that LLM calls, parsing, SBERT embedding, and scoring all work end-to-end
before scaling to the full pilot.

Usage:
    uv run python src/new_tests/scripts/run_drat_smoke.py configs/new_tests/drat_smoke.yaml --overwrite
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Add project root to path so `from src...` imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory, save_config
from src.dat_eval.cdat import SBERTEmbeddings, DEFAULT_CUES
from src.dat_eval.llm import (
    get_async_client,
    call_llm_async,
    extract_words_from_response,
)
from src.new_tests.drat import score_drat, compute_tau, drat_prompt


def resolve_noun_pool(config: dict) -> list[str]:
    """Resolve the random-noun pool from the config.

    Two ways to specify the pool:
      noun_pool: [list of strings]            -- explicit
      noun_pool_source: "cdat_default_cues"   -- named reference

    Crashes loudly if the source name is unknown.
    """
    if "noun_pool" in config and "noun_pool_source" in config:
        raise ValueError(
            "FATAL: specify either 'noun_pool' or 'noun_pool_source', not both"
        )
    if "noun_pool" in config:
        return list(config["noun_pool"])
    if "noun_pool_source" in config:
        source = config["noun_pool_source"]
        if source == "cdat_default_cues":
            return list(DEFAULT_CUES)
        raise ValueError(f"FATAL: unknown noun_pool_source: {source!r}")
    raise ValueError("FATAL: config must specify 'noun_pool' or 'noun_pool_source'")


async def run_pair_for_model(
    async_client,
    model: str,
    anchor_a: str,
    anchor_b: str,
    embeddings: SBERTEmbeddings,
    tau: float,
    n_min: int,
    temperature: float,
    max_tokens: int,
    seed: int | None,
    top_p: float,
    prompt_style: str = "default",
    reasoning: dict | None = None,
):
    prompt = drat_prompt(anchor_a, anchor_b, style=prompt_style)
    raw = await call_llm_async(
        async_client,
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        top_p=top_p,
        reasoning=reasoning,
    )
    words = extract_words_from_response(raw, expected_count=10)
    score = score_drat(words, anchor_a, anchor_b, embeddings, tau, n_min)
    return {
        "model": model,
        "anchor_a": anchor_a,
        "anchor_b": anchor_b,
        "raw_response": raw,
        "extracted_words": words,
        "score": score,
    }


async def main_async(config: dict, output_dir: Path) -> None:
    print("Loading SBERT...")
    embeddings = SBERTEmbeddings()

    noun_pool = resolve_noun_pool(config)
    print(f"Random-noun pool size: {len(noun_pool)}")

    # Per-pair tau calibration
    print("\nComputing per-pair thresholds...")
    pair_taus: dict[tuple[str, str], dict] = {}
    for pair in config["anchor_pairs"]:
        a, b = pair["anchor_a"], pair["anchor_b"]
        result = compute_tau(
            a, b, noun_pool, embeddings,
            percentile=config.get("tau_percentile", 90.0),
        )
        pair_taus[(a, b)] = result
        print(f"  ({a!r}, {b!r}): tau={result['tau']:.3f}")

    with open(output_dir / "tau_calibration.json", "w") as f:
        json.dump(
            {
                f"{a} | {b}": {
                    "tau": v["tau"],
                    "percentile": v["percentile"],
                    "noun_pool_size": v["noun_pool_size"],
                    "noun_utilities": v["noun_utilities"],
                }
                for (a, b), v in pair_taus.items()
            },
            f, indent=2,
        )

    # LLM calls — concurrent with a semaphore. Sequential when concurrency=1.
    n_models = len(config["models"])
    n_pairs = len(config["anchor_pairs"])
    concurrency = config.get("concurrency", 10)
    print(f"\nRunning {n_models} model(s) × {n_pairs} pair(s) "
          f"= {n_models * n_pairs} call(s) at concurrency={concurrency}...")
    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    max_retries = config.get("max_retries", 4)

    reasoning_models = set(config.get("reasoning_models", []))
    reasoning_cfg = config.get("reasoning")
    reasoning_mult = config.get("reasoning_max_tokens_multiplier", 4)

    async def bounded(model: str, pair: dict) -> dict:
        a, b = pair["anchor_a"], pair["anchor_b"]
        tau = pair_taus[(a, b)]["tau"]
        is_reasoning = model in reasoning_models
        eff_max_tokens = (
            config.get("max_tokens", 256) * reasoning_mult
            if is_reasoning
            else config.get("max_tokens", 256)
        )
        eff_reasoning = reasoning_cfg if is_reasoning else None
        async with sem:
            last_err: Exception | None = None
            for attempt in range(max_retries):
                try:
                    r = await run_pair_for_model(
                        async_client,
                        model=model,
                        anchor_a=a,
                        anchor_b=b,
                        embeddings=embeddings,
                        tau=tau,
                        n_min=config.get("n_min", 5),
                        temperature=config.get("temperature", 1.0),
                        max_tokens=eff_max_tokens,
                        seed=config.get("seed"),
                        top_p=config.get("top_p", 1.0),
                        prompt_style=config.get("prompt_style", "default"),
                        reasoning=eff_reasoning,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    msg = str(e).lower()
                    is_rate_limit = "429" in msg or "rate" in msg
                    is_transient = is_rate_limit or "503" in msg or "timeout" in msg
                    if not is_transient or attempt == max_retries - 1:
                        break
                    backoff = 2 ** attempt + 0.5  # 1.5, 2.5, 4.5, 8.5 ...
                    print(f"  retry {attempt+1}/{max_retries} for {model} on "
                          f"({a!r}, {b!r}) after {backoff:.1f}s: {type(e).__name__}")
                    await asyncio.sleep(backoff)
        if last_err is not None:
            print(f"  ERROR {model} | ({a!r}, {b!r}): {type(last_err).__name__}: {last_err}")
            return {
                "model": model,
                "anchor_a": a,
                "anchor_b": b,
                "raw_response": None,
                "extracted_words": [],
                "error": f"{type(last_err).__name__}: {last_err}",
                "score": {
                    "drat": 0.0,
                    "n_valid": 0,
                    "n_survivors": 0,
                    "survivors": [],
                    "scored_words": [],
                    "utilities": [],
                    "tau": tau,
                    "sufficient": False,
                    "reason": f"call failed: {type(last_err).__name__}",
                },
            }
        s = r["score"]
        survivors_str = (
            f"survivors={s['n_survivors']}/{s['n_valid']}"
            if s["sufficient"]
            else f"GATE FAIL: {s.get('reason', 'unknown')}"
        )
        print(f"  {model} | ({a!r}, {b!r}): DRAT={s['drat']:.2f} | {survivors_str}")
        return r

    tasks = [
        bounded(model, pair)
        for model in config["models"]
        for pair in config["anchor_pairs"]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Per-model summary
    summary: dict[str, dict] = {}
    for r in results:
        model = r["model"]
        summary.setdefault(model, {"drat_scores": [], "pair_results": []})
        summary[model]["drat_scores"].append(r["score"]["drat"])
        summary[model]["pair_results"].append({
            "anchor_a": r["anchor_a"],
            "anchor_b": r["anchor_b"],
            "drat": r["score"]["drat"],
            "n_survivors": r["score"]["n_survivors"],
            "sufficient": r["score"]["sufficient"],
        })
    for model in summary:
        scores = summary[model]["drat_scores"]
        summary[model]["mean_drat"] = float(np.mean(scores))
        summary[model]["n_pairs"] = len(scores)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPer-model DRAT summary:")
    for model, s in summary.items():
        print(f"  {model}: mean DRAT = {s['mean_drat']:.2f} over {s['n_pairs']} pair(s)")

    # Sanity-check criteria from the smoke-test plan
    print("\nSanity checks:")
    all_taus = [v["tau"] for v in pair_taus.values()]
    tau_in_range = all(0.10 < t < 0.40 for t in all_taus)
    print(f"  tau values in [0.10, 0.40]: {tau_in_range} ({all_taus})")
    drats = [r["score"]["drat"] for r in results]
    drat_in_range = all(0.0 <= d <= 200.0 for d in drats)
    print(f"  DRAT values in [0, 200]: {drat_in_range}")
    any_nonzero = any(d > 0 for d in drats)
    print(f"  At least one non-zero DRAT: {any_nonzero}")


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if "models" not in config:
        raise ValueError("FATAL: 'models' is required in config")
    if "anchor_pairs" not in config:
        raise ValueError("FATAL: 'anchor_pairs' is required in config")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    if debug:
        print(f"DEBUG MODE: output_dir = {output_dir}")

    asyncio.run(main_async(config, output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to YAML config")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
