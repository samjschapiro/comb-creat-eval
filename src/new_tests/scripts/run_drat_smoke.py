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


def _normalize_anchor_groups(config: dict) -> list[dict]:
    """Read anchor_groups (list with 'anchors': [...]) or anchor_pairs (list
    with 'anchor_a' / 'anchor_b'); return a canonical list of dicts each with
    an 'anchors' key (list of strings) and any metadata.
    """
    if "anchor_groups" in config:
        out = []
        for g in config["anchor_groups"]:
            if "anchors" not in g:
                raise ValueError(f"FATAL: anchor_groups entry missing 'anchors' list: {g}")
            out.append({**g, "anchors": list(g["anchors"])})
        return out
    if "anchor_pairs" in config:
        return [
            {**p, "anchors": [p["anchor_a"], p["anchor_b"]]}
            for p in config["anchor_pairs"]
        ]
    raise ValueError("FATAL: config must specify 'anchor_groups' or 'anchor_pairs'")


async def run_group_for_model(
    async_client,
    model: str,
    anchors: list[str],
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
    prompt = drat_prompt(list(anchors), style=prompt_style)
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
    score = score_drat(words, list(anchors), embeddings, tau, n_min=n_min)
    return {
        "model": model,
        "anchors": list(anchors),
        "raw_response": raw,
        "extracted_words": words,
        "score": score,
    }


# Backward-compatibility shim for any external callers still using the old API.
async def run_pair_for_model(async_client, model, anchor_a, anchor_b, embeddings,
                              tau, n_min, temperature, max_tokens, seed, top_p,
                              prompt_style="default", reasoning=None):
    r = await run_group_for_model(
        async_client, model, [anchor_a, anchor_b], embeddings, tau, n_min,
        temperature, max_tokens, seed, top_p, prompt_style, reasoning,
    )
    # Old-format keys for backward compat with downstream analysis scripts
    return {**r, "anchor_a": anchor_a, "anchor_b": anchor_b}


async def main_async(config: dict, output_dir: Path) -> None:
    print("Loading SBERT...")
    embeddings = SBERTEmbeddings()

    noun_pool = resolve_noun_pool(config)
    print(f"Random-noun pool size: {len(noun_pool)}")

    groups = _normalize_anchor_groups(config)
    n_anchors_per = len(groups[0]["anchors"]) if groups else 0
    print(f"Anchor groups: {len(groups)} (each with {n_anchors_per} anchors)")

    # Per-group tau calibration
    print("\nComputing per-group thresholds...")
    group_taus: dict[str, dict] = {}
    for g in groups:
        anchors = g["anchors"]
        result = compute_tau(
            list(anchors), noun_pool, embeddings,
            percentile=config.get("tau_percentile", 90.0),
        )
        key = " | ".join(anchors)
        group_taus[key] = result
        print(f"  ({key}): tau={result['tau']:.3f}")

    with open(output_dir / "tau_calibration.json", "w") as f:
        json.dump(
            {
                k: {
                    "tau": v["tau"],
                    "percentile": v["percentile"],
                    "noun_pool_size": v["noun_pool_size"],
                    "noun_utilities": v["noun_utilities"],
                    "n_anchors": v.get("n_anchors"),
                }
                for k, v in group_taus.items()
            },
            f, indent=2,
        )

    # LLM calls — concurrent with a semaphore. Sequential when concurrency=1.
    n_models = len(config["models"])
    n_groups = len(groups)
    concurrency = config.get("concurrency", 10)
    print(f"\nRunning {n_models} model(s) × {n_groups} group(s) "
          f"= {n_models * n_groups} call(s) at concurrency={concurrency}...")
    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    max_retries = config.get("max_retries", 4)

    reasoning_models = set(config.get("reasoning_models", []))
    reasoning_cfg = config.get("reasoning")
    reasoning_mult = config.get("reasoning_max_tokens_multiplier", 4)

    async def bounded(model: str, group: dict) -> dict:
        anchors = group["anchors"]
        key = " | ".join(anchors)
        tau = group_taus[key]["tau"]
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
                    r = await run_group_for_model(
                        async_client,
                        model=model,
                        anchors=list(anchors),
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
                    # Inject backward-compat keys for 2-anchor groups
                    if len(anchors) == 2:
                        r = {**r, "anchor_a": anchors[0], "anchor_b": anchors[1]}
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
                          f"({key!r}) after {backoff:.1f}s: {type(e).__name__}")
                    await asyncio.sleep(backoff)
        if last_err is not None:
            print(f"  ERROR {model} | ({key!r}): {type(last_err).__name__}: {last_err}")
            return {
                "model": model,
                "anchors": list(anchors),
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
        print(f"  {model} | ({key!r}): DRAT={s['drat']:.2f} | {survivors_str}")
        return r

    tasks = [
        bounded(model, group)
        for model in config["models"]
        for group in groups
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Per-model summary
    summary: dict[str, dict] = {}
    for r in results:
        model = r["model"]
        summary.setdefault(model, {"drat_scores": [], "group_results": []})
        summary[model]["drat_scores"].append(r["score"]["drat"])
        summary[model]["group_results"].append({
            "anchors": r.get("anchors") or [r.get("anchor_a"), r.get("anchor_b")],
            "drat": r["score"]["drat"],
            "n_survivors": r["score"]["n_survivors"],
            "sufficient": r["score"]["sufficient"],
        })
    for model in summary:
        scores = summary[model]["drat_scores"]
        summary[model]["mean_drat"] = float(np.mean(scores))
        summary[model]["n_groups"] = len(scores)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nPer-model DRAT summary:")
    for model, s in summary.items():
        print(f"  {model}: mean DRAT = {s['mean_drat']:.2f} over {s['n_groups']} group(s)")

    print("\nSanity checks:")
    all_taus = [v["tau"] for v in group_taus.values()]
    tau_in_range = all(0.10 < t < 0.50 for t in all_taus)
    print(f"  tau values in [0.10, 0.50]: {tau_in_range}")
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
    if "anchor_pairs" not in config and "anchor_groups" not in config:
        raise ValueError("FATAL: 'anchor_pairs' or 'anchor_groups' is required in config")

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
