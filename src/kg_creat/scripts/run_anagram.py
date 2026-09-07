"""Elicit anagram candidates for the exploratory-creativity task.

Anagram output is a JSON dict of strings (not triple-paths), so this runner mirrors run_elicit.py's
structure (resume-safe per model, M resamples x temperature, budget cap, PRICING estimate) but parses
with parse_anagrams and stores the raw candidate list. Scoring is a separate, judge-free pass
(score_anagram.py): deterministic letter-multiset validity + lexicon/Wikidata meaningfulness + novelty.

    python src/kg_creat/scripts/run_anagram.py configs/kg_creat/run_anagram.yaml --overwrite
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, init_directory, save_config  # noqa: E402
from src.dat_eval.llm import call_llm_async, get_async_client, model_id_to_key  # noqa: E402
from src.kg_creat.prompts import build_prompt  # noqa: E402
from src.kg_creat.parse import parse_anagrams  # noqa: E402
from src.kg_creat.scripts.run_elicit import (  # noqa: E402
    _local_mode, REASONING_MODELS, PRICING)

EST_INPUT_TOKENS = 230    # anagram prompt is short (single stimulus, no CREATE scaffolding)
EST_OUTPUT_TOKENS = 400   # open-ended list of short strings; capped by max_tokens


def estimate_model_cost(model_id: str, n_draws: int) -> float:
    if _local_mode():
        return 0.0
    pricing = PRICING.get(model_id)
    if pricing is None:
        return float("inf")
    in_price, out_price = pricing
    return (n_draws * EST_INPUT_TOKENS * in_price + n_draws * EST_OUTPUT_TOKENS * out_price) / 1_000_000


async def _run_one(async_client, sem, model_id, spec, max_tokens, temperature, sample_idx, reasoning):
    prompt_text = build_prompt(spec)
    messages = [{"role": "user", "content": prompt_text}]
    base = {k: spec.get(k) for k in ("prompt_id", "bundle_id", "regime", "mode",
                                     "u", "u_label", "domain_u")}
    base = {**base, "temperature": temperature, "sample_idx": sample_idx}
    async with sem:
        try:
            raw = await call_llm_async(async_client, messages=messages, model=model_id,
                                       temperature=temperature, max_tokens=max_tokens, reasoning=reasoning)
            if raw is None:
                return {**base, "raw_response": None, "candidates": [], "n_candidates": 0,
                        "parse_success": False, "api_error": "null content"}
            cands = parse_anagrams(raw)
            return {**base, "raw_response": raw, "candidates": cands, "n_candidates": len(cands),
                    "parse_success": len(cands) > 0, "api_error": None}
        except Exception as e:  # noqa: BLE001
            return {**base, "raw_response": None, "candidates": [], "n_candidates": 0,
                    "parse_success": False, "api_error": f"{type(e).__name__}: {e}"}


async def run_model(async_client, sem, model_id, specs, max_tokens, temperatures, n_samples,
                    reasoning, output_dir):
    model_dir = output_dir / model_id_to_key(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    responses_path = model_dir / "responses.json"
    if responses_path.exists():
        print(f"  {model_id}: responses.json exists, skipping")
        return json.loads(responses_path.read_text())

    draws = [(s, t, i) for s in specs for t in temperatures for i in range(n_samples)]
    print(f"  {model_id}: {len(specs)} anchors x {len(temperatures)} temps x {n_samples} samples "
          f"= {len(draws)} draws, max_tokens={max_tokens}, firing ...")
    t0 = time.time()
    results = await asyncio.gather(*[
        _run_one(async_client, sem, model_id, s, max_tokens, t, i, reasoning) for s, t, i in draws
    ])
    responses_path.write_text(json.dumps(results, indent=2, default=str))
    n_ok = sum(1 for r in results if r["parse_success"])
    n_api = sum(1 for r in results if r["api_error"])
    (model_dir / "summary.json").write_text(json.dumps({
        "model_id": model_id, "n_anchors": len(specs), "temperatures": temperatures,
        "n_samples": n_samples, "n_draws": len(draws), "n_parsed": n_ok,
        "n_api_fail": n_api, "elapsed_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"    done in {time.time()-t0:.1f}s — parsed={n_ok}/{len(draws)} api_fail={n_api}")
    return results


async def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    prompts_path = Path(config["upstream_dir"]) / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"FATAL: no prompts.json at {prompts_path} -- run sample_anagram.py first")

    if overwrite:
        output_dir = init_directory(config["output_dir"], overwrite=True)
    else:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    specs = [s for s in json.loads(prompts_path.read_text()) if s.get("mode") == "anagram"]
    if debug:
        specs = specs[:3]
        print(f"DEBUG: using {len(specs)} anchors")
    print(f"Loaded {len(specs)} anagram anchors")

    eval_cfg = config.get("eval", {})
    temperatures = eval_cfg.get("temperatures") or [eval_cfg.get("temperature", 0.7)]
    n_samples = eval_cfg.get("n_samples", 1)
    max_tokens = eval_cfg.get("max_tokens", 1024)
    concurrency = config.get("concurrency", 8)
    reasoning = config.get("reasoning", {"effort": "low", "exclude": True})
    budget_usd = config.get("budget_usd", 0.0)
    models = config["models"]
    draws_per_prompt = len(temperatures) * n_samples
    print(f"local_mode={_local_mode()}  models={len(models)}  concurrency={concurrency}  "
          f"budget=${budget_usd:.2f}  temps={temperatures}  M={n_samples}  ({draws_per_prompt}x/anchor)")

    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)
    cumulative, summaries = 0.0, []

    for model_id in models:
        done = (output_dir / model_id_to_key(model_id) / "responses.json").exists()
        if not done:
            est = estimate_model_cost(model_id, len(specs) * draws_per_prompt)
            reasoning_here = reasoning if model_id in REASONING_MODELS else None
            mt = max_tokens * 4 if model_id in REASONING_MODELS else max_tokens
            if budget_usd > 0 and cumulative + est > budget_usd:
                print(f"\nBUDGET CAP REACHED: next model {model_id} est ${est:.3f} "
                      f"would exceed ${budget_usd:.2f} (spent ~${cumulative:.3f}). Stopping.")
                break
            cumulative += est
            print(f"\n{model_id}  (est ${est:.4f}, cumulative ${cumulative:.4f})")
        else:
            reasoning_here, mt = None, max_tokens
            print(f"\n{model_id}  (already done)")
        results = await run_model(async_client, sem, model_id, specs, mt, temperatures, n_samples,
                                  reasoning_here, output_dir)
        summaries.append({"model_id": model_id, "n": len(results)})

    (output_dir / "run_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved responses to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.config_path, args.overwrite, args.debug))
