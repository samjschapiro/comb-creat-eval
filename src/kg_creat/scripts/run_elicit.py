"""Elicit model paths for the kg_creat constraint benchmark.

Downstream of ``sample_bundles.py``: renders each prompt spec, queries each model over
the OPEN KG (no graph in context), parses the emitted paths with CREATE's parser, and
saves raw + parsed responses per model. Resume-safe (skips models with responses.json),
budget-capped (PRICING), and free when pointed at a local server via LLM_BASE_URL.

    python src/kg_creat/scripts/run_elicit.py configs/kg_creat/run_elicit.yaml --overwrite
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, init_directory, save_config  # noqa: E402
from src.dat_eval.llm import call_llm_async, get_async_client, model_id_to_key  # noqa: E402
from src.kg_creat.prompts import build_prompt  # noqa: E402
from src.kg_creat.parse import parse_paths  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "safety"))
from cost_tracker import PRICING  # noqa: E402

EST_INPUT_TOKENS = 750    # rendered prompt (CREATE-scaffolded, ~2x longer)
EST_OUTPUT_TOKENS = 900   # k paths x ~3 triples in JSON

REASONING_MODELS = {
    "openai/o3", "openai/o3-mini", "openai/o4-mini",
    "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano",
    "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    "deepseek/deepseek-r1", "qwen/qwq-32b",
}


def _local_mode() -> bool:
    return bool(os.environ.get("LLM_BASE_URL"))


def estimate_model_cost(model_id: str, n_prompts: int) -> float:
    """USD estimate; 0 in local-server mode; +inf for unpriced models (cap refuses)."""
    if _local_mode():
        return 0.0
    pricing = PRICING.get(model_id)
    if pricing is None:
        return float("inf")
    in_price, out_price = pricing
    return (n_prompts * EST_INPUT_TOKENS * in_price + n_prompts * EST_OUTPUT_TOKENS * out_price) / 1_000_000


async def _run_one(async_client, sem, model_id, spec, max_tokens, temperature, reasoning):
    prompt_text = build_prompt(spec)
    messages = [{"role": "user", "content": prompt_text}]
    base = {k: spec.get(k) for k in ("prompt_id", "bundle_id", "regime", "mode",
                                     "u", "v", "u_label", "v_label", "h", "k", "constraint",
                                     "domain_u", "domain_v", "cross_domain")}
    async with sem:
        try:
            raw = await call_llm_async(async_client, messages=messages, model=model_id,
                                       temperature=temperature, max_tokens=max_tokens, reasoning=reasoning)
            if raw is None:
                return {**base, "raw_response": None, "paths": [], "n_paths": 0,
                        "parse_success": False, "api_error": "null content"}
            paths = parse_paths(raw)
            return {**base, "raw_response": raw,
                    "paths": [p.triples for p in paths], "n_paths": len(paths),
                    "parse_success": len(paths) > 0, "api_error": None}
        except Exception as e:  # noqa: BLE001
            return {**base, "raw_response": None, "paths": [], "n_paths": 0,
                    "parse_success": False, "api_error": f"{type(e).__name__}: {e}"}


async def run_model(async_client, sem, model_id, specs, max_tokens, temperature, reasoning, output_dir):
    model_dir = output_dir / model_id_to_key(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    responses_path = model_dir / "responses.json"
    if responses_path.exists():
        print(f"  {model_id}: responses.json exists, skipping")
        return json.loads(responses_path.read_text())

    print(f"  {model_id}: {len(specs)} prompts, max_tokens={max_tokens}, firing ...")
    t0 = time.time()
    results = await asyncio.gather(*[
        _run_one(async_client, sem, model_id, s, max_tokens, temperature, reasoning) for s in specs
    ])
    responses_path.write_text(json.dumps(results, indent=2))
    n_ok = sum(1 for r in results if r["parse_success"])
    n_api = sum(1 for r in results if r["api_error"])
    (model_dir / "summary.json").write_text(json.dumps({
        "model_id": model_id, "n_prompts": len(specs), "n_parsed": n_ok,
        "n_api_fail": n_api, "elapsed_seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"    done in {time.time()-t0:.1f}s — parsed={n_ok} api_fail={n_api}")
    return results


async def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    upstream_dir = Path(config["upstream_dir"])
    prompts_path = upstream_dir / "prompts.json"
    if not prompts_path.exists():
        raise FileNotFoundError(f"FATAL: no prompts.json at {prompts_path} -- run sample_bundles.py first")

    # Resume-safe: --overwrite starts fresh; otherwise keep the dir and add new models
    # (per-model skip below), so we can append models to an existing run without re-spending.
    if overwrite:
        output_dir = init_directory(config["output_dir"], overwrite=True)
    else:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    specs = json.loads(prompts_path.read_text())
    modes = config.get("modes")  # optional: restrict to a subset of modes (e.g. [analogy])
    if modes:
        specs = [s for s in specs if s["mode"] in modes]
        print(f"Filtered to modes {modes}: {len(specs)} prompts")
    if debug:
        specs = specs[:3]
        print(f"DEBUG: using {len(specs)} prompts")
    print(f"Loaded {len(specs)} prompt specs (open-vocabulary relations)")

    eval_cfg = config.get("eval", {})
    temperature = eval_cfg.get("temperature", 0.7)
    max_tokens = eval_cfg.get("max_tokens", 1500)
    concurrency = config.get("concurrency", 8)
    reasoning = config.get("reasoning", {"effort": "low", "exclude": True})
    budget_usd = config.get("budget_usd", 0.0)
    models = config["models"]
    print(f"local_mode={_local_mode()}  models={models}  concurrency={concurrency}  "
          f"budget=${budget_usd:.2f}  temp={temperature}")

    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)
    cumulative, summaries = 0.0, []

    for model_id in models:
        done = (output_dir / model_id_to_key(model_id) / "responses.json").exists()
        if not done:
            est = estimate_model_cost(model_id, len(specs))
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
        results = await run_model(async_client, sem, model_id, specs, mt, temperature, reasoning_here, output_dir)
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
