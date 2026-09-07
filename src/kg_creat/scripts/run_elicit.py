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
from src.kg_creat.parse import parse_items, parse_blend  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "safety"))
from cost_tracker import PRICING  # noqa: E402

EST_INPUT_TOKENS = 750    # rendered prompt (CREATE-scaffolded, ~2x longer)
EST_OUTPUT_TOKENS = 400   # measured: baseline ~400, analogy ~224, blending ~162 tok

REASONING_MODELS = {
    "openai/o3", "openai/o3-mini", "openai/o4-mini",
    "openai/gpt-5", "openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5.2",
    "openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.4-nano",
    "deepseek/deepseek-r1", "qwen/qwq-32b",
    # newer reasoning-capable models (over-including is safe: it only raises the token cap and passes a
    # low-effort reasoning param OpenRouter ignores when unsupported).
    "openai/gpt-5.6-sol", "x-ai/grok-4.6", "x-ai/grok-4.5",
    "google/gemini-3.1-pro-preview", "google/gemini-3.7-flash",
    "google/gemini-3-flash-preview", "google/gemini-2.5-pro",
    "qwen/qwen3-max", "qwen/qwen3-235b-a22b", "qwen/qwen3-30b-a3b", "z-ai/glm-4.6",
    # Claude 5 family (and opus-4.6) are extended-thinking by default -> at low max_tokens the thinking
    # consumes the budget and returns empty ("null content"); they need the 8x bump + a bounded param.
    "anthropic/claude-opus-5", "anthropic/claude-opus-4.6", "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-5", "anthropic/claude-sonnet-4.6", "anthropic/claude-sonnet-4.5",
    "anthropic/claude-fable-5",
    # Added 2026-09-05: both returned "null content" on the spread expansion (glm-4.5-air 48/90,
    # kimi-k2 12/90) with traces emitted -- the same thinking-eats-the-budget failure, so they need
    # the same headroom.
    "z-ai/glm-4.5-air", "moonshotai/kimi-k2",
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


async def _run_one(async_client, sem, model_id, spec, max_tokens, temperature, sample_idx, reasoning,
                   provider=None):
    prompt_text = build_prompt(spec)
    messages = [{"role": "user", "content": prompt_text}]
    base = {k: spec.get(k) for k in ("prompt_id", "bundle_id", "regime", "mode",
                                     "u", "v", "u_label", "v_label", "h", "k", "constraint",
                                     "domain_u", "domain_v", "cross_domain")}
    # temperature + sample_idx identify this draw; diversity is measured per (prompt_id, temperature)
    # over the M independent samples, so both must travel with the response.
    base = {**base, "temperature": temperature, "sample_idx": sample_idx}
    async with sem:
        try:
            if provider is not None:
                # non-OpenRouter route (LiteLLM gateway / Anthropic direct); see providers.py
                raw, reasoning_trace, usage = await provider.acall(
                    messages=messages, model=model_id, temperature=temperature, max_tokens=max_tokens)
            else:
                raw, reasoning_trace, usage = await call_llm_async(
                    async_client, messages=messages, model=model_id, temperature=temperature,
                    max_tokens=max_tokens, reasoning=reasoning, capture_reasoning=True, capture_usage=True)
            base["reasoning"] = reasoning_trace   # saved for analysis (also present when content is empty)
            base["usage"] = usage                 # actual billed tokens {in, out} -> cost ledger
            if raw is None:
                return {**base, "raw_response": None, "paths": [], "n_paths": 0,
                        "parse_success": False, "api_error": "null content"}
            mode = spec.get("mode")
            if mode == "blending":
                # Fusion blend: ONE object {concept, generic_space, structure, emergent}. Shaped like an
                # association item (structure = one path, emergent -> "inferences") plus two string
                # fields; single item per prompt. See docs/tracks/kg_creat/blending_fusion.md.
                blend = parse_blend(raw)
                if blend is None:
                    items = []
                else:
                    items = [{"paths": [blend["structure"].triples], "inferences": blend["emergent"],
                              "concept": blend["concept"], "generic_space": blend["generic_space"],
                              "tags": blend["tags"]}]
                flat = [it["paths"][0] for it in items]
                return {**base, "raw_response": raw, "items": items,
                        "paths": flat, "n_paths": len(flat), "n_items": len(items),
                        "parse_success": len(items) > 0, "api_error": None}
            # association / analogy emit a JSON array of item objects, each with its path(s) + emergent
            # "inferences". association: one "path" per item; analogy: a "path_a"/"path_b" PAIR. Paths
            # are flattened into "paths" (a,b interleaved) for scoring.
            path_keys = {"analogy": ("path_a", "path_b")}.get(mode, ("path",))
            items = parse_items(raw, path_keys)
            flat = [p.triples for it in items for p in it["paths"]]
            result = {**base, "raw_response": raw,
                      "items": [{"paths": [p.triples for p in it["paths"]],
                                 "inferences": it["inferences"],
                                 **{k: it[k] for k in ("projected", "invention", "projection") if k in it}}
                                for it in items],
                      "paths": flat, "n_paths": len(flat), "n_items": len(items),
                      "parse_success": len(items) > 0, "api_error": None}
            if mode == "analogy":
                result["pairs"] = [[it["paths"][0].triples, it["paths"][1].triples] for it in items]
                result["n_pairs"] = len(items)
            return result
        except Exception as e:  # noqa: BLE001
            return {**base, "raw_response": None, "paths": [], "n_paths": 0,
                    "parse_success": False, "api_error": f"{type(e).__name__}: {e}"}


async def run_model(async_client, sem, model_id, specs, max_tokens, temperatures, n_samples,
                    reasoning, output_dir, config_name="", provider=None):
    model_dir = output_dir / model_id_to_key(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    responses_path = model_dir / "responses.json"
    if responses_path.exists():
        print(f"  {model_id}: responses.json exists, skipping")
        return json.loads(responses_path.read_text()), 0.0

    draws = [(s, t, i) for s in specs for t in temperatures for i in range(n_samples)]
    print(f"  {model_id}: {len(specs)} prompts x {len(temperatures)} temps x {n_samples} samples "
          f"= {len(draws)} draws, max_tokens={max_tokens}, firing ...")
    t0 = time.time()
    results = await asyncio.gather(*[
        _run_one(async_client, sem, model_id, s, max_tokens, t, i, reasoning, provider)
        for s, t, i in draws
    ])
    # default=str so a single unexpected non-serializable value can never discard a whole model's
    # completed draws (a set once slipped through and crashed the write after all draws were spent).
    responses_path.write_text(json.dumps(results, indent=2, default=str))
    n_ok = sum(1 for r in results if r["parse_success"])
    n_api = sum(1 for r in results if r["api_error"])
    if n_ok == 0 and n_api == len(results):
        # Every draw failed at the API. Persisting this would make responses.json exist, which the
        # resume check reads as "already done" -- blocking retries and, in a shared pool directory,
        # silently adding an empty model to the pool. Fail loudly and write nothing.
        errs = sorted({str(r["api_error"])[:130] for r in results})
        raise RuntimeError(f"FATAL: {model_id}: all {len(results)} draws failed at the API; "
                           f"nothing written. Distinct errors: {errs}")
    # Actual billed tokens (reasoning tokens included in "out") -> persistent cost ledger. Only fresh
    # (non-skipped) runs reach here, so re-running a completed model never double-counts spend.
    in_tok = sum((r.get("usage") or {}).get("in", 0) for r in results)
    out_tok = sum((r.get("usage") or {}).get("out", 0) for r in results)
    n_trace = sum(1 for r in results if r.get("reasoning"))
    (model_dir / "summary.json").write_text(json.dumps({
        "model_id": model_id, "n_prompts": len(specs), "temperatures": temperatures,
        "n_samples": n_samples, "n_draws": len(draws), "n_parsed": n_ok,
        "n_api_fail": n_api, "in_tokens": in_tok, "out_tokens": out_tok,
        "n_reasoning_traces": n_trace, "elapsed_seconds": round(time.time() - t0, 1)}, indent=2))
    actual_cost = 0.0
    try:
        from src.kg_creat.cost_ledger import record
        e = record("elicit", model_id, len(results), in_tok, out_tok, config=config_name,
                   note=f"parsed={n_ok}/{len(draws)} api_fail={n_api} traces={n_trace}")
        actual_cost = e["cost_usd"] or 0.0
        cost_str = f"${e['cost_usd']:.4f}" if e["cost_usd"] is not None else "unpriced"
        print(f"    [ledger] {model_id}: {in_tok:,}+{out_tok:,} tok -> {cost_str}  (traces={n_trace}/{len(draws)})")
    except Exception as ex:  # noqa: BLE001
        print(f"    [ledger] record failed: {ex}")
    print(f"    done in {time.time()-t0:.1f}s — parsed={n_ok}/{len(draws)} api_fail={n_api}")
    return results, actual_cost


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
    # temperatures: a list for the decoding sweep; n_samples (M): resamples per (prompt, temp) for
    # set-level diversity. Back-compat: a scalar `temperature` / absent `n_samples` => single draw.
    temperatures = eval_cfg.get("temperatures") or [eval_cfg.get("temperature", 0.7)]
    n_samples = eval_cfg.get("n_samples", 1)
    max_tokens = eval_cfg.get("max_tokens", 1500)
    concurrency = config.get("concurrency", 8)
    # exclude=False so OpenRouter RETURNS the reasoning trace (we already pay for the tokens; keep them
    # for analysis). capture happens in _run_one via capture_reasoning=True.
    reasoning = config.get("reasoning", {"effort": "low", "exclude": False})
    budget_usd = config.get("budget_usd", 0.0)
    models = config["models"]
    draws_per_prompt = len(temperatures) * n_samples
    print(f"local_mode={_local_mode()}  models={models}  concurrency={concurrency}  "
          f"budget=${budget_usd:.2f}  temps={temperatures}  M={n_samples}  ({draws_per_prompt}x/prompt)")

    # A `provider:` block routes elicitation off OpenRouter WITHOUT touching LLM_BASE_URL, which would
    # also drag score.py's judge client along and disable the budget cap (see providers.py).
    provider_cfg = config.get("provider")
    provider = None
    if provider_cfg:
        from src.kg_creat.providers import build as build_provider
        provider = build_provider(provider_cfg)
        print(f"provider: {provider_cfg.get('kind')} "
              f"(effort={provider_cfg.get('effort')}, "
              f"thinking_budget={provider_cfg.get('thinking_budget_tokens')})")
    # reasoning models normally get 8x max_tokens for hidden thinking; an effort sweep needs an
    # explicit ceiling instead, since high effort blows past any multiple of a 1600-token base.
    absolute_mt = bool(eval_cfg.get("max_tokens_absolute"))

    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)
    # `spent` tracks ACTUAL billed cost of models run this session (from the ledger), not the flat
    # pre-estimate: reasoning models spend 10-24x EST_OUTPUT_TOKENS on hidden reasoning, so an
    # estimate-only cap never fires for them. We stop BEFORE a model whose estimate would push actual
    # spend over budget, bounding any overrun to that one model.
    spent, summaries = 0.0, []

    for model_id in models:
        done = (output_dir / model_id_to_key(model_id) / "responses.json").exists()
        if not done:
            est = estimate_model_cost(model_id, len(specs) * draws_per_prompt)
            reasoning_here = reasoning if model_id in REASONING_MODELS else None
            # Reasoning models spend part of the budget on hidden reasoning that still counts against
            # max_tokens; on the open-ended association task 4x (6400) left some prompts cut off mid-
            # reasoning -> empty content (e.g. gpt-5 returned null on 12/30). 8x gives headroom.
            mt = max_tokens if absolute_mt else (max_tokens * 8 if model_id in REASONING_MODELS
                                                 else max_tokens)
            if budget_usd > 0 and spent + est > budget_usd:
                print(f"\nBUDGET CAP REACHED: next model {model_id} est ${est:.3f} on top of "
                      f"${spent:.3f} actual spent would exceed ${budget_usd:.2f}. Stopping.")
                break
            print(f"\n{model_id}  (est ${est:.4f}, actual spent so far ${spent:.4f})")
        else:
            reasoning_here, mt = None, max_tokens
            print(f"\n{model_id}  (already done)")
        results, model_cost = await run_model(async_client, sem, model_id, specs, mt, temperatures,
                                              n_samples, reasoning_here, output_dir, provider=provider,
                                              config_name=Path(config_path).stem)
        spent += model_cost
        summaries.append({"model_id": model_id, "n": len(results), "cost_usd": round(model_cost, 4)})
        if budget_usd > 0 and spent >= budget_usd:
            print(f"\nBUDGET CAP REACHED: ${spent:.3f} actual >= ${budget_usd:.2f} after {model_id}. Stopping.")
            break

    (output_dir / "run_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved responses to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.config_path, args.overwrite, args.debug))
