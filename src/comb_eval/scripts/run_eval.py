"""Run the k-paths combinatorial creativity eval against LLMs via OpenRouter.

Step 2 of the pipeline: loads the graph and prompts, queries each model for
k distinct paths per prompt, and saves raw responses.

Features:
- Async with a concurrency semaphore (all prompts for a model fire in parallel).
- Reasoning config forwarded via OpenRouter's unified reasoning API; reasoning
  models get a larger max_tokens budget because hidden reasoning tokens count
  toward the cap.
- Budget cap via scripts/safety/cost_tracker.PRICING — run aborts before the
  next model would push cumulative estimated spend over the cap.
- Resume-safe: skips models whose responses.json already exists.

Usage:
    uv run python src/comb_eval/scripts/run_eval.py configs/comb_eval/run_eval.yaml
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, init_directory, save_config
from src.comb_eval.graph import load_graph
from src.comb_eval.llm import (
    call_llm_async,
    format_graph_prompt,
    format_query_prompt,
    get_async_client,
    model_id_to_key,
    parse_multi_path_response,
)
from src.comb_eval.prompts import EvalPrompt

# PRICING: model_id -> (input $/M, output $/M)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "safety"))
from cost_tracker import PRICING


# Rough per-call token estimates for the k-paths eval on a 150-node graph.
# The system prompt is the ~7.7K-char adjacency list (same for every call,
# so providers with prompt caching — Anthropic, OpenAI — will pay much less
# than this on repeat calls). We estimate without caching for a safe upper
# bound; actual spend will typically come in lower.
EST_INPUT_TOKENS = 2300   # ~2000 for the graph system prompt + ~300 for the query
EST_OUTPUT_TOKENS = 500   # k=5 paths in JSON, ~100 tokens per path entry


# Models whose hidden reasoning tokens count toward max_tokens. These need
# a larger budget or they silently truncate before finishing the JSON.
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


def max_tokens_for(model_id: str, base: int, reasoning_multiplier: int = 4) -> int:
    """Reasoning models get `base * reasoning_multiplier` to cover hidden tokens."""
    if model_id in REASONING_MODELS:
        return base * reasoning_multiplier
    return base


def estimate_model_cost(model_id: str, n_prompts: int) -> float:
    """Estimate USD cost of running n_prompts against model_id. Returns +inf
    if the model isn't in the pricing table (so the budget cap refuses to
    spend money on something we can't account for)."""
    pricing = PRICING.get(model_id)
    if pricing is None:
        return float("inf")
    in_price, out_price = pricing
    in_tokens = n_prompts * EST_INPUT_TOKENS
    out_tokens = n_prompts * EST_OUTPUT_TOKENS
    return (in_tokens * in_price + out_tokens * out_price) / 1_000_000


def load_prompts(path: Path) -> list[EvalPrompt]:
    with open(path) as f:
        data = json.load(f)
    return [EvalPrompt(**d) for d in data]


async def _run_one_prompt(
    async_client,
    sem: asyncio.Semaphore,
    model_id: str,
    graph_prompt: str,
    prompt: EvalPrompt,
    k_paths: int,
    temperature: float,
    max_tokens: int,
    reasoning: dict | None,
) -> dict:
    """Run a single prompt for a single model. Returns a serializable result dict."""
    query = format_query_prompt(prompt, k=k_paths)
    messages = [
        {"role": "system", "content": graph_prompt},
        {"role": "user", "content": query},
    ]
    async with sem:
        try:
            raw = await call_llm_async(
                async_client,
                messages=messages,
                model=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning=reasoning,
            )
            if raw is None:
                return {
                    "prompt_id": prompt.prompt_id,
                    "raw_response": None,
                    "paths": [],
                    "edge_label_sequences": [],
                    "n_paths_returned": 0,
                    "parse_success": False,
                    "api_error": "RuntimeError: provider returned null content",
                }
            parsed = parse_multi_path_response(raw)
            return {
                "prompt_id": prompt.prompt_id,
                "raw_response": raw,
                "paths": parsed.paths,
                "edge_label_sequences": parsed.edge_label_sequences,
                "n_paths_returned": parsed.n_paths_returned,
                "parse_success": parsed.parse_success,
                "api_error": None,
            }
        except Exception as e:
            return {
                "prompt_id": prompt.prompt_id,
                "raw_response": None,
                "paths": [],
                "edge_label_sequences": [],
                "n_paths_returned": 0,
                "parse_success": False,
                "api_error": f"{type(e).__name__}: {e}",
            }


async def run_model(
    async_client,
    sem: asyncio.Semaphore,
    model_id: str,
    graph_text: str,
    prompts: list[EvalPrompt],
    k_paths: int,
    temperature: float,
    max_tokens_base: int,
    reasoning_multiplier: int,
    reasoning: dict | None,
    output_dir: Path,
) -> dict:
    """Run all prompts for a model concurrently, write responses.json."""
    model_key = model_id_to_key(model_id)
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    responses_path = model_dir / "responses.json"
    if responses_path.exists():
        print(f"  {model_id}: responses.json exists, skipping")
        with open(responses_path) as f:
            existing = json.load(f)
        return {
            "model_id": model_id,
            "model_key": model_key,
            "n_prompts": len(existing),
            "n_success": sum(1 for r in existing if r.get("parse_success")),
            "n_parse_fail": sum(1 for r in existing if not r.get("parse_success") and r.get("api_error") is None),
            "n_api_fail": sum(1 for r in existing if r.get("api_error")),
            "skipped": True,
        }

    max_tokens = max_tokens_for(model_id, max_tokens_base, reasoning_multiplier)
    graph_prompt = format_graph_prompt(graph_text)

    print(f"  {model_id}: {len(prompts)} prompts, max_tokens={max_tokens}, firing...")
    t0 = time.time()

    tasks = [
        _run_one_prompt(
            async_client, sem, model_id, graph_prompt, p,
            k_paths=k_paths, temperature=temperature,
            max_tokens=max_tokens, reasoning=reasoning,
        )
        for p in prompts
    ]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    n_success = sum(1 for r in results if r["parse_success"])
    n_parse_fail = sum(1 for r in results if not r["parse_success"] and r["api_error"] is None)
    n_api_fail = sum(1 for r in results if r["api_error"])

    with open(responses_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "model_id": model_id,
        "model_key": model_key,
        "n_prompts": len(prompts),
        "n_success": n_success,
        "n_parse_fail": n_parse_fail,
        "n_api_fail": n_api_fail,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"    done in {elapsed:.1f}s — "
        f"success={n_success} parse_fail={n_parse_fail} api_fail={n_api_fail}"
    )
    return summary


async def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)

    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    G = load_graph(upstream_dir / "graph.json")
    graph_text = (upstream_dir / "graph_adjacency.txt").read_text()
    prompts = load_prompts(upstream_dir / "prompts.json")

    print(f"Loaded graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    print(f"Loaded {len(prompts)} prompts")
    print(f"Graph text: {len(graph_text)} chars")

    if debug:
        prompts = prompts[:3]
        print(f"DEBUG: using only {len(prompts)} prompts")

    eval_cfg = config["eval"]
    temperature = eval_cfg["temperature"]
    k_paths = eval_cfg["k_paths"]
    max_tokens_base = eval_cfg.get("max_tokens", 2048)
    concurrency = config.get("concurrency", 20)
    reasoning_multiplier = config.get("reasoning_max_tokens_multiplier", 4)
    reasoning_cfg = config.get("reasoning", {"effort": "low", "exclude": True})
    budget_usd = config.get("budget_usd", 0.0)
    models = config["models"]

    # Verify upstream generation k matches
    upstream_summary_path = upstream_dir / "results" / "generation_summary.json"
    if upstream_summary_path.exists():
        upstream_summary = json.loads(upstream_summary_path.read_text())
        upstream_k = upstream_summary.get("k_paths")
        if upstream_k is not None and upstream_k != k_paths:
            raise ValueError(
                f"FATAL: run_eval k_paths={k_paths} does not match upstream "
                f"generation k_paths={upstream_k}. Regenerate the eval set or "
                f"align configs."
            )

    print(f"Models: {len(models)}")
    print(f"k_paths: {k_paths}  temperature: {temperature}  concurrency: {concurrency}")
    if budget_usd > 0:
        print(f"Budget cap: ${budget_usd:.2f} (will stop before exceeding)")
    else:
        print(f"Budget cap: NONE")

    async_client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    cumulative_cost = 0.0
    all_summaries: list[dict] = []

    for model_id in models:
        # Budget pre-check: only count models that aren't already done
        model_dir = output_dir / model_id_to_key(model_id)
        already_done = (model_dir / "responses.json").exists()

        if not already_done:
            cost_est = estimate_model_cost(model_id, len(prompts))
            if budget_usd > 0 and (cumulative_cost + cost_est) > budget_usd:
                remaining = budget_usd - cumulative_cost
                print(f"\n{'='*60}")
                print(f"BUDGET CAP REACHED")
                print(f"  Cumulative spent (estimated): ${cumulative_cost:.2f}")
                print(f"  Budget cap: ${budget_usd:.2f}")
                print(f"  Next model ({model_id}) would cost: ${cost_est:.2f}")
                print(f"  Remaining: ${remaining:.2f}")
                print(f"  Stopping. Resume with a higher budget to continue.")
                print(f"{'='*60}")
                break
            cumulative_cost += cost_est
            print(
                f"\n{model_id}   (est cost: ${cost_est:.2f}, "
                f"cumulative: ${cumulative_cost:.2f})"
            )
        else:
            print(f"\n{model_id}   (already done, skipping)")

        summary = await run_model(
            async_client, sem,
            model_id=model_id,
            graph_text=graph_text,
            prompts=prompts,
            k_paths=k_paths,
            temperature=temperature,
            max_tokens_base=max_tokens_base,
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
    args = parser.parse_args()
    asyncio.run(main(args.config_path, args.overwrite, args.debug))
