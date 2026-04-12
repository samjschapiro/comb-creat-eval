"""Run the combinatorial creativity eval against LLMs via OpenRouter.

Step 2 of the pipeline: loads the graph and prompts, queries each model,
and saves raw responses.

Usage:
    uv run python src/comb_eval/scripts/run_eval.py configs/comb_eval/run_eval.yaml
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
from src.comb_eval.graph import load_graph
from src.comb_eval.llm import (
    call_llm,
    format_graph_prompt,
    format_query_prompt,
    parse_single_response,
)
from src.comb_eval.prompts import EvalPrompt


def load_prompts(path: Path) -> list[EvalPrompt]:
    """Load eval prompts from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [EvalPrompt(**d) for d in data]


def model_id_to_key(model_id: str) -> str:
    """Convert an OpenRouter model ID to a filesystem-safe key.

    E.g., 'openai/gpt-4o' -> 'openai_gpt-4o'
    """
    return model_id.replace("/", "_").replace(".", "-")


def run_model(
    model_id: str,
    graph_text: str,
    prompts: list[EvalPrompt],
    temperature: float,
    output_dir: Path,
) -> dict:
    """Run all prompts against a single model and save results.

    Returns summary dict with success/failure counts.
    """
    model_key = model_id_to_key(model_id)

    print(f"\n{'='*60}")
    print(f"Running {model_id} ({len(prompts)} prompts)")
    print(f"{'='*60}")

    graph_prompt = format_graph_prompt(graph_text)

    results = []
    n_success = 0
    n_parse_fail = 0
    n_api_fail = 0

    for prompt in tqdm(prompts, desc=model_key):
        query = format_query_prompt(prompt)

        messages = [
            {"role": "system", "content": graph_prompt},
            {"role": "user", "content": query},
        ]

        try:
            raw_response = call_llm(
                messages=messages,
                model=model_id,
                temperature=temperature,
            )
            parsed = parse_single_response(raw_response)

            results.append({
                "prompt_id": prompt.prompt_id,
                "raw_response": raw_response,
                "path": parsed.path,
                "edge_labels": parsed.edge_labels,
                "parse_success": parsed.parse_success,
                "api_error": None,
            })

            if parsed.parse_success:
                n_success += 1
            else:
                n_parse_fail += 1

        except Exception as e:
            n_api_fail += 1
            results.append({
                "prompt_id": prompt.prompt_id,
                "raw_response": None,
                "path": [],
                "edge_labels": [],
                "parse_success": False,
                "api_error": f"{type(e).__name__}: {e}",
            })
            traceback.print_exc()
            # Brief pause on API errors to avoid rate limits
            time.sleep(1)

    # Save results
    model_dir = output_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "responses.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = {
        "model_id": model_id,
        "model_key": model_key,
        "n_prompts": len(prompts),
        "n_success": n_success,
        "n_parse_fail": n_parse_fail,
        "n_api_fail": n_api_fail,
    }
    with open(model_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Success: {n_success}, Parse fail: {n_parse_fail}, API fail: {n_api_fail}")
    return summary


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)

    # Validate upstream
    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    # Load graph and prompts
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
    models = config["models"]

    all_summaries = []
    for model_id in models:
        summary = run_model(model_id, graph_text, prompts, temperature, output_dir)
        all_summaries.append(summary)

    # Save overall summary
    with open(output_dir / "run_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
