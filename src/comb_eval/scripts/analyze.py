"""Score LLM responses and run correlation analysis.

Step 3 of the pipeline: scores each model's responses using the
Creativity = Utility x Novelty metric, then correlates with external
benchmarks.

Usage:
    uv run python src/comb_eval/scripts/analyze.py configs/comb_eval/analyze.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from src.utils import load_config, init_directory, save_config
from src.comb_eval.graph import load_graph
from src.comb_eval.llm import LLMResponse
from src.comb_eval.prompts import EvalPrompt
from src.comb_eval.scoring import score_eval_set, aggregate_scores
from src.comb_eval.analysis import (
    load_benchmark_scores,
    load_eval_scores,
    full_correlation_analysis,
    compare_with_other_metrics,
)


def load_prompts(path: Path) -> list[EvalPrompt]:
    """Load eval prompts from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [EvalPrompt(**d) for d in data]


def load_responses(path: Path) -> list[LLMResponse]:
    """Load LLM responses from JSON."""
    with open(path) as f:
        data = json.load(f)
    return [
        LLMResponse(
            path=r["path"],
            edge_labels=r["edge_labels"],
            raw_response=r.get("raw_response", ""),
            parse_success=r["parse_success"],
        )
        for r in data
    ]


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)

    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    scoring_cfg = config["scoring"]
    analysis_cfg = config["analysis"]

    # Load graph (from the original eval set generation)
    # Walk up to find graph.json
    eval_set_dir = upstream_dir.parent  # run_v1 -> eval_set_v1/downstream -> eval_set_v1
    while not (eval_set_dir / "graph.json").exists():
        eval_set_dir = eval_set_dir.parent
        if eval_set_dir == Path("/"):
            raise FileNotFoundError("Could not find graph.json in upstream hierarchy")

    G = load_graph(eval_set_dir / "graph.json")
    prompts = load_prompts(eval_set_dir / "prompts.json")

    print(f"Loaded graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    print(f"Loaded {len(prompts)} prompts")

    # Score each model
    model_dirs = [d for d in upstream_dir.iterdir() if d.is_dir() and (d / "responses.json").exists()]
    print(f"Found {len(model_dirs)} model result directories")

    model_scores = {}  # model_key -> mean_creativity
    all_model_results = {}

    for model_dir in sorted(model_dirs):
        model_key = model_dir.name
        print(f"\nScoring {model_key}...")

        responses = load_responses(model_dir / "responses.json")

        if len(responses) != len(prompts):
            print(f"  WARNING: {len(responses)} responses vs {len(prompts)} prompts, skipping")
            continue

        scores = score_eval_set(
            G, prompts, responses,
            alpha_h=scoring_cfg["alpha_h"],
            alpha_r=scoring_cfg["alpha_r"],
            alpha_I=scoring_cfg["alpha_I"],
            alpha_X=scoring_cfg["alpha_X"],
        )

        agg = aggregate_scores(scores)
        model_scores[model_key] = agg["mean_creativity"]
        all_model_results[model_key] = agg

        print(f"  Creativity: {agg['mean_creativity']:.4f}")
        print(f"  Constraint satisfaction: {agg['constraint_satisfaction_rate']:.2%}")
        print(f"  Valid paths: {agg['valid_path_rate']:.2%}")
        print(f"  Errors: {agg['error_distribution']}")

        # Save per-model detailed scores
        model_out = output_dir / "results" / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        per_prompt_scores = [
            {
                "prompt_id": p.prompt_id,
                "creativity": s.creativity,
                "utility": s.utility,
                "novelty": s.novelty,
                "valid_path": s.valid_path,
                "error_type": s.error_type,
                "hop_count": s.hop_count,
                "surprise": s.surprise,
            }
            for p, s in zip(prompts, scores)
        ]
        with open(model_out / "per_prompt_scores.json", "w") as f:
            json.dump(per_prompt_scores, f, indent=2)
        with open(model_out / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

    # Save overall scores
    with open(output_dir / "results" / "model_scores.json", "w") as f:
        json.dump(model_scores, f, indent=2)
    with open(output_dir / "results" / "all_model_results.json", "w") as f:
        json.dump(all_model_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Model rankings by creativity score:")
    print(f"{'='*60}")
    for rank, (model, score) in enumerate(
        sorted(model_scores.items(), key=lambda x: -x[1]), 1
    ):
        print(f"  {rank}. {model}: {score:.4f}")

    # Correlation analysis
    benchmark_path = analysis_cfg["benchmark_file"]
    if Path(benchmark_path).exists():
        print(f"\n{'='*60}")
        print("Correlation analysis")
        print(f"{'='*60}")

        benchmarks = load_benchmark_scores(benchmark_path)
        eval_series = pd.Series(model_scores, name="comb_creativity")

        corr_results = full_correlation_analysis(
            eval_series,
            benchmarks,
            benchmark_columns=analysis_cfg["benchmark_columns"],
            n_bootstrap=analysis_cfg["n_bootstrap"],
            seed=analysis_cfg["seed"],
        )

        for col in analysis_cfg["benchmark_columns"]:
            if col in corr_results and "spearman_rho" in corr_results[col]:
                r = corr_results[col]
                print(f"  vs {col}: rho={r['spearman_rho']:.3f}, p={r['p_value']:.4f} (n={r['n_models']})")

        if "partial_arena_cw_controlling_overall" in corr_results:
            r = corr_results["partial_arena_cw_controlling_overall"]
            print(f"  partial (CW | Overall): rho={r['spearman_rho']:.3f}, p={r['p_value']:.4f}")

        with open(output_dir / "results" / "correlation_analysis.json", "w") as f:
            json.dump(corr_results, f, indent=2)

        # Compare with PACE/DAT if available
        pace_scores = None
        dat_scores = None
        if analysis_cfg.get("pace_scores_file") and Path(analysis_cfg["pace_scores_file"]).exists():
            pace_scores = load_eval_scores(analysis_cfg["pace_scores_file"])
        if analysis_cfg.get("dat_scores_file") and Path(analysis_cfg["dat_scores_file"]).exists():
            dat_scores = load_eval_scores(analysis_cfg["dat_scores_file"])

        if pace_scores is not None or dat_scores is not None:
            comparison = compare_with_other_metrics(
                eval_series, pace_scores, dat_scores, benchmarks
            )
            with open(output_dir / "results" / "metric_comparison.json", "w") as f:
                json.dump(comparison, f, indent=2)
            print("\nMetric comparison saved.")
    else:
        print(f"\nBenchmark file not found at {benchmark_path}, skipping correlation analysis.")

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
