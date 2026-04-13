"""Score LLM responses and run correlation analysis.

Step 3 of the pipeline: for each model, scores its k-paths responses using
intra-response diversity (edge-set Jaccard + normalized edit distance on
edge-label sequences), then correlates each metric with external benchmarks.

Each diversity metric is a separate hypothesis — both are reported.

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
    full_correlation_analysis,
)


def load_prompts(path: Path) -> list[EvalPrompt]:
    with open(path) as f:
        data = json.load(f)
    return [EvalPrompt(**d) for d in data]


def load_responses(path: Path) -> list[LLMResponse]:
    with open(path) as f:
        data = json.load(f)
    return [
        LLMResponse(
            paths=r.get("paths", []),
            edge_label_sequences=r.get("edge_label_sequences", []),
            n_paths_returned=r.get("n_paths_returned", 0),
            raw_response=r.get("raw_response") or "",
            parse_success=r.get("parse_success", False),
        )
        for r in data
    ]


# Headline metrics to report and correlate with benchmarks.
DIVERSITY_METRICS = ("mean_jaccard", "mean_edit_distance")


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)

    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    analysis_cfg = config["analysis"]

    # Walk up to find graph.json / prompts.json
    eval_set_dir = upstream_dir.parent
    while not (eval_set_dir / "graph.json").exists():
        eval_set_dir = eval_set_dir.parent
        if eval_set_dir == Path("/"):
            raise FileNotFoundError("Could not find graph.json in upstream hierarchy")

    G = load_graph(eval_set_dir / "graph.json")
    prompts = load_prompts(eval_set_dir / "prompts.json")

    print(f"Loaded graph ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    print(f"Loaded {len(prompts)} prompts")

    model_dirs = [d for d in upstream_dir.iterdir() if d.is_dir() and (d / "responses.json").exists()]
    print(f"Found {len(model_dirs)} model result directories")

    per_model_metrics: dict[str, dict] = {}  # model_key -> agg dict

    for model_dir in sorted(model_dirs):
        model_key = model_dir.name
        print(f"\nScoring {model_key}...")

        responses = load_responses(model_dir / "responses.json")
        if len(responses) != len(prompts):
            print(f"  WARNING: {len(responses)} responses vs {len(prompts)} prompts, skipping")
            continue

        scores = score_eval_set(G, prompts, responses)
        agg = aggregate_scores(scores)
        per_model_metrics[model_key] = agg

        print(
            f"  mean_jaccard={agg['mean_jaccard']:.4f}  "
            f"mean_edit={agg['mean_edit_distance']:.4f}  "
            f"solve_rate={agg['mean_solve_rate']:.2%}  "
            f"scorable={agg['n_scorable']}/{agg['n_prompts']}"
        )

        model_out = output_dir / "results" / model_key
        model_out.mkdir(parents=True, exist_ok=True)

        per_prompt = [
            {
                "prompt_id": p.prompt_id,
                "n_returned": s.n_returned,
                "n_valid": s.n_valid,
                "solve_rate": s.solve_rate,
                "mean_jaccard": s.mean_jaccard,
                "mean_edit_distance": s.mean_edit_distance,
                "error_distribution": s.error_distribution,
            }
            for p, s in zip(prompts, scores)
        ]
        with open(model_out / "per_prompt_scores.json", "w") as f:
            json.dump(per_prompt, f, indent=2)
        with open(model_out / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

    # Summary table across all models
    summary_df = pd.DataFrame.from_dict(per_model_metrics, orient="index")
    summary_df.to_csv(output_dir / "results" / "model_summary.csv")
    with open(output_dir / "results" / "all_model_results.json", "w") as f:
        json.dump(per_model_metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("Rankings by mean_jaccard:")
    print(f"{'='*60}")
    for rank, (model, m) in enumerate(
        sorted(per_model_metrics.items(), key=lambda x: -(x[1]["mean_jaccard"] if x[1]["mean_jaccard"] == x[1]["mean_jaccard"] else -1)),
        1,
    ):
        print(f"  {rank}. {model}: jaccard={m['mean_jaccard']:.4f} edit={m['mean_edit_distance']:.4f} solve={m['mean_solve_rate']:.2%}")

    # Correlation analysis — run for each diversity metric
    benchmark_path = analysis_cfg["benchmark_file"]
    if not Path(benchmark_path).exists():
        print(f"\nBenchmark file not found at {benchmark_path}, skipping correlation analysis.")
        print(f"\nAll results saved to {output_dir}")
        return

    print(f"\n{'='*60}")
    print("Correlation analysis")
    print(f"{'='*60}")

    benchmarks = load_benchmark_scores(benchmark_path)

    all_corr = {}
    for metric in DIVERSITY_METRICS:
        series = pd.Series(
            {m: v[metric] for m, v in per_model_metrics.items()},
            name=metric,
        )
        series = series.dropna()
        print(f"\n--- {metric} (n={len(series)}) ---")

        corr = full_correlation_analysis(
            series,
            benchmarks,
            benchmark_columns=analysis_cfg["benchmark_columns"],
            n_bootstrap=analysis_cfg["n_bootstrap"],
            seed=analysis_cfg["seed"],
        )
        all_corr[metric] = corr

        for col in analysis_cfg["benchmark_columns"]:
            r = corr.get(col, {})
            if "spearman_rho" in r:
                print(f"  vs {col}: rho={r['spearman_rho']:.3f}, p={r['p_value']:.4f} (n={r['n_models']})")

        for key, r in corr.items():
            if key.startswith("partial_"):
                print(f"  {key}: rho={r['spearman_rho']:.3f}, p={r['p_value']:.4f}")

    with open(output_dir / "results" / "correlation_analysis.json", "w") as f:
        json.dump(all_corr, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
