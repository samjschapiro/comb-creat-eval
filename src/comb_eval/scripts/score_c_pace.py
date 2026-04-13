"""Score C-PACE responses and run correlation analysis.

For each model × (seed, first_assoc, level) we score the chain with PACE's
internal-chain diversity (FastText cosine), then check constraint
satisfaction. Aggregation splits by level and also reports overall; both
`mean_chain_score_valid` (diversity over chains that satisfied constraints —
the creativity channel) and `constraint_satisfaction_rate` (capability
channel) are correlated independently against benchmarks.

Usage:
    uv run python src/comb_eval/scripts/score_c_pace.py configs/comb_eval/score_c_pace.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd

from src.utils import load_config, init_directory, save_config
from src.comb_eval.c_pace import (
    Constraints,
    CPaceResult,
    FastTextEmbeddings,
    aggregate_by_level,
    score_one,
)
from src.comb_eval.analysis import (
    load_benchmark_scores,
    full_correlation_analysis,
)


# Metrics we emit per model for downstream correlation. Each gets its own
# full_correlation_analysis pass against every benchmark column.
PER_LEVEL_METRICS = ("mean_chain_score_valid", "constraint_satisfaction_rate")
OVERALL_METRICS = ("mean_chain_score_valid", "mean_chain_score_all",
                   "constraint_satisfaction_rate")


def score_model_responses(data: dict, embeddings: FastTextEmbeddings) -> tuple[list[CPaceResult], dict]:
    """Score all chains for one model. Returns (per-chain results, agg dict)."""
    results: list[CPaceResult] = []
    for rec in data["results"]:
        seed = rec["seed"]
        for s2 in rec["stage2"]:
            if s2.get("api_error") is not None:
                continue
            chain = s2.get("chain", [])
            if len(chain) < 2:
                continue
            constraints = Constraints.from_dict(s2["constraints"])
            res = score_one(
                chain=chain,
                constraints=constraints,
                embeddings=embeddings,
                seed=seed,
                second_word=s2.get("first_assoc_word", chain[1] if len(chain) > 1 else ""),
            )
            results.append(res)
    agg = aggregate_by_level(results)
    return results, agg


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)

    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    analysis_cfg = config["analysis"]

    # FastText — path mirrors dat_eval's convention
    fasttext_path = config["fasttext_path"]
    embeddings = FastTextEmbeddings(fasttext_path)

    # Find model dirs with c_pace_responses.json
    model_dirs = [
        d for d in upstream_dir.iterdir()
        if d.is_dir() and (d / "c_pace_responses.json").exists()
    ]
    print(f"Found {len(model_dirs)} model result directories")

    per_model: dict[str, dict] = {}  # model_key -> agg

    for model_dir in sorted(model_dirs):
        model_key = model_dir.name
        with open(model_dir / "c_pace_responses.json") as f:
            data = json.load(f)

        print(f"\nScoring {model_key}...")
        results, agg = score_model_responses(data, embeddings)

        overall = agg["overall"]
        print(
            f"  overall: n={overall['n_chains']} "
            f"valid={overall['n_valid']} "
            f"sat_rate={overall['constraint_satisfaction_rate']:.2%} "
            f"diversity_valid={overall['mean_chain_score_valid']:.4f} "
            f"diversity_all={overall['mean_chain_score_all']:.4f}"
        )
        for lvl in sorted(agg["by_level"]):
            s = agg["by_level"][lvl]
            print(
                f"  L{lvl}: n={s['n_chains']} valid={s['n_valid']} "
                f"sat={s['constraint_satisfaction_rate']:.2%} "
                f"div_valid={s['mean_chain_score_valid']:.4f}"
            )

        per_model[model_key] = agg

        # Save per-model
        model_out = output_dir / "results" / model_key
        model_out.mkdir(parents=True, exist_ok=True)
        with open(model_out / "per_chain_scores.json", "w") as f:
            json.dump(
                [
                    {
                        "seed": r.seed,
                        "second_word": r.second_word,
                        "level": r.level,
                        "constraints": r.constraints,
                        "chain": r.chain,
                        "chain_score": r.chain_score,
                        "n_oov": r.n_oov,
                        "satisfied": r.satisfied,
                        "include_results": r.include_results,
                        "exclude_results": r.exclude_results,
                    }
                    for r in results
                ],
                f, indent=2,
            )
        with open(model_out / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

    # Summary table
    with open(output_dir / "results" / "all_model_aggregates.json", "w") as f:
        json.dump(per_model, f, indent=2)

    # Flat DataFrame for convenience
    rows = []
    for model_key, agg in per_model.items():
        row = {"model": model_key}
        for m in OVERALL_METRICS:
            row[f"overall_{m}"] = agg["overall"][m]
        for lvl, s in agg["by_level"].items():
            for m in PER_LEVEL_METRICS:
                row[f"L{lvl}_{m}"] = s[m]
        rows.append(row)
    summary_df = pd.DataFrame(rows).set_index("model")
    summary_df.to_csv(output_dir / "results" / "model_summary.csv")

    # --- correlation analysis
    benchmark_path = analysis_cfg["benchmark_file"]
    if not Path(benchmark_path).exists():
        print(f"\nBenchmark file not found: {benchmark_path}. Skipping correlations.")
        return

    benchmarks = load_benchmark_scores(benchmark_path)
    all_corr: dict = {}

    def _run_correlation(metric_name: str, series: pd.Series):
        series = series.dropna()
        if len(series) < 3:
            print(f"  {metric_name}: n={len(series)}, skipping")
            return None
        print(f"\n--- {metric_name} (n={len(series)}) ---")
        corr = full_correlation_analysis(
            series,
            benchmarks,
            benchmark_columns=analysis_cfg["benchmark_columns"],
            n_bootstrap=analysis_cfg["n_bootstrap"],
            seed=analysis_cfg["seed"],
        )
        for col in analysis_cfg["benchmark_columns"]:
            r = corr.get(col, {})
            if "spearman_rho" in r:
                print(f"  vs {col}: rho={r['spearman_rho']:.3f} "
                      f"p={r['p_value']:.4f} n={r['n_models']}")
        for key, r in corr.items():
            if key.startswith("partial_"):
                print(f"  {key}: rho={r['spearman_rho']:.3f} p={r['p_value']:.4f}")
        return corr

    # Overall metrics
    for m in OVERALL_METRICS:
        series = pd.Series(
            {k: v["overall"][m] for k, v in per_model.items()},
            name=f"overall_{m}",
        )
        corr = _run_correlation(f"overall_{m}", series)
        if corr is not None:
            all_corr[f"overall_{m}"] = corr

    # Per-level metrics
    levels = sorted({lvl for v in per_model.values() for lvl in v["by_level"]})
    for lvl in levels:
        for m in PER_LEVEL_METRICS:
            series = pd.Series(
                {k: v["by_level"][lvl][m] for k, v in per_model.items()},
                name=f"L{lvl}_{m}",
            )
            corr = _run_correlation(f"L{lvl}_{m}", series)
            if corr is not None:
                all_corr[f"L{lvl}_{m}"] = corr

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
