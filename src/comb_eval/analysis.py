"""Correlation analysis for combinatorial creativity evaluation.

Computes Spearman correlations between our eval scores and external benchmarks
(Arena CW, Arena Overall, MMLU-Pro, etc.), with bootstrap validation.
Follows the PACE paper's methodology (EMNLP 2025).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def load_benchmark_scores(path: str | Path) -> pd.DataFrame:
    """Load external benchmark scores from a JSON file.

    Expected format:
    {
        "model_name": {
            "arena_cw": 1234,
            "arena_overall": 1230,
            "mmlu_pro": 0.85,
            ...
        },
        ...
    }

    Returns:
        DataFrame with model names as index and benchmark columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark scores file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame.from_dict(data, orient="index")


def load_eval_scores(path: str | Path) -> pd.Series:
    """Load our eval creativity scores from a JSON file.

    Expected format:
    {
        "model_name": 1.234,
        ...
    }

    Returns:
        Series with model names as index and creativity scores as values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Eval scores file not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return pd.Series(data, name="comb_creativity")


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute Spearman rank correlation and p-value.

    Args:
        x: First variable (array-like).
        y: Second variable (array-like).

    Returns:
        (rho, p_value) tuple.
    """
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def partial_spearman(
    target: np.ndarray,
    predictor: np.ndarray,
    control: np.ndarray,
) -> tuple[float, float]:
    """Compute partial Spearman correlation controlling for a third variable.

    Correlates target with predictor after removing the effect of control
    from both, using rank residuals.

    Args:
        target: The variable we want to predict (e.g., Arena CW ranks).
        predictor: Our eval score ranks.
        control: The variable to control for (e.g., Arena Overall ranks).

    Returns:
        (rho, p_value) for the partial correlation.
    """
    # Rank all variables
    target_ranks = stats.rankdata(target)
    predictor_ranks = stats.rankdata(predictor)
    control_ranks = stats.rankdata(control)

    # Regress out control from both target and predictor (using ranks)
    # Residuals from regressing target_ranks on control_ranks
    slope_t, intercept_t, _, _, _ = stats.linregress(control_ranks, target_ranks)
    resid_target = target_ranks - (slope_t * control_ranks + intercept_t)

    slope_p, intercept_p, _, _, _ = stats.linregress(control_ranks, predictor_ranks)
    resid_predictor = predictor_ranks - (slope_p * control_ranks + intercept_p)

    # Spearman on residuals
    return spearman_correlation(resid_target, resid_predictor)


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    n_iterations: int = 500,
    seed: int = 42,
) -> dict:
    """Bootstrap validation of Spearman correlation.

    Resamples observations and recomputes Spearman rho, reporting mean,
    standard error, 95% CI, and significance ratio.

    Args:
        x: First variable.
        y: Second variable.
        n_iterations: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with keys: mean_rho, se, ci_lower, ci_upper, sig_ratio, all_rhos.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = []
    sig_count = 0

    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        rho, pval = spearman_correlation(x[idx], y[idx])
        rhos.append(rho)
        if pval < 0.05:
            sig_count += 1

    rhos = np.array(rhos)
    return {
        "mean_rho": float(np.mean(rhos)),
        "se": float(np.std(rhos, ddof=1)),
        "ci_lower": float(np.percentile(rhos, 2.5)),
        "ci_upper": float(np.percentile(rhos, 97.5)),
        "sig_ratio": sig_count / n_iterations,
        "all_rhos": rhos.tolist(),
    }


def full_correlation_analysis(
    eval_scores: pd.Series,
    benchmark_df: pd.DataFrame,
    benchmark_columns: list[str],
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """Run the full correlation analysis suite.

    For each benchmark:
        1. Spearman rho + p-value
        2. Bootstrap validation (mean, SE, 95% CI, significance ratio)

    Also computes partial correlations of eval vs Arena CW controlling
    for Arena Overall (if both are present).

    Args:
        eval_scores: Series of model creativity scores (index = model names).
        benchmark_df: DataFrame of benchmark scores (index = model names).
        benchmark_columns: Which benchmark columns to correlate with.
        n_bootstrap: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Dict with per-benchmark results and partial correlations.
    """
    # Align on common models
    common_models = eval_scores.index.intersection(benchmark_df.index)
    if len(common_models) == 0:
        raise ValueError("No models in common between eval scores and benchmarks")

    eval_aligned = eval_scores.loc[common_models].values
    results = {"n_models": len(common_models), "models": list(common_models)}

    for col in benchmark_columns:
        if col not in benchmark_df.columns:
            results[col] = {"error": f"Column '{col}' not found in benchmarks"}
            continue

        # Drop models with NaN in this benchmark
        bench_series = benchmark_df.loc[common_models, col]
        valid_mask = bench_series.notna()
        if valid_mask.sum() < 5:
            results[col] = {"error": f"Fewer than 5 models with valid '{col}' scores"}
            continue

        x = eval_aligned[valid_mask.values]
        y = bench_series[valid_mask].values

        rho, pval = spearman_correlation(x, y)
        bootstrap = bootstrap_correlation(x, y, n_iterations=n_bootstrap, seed=seed)
        # Remove raw rhos from the summary to save space
        bootstrap_summary = {k: v for k, v in bootstrap.items() if k != "all_rhos"}

        results[col] = {
            "spearman_rho": rho,
            "p_value": pval,
            "n_models": int(valid_mask.sum()),
            "bootstrap": bootstrap_summary,
        }

    # Partial correlation: eval vs arena_cw controlling for arena_overall
    if "arena_cw" in benchmark_df.columns and "arena_overall" in benchmark_df.columns:
        both_valid = (
            benchmark_df.loc[common_models, "arena_cw"].notna()
            & benchmark_df.loc[common_models, "arena_overall"].notna()
        )
        if both_valid.sum() >= 5:
            x = eval_aligned[both_valid.values]
            y_cw = benchmark_df.loc[common_models, "arena_cw"][both_valid].values
            y_overall = benchmark_df.loc[common_models, "arena_overall"][both_valid].values

            partial_rho, partial_pval = partial_spearman(y_cw, x, y_overall)
            results["partial_arena_cw_controlling_overall"] = {
                "spearman_rho": partial_rho,
                "p_value": partial_pval,
                "n_models": int(both_valid.sum()),
            }

    return results


def compare_with_other_metrics(
    eval_scores: pd.Series,
    pace_scores: pd.Series | None,
    dat_scores: pd.Series | None,
    benchmark_df: pd.DataFrame,
    target_benchmark: str = "arena_cw",
) -> dict:
    """Compare our eval with PACE and DAT scores.

    Computes:
        - Each metric's correlation with the target benchmark
        - Inter-metric correlations (our eval vs PACE, our eval vs DAT, PACE vs DAT)

    Args:
        eval_scores: Our combinatorial creativity scores.
        pace_scores: PACE scores (optional).
        dat_scores: DAT scores (optional).
        benchmark_df: External benchmark data.
        target_benchmark: Which benchmark to compare against.

    Returns:
        Dict with correlation comparisons.
    """
    results = {}

    all_metrics = {"comb_creativity": eval_scores}
    if pace_scores is not None:
        all_metrics["pace"] = pace_scores
    if dat_scores is not None:
        all_metrics["dat"] = dat_scores

    # Each metric vs target benchmark
    if target_benchmark in benchmark_df.columns:
        results["vs_benchmark"] = {}
        for name, scores in all_metrics.items():
            common = scores.index.intersection(benchmark_df.index)
            bench = benchmark_df.loc[common, target_benchmark]
            valid = bench.notna()
            if valid.sum() >= 5:
                rho, pval = spearman_correlation(
                    scores.loc[common][valid].values,
                    bench[valid].values,
                )
                results["vs_benchmark"][name] = {
                    "spearman_rho": rho,
                    "p_value": pval,
                    "n_models": int(valid.sum()),
                }

    # Inter-metric correlations
    metric_names = list(all_metrics.keys())
    results["inter_metric"] = {}
    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            name_a, name_b = metric_names[i], metric_names[j]
            scores_a, scores_b = all_metrics[name_a], all_metrics[name_b]
            common = scores_a.index.intersection(scores_b.index)
            if len(common) >= 5:
                rho, pval = spearman_correlation(
                    scores_a.loc[common].values,
                    scores_b.loc[common].values,
                )
                results["inter_metric"][f"{name_a}_vs_{name_b}"] = {
                    "spearman_rho": rho,
                    "p_value": pval,
                    "n_models": len(common),
                }

    return results
