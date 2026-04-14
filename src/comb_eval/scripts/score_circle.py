"""Score word-circle construction responses and run correlation analysis.

Per-model metrics (post-hoc, τ-configurable):
  - closure_rate:       fraction of circles where w_N → w_0 cosine >= τ_closure
  - edge_coherence_rate: mean fraction of intra-circle edges >= τ_edge
  - valid_circle_rate:  fraction with all edges OK AND closure OK AND distinct
                        AND excludes_seed AND n_in_vocab == n_words
  - mean_pairwise_diversity: avg FastText pairwise distance over N circle words
  - mean_pace_internal_score: PACE-style internal chain score
  - cross_trial_diversity: avg word-set Jaccard distance across the N_trials
                           circles for the same seed
  - mean_closure_cosine: raw mean FastText cosine of (w_N, w_0), τ-free

These are correlated against Arena CW, Arena overall, EQ-Bench CW, Mazur CW
v2, and Hivemind intra-model similarity. Also computes head-to-head vs PACE
(loaded from dat_eval's saved per-model scores) and the hierarchical
Y ~ PACE  vs  Y ~ PACE + circle_metric regression.

Usage:
    uv run python src/comb_eval/scripts/score_circle.py configs/comb_eval/score_circle.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import load_config, init_directory, save_config
from src.comb_eval.circle import (
    FastTextEmbeddings,
    score_circle,
    evaluate_at_thresholds,
    cross_trial_diversity,
)


def score_one_model(
    data: dict,
    embeddings: FastTextEmbeddings,
    tau_edge: float,
    tau_closure: float,
) -> dict:
    """Score one model's circle_responses.json into a flat aggregate dict."""
    n_words = int(data.get("n_words", 8))

    closure_rate_flags = []
    edge_coherence_rates = []
    valid_flags = []
    pairwise_divs = []
    pace_scores = []
    closure_cosines = []
    n_total_circles = 0

    # cross-trial diversity computed per-seed, averaged
    xtrial_values = []

    for rec in data["results"]:
        seed = rec["seed"]
        trial_scores = []
        for t in rec["trials"]:
            words = t.get("words", [])
            sc = score_circle(words, seed, embeddings)
            trial_scores.append(sc)

            n_total_circles += 1
            ev = evaluate_at_thresholds(sc, tau_edge, tau_closure, expected_n=n_words)
            closure_rate_flags.append(ev["closure_ok"])
            edge_coherence_rates.append(ev["edge_coherence_rate"])
            valid_flags.append(ev["valid_circle"])
            if not np.isnan(sc.pairwise_diversity):
                pairwise_divs.append(sc.pairwise_diversity)
            if not np.isnan(sc.pace_internal_score):
                pace_scores.append(sc.pace_internal_score)
            if not np.isnan(sc.closure_cosine):
                closure_cosines.append(sc.closure_cosine)

        xt = cross_trial_diversity(trial_scores)
        if not np.isnan(xt):
            xtrial_values.append(xt)

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "n_circles": n_total_circles,
        "closure_rate": _mean([1.0 if f else 0.0 for f in closure_rate_flags]),
        "edge_coherence_rate": _mean(edge_coherence_rates),
        "valid_circle_rate": _mean([1.0 if f else 0.0 for f in valid_flags]),
        "mean_pairwise_diversity": _mean(pairwise_divs),
        "mean_pace_internal_score": _mean(pace_scores),
        "mean_closure_cosine": _mean(closure_cosines),
        "cross_trial_diversity": _mean(xtrial_values),
    }


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 3:
        return float("nan"), float("nan"), len(df)
    r, p = stats.spearmanr(df["x"], df["y"])
    return float(r), float(p), len(df)


def _pearson(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 3:
        return float("nan"), float("nan"), len(df)
    r, p = stats.pearsonr(df["x"], df["y"])
    return float(r), float(p), len(df)


def _hierarchical(y: pd.Series, x1: pd.Series, x2: pd.Series):
    """R² gain from adding x2 to a model already containing x1."""
    df = pd.concat(
        [y.rename("y"), x1.rename("x1"), x2.rename("x2")], axis=1
    ).dropna()
    n = len(df)
    if n < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    yv = df["y"].values
    X1 = np.column_stack([np.ones(n), df["x1"].values])
    X2 = np.column_stack([np.ones(n), df["x1"].values, df["x2"].values])
    def r2_fit(X, y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        yh = X @ b
        ss_res = np.sum((y - yh) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot, ss_res
    r2_1, ss1 = r2_fit(X1, yv)
    r2_2, ss2 = r2_fit(X2, yv)
    F = ((ss1 - ss2) / 1) / (ss2 / (n - 3)) if ss2 > 0 else float("inf")
    p = 1 - stats.f.cdf(F, 1, n - 3) if np.isfinite(F) else 0.0
    return r2_1, r2_2, r2_2 - r2_1, p, n


def _sig(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""


HEADLINE_METRICS = (
    "closure_rate",
    "edge_coherence_rate",
    "valid_circle_rate",
    "mean_pairwise_diversity",
    "mean_pace_internal_score",
    "mean_closure_cosine",
    "cross_trial_diversity",
)


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    tau_edge = float(config.get("tau_edge", 0.2))
    tau_closure = float(config.get("tau_closure", 0.2))
    analysis_cfg = config["analysis"]
    fasttext_path = config.get("fasttext_path", "resources/crawl-300d-2M.vec")
    pace_scores_file = config.get("pace_scores_file")

    embeddings = FastTextEmbeddings(fasttext_path)

    # Per-model scoring
    model_dirs = [
        d for d in upstream_dir.iterdir()
        if d.is_dir() and (d / "circle_responses.json").exists()
    ]
    print(f"Found {len(model_dirs)} model result directories")
    print(f"τ_edge = {tau_edge}  τ_closure = {tau_closure}")

    per_model: dict[str, dict] = {}
    for model_dir in sorted(model_dirs):
        model_key = model_dir.name
        print(f"\nScoring {model_key}...")
        with open(model_dir / "circle_responses.json") as f:
            data = json.load(f)
        agg = score_one_model(data, embeddings, tau_edge, tau_closure)
        per_model[model_key] = agg
        print(
            f"  closure={agg['closure_rate']:.2%}  "
            f"edge_coh={agg['edge_coherence_rate']:.2%}  "
            f"valid={agg['valid_circle_rate']:.2%}  "
            f"pairwise_div={agg['mean_pairwise_diversity']:.3f}  "
            f"cross_trial={agg['cross_trial_diversity']:.3f}"
        )

        model_out = output_dir / "results" / model_key
        model_out.mkdir(parents=True, exist_ok=True)
        with open(model_out / "aggregate.json", "w") as f:
            json.dump(agg, f, indent=2)

    with open(output_dir / "results" / "all_model_aggregates.json", "w") as f:
        json.dump(per_model, f, indent=2)

    # Summary CSV
    summary_df = pd.DataFrame.from_dict(per_model, orient="index")
    summary_df.to_csv(output_dir / "results" / "model_summary.csv")

    # --- Correlations vs benchmarks
    benchmark_path = analysis_cfg["benchmark_file"]
    if not Path(benchmark_path).exists():
        print(f"\nBenchmark file not found: {benchmark_path}, skipping.")
        return
    with open(benchmark_path) as f:
        benchmarks = json.load(f)
    bm_df = pd.DataFrame.from_dict(benchmarks, orient="index")

    # Optional PACE comparison
    pace_series = None
    if pace_scores_file and Path(pace_scores_file).exists():
        ps = json.load(open(pace_scores_file))
        pace_series = pd.Series(
            {m: ps[m]["pace"] for m in per_model
             if isinstance(ps.get(m), dict) and ps[m].get("pace") is not None},
            name="PACE",
        )

    benchmark_cols = analysis_cfg["benchmark_columns"]

    print(f"\n========= Spearman + Pearson correlations (n=variable) =========")
    print(f"{'metric':<28s}  " + "  ".join(
        f"{c:<30s}" for c in benchmark_cols))
    for m in HEADLINE_METRICS + (("PACE",) if pace_series is not None else ()):
        series = (
            pace_series if m == "PACE"
            else pd.Series({k: v[m] for k, v in per_model.items()})
        )
        cells = []
        for c in benchmark_cols:
            rs, ps, n = _spearman(series, bm_df[c])
            rp, pp, _ = _pearson(series, bm_df[c])
            cells.append(f"ρ={rs:+.3f}{_sig(ps):<3s} r={rp:+.3f}{_sig(pp):<3s} n={n:2d}")
        print(f"{m:<28s}  " + "  ".join(f"{c:<30s}" for c in cells))

    # Hierarchical regression: does circle metric add info beyond PACE?
    if pace_series is not None:
        print(f"\n========= Hierarchical: Y ~ PACE  vs  Y ~ PACE + circle_metric =========")
        print(f"{'metric':<28s}  " + "  ".join(f"{c:<20s}" for c in benchmark_cols))
        for m in HEADLINE_METRICS:
            series = pd.Series({k: v[m] for k, v in per_model.items()})
            cells = []
            for c in benchmark_cols:
                r1, r2, d, p, n = _hierarchical(bm_df[c], pace_series, series)
                cells.append(f"ΔR²={d:+.3f}{_sig(p):<3s} (n={n})")
            print(f"{m:<28s}  " + "  ".join(f"{c:<20s}" for c in cells))

    # Save correlation outputs
    corr_out = {"per_metric": {}, "tau_edge": tau_edge, "tau_closure": tau_closure}
    for m in HEADLINE_METRICS:
        series = pd.Series({k: v[m] for k, v in per_model.items()})
        entry = {}
        for c in benchmark_cols:
            rs, ps, n = _spearman(series, bm_df[c])
            rp, pp, _ = _pearson(series, bm_df[c])
            entry[c] = {
                "spearman": {"rho": rs, "p": ps, "n": n},
                "pearson":  {"r": rp, "p": pp, "n": n},
            }
            if pace_series is not None:
                r1, r2, d, p, nh = _hierarchical(bm_df[c], pace_series, series)
                entry[c]["hierarchical_vs_PACE"] = {
                    "R2_pace": r1, "R2_combined": r2, "delta_R2": d, "F_p": p, "n": nh,
                }
        corr_out["per_metric"][m] = entry
    with open(output_dir / "results" / "correlation_analysis.json", "w") as f:
        json.dump(corr_out, f, indent=2)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
