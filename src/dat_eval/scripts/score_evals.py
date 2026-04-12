"""Score DAT, CDAT, and PACE responses and run correlation analysis.

Step 2 of the pipeline: loads raw responses, computes scores using
the appropriate embedding models, and correlates with Arena benchmarks.

Usage:
    uv run python src/dat_eval/scripts/score_evals.py configs/dat_eval/score_evals.yaml
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
from src.dat_eval.llm import model_id_to_key


def _find_temp_files(model_dir: Path, prefix: str) -> dict[str, Path]:
    """Find all per-temperature response files for a given prefix.

    Returns:
        Dict mapping temperature label (e.g. "0.9") to file path.
    """
    files = {}
    for f in model_dir.glob(f"{prefix}_responses_t*.json"):
        # Extract temp from filename: <prefix>_responses_t0-9.json -> "0.9"
        suffix = f.stem.replace(f"{prefix}_responses_t", "")
        temp_str = suffix.replace("-", ".")
        files[temp_str] = f
    # Backward compat: legacy un-suffixed file
    legacy = model_dir / f"{prefix}_responses.json"
    if legacy.exists() and not files:
        files["legacy"] = legacy
    return files


def score_dat_results(
    model_dir: Path,
    embeddings,
) -> dict | None:
    """Score DAT responses for one model across all temperatures.

    Returns:
        Dict mapping temperature -> per-temp score dict, plus "mean_score" pooled.
    """
    from src.dat_eval.dat import score_dat

    temp_files = _find_temp_files(model_dir, "dat")
    if not temp_files:
        return None

    by_temp = {}
    all_scores = []

    for temp_str, path in temp_files.items():
        with open(path) as f:
            responses = json.load(f)

        trial_scores = []
        for resp in responses:
            if resp["words"]:
                result = score_dat(resp["words"], embeddings)
                trial_scores.append(result)

        valid = [t["score"] for t in trial_scores if t["sufficient"]]
        mean = float(np.mean(valid)) if valid else 0.0
        by_temp[temp_str] = {
            "mean_score": mean,
            "n_trials": len(trial_scores),
            "n_sufficient": len(valid),
            "scores": valid,
        }
        all_scores.extend(valid)

    pooled = float(np.mean(all_scores)) if all_scores else 0.0
    return {
        "mean_score": pooled,
        "by_temperature": by_temp,
        "n_temperatures": len(by_temp),
    }


def score_cdat_results(
    model_dir: Path,
    embeddings,
) -> dict | None:
    """Score CDAT responses for one model across all temperatures."""
    from src.dat_eval.cdat import score_cdat

    temp_files = _find_temp_files(model_dir, "cdat")
    if not temp_files:
        return None

    by_temp = {}
    all_novelty = []
    all_approp = []

    for temp_str, path in temp_files.items():
        with open(path) as f:
            responses = json.load(f)

        cue_scores = []
        for cue, resp in responses.items():
            if resp["words"]:
                result = score_cdat(resp["words"], cue, embeddings)
                cue_scores.append(result)

        valid = [c for c in cue_scores if c["sufficient"]]
        nov = [c["novelty"] for c in valid]
        app = [c["appropriateness"] for c in valid]
        by_temp[temp_str] = {
            "mean_novelty": float(np.mean(nov)) if nov else 0.0,
            "mean_appropriateness": float(np.mean(app)) if app else 0.0,
            "n_cues": len(cue_scores),
            "n_sufficient": len(valid),
            "novelty_scores": nov,
            "appropriateness_scores": app,
        }
        all_novelty.extend(nov)
        all_approp.extend(app)

    return {
        "mean_novelty": float(np.mean(all_novelty)) if all_novelty else 0.0,
        "mean_appropriateness": float(np.mean(all_approp)) if all_approp else 0.0,
        "by_temperature": by_temp,
        "n_temperatures": len(by_temp),
    }


def score_pace_results(model_dir: Path, embeddings) -> dict | None:
    """Score PACE responses for one model. PACE typically uses a single temp (0.0)."""
    temp_files = _find_temp_files(model_dir, "pace")
    if not temp_files:
        return None

    # Use the first available temp file (PACE convention is single temp)
    pace_path = next(iter(temp_files.values()))

    from src.dat_eval.pace import score_model

    with open(pace_path) as f:
        responses = json.load(f)

    # Build the chains dict: seed -> list of 3 chains
    all_chains = {}
    for seed, data in responses.items():
        chains = []
        for chain_data in data.get("chains", []):
            chain = chain_data.get("chain", [])
            if len(chain) >= 3:
                chains.append(chain)
        if chains:
            all_chains[seed] = chains

    if not all_chains:
        return {"model_score": 0.0, "n_seeds": 0}

    result = score_model(all_chains, embeddings)
    return result


def spearman_corr(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Spearman correlation."""
    result = stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_iter: int = 500,
    seed: int = 42,
) -> dict:
    """Bootstrap Spearman correlation."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = []
    sig_count = 0

    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        rho, pval = spearman_corr(x[idx].tolist(), y[idx].tolist())
        rhos.append(rho)
        if pval < 0.05:
            sig_count += 1

    rhos = np.array(rhos)
    return {
        "mean_rho": float(np.mean(rhos)),
        "se": float(np.std(rhos, ddof=1)),
        "ci_lower": float(np.percentile(rhos, 2.5)),
        "ci_upper": float(np.percentile(rhos, 97.5)),
        "sig_ratio": sig_count / n_iter,
    }


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)

    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    glove_path = config.get("glove_path", "resources/glove.840B.300d.txt")
    fasttext_path = config.get("fasttext_path", "resources/crawl-300d-2M.vec")
    sbert_model = config.get("sbert_model", "all-mpnet-base-v2")
    benchmark_path = config.get("benchmark_file", "configs/comb_eval/benchmarks.json")
    evals_to_score = config.get("evals", ["dat", "cdat", "pace"])
    n_bootstrap = config.get("n_bootstrap", 500)

    # Find model directories
    model_dirs = [
        d for d in upstream_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    print(f"Found {len(model_dirs)} model directories")

    # Load embedding models once (lazy, but instantiate now)
    glove_emb = None
    sbert_emb = None
    fasttext_emb = None

    if "dat" in evals_to_score:
        from src.dat_eval.dat import GloVeEmbeddings
        glove_emb = GloVeEmbeddings(glove_path)
    if "cdat" in evals_to_score:
        from src.dat_eval.cdat import SBERTEmbeddings
        sbert_emb = SBERTEmbeddings(sbert_model)
    if "pace" in evals_to_score:
        from src.dat_eval.pace import FastTextEmbeddings
        fasttext_emb = FastTextEmbeddings(fasttext_path)

    # Score each model
    model_scores = {}  # model_key -> {dat, cdat_*, pace, *_by_temp}

    for model_dir in sorted(model_dirs):
        model_key = model_dir.name
        print(f"\n{'='*40}")
        print(f"Scoring {model_key}")

        scores = {}

        if "dat" in evals_to_score:
            print("  Scoring DAT...")
            dat = score_dat_results(model_dir, glove_emb)
            if dat:
                scores["dat"] = dat["mean_score"]
                # Per-temp scores
                for t, td in dat["by_temperature"].items():
                    scores[f"dat_t{t}"] = td["mean_score"]
                temps_str = ", ".join(f"t={t}: {d['mean_score']:.2f}" for t, d in dat["by_temperature"].items())
                print(f"    DAT pooled: {dat['mean_score']:.2f}  ({temps_str})")
                model_out = output_dir / "results" / model_key
                model_out.mkdir(parents=True, exist_ok=True)
                with open(model_out / "dat_scores.json", "w") as f:
                    json.dump(dat, f, indent=2)

        if "cdat" in evals_to_score:
            print("  Scoring CDAT...")
            cdat = score_cdat_results(model_dir, sbert_emb)
            if cdat:
                scores["cdat_novelty"] = cdat["mean_novelty"]
                scores["cdat_appropriateness"] = cdat["mean_appropriateness"]
                for t, td in cdat["by_temperature"].items():
                    scores[f"cdat_novelty_t{t}"] = td["mean_novelty"]
                    scores[f"cdat_approp_t{t}"] = td["mean_appropriateness"]
                temps_str = ", ".join(f"t={t}: nov={d['mean_novelty']:.2f}/app={d['mean_appropriateness']:.2f}" for t, d in cdat["by_temperature"].items())
                print(f"    CDAT pooled: nov={cdat['mean_novelty']:.2f}, app={cdat['mean_appropriateness']:.2f}  ({temps_str})")
                model_out = output_dir / "results" / model_key
                model_out.mkdir(parents=True, exist_ok=True)
                with open(model_out / "cdat_scores.json", "w") as f:
                    json.dump(cdat, f, indent=2)

        if "pace" in evals_to_score:
            print("  Scoring PACE...")
            pace = score_pace_results(model_dir, fasttext_emb)
            if pace:
                scores["pace"] = pace["model_score"]
                print(f"    PACE score: {pace['model_score']:.4f} ({pace['n_seeds']} seeds)")
                model_out = output_dir / "results" / model_key
                model_out.mkdir(parents=True, exist_ok=True)
                with open(model_out / "pace_scores.json", "w") as f:
                    json.dump(pace, f, indent=2)

        if scores:
            model_scores[model_key] = scores

    # Save all scores
    with open(output_dir / "results" / "all_scores.json", "w") as f:
        json.dump(model_scores, f, indent=2)

    # Print rankings
    for metric in ["dat", "cdat_novelty", "pace"]:
        ranked = sorted(
            [(k, v.get(metric, 0)) for k, v in model_scores.items()],
            key=lambda x: -x[1],
        )
        if any(s > 0 for _, s in ranked):
            print(f"\n{metric.upper()} Rankings:")
            for i, (model, score) in enumerate(ranked, 1):
                if score > 0:
                    print(f"  {i}. {model}: {score:.4f}")

    # Correlation analysis
    if not Path(benchmark_path).exists():
        print(f"\nBenchmark file not found at {benchmark_path}, skipping correlations.")
        return

    print(f"\n{'='*60}")
    print("Correlation Analysis")
    print(f"{'='*60}")

    with open(benchmark_path) as f:
        benchmarks = json.load(f)

    corr_results = {}

    # Build the list of metrics to correlate. Include pooled and per-temperature variants.
    # Discover per-temp metrics from the actual scores.
    base_metrics = ["dat", "cdat_novelty", "cdat_appropriateness", "pace"]
    per_temp_metrics = set()
    for scores in model_scores.values():
        for k in scores:
            if k.startswith(("dat_t", "cdat_novelty_t", "cdat_approp_t")):
                per_temp_metrics.add(k)
    metrics_to_correlate = base_metrics + sorted(per_temp_metrics)

    for metric in metrics_to_correlate:
        # Align model scores with benchmarks
        aligned_models = []
        metric_vals = []
        arena_cw_vals = []
        arena_overall_vals = []

        for model_key, scores in model_scores.items():
            if metric not in scores or scores[metric] == 0:
                continue
            if model_key not in benchmarks:
                continue
            bench = benchmarks[model_key]
            if "arena_cw" not in bench:
                continue

            aligned_models.append(model_key)
            metric_vals.append(scores[metric])
            arena_cw_vals.append(bench["arena_cw"])
            if "arena_overall" in bench:
                arena_overall_vals.append(bench["arena_overall"])

        if len(aligned_models) < 5:
            print(f"\n{metric}: only {len(aligned_models)} models with both scores, skipping")
            continue

        x = np.array(metric_vals)
        y_cw = np.array(arena_cw_vals)

        rho_cw, p_cw = spearman_corr(x.tolist(), y_cw.tolist())
        boot_cw = bootstrap_spearman(x, y_cw, n_iter=n_bootstrap)

        corr_results[metric] = {
            "vs_arena_cw": {
                "spearman_rho": rho_cw,
                "p_value": p_cw,
                "n_models": len(aligned_models),
                "bootstrap": boot_cw,
                "models": aligned_models,
            }
        }
        print(f"\n{metric.upper()} vs Arena CW: rho={rho_cw:.3f}, p={p_cw:.4f} (n={len(aligned_models)})")
        print(f"  Bootstrap: mean={boot_cw['mean_rho']:.3f}, 95% CI=[{boot_cw['ci_lower']:.3f}, {boot_cw['ci_upper']:.3f}]")

        # Also vs Arena Overall
        if len(arena_overall_vals) == len(aligned_models):
            y_all = np.array(arena_overall_vals)
            rho_all, p_all = spearman_corr(x.tolist(), y_all.tolist())
            boot_all = bootstrap_spearman(x, y_all, n_iter=n_bootstrap)
            corr_results[metric]["vs_arena_overall"] = {
                "spearman_rho": rho_all,
                "p_value": p_all,
                "n_models": len(aligned_models),
                "bootstrap": boot_all,
            }
            print(f"{metric.upper()} vs Arena Overall: rho={rho_all:.3f}, p={p_all:.4f}")

    # Inter-metric correlations
    metrics_available = [m for m in ["dat", "cdat_novelty", "pace"]
                         if any(m in v for v in model_scores.values())]
    if len(metrics_available) >= 2:
        print(f"\nInter-metric correlations:")
        corr_results["inter_metric"] = {}
        for i, m1 in enumerate(metrics_available):
            for m2 in metrics_available[i+1:]:
                common = [k for k in model_scores
                          if m1 in model_scores[k] and m2 in model_scores[k]
                          and model_scores[k][m1] > 0 and model_scores[k][m2] > 0]
                if len(common) >= 5:
                    x = [model_scores[k][m1] for k in common]
                    y = [model_scores[k][m2] for k in common]
                    rho, p = spearman_corr(x, y)
                    key = f"{m1}_vs_{m2}"
                    corr_results["inter_metric"][key] = {
                        "spearman_rho": rho,
                        "p_value": p,
                        "n_models": len(common),
                    }
                    print(f"  {m1} vs {m2}: rho={rho:.3f}, p={p:.4f} (n={len(common)})")

    with open(output_dir / "results" / "correlation_analysis.json", "w") as f:
        json.dump(corr_results, f, indent=2)

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
