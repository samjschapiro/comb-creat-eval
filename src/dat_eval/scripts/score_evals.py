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


def pearson_corr(x: list[float], y: list[float]) -> tuple[float, float]:
    """Compute Pearson correlation and p-value."""
    result = stats.pearsonr(x, y)
    return float(result.statistic), float(result.pvalue)


def _partial_p(r: float, n: int, k: int) -> float:
    """Two-sided p-value for a partial Pearson correlation with k controls.

    Uses t = r·sqrt((n - 2 - k)/(1 - r^2)) ~ t_{n - 2 - k}. This is the
    same form scipy.stats.pearsonr uses but with the residual-correlation
    df reduced by k, the number of variables regressed out.
    """
    df = n - 2 - k
    if df <= 0 or not (-1.0 < r < 1.0):
        return float("nan")
    t = r * np.sqrt(df / (1.0 - r * r))
    return float(2.0 * stats.t.sf(abs(t), df))


def partial_pearson(
    target: np.ndarray,
    predictor: np.ndarray,
    control: np.ndarray,
) -> tuple[float, float]:
    """Semi-partial Pearson correlation r(predictor, target | control)
    := r(predictor, target - control_predicted). The control's linear
    effect is removed from the target only; the predictor is left raw.
    This matches the paper's specificity definition. P-value uses
    df = n - 2 - k where k = 1 (the single control)."""
    slope_t, intercept_t, _, _, _ = stats.linregress(control, target)
    resid_target = target - (slope_t * control + intercept_t)

    r, _ = pearson_corr(predictor.tolist(), resid_target.tolist())
    return r, _partial_p(r, len(target), k=1)


def partial_spearman(
    target: np.ndarray,
    predictor: np.ndarray,
    control: np.ndarray,
) -> tuple[float, float]:
    """Spearman partial correlation of target ~ predictor, controlling for control.

    Method: rank all three variables, regress target ranks on control ranks and
    predictor ranks on control ranks, then compute Spearman correlation of the
    two sets of residuals. This isolates the target-predictor association that
    is not explainable by a linear relationship with the control variable.

    Equivalent to the closed-form:
        partial_rho = (rho_tp - rho_tc * rho_pc) /
                      sqrt((1 - rho_tc^2) * (1 - rho_pc^2))
    but the residual-based implementation is more stable with ranked ties.

    Args:
        target:    Variable we want to predict (e.g. Arena CW Elo).
        predictor: Variable whose predictive power we want to isolate (e.g. PACE).
        control:   Confound we want to remove (e.g. Arena Overall Elo).

    Returns:
        (partial_rho, p_value).
    """
    target_ranks = stats.rankdata(target)
    predictor_ranks = stats.rankdata(predictor)
    control_ranks = stats.rankdata(control)

    # Regress each ranked variable on the control ranks; take residuals
    slope_t, intercept_t, _, _, _ = stats.linregress(control_ranks, target_ranks)
    resid_target = target_ranks - (slope_t * control_ranks + intercept_t)

    slope_p, intercept_p, _, _, _ = stats.linregress(control_ranks, predictor_ranks)
    resid_predictor = predictor_ranks - (slope_p * control_ranks + intercept_p)

    return spearman_corr(resid_target.tolist(), resid_predictor.tolist())


def partial_pearson_multi(
    target: np.ndarray,
    predictor: np.ndarray,
    controls: list[np.ndarray],
) -> tuple[float, float]:
    """Semi-partial Pearson correlation r(predictor, target | controls)
    := r(predictor, target - controls_predicted). Regresses target on the
    stack of control variables via least squares, then correlates the
    raw predictor with the target residuals. This matches the paper's
    specificity definition. P-value uses df = n - 2 - k where k is the
    number of controls.
    """
    X = np.column_stack([np.ones(len(target))] + [np.asarray(c) for c in controls])
    beta_t, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid_target = target - X @ beta_t
    r, _ = pearson_corr(predictor.tolist(), resid_target.tolist())
    return r, _partial_p(r, len(target), k=len(controls))


def partial_spearman_multi(
    target: np.ndarray,
    predictor: np.ndarray,
    controls: list[np.ndarray],
) -> tuple[float, float]:
    """Spearman partial correlation controlling for multiple variables.

    Rank-transforms every variable, then applies the multi-control partial
    Pearson construction to the ranks.
    """
    tr = stats.rankdata(target)
    pr = stats.rankdata(predictor)
    cr = [stats.rankdata(c) for c in controls]
    return partial_pearson_multi(tr, pr, cr)


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
                # Per-temp scores (only record if temp had >=1 valid trial)
                for t, td in dat["by_temperature"].items():
                    if td["n_sufficient"] > 0:
                        scores[f"dat_t{t}"] = td["mean_score"]
                temps_str = ", ".join(
                    f"t={t}: {d['mean_score']:.2f}" if d["n_sufficient"] > 0 else f"t={t}: N/A(0 valid)"
                    for t, d in dat["by_temperature"].items()
                )
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
                    if td["n_sufficient"] > 0:
                        scores[f"cdat_novelty_t{t}"] = td["mean_novelty"]
                        scores[f"cdat_approp_t{t}"] = td["mean_appropriateness"]
                temps_str = ", ".join(
                    f"t={t}: nov={d['mean_novelty']:.2f}/app={d['mean_appropriateness']:.2f}"
                    if d["n_sufficient"] > 0
                    else f"t={t}: N/A(0 valid)"
                    for t, d in cdat["by_temperature"].items()
                )
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

    # Inject gated CDAT (Nakajima appropriateness gate, SBERT scoring) into
    # each model's score dict if cdat_gated_scores.json is present. The gate
    # is computed by src/dat_eval/scripts/cdat_gate.py and saved under
    # results/cdat_gated_scores.json. We use the SBERT (original CDAT
    # embedding) variant here.
    # Source-of-truth lives one level above the (--overwrite-able) output_dir
    # so it survives `--overwrite` runs of this script.
    gated_path = Path(config["upstream_dir"]) / "cdat_gated_scores.json"
    if not gated_path.exists():
        # Fallback for older layouts
        gated_path = output_dir / "results" / "cdat_gated_scores.json"
    if gated_path.exists():
        with open(gated_path) as f:
            gated = json.load(f)
        sbert = gated.get("sbert", {})
        for model_key, scores in model_scores.items():
            entry = sbert.get(model_key, {})
            scores["cdat"] = entry.get("cdat") if entry.get("cdat") is not None else 0

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
    base_metrics = ["dat", "cdat_novelty", "cdat_appropriateness", "cdat", "pace"]
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
        mmlu_pro_all_vals = []  # aligned 1:1 with aligned_models; None where missing
        eqbench_cw_vals = []
        eqbench_cw_keys = []
        hivemind_vals = []
        hivemind_keys = []
        mazur_vals = []
        mazur_keys = []

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
            mmlu_pro_all_vals.append(bench.get("mmlu_pro"))  # may be None
            if "eq_bench_cw" in bench:
                eqbench_cw_vals.append(bench["eq_bench_cw"])
                eqbench_cw_keys.append(model_key)
            if "hivemind_diversity" in bench:
                hivemind_vals.append(bench["hivemind_diversity"])
                hivemind_keys.append(model_key)
            elif "hivemind_intra_sim" in bench:
                # Fallback: convert legacy intra-sim to diversity (1 - sim)
                hivemind_vals.append(1.0 - bench["hivemind_intra_sim"])
                hivemind_keys.append(model_key)
            if "mazur_cw_v2" in bench:
                mazur_vals.append(bench["mazur_cw_v2"])
                mazur_keys.append(model_key)

        if len(aligned_models) < 5:
            print(f"\n{metric}: only {len(aligned_models)} models with both scores, skipping")
            continue

        x = np.array(metric_vals)
        y_cw = np.array(arena_cw_vals)

        rho_cw, p_cw = spearman_corr(x.tolist(), y_cw.tolist())
        r_cw, p_r_cw = pearson_corr(x.tolist(), y_cw.tolist())
        boot_cw = bootstrap_spearman(x, y_cw, n_iter=n_bootstrap)

        corr_results[metric] = {
            "vs_arena_cw": {
                "spearman_rho": rho_cw,
                "spearman_p": p_cw,
                "pearson_r": r_cw,
                "pearson_p": p_r_cw,
                "p_value": p_cw,  # legacy field
                "n_models": len(aligned_models),
                "bootstrap": boot_cw,
                "models": aligned_models,
            }
        }
        print(f"\n{metric.upper()} vs Arena CW: rho={rho_cw:.3f} (p={p_cw:.4f}), r={r_cw:.3f} (p={p_r_cw:.4f}), n={len(aligned_models)}")
        print(f"  Bootstrap rho: mean={boot_cw['mean_rho']:.3f}, 95% CI=[{boot_cw['ci_lower']:.3f}, {boot_cw['ci_upper']:.3f}]")

        # Also vs Arena Overall
        if len(arena_overall_vals) == len(aligned_models):
            y_all = np.array(arena_overall_vals)
            rho_all, p_all = spearman_corr(x.tolist(), y_all.tolist())
            r_all, p_r_all = pearson_corr(x.tolist(), y_all.tolist())
            boot_all = bootstrap_spearman(x, y_all, n_iter=n_bootstrap)
            corr_results[metric]["vs_arena_overall"] = {
                "spearman_rho": rho_all,
                "spearman_p": p_all,
                "pearson_r": r_all,
                "pearson_p": p_r_all,
                "p_value": p_all,  # legacy
                "n_models": len(aligned_models),
                "bootstrap": boot_all,
            }
            print(f"{metric.upper()} vs Arena Overall: rho={rho_all:.3f} (p={p_all:.4f}), r={r_all:.3f} (p={p_r_all:.4f})")

            # Partial correlation: metric vs Arena CW controlling for Arena Overall.
            # This isolates any creativity-specific signal from general capability.
            part_rho, part_p = partial_spearman(
                target=y_cw,
                predictor=x,
                control=y_all,
            )
            part_r, part_r_p = partial_pearson(
                target=y_cw,
                predictor=x,
                control=y_all,
            )
            corr_results[metric]["partial_cw_control_overall"] = {
                "spearman_rho": part_rho,
                "spearman_p": part_p,
                "pearson_r": part_r,
                "pearson_p": part_r_p,
                "p_value": part_p,  # legacy
                "n_models": len(aligned_models),
            }
            flag = ""
            if abs(rho_cw) > 0.2 and abs(part_rho) < 0.1:
                flag = "  <-- signal collapses, may be general-capability proxy"
            elif abs(part_rho) > 0.3:
                flag = "  <-- creativity-specific signal survives"
            print(f"{metric.upper()} partial (CW | Overall): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}){flag}")

        # Also correlate with EQ-Bench Creative Writing (often a subset of models)
        if len(eqbench_cw_vals) >= 5:
            eq_aligned_x = [
                val for mk, val in zip(aligned_models, metric_vals)
                if mk in eqbench_cw_keys
            ]
            # Arena Overall values for the EQ-Bench subset (for partial correlation)
            eq_aligned_overall = [
                bench for mk, bench in zip(aligned_models, arena_overall_vals)
                if mk in eqbench_cw_keys and len(arena_overall_vals) == len(aligned_models)
            ]
            x_eq = np.array(eq_aligned_x)
            y_eq = np.array(eqbench_cw_vals)
            rho_eq, p_eq = spearman_corr(x_eq.tolist(), y_eq.tolist())
            r_eq, p_r_eq = pearson_corr(x_eq.tolist(), y_eq.tolist())
            boot_eq = bootstrap_spearman(x_eq, y_eq, n_iter=n_bootstrap)
            corr_results[metric]["vs_eq_bench_cw"] = {
                "spearman_rho": rho_eq,
                "spearman_p": p_eq,
                "pearson_r": r_eq,
                "pearson_p": p_r_eq,
                "p_value": p_eq,
                "n_models": len(eqbench_cw_vals),
                "bootstrap": boot_eq,
            }
            print(f"{metric.upper()} vs EQ-Bench CW: rho={rho_eq:.3f} (p={p_eq:.4f}), r={r_eq:.3f} (p={p_r_eq:.4f}), n={len(eqbench_cw_vals)}")

            if len(eq_aligned_overall) == len(eqbench_cw_vals):
                part_rho, part_p = partial_spearman(
                    target=y_eq,
                    predictor=x_eq,
                    control=np.array(eq_aligned_overall),
                )
                part_r, part_r_p = partial_pearson(
                    target=y_eq,
                    predictor=x_eq,
                    control=np.array(eq_aligned_overall),
                )
                corr_results[metric]["partial_eqbench_control_overall"] = {
                    "spearman_rho": part_rho,
                    "spearman_p": part_p,
                    "pearson_r": part_r,
                    "pearson_p": part_r_p,
                    "p_value": part_p,
                    "n_models": len(eqbench_cw_vals),
                }
                print(f"{metric.upper()} partial EQ-Bench (| Overall): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f})")

        # Correlate with Hivemind output diversity (= 1 - intra-model similarity).
        # Higher diversity is expected for more creative models, so a valid
        # creativity metric should correlate POSITIVELY with hivemind_diversity.
        if len(hivemind_vals) >= 5:
            hm_aligned_x = [
                val for mk, val in zip(aligned_models, metric_vals)
                if mk in hivemind_keys
            ]
            hm_aligned_overall = [
                bench for mk, bench in zip(aligned_models, arena_overall_vals)
                if mk in hivemind_keys and len(arena_overall_vals) == len(aligned_models)
            ]
            x_hm = np.array(hm_aligned_x)
            y_hm = np.array(hivemind_vals)
            rho_hm, p_hm = spearman_corr(x_hm.tolist(), y_hm.tolist())
            r_hm, p_r_hm = pearson_corr(x_hm.tolist(), y_hm.tolist())
            boot_hm = bootstrap_spearman(x_hm, y_hm, n_iter=n_bootstrap)
            corr_results[metric]["vs_hivemind_diversity"] = {
                "spearman_rho": rho_hm,
                "spearman_p": p_hm,
                "pearson_r": r_hm,
                "pearson_p": p_r_hm,
                "p_value": p_hm,
                "n_models": len(hivemind_vals),
                "bootstrap": boot_hm,
                "note": "POSITIVE correlation expected — hivemind_diversity = 1 - intra_sim, higher = more diverse",
            }
            direction = "expected positive" if rho_hm > 0 else "WRONG DIRECTION"
            print(f"{metric.upper()} vs Hivemind diversity: rho={rho_hm:.3f} (p={p_hm:.4f}), r={r_hm:.3f} (p={p_r_hm:.4f}), n={len(hivemind_vals)}, {direction}")

            if len(hm_aligned_overall) == len(hivemind_vals):
                part_rho, part_p = partial_spearman(
                    target=y_hm,
                    predictor=x_hm,
                    control=np.array(hm_aligned_overall),
                )
                part_r, part_r_p = partial_pearson(
                    target=y_hm,
                    predictor=x_hm,
                    control=np.array(hm_aligned_overall),
                )
                corr_results[metric]["partial_hivemind_control_overall"] = {
                    "spearman_rho": part_rho,
                    "spearman_p": part_p,
                    "pearson_r": part_r,
                    "pearson_p": part_r_p,
                    "p_value": part_p,
                    "n_models": len(hivemind_vals),
                }
                print(f"{metric.upper()} partial Hivemind (| Overall): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f})")

        # --------------------------------------------------------------
        # Alternative capability control: MMLU-Pro instead of Arena Overall.
        # Arena Overall Elo is Arena-ecosystem-internal; MMLU-Pro is a
        # structurally independent knowledge/reasoning benchmark. If partial
        # correlations survive under BOTH controls, the creativity-specific
        # claim is much harder to dismiss as an Arena artifact.
        # --------------------------------------------------------------
        def _filter_with_mmlu(keys_with_metric, metric_values, bench_values, target_keys):
            """Return arrays (metric, bench, mmlu_pro) subset to models that
            have all three. target_keys is the list of model keys for which
            bench_values is defined."""
            out_m, out_b, out_mp = [], [], []
            bench_by_key = dict(zip(target_keys, bench_values))
            for mk, mv in zip(keys_with_metric, metric_values):
                if mk not in bench_by_key:
                    continue
                bench_row = benchmarks.get(mk, {})
                mp = bench_row.get("mmlu_pro")
                if mp is None:
                    continue
                out_m.append(mv)
                out_b.append(bench_by_key[mk])
                out_mp.append(mp)
            return out_m, out_b, out_mp

        def _filter_with_both(keys_with_metric, metric_values, bench_values, target_keys):
            """Return arrays (metric, bench, arena_overall, mmlu_pro) subset to
            models that have all four. Used for the BOTH-controls partial."""
            out_m, out_b, out_ao, out_mp = [], [], [], []
            bench_by_key = dict(zip(target_keys, bench_values))
            for mk, mv in zip(keys_with_metric, metric_values):
                if mk not in bench_by_key:
                    continue
                bench_row = benchmarks.get(mk, {})
                ao = bench_row.get("arena_overall")
                mp = bench_row.get("mmlu_pro")
                if ao is None or mp is None:
                    continue
                out_m.append(mv)
                out_b.append(bench_by_key[mk])
                out_ao.append(ao)
                out_mp.append(mp)
            return out_m, out_b, out_ao, out_mp

        def _add_both_partial(result_key, keys_with_metric, metric_values, bench_values, target_keys):
            """Compute partial correlation against a benchmark controlling for
            both Arena Overall AND MMLU-Pro simultaneously. Writes to
            corr_results[metric][result_key] if n >= 5."""
            m_vals, b_vals, ao_vals, mp_vals = _filter_with_both(
                keys_with_metric, metric_values, bench_values, target_keys)
            if len(m_vals) < 5:
                return
            target = np.array(b_vals)
            predictor = np.array(m_vals)
            controls = [np.array(ao_vals), np.array(mp_vals)]
            part_rho, part_p = partial_spearman_multi(target, predictor, controls)
            part_r, part_r_p = partial_pearson_multi(target, predictor, controls)
            corr_results[metric][result_key] = {
                "spearman_rho": part_rho, "spearman_p": part_p,
                "pearson_r": part_r, "pearson_p": part_r_p,
                "p_value": part_p, "n_models": len(m_vals),
            }
            nice = result_key.replace("partial_", "").replace("_control_both", "")
            print(f"{metric.upper()} partial {nice} (| Overall+MMLU): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(m_vals)}")

        # Arena CW partialled on MMLU-Pro
        m_vals, b_vals, mp_vals = _filter_with_mmlu(
            aligned_models, metric_vals, arena_cw_vals, aligned_models)
        if len(m_vals) >= 5:
            part_rho, part_p = partial_spearman(
                target=np.array(b_vals), predictor=np.array(m_vals), control=np.array(mp_vals))
            part_r, part_r_p = partial_pearson(
                target=np.array(b_vals), predictor=np.array(m_vals), control=np.array(mp_vals))
            corr_results[metric]["partial_cw_control_mmlu_pro"] = {
                "spearman_rho": part_rho, "spearman_p": part_p,
                "pearson_r": part_r, "pearson_p": part_r_p,
                "p_value": part_p, "n_models": len(m_vals),
            }
            print(f"{metric.upper()} partial (CW | MMLU-Pro): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(m_vals)}")

        _add_both_partial("partial_cw_control_both",
                          aligned_models, metric_vals, arena_cw_vals, aligned_models)

        # EQ-Bench CW partialled on MMLU-Pro
        m_vals, b_vals, mp_vals = _filter_with_mmlu(
            aligned_models, metric_vals, eqbench_cw_vals, eqbench_cw_keys)
        if len(m_vals) >= 5:
            part_rho, part_p = partial_spearman(
                target=np.array(b_vals), predictor=np.array(m_vals), control=np.array(mp_vals))
            part_r, part_r_p = partial_pearson(
                target=np.array(b_vals), predictor=np.array(m_vals), control=np.array(mp_vals))
            corr_results[metric]["partial_eqbench_control_mmlu_pro"] = {
                "spearman_rho": part_rho, "spearman_p": part_p,
                "pearson_r": part_r, "pearson_p": part_r_p,
                "p_value": part_p, "n_models": len(m_vals),
            }
            print(f"{metric.upper()} partial EQ-Bench (| MMLU-Pro): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(m_vals)}")

        _add_both_partial("partial_eqbench_control_both",
                          aligned_models, metric_vals, eqbench_cw_vals, eqbench_cw_keys)

        # Hivemind partialled on MMLU-Pro
        m_vals, b_vals, mp_vals = _filter_with_mmlu(
            aligned_models, metric_vals, hivemind_vals, hivemind_keys)
        if len(m_vals) >= 5:
            part_rho, part_p = partial_spearman(
                target=np.array(b_vals), predictor=np.array(m_vals), control=np.array(mp_vals))
            part_r, part_r_p = partial_pearson(
                target=np.array(b_vals), predictor=np.array(m_vals), control=np.array(mp_vals))
            corr_results[metric]["partial_hivemind_control_mmlu_pro"] = {
                "spearman_rho": part_rho, "spearman_p": part_p,
                "pearson_r": part_r, "pearson_p": part_r_p,
                "p_value": part_p, "n_models": len(m_vals),
            }
            print(f"{metric.upper()} partial Hivemind (| MMLU-Pro): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(m_vals)}")

        _add_both_partial("partial_hivemind_control_both",
                          aligned_models, metric_vals, hivemind_vals, hivemind_keys)

        # --------------------------------------------------------------
        # Mazur V2 Creative Writing (appendix-only; n ≈ 21). A third
        # creative-writing ground truth from outside the Arena ecosystem.
        # --------------------------------------------------------------
        if len(mazur_vals) >= 5:
            mz_aligned_x = [
                val for mk, val in zip(aligned_models, metric_vals)
                if mk in mazur_keys
            ]
            mz_aligned_overall = [
                bench for mk, bench in zip(aligned_models, arena_overall_vals)
                if mk in mazur_keys and len(arena_overall_vals) == len(aligned_models)
            ]
            x_mz = np.array(mz_aligned_x)
            y_mz = np.array(mazur_vals)
            rho_mz, p_mz = spearman_corr(x_mz.tolist(), y_mz.tolist())
            r_mz, p_r_mz = pearson_corr(x_mz.tolist(), y_mz.tolist())
            corr_results[metric]["vs_mazur_cw"] = {
                "spearman_rho": rho_mz, "spearman_p": p_mz,
                "pearson_r": r_mz, "pearson_p": p_r_mz,
                "p_value": p_mz, "n_models": len(mazur_vals),
            }
            print(f"{metric.upper()} vs Mazur CW v2: rho={rho_mz:.3f} (p={p_mz:.4f}), r={r_mz:.3f} (p={p_r_mz:.4f}), n={len(mazur_vals)}")

            if len(mz_aligned_overall) == len(mazur_vals):
                part_rho, part_p = partial_spearman(
                    target=y_mz, predictor=x_mz, control=np.array(mz_aligned_overall))
                part_r, part_r_p = partial_pearson(
                    target=y_mz, predictor=x_mz, control=np.array(mz_aligned_overall))
                corr_results[metric]["partial_mazur_control_overall"] = {
                    "spearman_rho": part_rho, "spearman_p": part_p,
                    "pearson_r": part_r, "pearson_p": part_r_p,
                    "p_value": part_p, "n_models": len(mazur_vals),
                }
                print(f"{metric.upper()} partial Mazur (| Overall): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(mazur_vals)}")

            # Mazur partial on MMLU-Pro
            m_vals, b_vals, mp_vals = _filter_with_mmlu(
                aligned_models, metric_vals, mazur_vals, mazur_keys)
            if len(m_vals) >= 5:
                part_rho, part_p = partial_spearman(
                    target=np.array(b_vals), predictor=np.array(m_vals), control=np.array(mp_vals))
                part_r, part_r_p = partial_pearson(
                    target=np.array(b_vals), predictor=np.array(m_vals), control=np.array(mp_vals))
                corr_results[metric]["partial_mazur_control_mmlu_pro"] = {
                    "spearman_rho": part_rho, "spearman_p": part_p,
                    "pearson_r": part_r, "pearson_p": part_r_p,
                    "p_value": part_p, "n_models": len(m_vals),
                }
                print(f"{metric.upper()} partial Mazur (| MMLU-Pro): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(m_vals)}")

            _add_both_partial("partial_mazur_control_both",
                              aligned_models, metric_vals, mazur_vals, mazur_keys)

    # =====================================================================
    # ARC-AGI v2 analysis (n_max = 10 in the current benchmarks file).
    # Uses a parallel inclusion gate ("model has metric + has arc_agi_v2"),
    # not the arena_cw gate that the main loop above uses, so models like
    # openai_gpt-5-2 / google_gemini-3-1-pro-preview that have ARC-AGI but
    # no Arena CW score still contribute to the ARC-AGI sample.
    # =====================================================================
    arc_keys = [k for k, v in benchmarks.items() if "arc_agi_v2" in v]
    print(f"\n{'='*60}")
    print(f"ARC-AGI v2 analysis (pool size {len(arc_keys)})")
    print(f"{'='*60}")

    for metric in metrics_to_correlate:
        aligned = [
            k for k in arc_keys
            if k in model_scores
            and metric in model_scores[k]
            and model_scores[k][metric] not in (0, None)
        ]
        if len(aligned) < 5:
            print(f"  {metric} vs ARC-AGI: only n={len(aligned)}, skipping")
            continue
        x = np.array([model_scores[k][metric] for k in aligned])
        y = np.array([benchmarks[k]["arc_agi_v2"] for k in aligned])
        rho, p = spearman_corr(x.tolist(), y.tolist())
        r, p_r = pearson_corr(x.tolist(), y.tolist())
        boot = bootstrap_spearman(x, y, n_iter=n_bootstrap)
        corr_results.setdefault(metric, {})["vs_arc_agi"] = {
            "spearman_rho": rho, "spearman_p": p,
            "pearson_r": r, "pearson_p": p_r,
            "p_value": p, "n_models": len(aligned),
            "bootstrap": boot, "models": aligned,
        }
        print(f"{metric.upper()} vs ARC-AGI: rho={rho:.3f} (p={p:.4f}), r={r:.3f} (p={p_r:.4f}), n={len(aligned)}")

        # Partials are skipped for ARC-AGI when n < 7 (one control) or n < 8
        # (two controls): at n=5, partial rank correlations on residuals
        # routinely return |rho|=1 or NaN as a small-sample artifact, not a
        # real signal. The current ARC-AGI ∩ {Arena Overall, MMLU-Pro}
        # intersection sits at n=6, so the partials are intentionally
        # omitted here and the raw n=10 correlation is the reportable
        # quantity.
        sub_overall = [k for k in aligned if "arena_overall" in benchmarks[k]]
        sub_mmlu = [k for k in aligned if "mmlu_pro" in benchmarks[k]]
        sub_both = [k for k in aligned
                    if "arena_overall" in benchmarks[k]
                    and "mmlu_pro" in benchmarks[k]]
        if len(sub_overall) >= 7:
            xs = np.array([model_scores[k][metric] for k in sub_overall])
            ys = np.array([benchmarks[k]["arc_agi_v2"] for k in sub_overall])
            ao = np.array([benchmarks[k]["arena_overall"] for k in sub_overall])
            part_rho, part_p = partial_spearman(target=ys, predictor=xs, control=ao)
            part_r, part_r_p = partial_pearson(target=ys, predictor=xs, control=ao)
            corr_results[metric]["partial_arc_agi_control_overall"] = {
                "spearman_rho": part_rho, "spearman_p": part_p,
                "pearson_r": part_r, "pearson_p": part_r_p,
                "p_value": part_p, "n_models": len(sub_overall),
            }
            print(f"{metric.upper()} partial ARC-AGI (| Overall): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(sub_overall)}")
        if len(sub_mmlu) >= 7:
            xs = np.array([model_scores[k][metric] for k in sub_mmlu])
            ys = np.array([benchmarks[k]["arc_agi_v2"] for k in sub_mmlu])
            mp = np.array([benchmarks[k]["mmlu_pro"] for k in sub_mmlu])
            part_rho, part_p = partial_spearman(target=ys, predictor=xs, control=mp)
            part_r, part_r_p = partial_pearson(target=ys, predictor=xs, control=mp)
            corr_results[metric]["partial_arc_agi_control_mmlu_pro"] = {
                "spearman_rho": part_rho, "spearman_p": part_p,
                "pearson_r": part_r, "pearson_p": part_r_p,
                "p_value": part_p, "n_models": len(sub_mmlu),
            }
            print(f"{metric.upper()} partial ARC-AGI (| MMLU-Pro): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(sub_mmlu)}")
        if len(sub_both) >= 8:
            xs = np.array([model_scores[k][metric] for k in sub_both])
            ys = np.array([benchmarks[k]["arc_agi_v2"] for k in sub_both])
            ao = np.array([benchmarks[k]["arena_overall"] for k in sub_both])
            mp = np.array([benchmarks[k]["mmlu_pro"] for k in sub_both])
            part_rho, part_p = partial_spearman_multi(ys, xs, [ao, mp])
            part_r, part_r_p = partial_pearson_multi(ys, xs, [ao, mp])
            corr_results[metric]["partial_arc_agi_control_both"] = {
                "spearman_rho": part_rho, "spearman_p": part_p,
                "pearson_r": part_r, "pearson_p": part_r_p,
                "p_value": part_p, "n_models": len(sub_both),
            }
            print(f"{metric.upper()} partial ARC-AGI (| Overall+MMLU): rho={part_rho:.3f} (p={part_p:.4f}), r={part_r:.3f} (p={part_r_p:.4f}), n={len(sub_both)}")

    # ARC-AGI vs every other benchmark in the suite (benchmark-vs-benchmark).
    print(f"\nARC-AGI vs other benchmarks:")
    arc_vs_bench = {}
    other_benchmarks = [
        "arena_overall", "arena_cw", "eq_bench_cw", "eq_bench_cw_rubric",
        "mmlu_pro", "hivemind_diversity", "mazur_cw_v2",
    ]

    def _bench_value(model_bench, field):
        if field == "hivemind_diversity":
            if "hivemind_diversity" in model_bench:
                return model_bench["hivemind_diversity"]
            if "hivemind_intra_sim" in model_bench:
                return 1.0 - model_bench["hivemind_intra_sim"]
            return None
        return model_bench.get(field)

    for field in other_benchmarks:
        sub = [k for k in arc_keys if _bench_value(benchmarks[k], field) is not None]
        if len(sub) < 5:
            arc_vs_bench[field] = {"n_models": len(sub), "skipped_reason": "n<5"}
            print(f"  ARC-AGI vs {field}: only n={len(sub)}, skipping")
            continue
        x = np.array([benchmarks[k]["arc_agi_v2"] for k in sub])
        y = np.array([_bench_value(benchmarks[k], field) for k in sub])
        rho, p = spearman_corr(x.tolist(), y.tolist())
        r, p_r = pearson_corr(x.tolist(), y.tolist())
        boot = bootstrap_spearman(x, y, n_iter=n_bootstrap)
        arc_vs_bench[field] = {
            "spearman_rho": rho, "spearman_p": p,
            "pearson_r": r, "pearson_p": p_r,
            "p_value": p, "n_models": len(sub),
            "bootstrap": boot, "models": sub,
        }
        print(f"  ARC-AGI vs {field}: rho={rho:.3f} (p={p:.4f}), r={r:.3f} (p={p_r:.4f}), n={len(sub)}")
    corr_results["arc_agi_vs_benchmarks"] = arc_vs_bench

    # Inter-metric correlations
    metrics_available = [m for m in ["dat", "cdat", "pace"]
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
                    r, r_p = pearson_corr(x, y)
                    key = f"{m1}_vs_{m2}"
                    corr_results["inter_metric"][key] = {
                        "spearman_rho": rho,
                        "pearson_r": r,
                        "pearson_p": r_p,
                        "p_value": p,
                        "n_models": len(common),
                    }
                    print(f"  {m1} vs {m2}: rho={rho:.3f}, r={r:.3f} (n={len(common)})")

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
