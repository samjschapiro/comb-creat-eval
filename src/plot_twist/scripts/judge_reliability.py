"""Inter-judge reliability + IRT-style judge diagnostics for the rubric ensemble.

Operates on a rubric_scores.json (the per-story `by_judge` raw ratings) and, per
rubric dimension, reports the standard inter-rater reliability suite plus an
IRT-style judge-facet diagnostic (severity / consistency). Run on any scored set
(human gold and/or LLM).

Metrics per dimension (judges = raters, stories = items, ratings 1-5):
  - Krippendorff's alpha (ordinal)  -- chance-corrected agreement, handles missing
  - ICC(2,k)                        -- reliability of the 3-judge MEAN (two-way random)
  - Fleiss' kappa                   -- categorical agreement across judges
  - Kendall's W                     -- concordance of the judges' rankings
  - pairwise Spearman               -- per judge pair
IRT-style "judge response" facet (per judge):
  - severity   = mean rating offset from the grand mean (negative = harsher)
  - consistency = Spearman(judge, leave-one-out mean of the other judges)

Usage:
    python src/plot_twist/scripts/judge_reliability.py configs/plot_twist/judge_reliability.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.utils import init_directory, load_config, save_config

DIMENSIONS = ("surprise", "coherence", "prose_quality", "overall")


def _matrix(scores: list[dict], judges: list[str], dim: str) -> np.ndarray:
    """raters x items matrix of ratings (np.nan where a judge failed)."""
    M = np.full((len(judges), len(scores)), np.nan)
    for j, judge in enumerate(judges):
        for i, s in enumerate(scores):
            v = (s["by_judge"].get(judge) or {}).get(dim)
            if v is not None:
                M[j, i] = float(v)
    return M


def _icc_2k(M: np.ndarray) -> float:
    """ICC(2,k): two-way random, average-measures, absolute agreement. Complete cases."""
    X = M[:, ~np.isnan(M).any(axis=0)].T  # items x raters
    n, k = X.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = X.mean()
    ms_rows = k * ((X.mean(axis=1) - grand) ** 2).sum() / (n - 1)          # between items
    ms_cols = n * ((X.mean(axis=0) - grand) ** 2).sum() / (k - 1)          # between raters
    resid = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + grand
    ms_err = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_rows + (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom != 0 else float("nan")


def _fleiss_kappa(M: np.ndarray, cats=(1, 2, 3, 4, 5)) -> float:
    X = M[:, ~np.isnan(M).any(axis=0)]  # raters x items, complete cases
    k, n = X.shape
    if n < 1 or k < 2:
        return float("nan")
    counts = np.array([[ (X[:, i] == c).sum() for c in cats] for i in range(n)])  # items x cats
    p_j = counts.sum(axis=0) / (n * k)
    P_i = (((counts ** 2).sum(axis=1) - k) / (k * (k - 1)))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if (1 - P_e) != 0 else float("nan")


def _kendall_w(M: np.ndarray) -> float:
    X = M[:, ~np.isnan(M).any(axis=0)]  # raters x items
    k, n = X.shape
    if n < 2 or k < 2:
        return float("nan")
    ranks = np.apply_along_axis(lambda r: _rankdata(r), 1, X)  # rank items within each judge
    Rsum = ranks.sum(axis=0)
    S = ((Rsum - Rsum.mean()) ** 2).sum()
    # tie correction
    T = 0.0
    for r in X:
        _, cnt = np.unique(r, return_counts=True)
        T += (cnt ** 3 - cnt).sum()
    denom = k ** 2 * (n ** 3 - n) - k * T
    return float(12 * S / denom) if denom != 0 else float("nan")


def _rankdata(a):
    from scipy.stats import rankdata
    return rankdata(a)


def analyze(label: str, path: Path) -> dict:
    data = json.loads(path.read_text())
    scores = data["scores"]
    judges = data["judge_models"]
    import krippendorff

    out = {"label": label, "n_items": len(scores), "judges": judges, "dimensions": {}}
    print(f"\n===== {label}  (n={len(scores)} stories, {len(judges)} judges) =====")
    print(f"{'dim':<14}{'Kripp-a':>9}{'ICC(2,k)':>10}{'Fleiss-k':>10}{'Kendall-W':>11}")
    for dim in DIMENSIONS:
        M = _matrix(scores, judges, dim)
        alpha = float(krippendorff.alpha(reliability_data=M, level_of_measurement="ordinal"))
        icc = _icc_2k(M)
        fk = _fleiss_kappa(M)
        kw = _kendall_w(M)
        # pairwise spearman
        pair = {}
        for a, b in combinations(range(len(judges)), 2):
            mask = ~np.isnan(M[a]) & ~np.isnan(M[b])
            if mask.sum() >= 3 and np.std(M[a][mask]) > 0 and np.std(M[b][mask]) > 0:
                rho = float(spearmanr(M[a][mask], M[b][mask]).statistic)
            else:
                rho = float("nan")
            pair[f"{judges[a].split('/')[-1]}~{judges[b].split('/')[-1]}"] = rho
        out["dimensions"][dim] = {
            "krippendorff_alpha_ordinal": alpha, "icc_2k": icc,
            "fleiss_kappa": fk, "kendall_w": kw, "pairwise_spearman": pair,
        }
        print(f"{dim:<14}{alpha:>9.3f}{icc:>10.3f}{fk:>10.3f}{kw:>11.3f}")

    # IRT-style judge facet on OVERALL: severity (mean offset) + consistency (LOO Spearman)
    M = _matrix(scores, judges, "overall")
    grand = np.nanmean(M)
    facet = {}
    print(f"\n  judge facet (dimension=overall): severity = mean offset, consistency = Spearman vs others")
    for j, judge in enumerate(judges):
        sev = float(np.nanmean(M[j]) - grand)
        others = np.nanmean(np.delete(M, j, axis=0), axis=0)
        mask = ~np.isnan(M[j]) & ~np.isnan(others)
        cons = float(spearmanr(M[j][mask], others[mask]).statistic) if mask.sum() >= 3 else float("nan")
        facet[judge] = {"severity": sev, "consistency_vs_others": cons, "mean_rating": float(np.nanmean(M[j]))}
        print(f"    {judge:<32} severity={sev:+.3f}  consistency={cons:.3f}  mean={np.nanmean(M[j]):.2f}")
    out["judge_facet_overall"] = facet
    return out


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "scores_files"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    results = [analyze(label, Path(p)) for label, p in cfg["scores_files"].items()]
    (out / "reliability.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out/'reliability.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
