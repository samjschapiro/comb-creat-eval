"""Graded Response Model (GRM) judge diagnostics -- the AGC IRT framework applied
to our rubric ensemble (Samejima GRM; cf. "Diagnosing the Reliability of
LLM-as-a-Judge via Item Response Theory", arXiv:2602.00521).

We treat the 3 judges as the GRM "items"/replications and each story as a subject
with latent twist-quality theta. Per dimension we estimate:

  - discrimination alpha_j   : how sharply judge j separates quality
  - thresholds beta_{j,k}     : where judge j switches between adjacent scores
                                (top threshold = severity; higher = harsher)
  - latent quality theta_i    : judge-calibrated quality per story
  - marginal reliability rho   : Var(theta_hat) / (Var(theta_hat) + E[SE^2]),
                                 with SE from the GRM test information (AGC Phase-1
                                 reliability; acceptance rho >= 0.70)

NOT computable from current data (stated, not faked):
  - Prompt-consistency C_v   : needs the SAME judge re-run under prompt variants.
  - Human-alignment (W1 D_v, theta-range ratio): needs a human-rated subset.

Usage:
    python src/plot_twist/scripts/grm_irt.py configs/plot_twist/grm_irt.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.utils import init_directory, load_config, save_config

DIMENSIONS = ("surprise", "coherence", "overall")


def _grm_info_at(theta: float, alpha: np.ndarray, beta: np.ndarray) -> float:
    """Total GRM test information at theta (sum over judges). Samejima."""
    total = 0.0
    for a, b in zip(alpha, beta):
        Pstar = 1.0 / (1.0 + np.exp(-a * (theta - b)))      # P(Y>=k), k=1..K-1
        Ps = np.concatenate([[1.0], Pstar, [0.0]])           # P*_0..P*_K
        Q = 1.0 - Ps
        num = (Ps[:-1] * Q[:-1] - Ps[1:] * Q[1:]) ** 2
        den = np.clip(Ps[:-1] - Ps[1:], 1e-9, None)          # category prob P_k
        total += a * a * np.sum(num / den)
    return float(total)


def _fit(scores, judges, dim):
    import girth

    rows, med = [], []
    for s in scores:
        col = [(s["by_judge"].get(j) or {}).get(dim) for j in judges]
        if all(v is not None for v in col):
            rows.append(col)
            med.append(s.get(dim))
    M = np.array(rows, dtype=int).T  # items(judges) x respondents(stories)
    res = girth.grm_mml(M)
    alpha = np.asarray(res["Discrimination"], dtype=float)
    beta = np.asarray(res["Difficulty"], dtype=float)       # judges x (K-1)
    theta = np.asarray(res["Ability"], dtype=float)         # per story
    se = np.array([1.0 / np.sqrt(_grm_info_at(t, alpha, beta)) for t in theta])
    rho = float(np.var(theta) / (np.var(theta) + np.mean(se ** 2)))
    # does the IRT-calibrated quality agree with the simple median aggregate?
    med = np.array(med, dtype=float)
    rho_med = float(spearmanr(theta, med).statistic) if np.std(med) > 0 else float("nan")
    return {
        "n": int(M.shape[1]),
        "discrimination": {j: float(a) for j, a in zip(judges, alpha)},
        "thresholds": {j: [float(x) for x in b] for j, b in zip(judges, beta)},
        "severity_top_threshold": {j: float(b[-1]) for j, b in zip(judges, beta)},
        "marginal_reliability_rho": rho,
        "spearman_theta_vs_median": rho_med,
    }


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "scores_files"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    results = {}
    for label, path in cfg["scores_files"].items():
        data = json.loads(Path(path).read_text())
        judges = data["judge_models"]
        print(f"\n===== {label}  (judges: {', '.join(j.split('/')[-1] for j in judges)}) =====")
        results[label] = {}
        for dim in DIMENSIONS:
            r = _fit(data["scores"], judges, dim)
            results[label][dim] = r
            print(f"\n  [{dim}]  n={r['n']}   marginal reliability rho={r['marginal_reliability_rho']:.3f}"
                  f"   (theta vs median rho={r['spearman_theta_vs_median']:.3f})")
            print(f"    {'judge':<32}{'discrim a':>11}{'severity b_top':>16}")
            for j in judges:
                print(f"    {j:<32}{r['discrimination'][j]:>11.2f}{r['severity_top_threshold'][j]:>16.2f}")

    (out / "grm_irt.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved: {out/'grm_irt.json'}")
    print("\nNOTE: AGC Phase-1 prompt-consistency (C_v) needs prompt-variant judge runs;")
    print("      AGC Phase-2 human-alignment (Wasserstein D_v, theta-range ratio) needs a human-rated subset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
