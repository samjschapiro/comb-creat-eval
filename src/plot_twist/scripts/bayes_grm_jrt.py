"""Bayesian Graded Response Model (Judge Response Theory; Myszkowski & Storme 2019)
to correct systematic per-judge severity and discrimination in the rubric ensemble.

Fits a GRM over the combined set (every LLM model + the human gold set) so all units
sit on ONE judge-corrected latent scale:

  rating_{i,j} ~ OrderedLogistic( eta = alpha_j * theta_i , cutpoints_j )
    theta_i  ~ Normal(0,1)            # per-unit (per-story) latent ability
    alpha_j  ~ LogNormal             # per-judge discrimination (>0)
    cutpoints_j (ordered)            # per-judge severity thresholds

Recovers per-unit ability theta alongside per-judge discrimination and severity.
The corrected per-model score is mean(theta); we verify it preserves the raw
leaderboard rank order while flattening per-judge scale differences (Figure 2 +
rank-preservation Spearman).

Usage:
    python src/plot_twist/scripts/bayes_grm_jrt.py configs/plot_twist/bayes_grm_jrt.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from src.utils import init_directory, load_config, save_config

BATLOW_BLUE, BATLOW_ORANGE = "#103D5F", "#EE9D6B"


def _source_of(story_id: str) -> str:
    return story_id.split("__t")[0] if "__t" in story_id else "human"


def _short(s: str) -> str:
    return "Human gold" if s == "human" else s.split("_", 1)[-1] if "_" in s else s


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    import pymc as pm
    import pytensor.tensor as pt
    import arviz as az

    cfg = load_config(config_path)
    for f in ("output_dir", "scores_files", "dimension"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)
    dim = cfg["dimension"]
    K = 5

    # --- assemble complete-case observations across all scored sets ---
    judges, story_ids, sources, raw_agg = None, [], [], []
    rows = []  # (story_index, judge_index, rating0)
    for path in cfg["scores_files"].values():
        data = json.loads(Path(path).read_text())
        if judges is None:
            judges = data["judge_models"]
        for s in data["scores"]:
            vals = [(s["by_judge"].get(j) or {}).get(dim) for j in judges]
            if any(v is None for v in vals):
                continue
            si = len(story_ids)
            story_ids.append(s["story_id"])
            sources.append(_source_of(s["story_id"]))
            raw_agg.append(s.get(dim))  # raw median aggregate, for rank-preservation check
            for ji, v in enumerate(vals):
                rows.append((si, ji, int(v) - 1))
    story_idx = np.array([r[0] for r in rows])
    judge_idx = np.array([r[1] for r in rows])
    y_obs = np.array([r[2] for r in rows])
    n_stories, n_judges = len(story_ids), len(judges)
    raw_agg = np.array(raw_agg, dtype=float)
    print(f"GRM over {n_stories} units x {n_judges} judges = {len(rows)} ratings (dim={dim})")

    # --- Bayesian GRM ---
    with pm.Model() as model:
        theta = pm.Normal("theta", 0.0, 1.0, shape=n_stories)
        alpha = pm.LogNormal("alpha", 0.0, 0.4, shape=n_judges)
        b1 = pm.Normal("b1", 0.0, 2.0, shape=n_judges)
        gaps = pm.HalfNormal("gaps", 1.5, shape=(n_judges, K - 2))
        cuts = pm.Deterministic("cuts", pt.concatenate(
            [b1[:, None], b1[:, None] + pt.cumsum(gaps, axis=1)], axis=1))  # (n_judges, K-1)

        a_o = alpha[judge_idx]
        th_o = theta[story_idx]
        c_o = cuts[judge_idx]                                   # (Nobs, K-1)
        Pge = pm.math.sigmoid(a_o[:, None] * (th_o[:, None] - c_o))  # P(Y>=2..K)
        ones = pt.ones((len(rows), 1))
        zeros = pt.zeros((len(rows), 1))
        Pfull = pt.concatenate([ones, Pge, zeros], axis=1)     # P(Y>=1..K+1)
        p = pt.clip(Pfull[:, :-1] - Pfull[:, 1:], 1e-9, 1.0)
        pm.Categorical("obs", p=p, observed=y_obs)

        idata = pm.sample(
            draws=cfg.get("draws", 600), tune=cfg.get("tune", 600),
            chains=cfg.get("chains", 2), cores=1, target_accept=0.9,
            random_seed=cfg.get("seed", 0), progressbar=False,
        )

    post = idata.posterior
    theta_hat = post["theta"].mean(("chain", "draw")).values
    theta_hdi = az.hdi(idata, var_names=["theta"])["theta"].values  # (n_stories, 2)
    alpha_hat = post["alpha"].mean(("chain", "draw")).values
    cuts_hat = post["cuts"].mean(("chain", "draw")).values          # (n_judges, K-1)
    severity = cuts_hat.mean(axis=1)                                # higher = harsher

    print("\nper-judge discrimination (alpha) and severity (mean cutpoint):")
    for j, a, sev in zip(judges, alpha_hat, severity):
        print(f"  {j:<32} alpha={a:.2f}  severity={sev:+.2f}")

    # --- per-source corrected (mean theta) vs raw (mean median) leaderboard ---
    sources_arr = np.array(sources)
    uniq = sorted(set(sources), key=lambda s: -np.mean(theta_hat[sources_arr == s]))
    table = []
    for s in uniq:
        m = sources_arr == s
        table.append({
            "source": s, "n": int(m.sum()),
            "corrected_theta": float(theta_hat[m].mean()),
            "raw_mean": float(raw_agg[m].mean()),
        })
    rho_rank = spearmanr([t["corrected_theta"] for t in table], [t["raw_mean"] for t in table]).statistic
    print(f"\nrank preservation (corrected theta vs raw median), Spearman = {rho_rank:.3f}")
    print(f"{'source':<26}{'n':>4}{'corrected_theta':>17}{'raw_mean':>10}")
    for t in table:
        print(f"{_short(t['source']):<26}{t['n']:>4}{t['corrected_theta']:>17.3f}{t['raw_mean']:>10.2f}")

    json.dump({
        "dimension": dim, "judges": judges,
        "discrimination": {j: float(a) for j, a in zip(judges, alpha_hat)},
        "severity": {j: float(s) for j, s in zip(judges, severity)},
        "rank_preservation_spearman": float(rho_rank),
        "leaderboard": table,
    }, open(out / "bayes_grm_jrt.json", "w"), indent=2)

    # --- Figure 2: (A) corrected leaderboard with HDI; (B) rank preservation ---
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(18, 8))
    order = table
    means = [t["corrected_theta"] for t in order]
    labels = [_short(t["source"]) for t in order]
    # HDI per source: mean of unit HDIs (approx CI on the corrected score)
    los, his = [], []
    for t in order:
        m = sources_arr == t["source"]
        los.append(means[order.index(t)] - (theta_hat[m] - theta_hdi[m, 0]).mean())
        his.append(means[order.index(t)] + (theta_hdi[m, 1] - theta_hat[m]).mean())
    cols = [BATLOW_ORANGE if t["source"] == "human" else BATLOW_BLUE for t in order]
    yerr = [np.array(means) - np.array(los), np.array(his) - np.array(means)]
    axA.bar(range(len(order)), means, color=cols, width=0.72,
            yerr=yerr, error_kw=dict(ecolor="#555", lw=1.5, capsize=4))
    axA.set_xticks(range(len(order)))
    axA.set_xticklabels(labels, rotation=35, ha="right", fontsize=16)
    axA.tick_params(axis="y", labelsize=16)
    axA.set_ylabel(r"Judge-corrected ability  $\theta$", fontsize=22, labelpad=10)
    axA.set_title("(A) Corrected leaderboard (Bayesian GRM)", fontsize=20)
    from matplotlib.patches import Patch
    axA.legend(handles=[Patch(color=BATLOW_ORANGE, label="Human gold"),
                        Patch(color=BATLOW_BLUE, label="LLM")], fontsize=15, loc="upper right")

    axB.scatter([t["raw_mean"] for t in order], means, s=120,
                color=[BATLOW_ORANGE if t["source"] == "human" else BATLOW_BLUE for t in order],
                zorder=3)
    for t, mn in zip(order, means):
        axB.annotate(_short(t["source"]), (t["raw_mean"], mn), fontsize=11,
                     xytext=(5, 3), textcoords="offset points")
    axB.set_xlabel(r"Raw median aggregate", fontsize=22, labelpad=10)
    axB.set_ylabel(r"Corrected $\theta$", fontsize=22, labelpad=10)
    axB.tick_params(labelsize=16)
    axB.set_title(f"(B) Rank preservation  (Spearman $\\rho$={rho_rank:.2f})", fontsize=20)
    fig.tight_layout()
    p_fig = out / "fig2_grm_correction.png"
    fig.savefig(p_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out/'bayes_grm_jrt.json'}\n       {p_fig}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
