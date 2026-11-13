"""Test-retest reliability of Kombine scores across two disjoint item sets (rebuttal, reviewer cGKy Q1).

test30   = the paper's 30 items per task (data/kg_creat/human_study/paper_scores)
retest30 = 30 new cross-domain anchor pairs, disjoint at the entity level, shared by all three tasks
           (data/kg_creat/kombine_retest30/scores; scored by the paper's panel + Haiku-4.5 factuality)

The same 10 models answered both sets. Originality is pool-relative, so for a like-for-like comparison
test30 originality is RECOMPUTED over a pool of the same 10 models only (no other models, no humans), which
is the pool retest30 was scored against. Every other dimension is pool-free and is used as stored.

Reports, per task and overall:
  * per model: score on each set, the difference, and an item-bootstrap 95% CI on the difference
  * across the 10 models: Pearson r, Spearman rho, ICC(A,1) (absolute agreement) between the two sets
  * the item-sampling standard error of a model's score within one set of 30 items (the noise floor)
and the same for utility (the pass rate) alone, which is the pool-free core of the composite.

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_retest
(needs MLX for the test30 originality recomputation; run rescore_split_originality on the retest scores first)
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.compute_composite import TASK_DIMS, _mean, _ok, artifact_dims
from src.kg_creat.scripts.robustness_originality_pool import ItemPool, apply, collect, load, score_variant

RETEST = Path("data/kg_creat/kombine_retest30/scores")
OUT = Path("data/kg_creat/kombine_retest30/analysis/test_retest.json")
# these three were drawn on OpenRouter at T=0.9 with an 8k thinking cap for test30, and on the direct
# Anthropic API (explicit thinking budget, no temperature) for retest30: sampling conditions changed too.
COND_CHANGED = {"anthropic_claude-fable-5", "anthropic_claude-opus-4-5", "anthropic_claude-opus-4-6"}
TASKS = list(TASK_DIMS)
B = 2000


def scores(recs):
    raw = artifact_dims(recs)
    pt = {t: 100.0 * _mean([raw[t][k] for k in ds if _ok(raw[t].get(k))]) for t, ds in TASK_DIMS.items()}
    pt["overall"] = _mean([pt[t] for t in TASKS])
    for t in TASKS:
        pt[f"{t}.utility"] = 100.0 * raw[t]["utility"]
    pt["utility"] = _mean([pt[f"{t}.utility"] for t in TASKS])
    return pt


def by_item(recs):
    out = {}
    for r in recs:
        out.setdefault((r["mode"], r["prompt_id"]), []).append(r)
    return out


def boot(recs, rng):
    """Resample items within each task (with replacement); returns the records of one bootstrap draw."""
    groups = by_item(recs)
    keys_by_mode = {}
    for k in groups:
        keys_by_mode.setdefault(k[0], []).append(k)
    out = []
    for ks in keys_by_mode.values():
        for i in rng.integers(0, len(ks), len(ks)):
            out += groups[ks[i]]
    return out


def icc_a1(x, y):
    """ICC(A,1), two-way, absolute agreement, single measure (McGraw & Wong), for n targets x 2 sets."""
    X = np.column_stack([x, y]).astype(float)
    n, k = X.shape
    gm = X.mean()
    msr = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def main():
    models = sorted(d.name for d in RETEST.iterdir() if (d / "path_scores.json").exists())
    print(f"{len(models)} models: {models}")
    re_recs = {m: json.loads((RETEST / m / "path_scores.json").read_text()) for m in models}
    assert all(any(r.get("em_originality") is not None for r in re_recs[m]) for m in models), \
        "run rescore_split_originality on the retest scores first"

    # test30 with originality recomputed over the same 10-model pool (paper's in-pool rule)
    _paper_models, ents, recs, resp = load()
    missing = set(models) - set(recs)
    assert not missing, missing
    pa, pb, pe, ta, ts = collect(ents, resp)
    embed = get_embedder()
    pools = tuple({pid: ItemPool(c, embed) for pid, c in p.items()} for p in (pa, pb, pe))
    upd = score_variant(pools, (ta, ts), set(models), set(models), exclude_identical=True)
    te_recs = {m: apply(recs[m], upd[m]) for m in models}

    keys = TASKS + ["overall"] + [f"{t}.utility" for t in TASKS] + ["utility"]
    te = {m: scores(te_recs[m]) for m in models}
    re_ = {m: scores(re_recs[m]) for m in models}

    rng = np.random.default_rng(0)
    bt = {m: [scores(boot(te_recs[m], rng)) for _ in range(B)] for m in models}
    br = {m: [scores(boot(re_recs[m], rng)) for _ in range(B)] for m in models}

    res = {"models": models, "cond_changed": sorted(COND_CHANGED & set(models)), "B": B,
           "per_model": {}, "agreement": {}, "item_se": {}}
    for m in models:
        res["per_model"][m] = {}
        for k in keys:
            d = np.array([b[k] - a[k] for a, b in zip(bt[m], br[m])])
            res["per_model"][m][k] = {"test": te[m][k], "retest": re_[m][k], "diff": re_[m][k] - te[m][k],
                                      "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    for k in keys:
        x = [te[m][k] for m in models]
        y = [re_[m][k] for m in models]
        sub = [m for m in models if m not in COND_CHANGED]
        res["agreement"][k] = {
            "pearson": float(pearsonr(x, y)[0]), "spearman": float(spearmanr(x, y)[0]), "icc_a1": icc_a1(x, y),
            "mean_test": float(np.mean(x)), "mean_retest": float(np.mean(y)),
            "mean_abs_diff": float(np.mean(np.abs(np.subtract(y, x)))),
            "n_ci_excl_0": sum(1 for m in models if not (res["per_model"][m][k]["ci"][0] <= 0 <= res["per_model"][m][k]["ci"][1])),
            "same_conditions_only": {"n": len(sub),
                                     "pearson": float(pearsonr([te[m][k] for m in sub], [re_[m][k] for m in sub])[0]) if len(sub) > 2 else None,
                                     "spearman": float(spearmanr([te[m][k] for m in sub], [re_[m][k] for m in sub])[0]) if len(sub) > 2 else None}}
        res["item_se"][k] = float(np.median([np.std([b[k] for b in bt[m]]) for m in models]))

    print(f"\n{'measure':22s} {'r':>6s} {'rho':>6s} {'ICC':>6s} {'mean T':>7s} {'mean R':>7s} {'|d|':>6s} {'SE':>5s} ci!0  (same-cond r/rho)")
    for k in keys:
        a = res["agreement"][k]
        s = a["same_conditions_only"]
        print(f"{k:22s} {a['pearson']:6.2f} {a['spearman']:6.2f} {a['icc_a1']:6.2f} {a['mean_test']:7.1f} {a['mean_retest']:7.1f} "
              f"{a['mean_abs_diff']:6.1f} {res['item_se'][k]:5.1f} {a['n_ci_excl_0']:3d}   ({s['pearson']:.2f}/{s['spearman']:.2f})")
    print("\nper model, overall: test -> retest (diff, 95% CI)")
    for m in sorted(models, key=lambda m: -te[m]["overall"]):
        p = res["per_model"][m]["overall"]
        print(f"  {m:36s} {p['test']:5.1f} -> {p['retest']:5.1f}  ({p['diff']:+5.1f} [{p['ci'][0]:+.1f}, {p['ci'][1]:+.1f}])"
              + ("  *conditions changed" if m in COND_CHANGED else ""))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
