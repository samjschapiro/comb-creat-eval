"""Rebuttal (reviewer cGKy): is the Kombine leaderboard an artifact of the equal-weight composite and
the hard zero on utility failure?

Starts from the paper's scores (data/kg_creat/human_study/paper_scores, the 35 models; humans are
reported separately and excluded here) and recomputes the 35-model ranking under:

  gating variants   paper = utility-gated (a failed artifact scores 0 on every other dimension)
                    ungated = every dimension averaged over all artifacts, pass or fail
                    among_passed = non-utility dimensions averaged over passed artifacts only
                                   (utility itself still enters as the pass rate)
  random weights    10,000 draws of Dirichlet(1) weights over the dimensions within each task and
                    Dirichlet(1) weights over the three tasks (paper gating)
  alternatives      utility-only (the task's utility dims), originality-only (base + emergent
                    originality), geometric mean of the three task scores, task weights proportional
                    to 1 / across-model variance of the task score

For each: Spearman and Kendall vs the paper ranking, top-1 identity, top-5 and top-10 overlap.
Writes data/kg_creat/kombine_test30/analysis/rebuttal_composite_robustness.json.

    .venv/bin/python -m src.kg_creat.scripts.rebuttal_composite_robustness
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.kg_creat.scripts.compute_composite import TASKS, TASK_DIMS, _mean, _ok

SCORES = Path("data/kg_creat/human_study/paper_scores")
OUT = Path("data/kg_creat/kombine_test30/analysis/rebuttal_composite_robustness.json")
N_DRAWS = 10_000
SEED = 0


def dims(recs, gate):
    """Per-task dimension fractions for one model. gate in {'paper', 'ungated', 'among_passed'}."""
    out = {}
    for label, mode in TASKS:
        rs = [r for r in recs if r.get("mode") == mode]
        if mode == "analogy":
            arts = [r for r in rs if "pair_sat" in r]
            passed = [r.get("pair_sat") is True for r in arts]
        else:
            arts = [r for r in rs if r.get("triples")]
            passed = [r.get("sat") is True for r in arts]

        def val(r, key):
            x = r.get(key)
            if key == "blend_integration":
                return ((x or 1) - 1) / 2 if _ok(x) else None
            if key in ("invention_utility", "invention_integration", "blend_utility"):
                return float(bool(x)) if _ok(x) else None
            return min(1.0, max(0.0, x)) if _ok(x) else None

        def agg(key):
            if gate == "paper":
                return _mean([(val(r, key) if p else 0.0) for r, p in zip(arts, passed) if val(r, key) is not None])
            if gate == "ungated":
                return _mean([val(r, key) for r in arts if val(r, key) is not None])
            return _mean([val(r, key) for r, p in zip(arts, passed) if p and val(r, key) is not None])

        d = {"utility": _mean([float(p) for p in passed]) if arts else float("nan"),
             "surprise": agg("R"), "originality": agg("originality")}
        if label in ("analogy", "blending"):
            d["em_originality"] = agg("em_originality")
        if label == "analogy":
            d["em_utility"] = agg("invention_utility")
            d["em_integration"] = agg("invention_integration")
        elif label == "blending":
            d["em_utility"] = agg("blend_utility")
            d["em_integration"] = agg("blend_integration")
        out[label] = {k: (0.0 if not _ok(v) else v) for k, v in d.items()}   # a model with no passed artifact
    return out


def task_scores(D, w_dims=None):
    """D: model -> task -> dim -> value. Returns array (models x 3) of task scores (0-100)."""
    out = []
    for m in D:
        row = []
        for task in ("association", "analogy", "blending"):
            ks = TASK_DIMS[task]
            v = np.array([D[m][task][k] for k in ks])
            w = np.ones(len(ks)) / len(ks) if w_dims is None else w_dims[task]
            row.append(100 * float(v @ w))
        out.append(row)
    return np.array(out)


def compare(base_scores, alt_scores, names):
    rb = (-base_scores).argsort().argsort()
    ra = (-alt_scores).argsort().argsort()
    top = lambda s, k: set(np.argsort(-s)[:k])
    return {"spearman": float(spearmanr(base_scores, alt_scores)[0]),
            "kendall": float(kendalltau(base_scores, alt_scores)[0]),
            "top1_same": bool(np.argmax(base_scores) == np.argmax(alt_scores)),
            "top1": names[int(np.argmax(alt_scores))],
            "top5_overlap": len(top(base_scores, 5) & top(alt_scores, 5)),
            "top10_overlap": len(top(base_scores, 10) & top(alt_scores, 10)),
            "max_rank_shift": int(np.abs(rb - ra).max()),
            "mean_abs_rank_shift": float(np.abs(rb - ra).mean())}


def main():
    models = sorted(d.name for d in SCORES.iterdir()
                    if (d / "path_scores.json").exists() and not d.name.startswith("human"))
    recs = {m: json.loads((SCORES / m / "path_scores.json").read_text()) for m in models}
    D = {g: {m: dims(recs[m], g) for m in models} for g in ("paper", "ungated", "among_passed")}

    paper_T = task_scores(D["paper"])
    paper = paper_T.mean(1)
    # sanity: reproduces the paper composite
    comp = json.loads((SCORES / "composite.json").read_text())["per_model"]
    ref = np.array([comp[m]["overall"] for m in models])
    assert np.allclose(paper, ref, atol=1e-6), float(np.abs(paper - ref).max())
    print(f"reproduced paper composite for {len(models)} models (max |diff| {np.abs(paper - ref).max():.2e})")

    res = {"n_models": len(models), "paper_top5": [models[i] for i in np.argsort(-paper)[:5]], "variants": {}}

    def add(name, scores):
        res["variants"][name] = compare(paper, scores, models)

    add("ungated", task_scores(D["ungated"]).mean(1))
    add("among_passed", task_scores(D["among_passed"]).mean(1))

    util_w = {t: np.array([1.0 if k in ("utility", "em_utility", "em_integration") else 0.0 for k in TASK_DIMS[t]])
              for t in TASK_DIMS}
    orig_w = {t: np.array([1.0 if k in ("originality", "em_originality") else 0.0 for k in TASK_DIMS[t]])
              for t in TASK_DIMS}
    for w in (util_w, orig_w):
        for t in w:
            w[t] = w[t] / w[t].sum()
    add("utility_only", task_scores(D["paper"], util_w).mean(1))
    add("originality_only_gated", task_scores(D["paper"], orig_w).mean(1))
    add("originality_only_ungated", task_scores(D["ungated"], orig_w).mean(1))
    add("geometric_mean_of_tasks", np.exp(np.log(np.clip(paper_T, 1e-6, None)).mean(1)))
    iv = 1 / paper_T.var(0)
    add("task_weights_inverse_variance", paper_T @ (iv / iv.sum()))
    for i, t in enumerate(("association", "analogy", "blending")):
        add(f"single_task_{t}", paper_T[:, i])

    # why gate: without it, originality rewards failed artifacts. Correlate ungated originality with utility.
    u = task_scores(D["paper"], util_w).mean(1)
    ou = task_scores(D["ungated"], orig_w).mean(1)
    res["ungated_originality_vs_utility_spearman"] = float(spearmanr(ou, u)[0])
    res["ungated_originality_vs_utility_p"] = float(spearmanr(ou, u)[1])
    # per-artifact: is a failed artifact more "original" than a passed one? (pooled over models, per task)
    fail_vs_pass = {}
    for label, mode in TASKS:
        po, fo = [], []
        for m in models:
            for r in recs[m]:
                if r.get("mode") != mode or not _ok(r.get("originality")):
                    continue
                ok = r.get("pair_sat") if mode == "analogy" else r.get("sat")
                if mode == "analogy" and "pair_sat" not in r:
                    continue
                (po if ok is True else fo).append(r["originality"])
        fail_vs_pass[label] = {"orig_passed": float(np.mean(po)), "orig_failed": float(np.mean(fo)),
                               "n_passed": len(po), "n_failed": len(fo)}
    res["originality_failed_vs_passed"] = fail_vs_pass

    # random Dirichlet weights
    rng = np.random.default_rng(SEED)
    sp, kt, top1, top5, top10 = [], [], [], [], []
    for _ in range(N_DRAWS):
        w_dims = {t: rng.dirichlet(np.ones(len(TASK_DIMS[t]))) for t in TASK_DIMS}
        w_task = rng.dirichlet(np.ones(3))
        s = task_scores(D["paper"], w_dims) @ w_task
        c = compare(paper, s, models)
        sp.append(c["spearman"]); kt.append(c["kendall"]); top1.append(c["top1_same"])
        top5.append(c["top5_overlap"]); top10.append(c["top10_overlap"])
    q = lambda x: {"median": float(np.median(x)), "p5": float(np.percentile(x, 5)), "p95": float(np.percentile(x, 95)),
                   "min": float(np.min(x))}
    res["dirichlet"] = {"n_draws": N_DRAWS, "spearman": q(sp), "kendall": q(kt),
                        "top1_same_rate": float(np.mean(top1)),
                        "top5_overlap": q(top5), "top5_overlap_ge4_rate": float(np.mean(np.array(top5) >= 4)),
                        "top10_overlap": q(top10)}
    top1_counts = {}
    rng = np.random.default_rng(SEED)
    for _ in range(N_DRAWS):
        w_dims = {t: rng.dirichlet(np.ones(len(TASK_DIMS[t]))) for t in TASK_DIMS}
        w_task = rng.dirichlet(np.ones(3))
        s = task_scores(D["paper"], w_dims) @ w_task
        top1_counts[models[int(np.argmax(s))]] = top1_counts.get(models[int(np.argmax(s))], 0) + 1
    res["dirichlet"]["top1_distribution"] = dict(sorted(top1_counts.items(), key=lambda kv: -kv[1]))

    OUT.write_text(json.dumps(res, indent=1))
    print(f"\npaper top-5: {res['paper_top5']}")
    print(f"\n{'variant':32s} {'spearman':>8s} {'kendall':>8s} {'top1':>5s} {'top5':>5s} {'top10':>6s} {'maxΔrank':>8s}  top-1 model")
    for k, c in res["variants"].items():
        print(f"{k:32s} {c['spearman']:8.3f} {c['kendall']:8.3f} {str(c['top1_same']):>5s} {c['top5_overlap']:>5d} "
              f"{c['top10_overlap']:>6d} {c['max_rank_shift']:>8d}  {c['top1']}")
    d = res["dirichlet"]
    print(f"\nDirichlet(1) weights, {N_DRAWS} draws: Spearman median {d['spearman']['median']:.3f} "
          f"[5th pct {d['spearman']['p5']:.3f}, min {d['spearman']['min']:.3f}]; Kendall median {d['kendall']['median']:.3f} "
          f"[5th {d['kendall']['p5']:.3f}]; top-1 unchanged {100 * d['top1_same_rate']:.1f}%; "
          f"top-5 overlap >=4 in {100 * d['top5_overlap_ge4_rate']:.1f}% (median {d['top5_overlap']['median']:.0f}/5)")
    print("top-1 distribution:", d["top1_distribution"])
    print(f"ungated originality vs utility across models: Spearman {res['ungated_originality_vs_utility_spearman']:+.3f} "
          f"(p={res['ungated_originality_vs_utility_p']:.2g})")
    for t, v in res["originality_failed_vs_passed"].items():
        print(f"  {t:12s} mean originality: failed {v['orig_failed']:.3f} (n={v['n_failed']}) vs passed {v['orig_passed']:.3f} (n={v['n_passed']})")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
