"""Utility-gated, equal-weight per-dimension z-composite over Kombine scores.

Downstream of ``score.py``. For each model it reads ``<scores_dir>/<model>/path_scores.json`` and
computes, per task (association / analogy / blending), the four creativity dimensions at the ARTIFACT
level, with surprise / originality / emergent **utility-gated**: each artifact's value counts only if
that artifact passed utility (a genuine, factual connection for association; a valid isomorphism for
analogy; a genuine fusion for blending), and 0 otherwise. This stops novelty from failed artifacts --
e.g. rare-but-malformed elements from a model that mostly fails -- inflating the score.

Each (task, dimension) is then z-scored across the scored models, and the z-scores are averaged with
equal weight into a per-task composite and an overall composite. Dimensions with zero variance or any
undefined value across models are skipped (in the current scorer nothing is: blend surprise is the mean
distance from each input to the blend's generic space, which the model writes, so it varies by model).

    python src/kg_creat/scripts/compute_composite.py data/kg_creat/kombine_v1/scores

Writes ``<scores_dir>/composite.json`` and prints the ranking. The association task is keyed
``baseline`` internally (a legacy mode name); it is reported as "association".
"""

import argparse
import json
import math
from pathlib import Path

# (task label, internal mode name). "baseline" is the legacy mode key for association.
TASKS = [("association", "baseline"), ("analogy", "analogy"), ("blending", "blending")]
# Dimensions entering the composite per task. Blending surprise is excluded (see module docstring).
# Emergent creativity is kept as SEPARATE dimensions (paper 05_benchmark.tex): utility (J^utl) and
# integration quality (J^qua). Emergent originality O(h)/O(c') is not separately scored.
TASK_DIMS = {
    "association": ["utility", "surprise", "originality"],
    # "originality" is the BASE artifact; "em_originality" is the emergent invention (kept separate).
    "analogy": ["utility", "surprise", "originality", "em_originality", "em_utility", "em_integration"],
    "blending": ["utility", "surprise", "originality", "em_originality", "em_utility", "em_integration"],
}


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _mean(xs):
    xs = [x for x in xs if _ok(x)]
    return (sum(xs) / len(xs)) if xs else float("nan")


def _std(xs):
    xs = [x for x in xs if _ok(x)]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def artifact_dims(recs: list) -> dict:
    """Per-task utility-gated dimension values for one model.

    The artifact and its utility flag are task-specific: a path (``sat``) for association, a pair head
    (``pair_sat``) for analogy, the single structure record (``sat``) for blending. Gated value =
    ``value`` if the artifact passed utility else 0, averaged over the model's artifacts for the task.
    """
    out = {}
    for label, mode in TASKS:
        rs = [r for r in recs if r.get("mode") == mode]
        if mode == "analogy":
            arts = [r for r in rs if "pair_sat" in r]
            passed = [r.get("pair_sat") is True for r in arts]
        else:
            arts = [r for r in rs if r.get("triples")]
            passed = [r.get("sat") is True for r in arts]

        # Every dimension is a FRACTION OF ITS MAXIMUM in [0,1] (utility-gated), so the score is
        # STATIONARY -- it does not depend on the pool of models (unlike a z-score).
        def gated01(key):  # gated mean of a value already bounded in [0,1] (cosine distance)
            return _mean([(min(1.0, max(0.0, r[key])) if p else 0.0)
                          for r, p in zip(arts, passed) if _ok(r.get(key))])

        d = {"utility": _mean([1.0 if p else 0.0 for p in passed]) if arts else float("nan"),
             "surprise": gated01("R"), "originality": gated01("originality")}
        if label in ("analogy", "blending"):
            # emergent-invention ORIGINALITY O(h)/O(c'), gated -- kept separate from base originality.
            d["em_originality"] = gated01("em_originality")
        if label == "analogy":
            # emergent utility J^utl_an and integration quality J^qua_an of the invention h, gated.
            d["em_utility"] = _mean([(int(bool(r.get("invention_utility"))) if p else 0.0)
                                     for r, p in zip(arts, passed) if _ok(r.get("invention_utility"))])
            d["em_integration"] = _mean([(int(bool(r.get("invention_integration"))) if p else 0.0)
                                         for r, p in zip(arts, passed) if _ok(r.get("invention_integration"))])
        elif label == "blending":
            # emergent utility J^utl_bl and integration quality J^qua_bl (scope in {1,2,3} ->
            # (scope-1)/2 in {0,.5,1}), gated.
            d["em_utility"] = _mean([(int(bool(r.get("blend_utility"))) if p else 0.0)
                                     for r, p in zip(arts, passed) if _ok(r.get("blend_utility"))])
            d["em_integration"] = _mean([(((r.get("blend_integration") or 1) - 1) / 2 if p else 0.0)
                                         for r, p in zip(arts, passed) if _ok(r.get("blend_integration"))])
        d["_n_artifacts"] = len(arts)
        out[label] = d
    return out


def compute(scores_dir: Path) -> dict:
    model_dirs = sorted(d for d in scores_dir.iterdir() if (d / "path_scores.json").exists())
    models = [d.name for d in model_dirs]
    raw = {m: artifact_dims(json.loads((scores_dir / m / "path_scores.json").read_text()))
           for m in models}

    # STATIONARY % score: each task = mean of its [0,1] dimension fractions x 100; overall = mean of
    # the per-task %s (equal weight per task). No pool-dependent normalisation.
    skipped = [f"{task}.{k}" for task, dims in TASK_DIMS.items() for k in dims
               if any(not _ok(raw[m][task].get(k)) for m in models)]
    result = {"scores_dir": str(scores_dir), "models": models, "skipped_dims": skipped,
              "scale": "percent_of_max", "per_model": {}}
    for m in models:
        per_task = {}
        for task, dims in TASK_DIMS.items():
            vals = [raw[m][task][k] for k in dims if _ok(raw[m][task].get(k))]
            per_task[task] = 100.0 * _mean(vals) if vals else None
        overall = _mean([v for v in per_task.values() if _ok(v)])
        result["per_model"][m] = {"raw": raw[m], "per_task": per_task, "overall": overall}
    result["ranking"] = sorted(models, key=lambda m: result["per_model"][m]["overall"] or -1e9,
                               reverse=True)
    return result


def main(scores_dir: str, allow_dropped_dims: bool = False):
    scores_dir = Path(scores_dir)
    res = compute(scores_dir)

    # A dimension that silently disappears changes what the composite MEANS, and the leaderboard still
    # prints. This happened for real: `em_originality` is written by rescore_split_originality.py, which
    # had not been run on newly-scored models, so the composite quietly fell from 6 dimensions to 4 on
    # two tasks. Compare against the previous run and refuse unless the drop is asked for.
    prev = scores_dir / "composite.json"
    if prev.exists() and not allow_dropped_dims:
        was = set(json.loads(prev.read_text()).get("skipped_dims") or [])
        newly = sorted(set(res["skipped_dims"]) - was)
        if newly:
            raise SystemExit(
                f"FATAL: {len(newly)} dimension(s) present in the previous composite are now undefined "
                f"for at least one model: {', '.join(newly)}.\nThe composite would silently change "
                f"meaning. Usually this means a model was scored without a follow-up scorer having been "
                f"run over the whole pool (e.g. rescore_split_originality.py for em_originality). Fix "
                f"that, or pass --allow-dropped-dims if the drop is intended.")
    (scores_dir / "composite.json").write_text(json.dumps(res, indent=2))

    if res["skipped_dims"]:
        print(f"Skipped (constant/undefined) dimensions: {', '.join(res['skipped_dims'])}")
    print(f"\n{'model':32s} {'assoc':>7s} {'analogy':>8s} {'blend':>7s} {'OVERALL':>8s}")
    for m in res["ranking"]:
        pm = res["per_model"][m]
        pt = pm["per_task"]
        def f(x):
            return f"{x:>7.2f}" if isinstance(x, float) else f"{'n/a':>7s}"
        print(f"{m:32s} {f(pt['association'])} {f(pt['analogy']):>8s} {f(pt['blending'])} "
              f"{pm['overall']:>8.2f}")
    print(f"\nWrote {scores_dir / 'composite.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir", type=str, help="e.g. data/kg_creat/kombine_v1/scores")
    ap.add_argument("--allow-dropped-dims", action="store_true",
                    help="permit a dimension the previous composite had to become undefined")
    a = ap.parse_args()
    main(a.scores_dir, a.allow_dropped_dims)
