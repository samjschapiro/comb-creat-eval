"""Utility-gated, equal-weight per-dimension z-composite over Kombine scores.

Downstream of ``score.py``. For each model it reads ``<scores_dir>/<model>/path_scores.json`` and
computes, per task (association / analogy / blending), the four creativity dimensions at the ARTIFACT
level, with surprise / originality / emergent **utility-gated**: each artifact's value counts only if
that artifact passed utility (a genuine, factual connection for association; a valid isomorphism for
analogy; a genuine fusion for blending), and 0 otherwise. This stops novelty from failed artifacts --
e.g. rare-but-malformed elements from a model that mostly fails -- inflating the score.

Each (task, dimension) is then z-scored across the scored models, and the z-scores are averaged with
equal weight into a per-task composite and an overall composite. Dimensions with zero variance or any
undefined value across models are skipped (this drops blending's surprise, which is d_cos(u,v): fixed
per item, so once gated it is collinear with utility).

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
TASK_DIMS = {
    "association": ["utility", "surprise", "originality"],
    "analogy": ["utility", "surprise", "originality", "emergent"],
    "blending": ["utility", "originality", "emergent"],
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

        def gated(key):
            return _mean([(r[key] if p else 0.0) for r, p in zip(arts, passed) if _ok(r.get(key))])

        d = {"utility": _mean([1.0 if p else 0.0 for p in passed]) if arts else float("nan"),
             "surprise": gated("R"), "originality": gated("originality")}
        if label != "association":
            d["emergent"] = gated("emergent_count")
        d["_n_artifacts"] = len(arts)
        out[label] = d
    return out


def compute(scores_dir: Path) -> dict:
    model_dirs = sorted(d for d in scores_dir.iterdir() if (d / "path_scores.json").exists())
    models = [d.name for d in model_dirs]
    raw = {m: artifact_dims(json.loads((scores_dir / m / "path_scores.json").read_text()))
           for m in models}

    # z-score each (task, dim) across models; skip constant / any-undefined columns.
    z = {m: [] for m in models}
    per_task = {m: {t: [] for t, _ in TASKS} for m in models}
    skipped = []
    for task, dims in TASK_DIMS.items():
        for k in dims:
            col = [raw[m][task][k] for m in models]
            if any(not _ok(v) for v in col) or _std(col) == 0:
                skipped.append(f"{task}.{k}")
                continue
            mu, sd = _mean(col), _std(col)
            for m in models:
                zz = (raw[m][task][k] - mu) / sd
                z[m].append(zz)
                per_task[m][task].append(zz)

    result = {"scores_dir": str(scores_dir), "models": models, "skipped_dims": skipped,
              "per_model": {}}
    for m in models:
        result["per_model"][m] = {
            "raw": raw[m],
            "per_task": {t: (_mean(v) if v else None) for t, v in per_task[m].items()},
            "overall": _mean(z[m]) if z[m] else None,
        }
    result["ranking"] = sorted(models, key=lambda m: result["per_model"][m]["overall"] or -1e9,
                               reverse=True)
    return result


def main(scores_dir: str):
    scores_dir = Path(scores_dir)
    res = compute(scores_dir)
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
    main(ap.parse_args().scores_dir)
