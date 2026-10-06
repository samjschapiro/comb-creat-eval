"""Human vs. model creativity on the paper's composite, per dimension, gated and among passed answers.

Reads data/kg_creat/human_study/combined_scores (rescore_originality_combined.py --pool combined, then
compute_composite.py): every entity's originality is against the same human + model pool, on the 86 items both
answered. Writes data/kg_creat/human_study/analysis/human_vs_models.json:

  gated          the composite's own inputs (a dimension counts 0 when the answer failed utility)
  among_passed   mean over answers that PASSED utility only -- how original/surprising valid answers are

    .venv/bin/python -m src.kg_creat.scripts.analyze_human_vs_models
"""
import glob
import json
import os

import numpy as np

SCORES = "data/kg_creat/human_study/paper_scores"   # the paper's scores: shared pool, skips failed, all 90 model items (build_paper_scores.py)
OUT = "data/kg_creat/human_study/analysis/human_vs_models.json"
TASKS = [("association", "baseline"), ("analogy", "analogy"), ("blending", "blending")]


def among_passed(recs):
    out = {}
    for t, mode in TASKS:
        rs = [r for r in recs if r.get("mode") == mode]
        arts = [r for r in rs if "pair_sat" in r] if mode == "analogy" else [r for r in rs if r.get("triples")]
        ok = [(r.get("pair_sat") if mode == "analogy" else r.get("sat")) is True for r in arts]
        for k, label in (("originality", "originality"), ("R", "surprise"), ("em_originality", "em_originality")):
            v = [r[k] for r, p in zip(arts, ok) if p and r.get(k) is not None]
            if v:
                out[f"{t}.{label}"] = float(np.mean(v))
    return out


def compare(h, xs):
    return {"human": round(h, 4), "model_median": round(float(np.median(xs)), 4),
            "model_min": round(min(xs), 4), "model_max": round(max(xs), 4),
            "models_below_human": int(sum(x < h for x in xs)), "n_models": len(xs)}


def human_rows(comp, models):
    """Composite of kept humans and of the kept humans who passed the Breakfast + Lunch screen (blend judge on the
    warm-up, data/kg_creat/human_study/scores), with each group's rank among the models + that group."""
    from src.kg_creat.scripts.compute_composite import TASK_DIMS, _mean, _ok, artifact_dims
    root = "data/kg_creat/human_study"
    idx = {r["sample_idx"]: r["session_id"] for r in json.load(open(f"{root}/score_input/human_kept/responses.json"))}
    passed = {idx[r["sample_idx"]] for r in json.load(open(f"{root}/scores/human_kept/path_scores.json"))
              if r["prompt_id"] == "blend_ctrl_1" and r.get("generic_ok") is True}
    kept = json.load(open(f"{SCORES}/human_kept/path_scores.json"))

    def composite(recs):
        raw = artifact_dims(recs)
        pt = {t: 100 * _mean([raw[t][k] for k in ds if _ok(raw[t].get(k))]) for t, ds in TASK_DIMS.items()}
        return {"overall": _mean(list(pt.values())), **pt}
    out = {}
    for name, recs, n in (("all", kept, len(idx)),
                          ("filtered", [r for r in kept if idx.get(r.get("sample_idx")) in passed], len(passed))):
        c = composite(recs)
        out[name] = {"n_sessions": n, **{v: {"score": round(c[v], 2),
                     "rank": 1 + sum((comp[m]["overall"] if v == "overall" else comp[m]["per_task"][v]) > c[v]
                                     for m in models), "of": len(models) + 1}
                     for v in ("overall", "association", "analogy", "blending")}}
    return out


def main():
    comp = json.load(open(f"{SCORES}/composite.json"))
    pm = comp["per_model"]
    models = [m for m in pm if not m.startswith("human")]
    res = {"composite_rank": comp["ranking"],
           "overall": {m: round(pm[m]["overall"], 2) for m in comp["ranking"]}, "gated": {}, "among_passed": {}}
    for t, dims in pm[models[0]]["raw"].items():
        for d, v in dims.items():
            if not d.startswith("_"):
                res["gated"][f"{t}.{d}"] = {g: compare(pm[g]["raw"][t][d], [pm[m]["raw"][t][d] for m in models])
                                            for g in ("human_kept", "human_flagged")}
    ap = {os.path.basename(os.path.dirname(f)): among_passed(json.load(open(f)))
          for f in glob.glob(f"{SCORES}/*/path_scores.json")}
    for key in ap["human_kept"]:
        res["among_passed"][key] = {g: compare(ap[g][key], [ap[m][key] for m in models if key in ap[m]])
                                    for g in ("human_kept", "human_flagged") if key in ap[g]}
    # human groups for the paper's table: all kept, and kept who passed the pre-registered Breakfast + Lunch screen
    res["human_rows"] = human_rows(pm, models)
    json.dump(res, open(OUT, "w"), indent=1)
    for sect in ("gated", "among_passed"):
        print(sect)
        for k, v in res[sect].items():
            h = v["human_kept"]
            print(f"  {k:28s} human {h['human']:.3f}  models median {h['model_median']:.3f}  "
                  f"humans above {h['models_below_human']}/{h['n_models']}")


if __name__ == "__main__":
    main()
