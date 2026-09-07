"""Flat hierarchy or steep-but-deviant? Mednick's own rival explanation, tested.

Mednick (1962, pp. 222-223) predicts creativity from a FLAT associative hierarchy: past the one or
two conventional responses, associative strength decays slowly, so the person keeps producing less
probable, more remote -- but still apt -- associations. He raises the rival himself: a STEEP-BUT-
DEVIANT hierarchy, one strong unusual response and nothing behind it. Flat is a "multi-producer",
steep-deviant a "one-shot producer".

Pool-relative originality alone cannot separate them: a model scores high either way. Two things can.

  BREADTH        a multi-producer ranges over many distinct associative elements; a one-shot producer
                 reuses a narrow, odd repertoire. Measured as the type-token ratio of a model's
                 association elements across items, and the spread of its own element cloud.
  APTNESS        Mednick's remote elements are the ones a creative solution is built from, so they
                 are apt, not merely odd -- "it is only when conditions are such that this answer is
                 useful that we can also call it creative" (p. 221). So originality computed over the
                 model's VALID association paths should carry the predictive signal, and originality
                 over its INVALID paths should not. If both predict equally, the measure is picking
                 up deviance rather than flatness.

    .venv_mlx/bin/python -m src.kg_creat.scripts.test_flat_vs_deviant
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

import src.kg_creat.scripts.analyze_facet_correlations as F

SCORES = "data/kg_creat/kombine_test30/scores"
OUT = Path("data/kg_creat/kombine_test30/analysis/flat_vs_deviant.json")


def association_records(model):
    """Per-artifact association records: originality, validity, and the path's elements."""
    out = []
    for r in json.load(open(f"{SCORES}/{model}/path_scores.json")):
        if r.get("mode") != "baseline" or not r.get("triples"):
            continue
        els = []
        for tp in r["triples"]:
            if len(tp) == 3:
                els += [str(tp[0]).lower().strip(), str(tp[1]).lower().strip(),
                        str(tp[2]).lower().strip()]
        out.append({"orig": r.get("originality"), "sat": r.get("sat") is True,
                    "els": els, "item": (r.get("u_label"), r.get("v_label"))})
    return out


def main():
    models = sorted({f.split("/")[-2] for f in glob.glob(f"{SCORES}/*/path_scores.json")})
    V = {m: F.ungated_dims(m) for m in models}
    recs = {m: association_records(m) for m in models}

    rows = {}
    for m in models:
        rs = recs[m]
        anchors = {x.lower() for r_ in rs for x in r_["item"] if x}
        toks = [e for r_ in rs for e in r_["els"] if e and e not in anchors]
        ok = [r_["orig"] for r_ in rs if r_["sat"] and r_["orig"] is not None]
        bad = [r_["orig"] for r_ in rs if not r_["sat"] and r_["orig"] is not None]
        rows[m] = {
            "ttr": len(set(toks)) / len(toks) if toks else np.nan,   # breadth of the repertoire
            "n_tokens": len(toks), "n_types": len(set(toks)),
            "orig_valid": float(np.mean(ok)) if ok else np.nan,
            "orig_invalid": float(np.mean(bad)) if bad else np.nan,
            "n_valid": len(ok), "n_invalid": len(bad),
            "orig_all": V[m]["association"]["originality"],
            "assoc_util": V[m]["association"]["utility"],
            "blend_util": V[m]["blending"]["utility"],
            "analogy_util": V[m]["analogy"]["utility"]}

    def col(k, keep=None):
        ms = [m for m in models if not np.isnan(rows[m][k])] if keep is None else keep
        return ms, np.array([rows[m][k] for m in ms])

    def report(k, label, outcome="blend_util"):
        ms, x = col(k)
        y = np.array([rows[m][outcome] for m in ms])
        r, p = pearsonr(x, y); rho, rp = spearmanr(x, y)
        print(f"  {label:44s} r={r:+.2f} (p={p:.4f})  rho={rho:+.2f}  n={len(ms)}")
        return {"r": float(r), "p": float(p), "spearman": float(rho), "n": len(ms)}

    res = {"n_models": len(models), "per_model": rows}
    print(f"{len(models)} models\n")
    print("APTNESS TEST -- does the signal live in originality over VALID or INVALID paths?")
    print("  (outcome: blending utility)")
    res["orig_valid"] = report("orig_valid", "originality over VALID association paths")
    res["orig_invalid"] = report("orig_invalid", "originality over INVALID association paths")
    res["orig_all"] = report("orig_all", "originality over all paths (the reported measure)")

    print("\nBREADTH TEST -- is the model a multi-producer or a one-shot producer?")
    res["ttr"] = report("ttr", "type-token ratio of association elements")
    ms, t = col("ttr")
    o = np.array([rows[m]["orig_all"] for m in ms])
    r, p = pearsonr(t, o)
    res["ttr_vs_originality"] = {"r": float(r), "p": float(p)}
    print(f"  {'breadth vs originality (are they the same thing?)':44s} r={r:+.2f} (p={p:.4f})")

    print("\nDO BREADTH AND ORIGINALITY BOTH MATTER? (blending utility on both, standardised)")
    y = np.array([rows[m]["blend_util"] for m in ms])
    X = np.c_[np.ones(len(ms)), (t - t.mean()) / t.std(), (o - o.mean()) / o.std()]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  beta(breadth) = {beta[1]:+.3f}   beta(originality) = {beta[2]:+.3f}   R^2 = {ss:.2f}")
    res["joint_model"] = {"beta_breadth": float(beta[1]), "beta_originality": float(beta[2]),
                          "r2": float(ss)}

    print("\nMOST / LEAST BROAD REPERTOIRES")
    order = sorted(ms, key=lambda m: -rows[m]["ttr"])
    for m in order[:4] + order[-4:]:
        d = rows[m]
        print(f"    {m:36s} ttr={d['ttr']:.3f} ({d['n_types']}/{d['n_tokens']})  "
              f"orig={d['orig_all']:.3f}  blend_util={d['blend_util']:.2f}")
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
