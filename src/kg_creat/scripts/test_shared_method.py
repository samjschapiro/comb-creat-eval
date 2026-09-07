"""Is the association-originality result an artifact of shared measurement?

The concern: association originality and the downstream novelty facets are BOTH pool-relative
embedding distances over the same model's own text. A model with an unusual lexical register would
score high on both for reasons that have nothing to do with creative ability. That confound has two
separable parts, and they need different tests:

  (1) SHARED ENCODER   one encoder's geometry could manufacture the correlation. Tested by
                       recomputing under two held-out encoders, and -- the decisive version -- by
                       measuring the PREDICTOR with one encoder and the OUTCOME with a different one.
                       If a cross-encoder correlation survives, no single geometry produced it.

  (2) SHARED STYLE     a model that writes unusually writes unusually everywhere. Changing encoder
                       does NOT fix this; both encoders would flag the same model. Two things speak
                       to it instead: the headline cell's outcome is a JUDGE verdict, not an
                       embedding, and association SURPRISE -- same encoder, same text, same pipeline
                       -- predicts nothing. A blanket style artifact would move surprise too.

    .venv_mlx/bin/python -m src.kg_creat.scripts.test_shared_method
"""
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.embedding_robustness import ENCODERS, artifacts, score_with

OUT = Path("data/kg_creat/kombine_test30/analysis/shared_method.json")
SCORING_ENCODER = ENCODERS[0]      # the one everything was scored with


def per_model(recs):
    """Ungated per-model means, matching analyze_facet_correlations: every artifact counts, pass or
    fail. Utility is the judge pass rate and carries no encoder dependence."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in recs:
        acc[r["model"]][(r["task"], "utility")].append(bool(r["sat"]))
        for dim, key in (("surprise", "R"), ("originality", "originality")):
            if r.get(key) is not None:
                acc[r["model"]][(r["task"], dim)].append(r[key])
    return {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in acc.items()}


def corr(pm_x, pm_y, kx, ky, models):
    a = np.array([pm_x[m][kx] for m in models])
    b = np.array([pm_y[m][ky] for m in models])
    r, p = pearsonr(a, b)
    return float(r), float(p)


def main():
    base = artifacts()
    per_enc = {}
    for name in ENCODERS:
        recs = deepcopy(base)
        score_with(recs, get_embedder(name))
        per_enc[name] = per_model(recs)
        print(f"  scored under {name.split('/')[-1]}")
    models = sorted(set.intersection(*(set(p) for p in per_enc.values())))
    short = {n: n.split("/")[-1].replace("-4bit", "").replace("-mlx", "") for n in ENCODERS}
    res = {"n_models": len(models), "encoders": ENCODERS}
    print(f"\n{len(models)} models\n")

    A = ("association", "originality")
    print("1. THE HEADLINE CELL -- outcome is a JUDGE verdict, so the encoder only enters the predictor")
    res["headline"] = {}
    for out in (("blending", "utility"), ("analogy", "utility")):
        row = []
        for e in ENCODERS:
            r, p = corr(per_enc[e], per_enc[e], A, out, models)
            row.append(f"{short[e]} {r:+.2f} (p={p:.3f})")
            res["headline"][f"{short[e]}->{out[1]}_{out[0]}"] = {"r": r, "p": p}
        print(f"   assoc.originality -> {out[0]}.{out[1]:11s}  " + "   ".join(row))

    print("\n2. THE SHARED-METHOD CELL -- both sides embedding-derived. Cross-encoder is the real test:")
    print("   predictor measured with the ROW encoder, outcome with the COLUMN encoder")
    res["cross_encoder"] = {}
    for out in (("analogy", "originality"), ("blending", "surprise")):
        print(f"\n   assoc.originality -> {out[0]}.{out[1]}")
        print("      predictor \\ outcome   " + "  ".join(f"{short[e][:16]:>18s}" for e in ENCODERS))
        for ex in ENCODERS:
            cells = []
            for ey in ENCODERS:
                r, p = corr(per_enc[ex], per_enc[ey], A, out, models)
                cells.append(f"{r:+.2f}{'*' if p < 0.05 else ' '}")
                res["cross_encoder"][f"{short[ex]}|{short[ey]}|{out[0]}.{out[1]}"] = {"r": r, "p": p}
            print(f"      {short[ex][:20]:20s} " + "  ".join(f"{c:>18s}" for c in cells))
        off = [res["cross_encoder"][f"{short[a]}|{short[b]}|{out[0]}.{out[1]}"]["r"]
               for a in ENCODERS for b in ENCODERS if a != b]
        print(f"      off-diagonal (different encoders each side): {min(off):+.2f} to {max(off):+.2f}")

    print("\n3. THE INTERNAL CONTROL -- same encoder, same text, same pipeline, different dimension.")
    print("   If an unusual lexical register drove everything, SURPRISE would move too.")
    res["control"] = {}
    for src in (("association", "surprise"), ("association", "utility")):
        row = []
        for e in ENCODERS:
            r, p = corr(per_enc[e], per_enc[e], src, ("blending", "utility"), models)
            row.append(f"{short[e]} {r:+.2f} (p={p:.3f})")
            res["control"][f"{short[e]}|{src[0]}.{src[1]}"] = {"r": r, "p": p}
        print(f"   {src[0]}.{src[1]:11s} -> blending.utility  " + "   ".join(row))

    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
