"""Paired within-endpoint analysis: the controlled comparison the design was built for.

Every constrained cell was administered on the SAME 30 endpoint bundles as the unconstrained
baseline, so creativity under a constraint can be differenced against the baseline *for the same
endpoints and the same model* rather than compared across marginal means. That pairing is what
removes endpoint difficulty as a confound: some entity pairs are simply richer than others, and a
marginal comparison lets that variance leak into the constraint effect.

Unit of analysis = (model, bundle), giving 8 x 30 = 240 paired observations per constraint type.

Also runs the matched-difficulty contrast: rare-inclusion vs ordering rule out nearly the same
share of default behaviour, so differencing THOSE two on shared endpoints isolates constraint type
from restrictiveness.

    .venv_mlx/bin/python src/kg_creat/scripts/paired_analysis.py data/kg_creat/scores_regimeA_all
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

MODES = ["categorical", "exclusion", "inclusion_rare", "inclusion", "ordering"]
LABEL = {"categorical": "Categorical", "exclusion": "Exclusion", "inclusion": "Inclusion (common)",
         "inclusion_rare": "Inclusion (rare)", "ordering": "Ordering"}
SHORT = {"categorical": "categ", "exclusion": "excl", "inclusion": "incl", 
         "inclusion_rare": "incl_rare", "ordering": "order"}


def cell_creativity(recs):
    """E[R·U] over one cell's paths (utility is judge-gated, so failures contribute zero)."""
    vals = [(r["R"] if r["sat"] else 0.0) for r in recs
            if r.get("sat") is not None and r.get("R") is not None
            and not (isinstance(r["R"], float) and math.isnan(r["R"]))]
    return sum(vals) / len(vals) if vals else None


def collect(scores_dir):
    """-> {mode: {(model, bundle): creativity}} for every cell including baseline."""
    out = defaultdict(dict)
    for md in sorted(Path(scores_dir).glob("*/path_scores.json")):
        model = md.parent.name
        by = defaultdict(list)
        for r in json.loads(md.read_text()):
            if r["regime"] == "A":
                by[(r["mode"], r["bundle_id"])].append(r)
        for (mode, bundle), recs in by.items():
            c = cell_creativity(recs)
            if c is not None:
                out[mode][(model, bundle)] = c
    return out


def paired(a: dict, b: dict):
    """Differences a-b over keys present in both (the matched endpoints)."""
    keys = sorted(set(a) & set(b))
    return [a[k] - b[k] for k in keys]


def wilcoxon(d):
    """Two-sided Wilcoxon signed-rank p, or None if scipy is unavailable."""
    try:
        from scipy.stats import wilcoxon as w
    except Exception:  # noqa: BLE001
        return None
    nz = [x for x in d if x != 0]
    if len(nz) < 10:
        return None
    return float(w(nz).pvalue)


def summarize(d, label):
    n = len(d)
    mean = sum(d) / n
    sd = (sum((x - mean) ** 2 for x in d) / (n - 1)) ** 0.5 if n > 1 else 0.0
    se = sd / math.sqrt(n)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    frac_down = sum(1 for x in d if x < 0) / n
    p = wilcoxon(d)
    ptxt = "n/a" if p is None else (f"{p:.2e}" if p < 1e-4 else f"{p:.4f}")
    print(f"  {label:22s} n={n:4d}  Δ={mean:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"decreased in {100*frac_down:5.1f}% of pairs   p={ptxt}")
    return mean


def main(scores_dir):
    data = collect(scores_dir)
    base = data["baseline"]

    print("PAIRED Δ CREATIVITY vs the SAME endpoints, unconstrained")
    print("(unit = one model on one endpoint bundle; both cells share the identical (u,v))\n")
    for m in MODES:
        summarize(paired(data[m], base), LABEL[m])

    print("\nMATCHED-DIFFICULTY CONTRAST (same endpoints, both constraints rule out ~99% of")
    print("default behaviour — so any difference is constraint TYPE, not restrictiveness)\n")
    summarize(paired(data["ordering"], data["inclusion_rare"]), "ordering − rare-incl")

    print("\nPER-MODEL paired Δ creativity vs baseline\n")
    models = sorted({k[0] for k in base})
    print(f"  {'model':34s}" + "".join(f"{SHORT[m]:>11s}" for m in MODES))
    for mdl in models:
        row = []
        for m in MODES:
            a = {k: v for k, v in data[m].items() if k[0] == mdl}
            b = {k: v for k, v in base.items() if k[0] == mdl}
            d = paired(a, b)
            row.append(sum(d) / len(d) if d else float("nan"))
        print(f"  {mdl:34s}" + "".join(f"{x:>+11.4f}" for x in row))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_regimeA_all")
