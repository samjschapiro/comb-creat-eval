"""The corpus-wide numbers behind the "How Blends Fail" memo.

That memo was written from one-off scans, so it silently went stale when the pool grew. Everything it
quotes corpus-wide is computed here instead: how many blends claim a shared `uv` slot, how many survive
the panel's scope verification, the two failure themes in the judges' free text, the per-model genuine-
fusion rate, and the per-pair spread.

The two failure themes are keyword matches over the judges' explanations, not an independent labelling
pass, and they overlap -- reported that way here and in the memo.

    .venv/bin/python -m src.kg_creat.scripts.analyze_blend_integration
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.scripts.analyze_failure_modes import SCORES
from src.kg_creat.scripts.plot_radar import DISPLAY

RESP = Path("data/kg_creat/kombine_test30/responses")
OUT = Path("data/kg_creat/kombine_test30/analysis/blend_integration.json")
ONE_SIDED = re.compile(r"one[- ]sided|unbalanced|only one input|not.{0,20}instantiat|asymmetric", re.I)
ABSURD = re.compile(r"absurd|incoherent|nonsens|literal|category error|not a coherent", re.I)


def main():
    uv = {}
    for f in sorted(RESP.glob("*/responses.json")):
        for r in json.loads(f.read_text()):
            if r.get("mode") == "blending" and r.get("items"):
                tags = r["items"][0].get("tags") or []
                uv[(f.parent.name, r["u_label"], r["v_label"])] = sum(t == "uv" for t in tags)

    scope, per_model, per_pair, reasons = [], defaultdict(list), defaultdict(list), []
    for f in sorted(SCORES.glob("*/path_scores.json")):
        m = f.parent.name
        for r in json.loads(f.read_text()):
            if r.get("mode") != "blending" or r.get("blend_integration") is None:
                continue
            sc = int(r["blend_integration"])
            scope.append(sc)
            per_model[m].append(sc >= 2)
            per_pair[(r["u_label"], r["v_label"])].append(sc >= 2)
            if sc == 1:
                reasons.append(" ".join((j.get("explanation") or "") for j in (r.get("blend_judges") or [])
                                        if isinstance(j, dict)))
    n = len(scope)
    c = Counter(scope)
    n_uv = sum(1 for v in uv.values() if v)
    print(f"BLENDS SCORED: {n}   (blends emitting at least one `uv` triple: {n_uv} of {len(uv)})")
    print(f"  scope 3 {c[3]} ({100*c[3]/n:.0f}%)   scope 2 {c[2]} ({100*c[2]/n:.0f}%)   "
          f"scope 1 {c[1]} ({100*c[1]/n:.0f}%)")
    print(f"  genuine fusion (scope >= 2): {c[2]+c[3]} ({100*(c[2]+c[3])/n:.0f}%)   "
          f"faked slot (scope 1): {c[1]} ({100*c[1]/n:.0f}%)")

    n1 = len(reasons)
    one = sum(bool(ONE_SIDED.search(t)) for t in reasons)
    ab = sum(bool(ABSURD.search(t)) for t in reasons)
    print(f"\nFAILURE THEMES across the {n1} scope-1 blends (keyword themes, overlapping)")
    print(f"  one-sided / unbalanced schema  {one} ({100*one/n1:.0f}%)")
    print(f"  categorical absurdity          {ab} ({100*ab/n1:.0f}%)")

    pm = {m: float(np.mean(v)) for m, v in per_model.items()}
    order = sorted(pm, key=lambda m: -pm[m])
    print(f"\nGENUINE-FUSION RATE PER MODEL (scope >= 2), {len(pm)} models")
    print("  best:  " + ", ".join(f"{DISPLAY.get(m, m)} {100*pm[m]:.0f}%" for m in order[:5]))
    print("  worst: " + ", ".join(f"{DISPLAY.get(m, m)} {100*pm[m]:.0f}%" for m in order[-3:]))
    print(f"  spread {100*min(pm.values()):.0f}%-{100*max(pm.values()):.0f}%")

    pp = {k: (int(sum(v)), len(v)) for k, v in per_pair.items()}
    best = sorted(pp, key=lambda k: -(pp[k][0] / pp[k][1]))
    print("\nEASIEST AND HARDEST ANCHOR PAIRS (models achieving genuine fusion)")
    for k in best[:3] + best[-3:]:
        a, b = pp[k]
        print(f"  {k[0] + ' + ' + k[1]:44s} {a}/{b}")
    OUT.write_text(json.dumps(
        {"n_blends": n, "n_emitting_uv": n_uv, "scope_counts": {str(k): v for k, v in c.items()},
         "genuine_pct": round(100 * (c[2] + c[3]) / n, 1), "scope1_pct": round(100 * c[1] / n, 1),
         "themes": {"n_scope1": n1, "one_sided": one, "one_sided_pct": round(100 * one / n1, 1),
                    "absurdity": ab, "absurdity_pct": round(100 * ab / n1, 1)},
         "per_model_genuine_pct": {m: round(100 * v, 1) for m, v in pm.items()},
         "per_pair_genuine": {f"{a} + {b}": pp[(a, b)] for a, b in pp}}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
