"""Thinking-effort study: does more reasoning effort help combinatorial creativity?

Two subjects (gpt-5.6-sol, gpt-6-astra-flex) x three effort levels (low/medium/high),
scored in one pooled pass so that pool-relative originality is comparable across levels.

Reports, per (model, effort), the utility and originality of each task:
  association  U = sat (well-formed AND factual)
  analogy      U = pair_sat (pair-level: structural match AND factual), deduped per pair
  blending     U = generic_ok (the panel's generic-space gate), plus scope-3 rate

`unjudged` is printed as a data-integrity check: it must be 0 after the haiku
re-judge. A nonzero rate that varies with effort silently manufactures an
"effort hurts" result, which is exactly the bug this study hit the first time.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

SCORES = Path(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/effort_study/scores")
EFFORTS = ("low", "medium", "high")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def load(d):
    return json.loads((d / "path_scores.json").read_text())


def pair_units(recs):
    """Analogy is scored per pair; collapse the duplicated per-path rows."""
    seen = {}
    for r in recs:
        key = (r["prompt_id"], r["sample_idx"], r.get("pair_idx"))
        seen.setdefault(key, r)
    return list(seen.values())


def main():
    rows = defaultdict(dict)
    for d in sorted(SCORES.iterdir()):
        if not (d / "path_scores.json").exists():
            continue
        model, effort = d.name.rsplit("__", 1)
        recs = load(d)
        by_mode = defaultdict(list)
        for r in recs:
            by_mode[r["mode"]].append(r)

        assoc = by_mode["baseline"]
        anal = pair_units(by_mode["analogy"])
        blend = by_mode["blending"]

        rows[model][effort] = {
            "assoc_n": len(assoc),
            "assoc_unjudged": mean([r.get("channel") == "unjudged" for r in assoc]),
            "assoc_wf": mean([bool(r.get("well_formed")) for r in assoc]),
            "assoc_U": mean([bool(r.get("sat")) for r in assoc]),
            "assoc_orig": mean([r.get("originality") for r in assoc]),
            "assoc_orig_gated": mean([r.get("originality") for r in assoc if r.get("sat")]),
            "anal_n": len(anal),
            "anal_U": mean([bool(r.get("pair_sat")) for r in anal]),
            "anal_orig": mean([r.get("originality") for r in anal]),
            "blend_n": len(blend),
            "blend_U": mean([bool(r.get("generic_ok")) for r in blend]),
            "blend_scope3": mean([r.get("blend_integration") == 3 for r in blend]),
            "blend_orig": mean([r.get("originality") for r in blend]),
            "blend_emergent": mean([r.get("emergent_count") or 0 for r in blend]),
        }

    for model, per_effort in rows.items():
        print(f"\n{'=' * 78}\n{model}\n{'=' * 78}")
        hdr = f"{'metric':<26}" + "".join(f"{e:>16}" for e in EFFORTS)
        print(hdr)
        print("-" * len(hdr))
        spec = [
            ("association  n", "assoc_n", "{:.0f}"),
            ("  UNJUDGED (must be 0)", "assoc_unjudged", "{:.1%}"),
            ("  well-formed", "assoc_wf", "{:.1%}"),
            ("  utility", "assoc_U", "{:.1%}"),
            ("  originality", "assoc_orig", "{:.3f}"),
            ("  originality (gated)", "assoc_orig_gated", "{:.3f}"),
            ("analogy  n (pairs)", "anal_n", "{:.0f}"),
            ("  utility", "anal_U", "{:.1%}"),
            ("  originality", "anal_orig", "{:.3f}"),
            ("blending  n", "blend_n", "{:.0f}"),
            ("  utility (generic_ok)", "blend_U", "{:.1%}"),
            ("  scope-3 rate", "blend_scope3", "{:.1%}"),
            ("  originality", "blend_orig", "{:.3f}"),
            ("  emergent/blend", "blend_emergent", "{:.2f}"),
        ]
        for label, key, fmt in spec:
            cells = ""
            for e in EFFORTS:
                v = per_effort.get(e, {}).get(key)
                cells += f"{(fmt.format(v) if v is not None else '--'):>16}"
            print(f"{label:<26}{cells}")


if __name__ == "__main__":
    main()
