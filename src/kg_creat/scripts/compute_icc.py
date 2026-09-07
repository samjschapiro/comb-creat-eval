"""Inter-judge reliability of the 3-judge panel: ICC(2,3) per subjective dimension.

The panel's per-judge verdicts are persisted on every scored artifact (``blend_judges``,
``invention_judges``), so reliability is recomputed from the same records the majority votes came
from -- it is never re-elicited. ICC(2,3) is the two-way random-effects, absolute-agreement
reliability of the MEAN of three raters (Shrout & Fleiss 1979), which is what a majority vote of
three approximates:

    ICC(2,k) = (MSR - MSE) / (MSR + (MSC - MSE) / n)

with n artifacts as targets and k = 3 judges as a random sample of raters. Artifacts where any judge
abstained are dropped, since ICC needs a complete targets x raters matrix.

    .venv/bin/python -m src.kg_creat.scripts.compute_icc
"""
import json
from pathlib import Path

import numpy as np

SCORES = Path("data/kg_creat/kombine_test30/scores")
OUT = Path("data/kg_creat/kombine_test30/analysis/judge_icc.json")

# (paper label, judge-list field, per-judge key, cast) -- the wire keys are deliberately not renamed
# (judge.py notes this), so the mapping to the paper's symbols lives here.
DIMS = [
    (r"Blend utility ($J^\text{gen}_\text{bl}$)",                    "blend_judges",     "generic_ok", float),
    (r"Analogy utility ($J^\text{utl}_\text{an}$)",                  "invention_judges", "coherent",   float),
    (r"Blend semantic integration quality ($J^\text{qua}_\text{bl}$)", "blend_judges",   "scope",      float),
    (r"Analogy integration quality ($J^\text{qua}_\text{an}$)",      "invention_judges", "valid",      float),
    (r"Blend utility ($J^\text{utl}_\text{bl}$)",                    "blend_judges",     "coherent",   float),
]


def icc_2k(M: np.ndarray) -> float:
    """ICC(2,k), two-way random effects, absolute agreement, average of k raters."""
    n, k = M.shape
    gm = M.mean()
    MSR = k * ((M.mean(axis=1) - gm) ** 2).sum() / (n - 1)          # between targets
    MSC = n * ((M.mean(axis=0) - gm) ** 2).sum() / (k - 1)          # between judges
    resid = M - M.mean(axis=1, keepdims=True) - M.mean(axis=0, keepdims=True) + gm
    MSE = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = MSR + (MSC - MSE) / n
    return float((MSR - MSE) / denom) if denom else float("nan")


def band(v: float) -> str:
    return "poor" if v < 0.40 else "fair" if v < 0.60 else "good" if v < 0.75 else "excellent"


def collect(field: str, key: str):
    rows, judges = [], None
    for f in sorted(SCORES.glob("*/path_scores.json")):
        for r in json.loads(f.read_text()):
            js = r.get(field)
            if not js:
                continue
            d = {j["model"]: j.get(key) for j in js if isinstance(j, dict) and j.get("model")}
            if judges is None:
                judges = sorted(d)
            if len(d) != len(judges) or any(d.get(m) is None for m in judges):
                continue                                     # ICC needs a complete row
            rows.append([float(d[m]) for m in judges])
    return np.array(rows, float), judges


def main():
    out, lines = {}, []
    print(f"{'dimension':52s} {'n':>6s} {'ICC(2,3)':>9s}  band")
    for label, field, key, _ in DIMS:
        M, judges = collect(field, key)
        if len(M) < 3:
            print(f"  {label:50s} insufficient data"); continue
        v = icc_2k(M)
        out[label] = {"icc_2_3": round(v, 3), "n_artifacts": int(len(M)),
                      "judges": judges, "band": band(v)}
        print(f"{label:52s} {len(M):6d} {v:9.3f}  {band(v)}")
        lines.append((label, v, band(v)))
    vals = [v for _, v, _ in lines]
    print(f"\nrange {min(vals):.2f}-{max(vals):.2f}; judges = {judges}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")
    print("\n--- LaTeX rows, highest first ---")
    for label, v, b in sorted(lines, key=lambda t: -t[1]):
        print(f"{label} & {v:.2f} & {b} \\\\")


if __name__ == "__main__":
    main()
