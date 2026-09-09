"""Is DAT's association with Kombine SPECIFIC, or just general capability?

DAT correlates ~+0.66 with the Kombine composite. That is only evidence about creativity if it
survives partialling out how good the model is in general -- otherwise "divergent thinking predicts
creative performance" reduces to "better models are better".

Reports, for each Kombine task/facet:
  r          zero-order correlation with DAT
  sr         SEMI-PARTIAL: Kombine vs the part of DAT that general capability does NOT explain
             sr = (r_YX - r_YZ r_XZ) / sqrt(1 - r_XZ^2)
  pr         partial (both sides residualised), for reference

Controls (Z) differ wildly in coverage, so each is reported with its own n. External capability
benchmarks are the right control but are thin on this pool; RAT is in-battery and better powered but
is itself one of the constructs under test, so it is the conservative bound rather than the clean one.

    .venv/bin/python -m src.kg_creat.scripts.analyze_dat_specificity
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, t as tdist

from src.kg_creat.scripts.analyze_cognitive_facets import (
    TASKS, canon, load_cdat, load_dat, load_drat, load_facets_ungated, load_kombine, load_rat)

BENCH = "configs/comb_eval/benchmarks.json"


def load_bench(name):
    d = json.loads(Path(BENCH).read_text())
    return {canon(m): v[name] for m, v in d.items() if isinstance(v, dict) and v.get(name) is not None}


def semipartial(y, x, z):
    """(sr, pr, n, p_sr). sr: y vs x with z removed from X ONLY."""
    n = len(y)
    if n < 6:
        return None
    ryx = pearsonr(y, x)[0]; ryz = pearsonr(y, z)[0]; rxz = pearsonr(x, z)[0]
    den_sr = math.sqrt(max(1 - rxz ** 2, 1e-12))
    sr = (ryx - ryz * rxz) / den_sr
    pr = (ryx - ryz * rxz) / (math.sqrt(max(1 - ryz ** 2, 1e-12)) * den_sr)
    df = n - 3
    tv = sr * math.sqrt(df / max(1 - sr ** 2, 1e-12))
    return sr, pr, n, ryx, float(2 * tdist.sf(abs(tv), df))


def main():
    komb = load_kombine(Path("data/kg_creat/kombine_test30/scores"))
    dat = load_dat()
    facets = load_facets_ungated(Path("data/kg_creat/kombine_test30/scores"))
    controls = {
        "MMLU-Pro": load_bench("mmlu_pro"),
        "Arena (overall)": load_bench("arena_overall"),
        "RAT (in-battery)": load_rat(),
    }

    series = {t: {m: v[t] for m, v in komb.items() if v.get(t) is not None} for t in TASKS}
    series["overall"] = {m: v["overall"] for m, v in komb.items() if v.get("overall") is not None}
    for k in sorted(facets, key=lambda k: (TASKS.index(k[0]), k[1])):
        series[f"{k[0][:5]}.{k[1]}"] = facets[k]

    for cname, ctrl in controls.items():
        print(f"\n{'=' * 78}\nCONTROL: {cname}   (DAT n={len(set(dat) & set(ctrl))} overlapping)\n{'=' * 78}")
        print(f"  {'outcome':26s}{'r(DAT)':>9}{'sr':>9}{'partial':>9}{'n':>5}{'p(sr)':>9}")
        print("  " + "-" * 67)
        for label, s in series.items():
            ms = sorted(set(s) & set(dat) & set(ctrl))
            res = semipartial(np.array([s[m] for m in ms]),
                              np.array([dat[m] for m in ms]),
                              np.array([ctrl[m] for m in ms])) if len(ms) >= 6 else None
            if not res:
                print(f"  {label:26s}{'--':>9}{'--':>9}{'--':>9}{len(ms):>5}"); continue
            sr, pr, n, ryx, p = res
            star = "*" if p < 0.05 else ""
            print(f"  {label:26s}{ryx:>+9.2f}{sr:>+9.2f}{pr:>+9.2f}{n:>5}{p:>8.3f}{star}")
        # how much of DAT is capability?
        ms = sorted(set(dat) & set(ctrl))
        if len(ms) >= 6:
            r = pearsonr([dat[m] for m in ms], [ctrl[m] for m in ms])[0]
            print(f"\n  DAT vs {cname}: r = {r:+.2f} (n={len(ms)}) -> "
                  f"{100 * r ** 2:.0f}% of DAT variance is shared with this control")


if __name__ == "__main__":
    main()
