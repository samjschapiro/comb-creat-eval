"""Kombine as a fourth creative-achievement construct for the semantic distance tests paper.

Treats the Kombine benchmark (association / analogy / blending) as a benchmark, the way the paper
treats Arena CW or LiveIdeaBench, and scores DAT, CDAT, PACE and RAT on it:

  validity     r(test, Kombine)
  specificity  r(test, Kombine residualized on MMLU-Pro)   -- MMLU-Pro ONLY; Arena Overall is no
               longer a control (common-method bias with Arena CW, rebuttal to F63x).

Also asks whether Kombine is a distinct construct: its correlation with the six existing benchmarks
and with MMLU-Pro.

Test scores (reused from src/kg_creat/scripts/analyze_cognitive_facets.py, which fixed three bugs):
  DAT, CDAT  at temperature 1.0 (closest to Kombine's 0.9), >=70% of trials parsed; CDAT is novelty
             at 1.0, kept only if the appropriateness gate passed at 1.0.
  RAT        zero-shot strict accuracy (temp 0). gpt-5.6-sol is dropped: its 43% is the 32-token-cap
             artifact (13/13 correct on the items it answered, the rest empty).
  PACE       model_score from pace_scores.json (temp 0).
These are NOT the paper's Table 1 scores, which average temperatures and z-score three embeddings.

MMLU-Pro: configs/comb_eval/benchmarks.json, plus five Kombine models matched by name in the
TIGER-Lab leaderboard (data/dat_eval/mmlu_pro_raw.json). Thinking-only variants are not matched.

    .venv/bin/python -m src.dat_eval.scripts.kombine_as_benchmark
"""
import glob
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from src.kg_creat.scripts.analyze_cognitive_facets import (
    canon, load_cdat, load_dat, load_facets_ungated, load_kombine, load_rat)

SCORES = Path("data/kg_creat/kombine_test30/scores")
OUT = Path("data/dat_eval/kombine_benchmark/results.json")
BENCH = json.load(open("configs/comb_eval/benchmarks.json"))

# Kombine model -> exact TIGER-Lab MMLU-Pro row, for models missing from benchmarks.json.
MMLU_RAW_MATCH = {
    "moonshotai_kimi-k2": "Kimi-K2-Instruct",
    "z-ai_glm-4-5-air": "GLM-4.5-Air",
    "google_gemini-3-1-pro-preview": "Gemini-3.1-Pro",
    "google_gemini-3-flash-preview": "Gemini-3-Flash(12/25)",
    "openai_gpt-5-2": "GPT-5.2",
}
RAT_ARTIFACTS = {"openai_gpt-5-6-sol"}

EXISTING = {  # the paper's six benchmarks, by construct
    "Arena CW": "arena_cw", "EQ-Bench CW": "eq_bench_cw", "Mazur CW": "mazur_cw_v2",
    "Hivemind": "hivemind_diversity", "NoveltyBench": "noveltybench_utility",
    "LiveIdeaBench": "liveideabench", "MMLU-Pro": "mmlu_pro",
}
# paper terms for the Kombine criteria (code key -> paper name)
FACET_NAMES = {"utility": "factual", "surprise": "surprise", "originality": "originality",
               "em_originality": "emergent originality"}


def ok(v):
    return v not in (None, "", "---") and not (isinstance(v, float) and math.isnan(v))


def load_mmlu() -> dict:
    out = {canon(m): float(b["mmlu_pro"]) for m, b in BENCH.items() if ok(b.get("mmlu_pro"))}
    raw = json.load(open("data/dat_eval/mmlu_pro_raw.json"))
    for m, row in MMLU_RAW_MATCH.items():
        out.setdefault(m, float(raw[row]))
    return out


def load_pace() -> dict:
    vals = {}
    for p in glob.glob("data/dat_eval/*/downstream/*/results/*/pace_scores.json"):
        v = json.loads(Path(p).read_text()).get("model_score")
        if ok(v):
            vals.setdefault(canon(Path(p).parent.name), []).append(float(v))
    return {m: float(np.mean(v)) for m, v in vals.items()}


def p_of(r, n, k):
    df = n - 2 - k
    if df <= 0 or not -1 < r < 1:
        return float("nan")
    return float(2 * stats.t.sf(abs(r * math.sqrt(df / (1 - r * r))), df))


def min_detectable_r(n, k, alpha=0.05, power=0.80):
    """Smallest |r| the two-sided t-test detects with the given power (Fisher-z approximation)."""
    if n - 3 - k <= 0:
        return float("nan")
    z = (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / math.sqrt(n - 3 - k)
    return float(np.tanh(z))


def score(test: dict, target: dict, mmlu: dict) -> dict:
    v_ms = sorted(set(test) & set(target))
    s_ms = sorted(set(v_ms) & set(mmlu))
    out = {}
    if len(v_ms) >= 5:
        x = np.array([test[m] for m in v_ms]); y = np.array([target[m] for m in v_ms])
        r = float(stats.pearsonr(x, y).statistic)
        out["validity"] = {"r": r, "p": p_of(r, len(v_ms), 0), "n": len(v_ms),
                           "min_detectable_r": min_detectable_r(len(v_ms), 0)}
    if len(s_ms) >= 5:
        x = np.array([test[m] for m in s_ms]); y = np.array([target[m] for m in s_ms])
        g = np.array([mmlu[m] for m in s_ms])
        X = np.column_stack([np.ones(len(g)), g])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        r = float(stats.pearsonr(x, resid).statistic)
        out["specificity"] = {"r": r, "p": p_of(r, len(s_ms), 1), "n": len(s_ms),
                              "min_detectable_r": min_detectable_r(len(s_ms), 1)}
    return out


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""


def fmt(c):
    return f"{c['r']:+.2f}{star(c['p']):<3s} n={c['n']:<2d}" if c else f"{'--':^14s}"


def main():
    komb = load_kombine(SCORES)
    mmlu = load_mmlu()
    rat = {m: v for m, v in load_rat().items() if m not in RAT_ARTIFACTS}
    tests = {"DAT": load_dat(), "CDAT": load_cdat(), "PACE": load_pace(), "RAT": rat}
    targets = {f"Kombine {t}": {m: v[t] for m, v in komb.items() if ok(v.get(t))}
               for t in ("overall", "association", "analogy", "blending")}
    facets = load_facets_ungated(SCORES)
    facet_targets = {f"{t[:5]}. {FACET_NAMES[k]}": d for (t, k), d in facets.items() if k in FACET_NAMES}
    bench = {name: {canon(m): float(b[f]) for m, b in BENCH.items() if ok(b.get(f))}
             for name, f in EXISTING.items()}

    res = {"n_kombine_models": len(komb),
           "coverage": {n: len(set(komb) & set(d)) for n, d in {**tests, "MMLU-Pro": mmlu}.items()},
           "coverage_with_mmlu": {n: len(set(komb) & set(d) & set(mmlu)) for n, d in tests.items()},
           "tests_vs_kombine": {}, "tests_vs_facets": {}, "kombine_vs_benchmarks": {}}

    print(f"Kombine models: {len(komb)};  with MMLU-Pro: {len(set(komb) & set(mmlu))}")
    for n in tests:
        print(f"  {n:5s} coverage {res['coverage'][n]:2d}   with MMLU-Pro {res['coverage_with_mmlu'][n]:2d}")

    print("\nTESTS vs KOMBINE   validity r | specificity r|MMLU-Pro   (*p<.05 **p<.01 ***p<.001)")
    print(f"  {'':22s}" + "".join(f"{n:>32s}" for n in tests))
    for tname, tgt in targets.items():
        row = f"  {tname:22s}"
        for n, d in tests.items():
            c = score(d, tgt, mmlu)
            res["tests_vs_kombine"].setdefault(tname, {})[n] = c
            row += f"  {fmt(c.get('validity'))} | {fmt(c.get('specificity'))}"
        print(row)

    print("\nTESTS vs KOMBINE FACETS (ungated)   validity | specificity  -- exploratory, BH below")
    all_p = []
    for fname in sorted(facet_targets):
        row = f"  {fname:28s}"
        for n, d in tests.items():
            c = score(d, facet_targets[fname], mmlu)
            res["tests_vs_facets"].setdefault(fname, {})[n] = c
            for kind in ("validity", "specificity"):
                if kind in c:
                    all_p.append(c[kind]["p"])
            row += f"  {fmt(c.get('validity'))} | {fmt(c.get('specificity'))}"
        print(row)
    ps = sorted(p for p in all_p if not math.isnan(p))
    thresh = max([p for i, p in enumerate(ps, 1) if p <= 0.05 * i / len(ps)], default=0.0)
    res["facet_bh"] = {"n_tests": len(ps), "threshold_p": thresh,
                       "n_surviving": sum(p <= thresh for p in ps) if thresh else 0}
    print(f"  BH q=0.05 over {len(ps)} facet tests: {res['facet_bh']['n_surviving']} survive "
          f"(threshold p <= {thresh:.4f})")

    print("\nIS KOMBINE A DISTINCT CONSTRUCT?  r(Kombine, existing benchmark)")
    for tname in ("Kombine overall", "Kombine association", "Kombine analogy", "Kombine blending"):
        row = f"  {tname:22s}"
        for bname, d in bench.items():
            ms = sorted(set(targets[tname]) & set(d))
            if len(ms) < 5:
                row += f"  {bname}: --"
                continue
            r = float(stats.pearsonr([targets[tname][m] for m in ms], [d[m] for m in ms]).statistic)
            p = p_of(r, len(ms), 0)
            res["kombine_vs_benchmarks"].setdefault(tname, {})[bname] = {"r": r, "p": p, "n": len(ms)}
            row += f"  {bname}: {r:+.2f}{star(p)} (n={len(ms)})"
        print(row)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
