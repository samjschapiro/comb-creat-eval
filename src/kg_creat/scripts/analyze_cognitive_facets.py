"""Which cognitive facet does each Kombine task load on?

Correlates each Kombine task score (association / analogy / blending) against three reference tests
administered to the same models:

  DAT   divergent thinking      mean pairwise semantic distance of 10 deliberately unrelated nouns
  CDAT  constrained divergent    same, but the nouns must also fit a given cue -- divergence under a
                                 relevance constraint, so it is the appropriateness half of DAT
  RAT   convergent thinking      zero-shot accuracy on 30 Remote Associates items
  DRAT  relevance-gated diverg.  10 nouns, each kept if its similarity to its NEAREST anchor clears a
                                 per-group tau (the 90th percentile of a random-noun null under the
                                 same max rule); the score is then the mean pairwise distance among
                                 survivors. NOT "applies to all anchors" -- the gate is max over
                                 anchors, so a word need only be near ONE. tau therefore rises with
                                 how semantically broad the anchor set is, and across 194 groups it
                                 correlates -0.73 with the survivor count: item breadth, not model
                                 ability, drives most of the variance. Read this column with care.

Correlations are reported per FACET, not only per task composite: a task composite averages utility,
surprise and originality (plus the emergent dimensions), and those pull in different directions, so a
task-level r can hide the facet that is actually loading.

Model-level correlation over the models that have both. n is small, so this reports CIs and treats
the result as exploratory: with ~20 models r_crit is about 0.44 at p<0.05, and the three reference
tests are themselves correlated, so the loadings are not independent tests.

    .venv/bin/python -m src.kg_creat.scripts.analyze_cognitive_facets
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from src.kg_creat.model_names import DISPLAY
from scipy.stats import pearsonr, spearmanr

TASKS = ["association", "analogy", "blending"]


def canon(m: str) -> str:
    m = str(m).replace("/", "_").replace(".", "-").lower()
    if m.startswith("claude-"):
        m = "anthropic_" + m
    elif m.startswith("gpt-"):
        m = "openai_" + m
    return m


def load_kombine(scores_dir: Path) -> dict:
    comp = json.loads((scores_dir / "composite.json").read_text())
    out = {}
    for m, v in comp["per_model"].items():
        out[canon(m)] = {**{t: v["per_task"].get(t) for t in TASKS}, "overall": v["overall"]}
    return out


MIN_VALID = 0.70  # a model whose responses mostly failed to parse has a score computed from a
                  # minority of its trials; several OpenRouter providers dropped or nulled 30-90% of
                  # calls, and those scores are not comparable to a model with 40/40 clean draws.
TEMP = "1.0"   # DAT/CDAT are collected at 1.0/1.5/2.0; Kombine runs at 0.9, so the sweep's high end
               # is far outside the operating point being correlated. Fix both at 1.0, the closest
               # available and the only temperature every gate-passing model shares.


def load_dat() -> dict:
    out = {}
    for p in glob.glob("data/dat_eval/*/downstream/*/results/*/dat_scores.json"):
        d = json.loads(Path(p).read_text())
        t = (d.get("by_temperature") or {}).get(TEMP, {})
        v, n, ns = t.get("mean_score"), t.get("n_trials") or 0, t.get("n_sufficient") or 0
        if v is None or (isinstance(v, float) and math.isnan(v)) or not n or ns / n < MIN_VALID:
            continue
        out[canon(Path(p).parent.name)] = v
    return out


def load_cdat(variant: str = "sbert") -> dict:
    """CDAT novelty at TEMP, gated at TEMP.

    The stored `cdat` averages over whichever temperatures a model passed, which differs by model
    (49 passed all three, 6 passed only 1.0, 11 none) -- so models were not compared at a common
    operating point, and weaker models were scored at the easier low temperatures. Here the score is
    the novelty at one fixed temperature, kept only if the appropriateness gate passed AT that same
    temperature.
    """
    passed = set()
    for p in glob.glob("data/dat_eval/*/cdat_gated_scores.json"):
        for m, v in json.loads(Path(p).read_text()).get(variant, {}).items():
            if isinstance(v, dict) and (v.get("gate") or {}).get(TEMP, {}).get("passed"):
                passed.add(canon(m))
    out = {}
    for p in glob.glob("data/dat_eval/*/downstream/*/results/*/cdat_scores.json"):
        m = canon(Path(p).parent.name)
        if m not in passed:
            continue
        t = (json.loads(Path(p).read_text()).get("by_temperature") or {}).get(TEMP, {})
        v, n, ns = t.get("mean_novelty"), t.get("n_cues") or 0, t.get("n_sufficient") or 0
        if v is None or (isinstance(v, float) and math.isnan(v)) or not n or ns / n < MIN_VALID:
            continue
        out[m] = v
    return out


def load_rat() -> dict:
    out = {}
    for p in glob.glob("data/new_tests/rat/*/summary.json"):
        for m, v in json.loads(Path(p).read_text()).items():
            if isinstance(v, dict) and v.get("zs_accuracy_strict") is not None and v.get("n_errors", 0) < 15:
                out.setdefault(canon(m), []).append(v["zs_accuracy_strict"])
    return {m: float(np.mean(v)) for m, v in out.items()}


def load_drat() -> dict:
    """Mean DRAT including zeros -- a zero means fewer than n_min words survived the utility
    threshold, which is a genuine low score, not a parse failure. Models whose responses did not
    parse at all (mean words < 5) are dropped instead of being scored 0."""
    per, words = {}, {}
    for p in glob.glob("data/new_tests/drat/*/raw_results.json"):
        for r in json.loads(Path(p).read_text()):
            s = r.get("score")
            if isinstance(s, dict):
                val, nv = s.get("drat", 0.0) or 0.0, s.get("n_valid", 0)
            else:
                val, nv = (s or 0.0), len(r.get("extracted_words") or [])
            k = canon(r["model"])
            per.setdefault(k, []).append(val)
            words.setdefault(k, []).append(nv)
    return {m: float(np.mean(v)) for m, v in per.items()
            if np.mean(words[m]) >= 5}



def load_facets_ungated(scores_dir: Path) -> dict:
    """Per-task facet values WITHOUT utility gating.

    compute_composite zeroes surprise/originality/emergent on any artifact that failed utility. That
    is right for a leaderboard -- novelty from a broken artifact should not score -- but it makes the
    facets near-collinear (within-task pairwise r ~ 0.95-0.97), because every one of them is then
    mostly re-expressing the utility rate. Correlating gated facets against an external test cannot
    tell you WHICH facet loads. Here each facet is the plain mean over the artifacts that have it.
    """
    out = {}
    for md in sorted(Path(scores_dir).iterdir()):
        f = md / "path_scores.json"
        if not f.exists():
            continue
        recs = json.loads(f.read_text())
        for task, mode in (("association", "baseline"), ("analogy", "analogy"), ("blending", "blending")):
            rs = [r for r in recs if r.get("mode") == mode]
            if mode == "analogy":
                arts = [r for r in rs if "pair_sat" in r]
                passed = [r.get("pair_sat") is True for r in arts]
            else:
                arts = [r for r in rs if r.get("triples")]
                passed = [r.get("sat") is True for r in arts]
            if not arts:
                continue

            def mean_of(key):
                v = [r[key] for r in arts if r.get(key) is not None
                     and not (isinstance(r[key], float) and math.isnan(r[key]))]
                return float(np.mean(v)) if v else None

            d = {"utility": float(np.mean(passed)),
                 "surprise": mean_of("R"), "originality": mean_of("originality")}
            if task in ("analogy", "blending"):
                d["em_originality"] = mean_of("em_originality")
            if task == "analogy":
                d["em_utility"] = mean_of("invention_utility")
                d["em_integration"] = mean_of("invention_integration")
            elif task == "blending":
                d["em_utility"] = mean_of("blend_utility")
                sc = [r["blend_integration"] for r in arts if r.get("blend_integration") is not None]
                d["em_integration"] = float(np.mean([(x - 1) / 2 for x in sc])) if sc else None
            for k, v in d.items():
                if v is not None:
                    out.setdefault((task, k), {})[canon(md.name)] = v
    return out


def ci(r, n):
    if n < 4:
        return (float("nan"), float("nan"))
    z = np.arctanh(r); se = 1 / math.sqrt(n - 3)
    return tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default="data/kg_creat/kombine_test30/scores")
    a = ap.parse_args()
    komb = load_kombine(Path(a.scores_dir))
    refs = {"DAT": load_dat(), "CDAT": load_cdat(), "RAT": load_rat(), "DRAT": load_drat()}
    facets = load_facets_ungated(Path(a.scores_dir))

    print(f"Kombine models: {len(komb)}")
    for name, d in refs.items():
        print(f"  {name:18s} overlap {len(set(komb) & set(d)):2d}")
    common = set(komb)
    for d in refs.values():
        common &= set(d)
    print(f"\n  models with ALL THREE reference tests: {len(common)}")

    def cell(series, d):
        ms = sorted(set(series) & set(d))
        if len(ms) < 5:
            return None
        r, p = pearsonr([series[m] for m in ms], [d[m] for m in ms])
        return r, p, len(ms)

    print("\nTASK COMPOSITES")
    print(f"  {'':16s}" + "".join(f"{n:>16}" for n in refs))
    for t in TASKS + ["overall"]:
        series = {m: v[t] for m, v in komb.items() if v.get(t) is not None}
        row = f"  {t:16s}"
        for d in refs.values():
            c = cell(series, d)
            row += f"{'--':>16}" if not c else f"{f'{c[0]:+.2f} n={c[2]}':>16}"
        print(row)

    print("\nFACETS -- UNGATED (BH-corrected across the whole facet grid)")
    keys = sorted(facets, key=lambda k: (TASKS.index(k[0]), k[1]))
    cells = {(k, n): cell(facets[k], d) for k in keys for n, d in refs.items()}
    ps = sorted((c[1], key) for key, c in cells.items() if c)
    crit, mtests = {}, len(ps)
    thresh = 0.0
    for i, (pv, key) in enumerate(ps, 1):
        if pv <= 0.05 * i / mtests:
            thresh = pv
    print(f"  {'':30s}" + "".join(f"{n:>16}" for n in refs))
    print("  " + "-" * (30 + 16 * len(refs)))
    for k in keys:
        row = f"  {k[0][:5] + '.' + k[1]:30s}"
        for n in refs:
            c = cells[(k, n)]
            if not c:
                row += f"{'--':>16}"
            else:
                mark = "**" if c[1] <= thresh else ("*" if c[1] < 0.05 else "")
                row += f"{f'{c[0]:+.2f} n={c[2]}{mark}':>16}"
        print(row)
    print(f"\n  * p<0.05 uncorrected;  ** survives Benjamini-Hochberg at q=0.05 over "
          f"{mtests} facet tests (threshold p<={thresh:.4f}).")

    print("\nPER-MODEL REFERENCE SCORES  (-- = no usable data: missing, or <70% of trials parsed)")
    print(f"  {'model':26s}{'DAT':>9}{'CDAT':>9}{'RAT':>9}{'DRAT':>9}   Kombine")
    print("  " + "-" * 71)
    order = sorted(komb, key=lambda m: -(komb[m].get("overall") or -1e9))
    for m in order:
        cells = ""
        for name, d in refs.items():
            v = d.get(m)
            cells += f"{'--':>9}" if v is None else (f"{v:>9.1%}" if name == "RAT" else f"{v:>9.1f}")
        ov = komb[m].get("overall")
        print(f"  {DISPLAY.get(m, m):26s}{cells}   {ov:>7.1f}")
    n_missing = {n: sum(1 for m in komb if m not in d) for n, d in refs.items()}
    print("\n  models lacking usable data: " + ", ".join(f"{n} {c}" for n, c in n_missing.items()))

    print("\nReference tests against each other (are they measuring different things?):")
    names = list(refs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ms = sorted(set(refs[names[i]]) & set(refs[names[j]]) & set(komb))
            if len(ms) < 4:
                continue
            r, p = pearsonr([refs[names[i]][m] for m in ms], [refs[names[j]][m] for m in ms])
            print(f"  {names[i]:18s} vs {names[j]:18s} r={r:+.2f} (n={len(ms)}, p={p:.3f})")


if __name__ == "__main__":
    main()
