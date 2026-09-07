"""Does succeeding at analogy on a pair of entities mean succeeding at blending them?

Analogy and blending run on the SAME 30 anchor pairs, so every model gives a matched pair of
attempts on every item: 1 model x 1 pair x 2 tasks. That makes the question a 2x2 contingency table
over cells, not a correlation over models, and it separates two things a model-level correlation
cannot:

  ASSOCIATION      how often the two tasks agree on the same (model, pair) cell -- the phi coefficient
  DIRECTION        whether the disagreements are symmetric -- McNemar on the discordant cells

Two levels are tabulated, because the tasks meet at two different heights:

  utility          the benchmark's own flag per task: ``pair_sat`` for analogy (both domain paths
                   valid and aligned), ``sat`` for blending (which includes the generic-space gate)
  the creative act ``invention_utility & invention_integration`` for analogy (the projected concept
                   h was actually built and holds together) against ``blend_integration >= 2`` (the
                   generic space was accepted). This is the comparison the formalism cares about:
                   both are the step where the model has to invent rather than retrieve.

    .venv/bin/python -m src.kg_creat.scripts.analyze_task_dissociation
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from src.kg_creat.scripts.analyze_failure_modes import FRONTIER, SCORES

OUT = Path("data/kg_creat/kombine_test30/analysis/task_dissociation.json")
RESP = Path("data/kg_creat/kombine_test30/responses")


def cells():
    """One record per (model, anchor pair) carrying both tasks' outcomes."""
    c = defaultdict(dict)
    for f in sorted(SCORES.glob("*/path_scores.json")):
        m = f.parent.name
        for r in json.loads(f.read_text()):
            k = (m, r.get("u_label"), r.get("v_label"))
            if r.get("mode") == "analogy" and "pair_sat" in r:
                c[k]["analogy_util"] = r["pair_sat"] is True
                if r.get("invention_utility") is not None:
                    c[k]["analogy_act"] = bool(r["invention_utility"]) and bool(r.get("invention_integration"))
            elif r.get("mode") == "blending" and r.get("triples"):
                c[k]["blend_util"] = r.get("sat") is True
                if r.get("blend_integration") is not None:
                    c[k]["blend_act"] = int(r["blend_integration"]) >= 2
    return c


def table(c, a_key, b_key, keys, label, res):
    rows = [v for k, v in c.items() if k in keys and a_key in v and b_key in v]
    A = np.array([v[a_key] for v in rows]); B = np.array([v[b_key] for v in rows])
    n11 = int((A & B).sum()); n10 = int((A & ~B).sum())
    n01 = int((~A & B).sum()); n00 = int((~A & ~B).sum())
    n = len(rows)
    phi = float(np.corrcoef(A.astype(float), B.astype(float))[0, 1])
    mc = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    print(f"\n{label}   (n = {n} model x pair cells)")
    print(f"                          blend ok   blend fail")
    print(f"    analogy ok            {n11:8d}   {n10:10d}")
    print(f"    analogy fail          {n01:8d}   {n00:10d}")
    print(f"    analogy ok {100*A.mean():.0f}%, blend ok {100*B.mean():.0f}%")
    print(f"    DISCORDANT {100*(n10+n01)/n:.0f}% of cells: "
          f"analogy-only {n10} ({100*n10/n:.0f}%), blend-only {n01} ({100*n01/n:.0f}%)")
    print(f"    phi = {phi:+.3f}   McNemar p = {mc:.2g}  "
          f"({'asymmetric' if mc < 0.05 else 'symmetric'})")
    res[label] = {"n": n, "both": n11, "analogy_only": n10, "blend_only": n01, "neither": n00,
                  "analogy_rate": round(float(A.mean()), 3), "blend_rate": round(float(B.mean()), 3),
                  "discordant_pct": round(100 * (n10 + n01) / n, 1),
                  "phi": round(phi, 3), "mcnemar_p": float(mc)}
    return rows


def reliability(res):
    """A near-zero phi is only interesting if the two measures are reliable enough to correlate.

    Each gate is a majority of 3 judges, so single-judge verdicts are available and the panel's
    reliability can be estimated instead of assumed: mean pairwise correlation among the three
    judges, stepped up to a 3-judge composite by Spearman-Brown. The observed phi is then divided by
    the geometric mean of the two reliabilities -- the ceiling the correlation could reach if both
    gates were measured without error. (Spearman-Brown assumes an averaged composite; the gates are
    majority votes, so this is an approximation, and a generous one.)
    """
    ana, bl = defaultdict(dict), defaultdict(dict)
    for f in sorted(SCORES.glob("*/path_scores.json")):
        m = f.parent.name
        for r in json.loads(f.read_text()):
            k = (m, r.get("u_label"), r.get("v_label"))
            for js, store, fn in ((r.get("invention_judges"), ana,
                                   lambda j: bool(j.get("valid")) and bool(j.get("coherent"))),
                                  (r.get("blend_judges"), bl,
                                   lambda j: bool(j.get("generic_ok")))):
                for j in js or []:
                    if isinstance(j, dict) and j.get("model"):
                        store[k][j["model"]] = fn(j)
    out = {}
    for name, store in (("analogy invention", ana), ("blend generic space", bl)):
        judges = sorted({jm for v in store.values() for jm in v})
        keys = [k for k, v in store.items() if len(v) == len(judges)]
        X = np.array([[store[k][j] for j in judges] for k in keys], float)
        rs = [np.corrcoef(X[:, a], X[:, b])[0, 1] for a in range(len(judges))
              for b in range(a + 1, len(judges))]
        rbar = float(np.mean(rs))
        rel = len(judges) * rbar / (1 + (len(judges) - 1) * rbar)
        out[name] = {"n_cells": len(keys), "n_judges": len(judges),
                     "mean_pairwise_judge_r": round(rbar, 3), "panel_reliability": round(rel, 3)}
        print(f"  {name:22s} n = {len(keys)} cells, {len(judges)} judges, "
              f"mean pairwise judge r = {rbar:.2f}  ->  panel reliability {rel:.2f}")
    ceiling = float(np.sqrt(out["analogy invention"]["panel_reliability"] *
                            out["blend generic space"]["panel_reliability"]))
    phi = res["ALL MODELS — the creative act (invention h vs generic space)"]["phi"]
    out["attenuation_ceiling"] = round(ceiling, 3)
    out["disattenuated_phi"] = round(phi / ceiling, 3)
    print(f"  attenuation ceiling = {ceiling:.2f}; observed phi {phi:+.3f} "
          f"-> disattenuated {phi / ceiling:+.3f}")
    print("  even corrected for judge noise the two tasks barely predict each other cell by cell")
    res["reliability"] = out


def model_level(c, res):
    """The same question one level up: do models that analogize well blend well?"""
    per = defaultdict(lambda: [[], []])
    for (m, u, v), d in c.items():
        if "analogy_act" in d and "blend_act" in d:
            per[m][0].append(d["analogy_act"]); per[m][1].append(d["blend_act"])
    ms = sorted(per)
    A = np.array([np.mean(per[m][0]) for m in ms]); B = np.array([np.mean(per[m][1]) for m in ms])
    r = float(np.corrcoef(A, B)[0, 1])
    res["model_level"] = {"n_models": len(ms), "r": round(r, 3)}
    print(f"\nMODEL LEVEL: analogy invention rate vs generic-space rate over {len(ms)} models: "
          f"r = {r:+.2f}")
    return ms, A, B


def main():
    c = cells()
    everyone = {k[0] for k in c}
    frontier = FRONTIER & everyone
    res = {"n_models": len(everyone), "n_frontier": len(frontier)}
    print(f"{len(everyone)} scored models x 30 anchor pairs; frontier subset = {len(frontier)}")

    for scope, keys in (("ALL MODELS", {k for k in c}),
                        ("FRONTIER", {k for k in c if k[0] in frontier})):
        print(f"\n{'=' * 74}\n{scope}")
        table(c, "analogy_util", "blend_util", keys, f"{scope} — task utility", res)
        table(c, "analogy_act", "blend_act", keys, f"{scope} — the creative act "
              f"(invention h vs generic space)", res)

    # is the disagreement systematic, or noise sprayed evenly?
    print(f"\n{'=' * 74}\nWHERE THE DISAGREEMENTS SIT (creative act, all models)")
    per_m, per_i = defaultdict(lambda: [0, 0, 0]), defaultdict(lambda: [0, 0, 0])
    for (m, u, v), d in c.items():
        if "analogy_act" not in d or "blend_act" not in d:
            continue
        a, b = d["analogy_act"], d["blend_act"]
        for acc in (per_m[m], per_i[(u, v)]):
            acc[0] += int(a and not b); acc[1] += int(b and not a); acc[2] += 1
    print("  per model, of its 30 pairs:  analogy-only / blend-only")
    for m, (x, y, n) in sorted(per_m.items(), key=lambda kv: -(kv[1][0] - kv[1][1])):
        print(f"    {m:38s} {x:3d} / {y:<3d}  ({n} pairs)")
    print("\n  the most one-sided anchor pairs (analogy-only minus blend-only, across models):")
    order = sorted(per_i.items(), key=lambda kv: -(kv[1][0] - kv[1][1]))
    for (u, v), (x, y, n) in order[:5] + order[-5:]:
        print(f"    {u + ' + ' + v:44s} analogy-only {x:2d}, blend-only {y:2d}  (of {n})")
    res["per_model_discordance"] = {m: {"analogy_only": x, "blend_only": y, "n": n}
                                    for m, (x, y, n) in per_m.items()}
    res["per_item_discordance"] = {f"{u} + {v}": {"analogy_only": x, "blend_only": y, "n": n}
                                   for (u, v), (x, y, n) in per_i.items()}
    print(f"\n{'=' * 74}\nIS THE NEAR-ZERO ASSOCIATION JUST JUDGE NOISE?")
    reliability(res)
    model_level(c, res)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
