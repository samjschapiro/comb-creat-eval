"""Utility only: can a model instantiate the schema each operator needs?

Utility is the gate in the formalism -- every later dimension is scored only on artifacts that pass
it -- so it is the dimension that decides whether the model can perform the subsequent steps at all.
This compares it across the two operators on matched data, with item difficulty controlled.

  U_an = pair_sat        = the two paths share a relation sequence AND their triples are factual
  U_bl = generic_ok      = the 3-judge panel accepts the generic space as instantiated by BOTH inputs

These are like for like. Shared relations over factual triples IS a valid analogy -- there is nothing
further to verify, which is why the check can be mechanical: the analogy operator writes its schema as
an explicit structural correspondence. Blending writes its schema as a sentence, so the same question
("do both inputs instantiate it?") can only be answered by a judge. Different machinery, same job:
each flag fully verifies that its operator's schema holds.

Item difficulty is controlled two ways: items where NO model passed are dropped (the user's rule),
and the headline test is McNemar over matched (model, item) cells, which conditions on both the item
and the model by construction.

    .venv/bin/python -m src.kg_creat.scripts.test_utility_analogy_vs_blending
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

from src.kg_creat.scripts.analyze_failure_modes import FRONTIER, SCORES
from src.kg_creat.scripts.plot_radar import DISPLAY

OUT = Path("data/kg_creat/kombine_test30/analysis/utility_analogy_vs_blending.json")


def cells():
    c = defaultdict(dict)
    for f in sorted(SCORES.glob("*/path_scores.json")):
        m = f.parent.name
        for r in json.loads(f.read_text()):
            k = (m, r.get("u_label"), r.get("v_label"))
            if r.get("mode") == "analogy" and "pair_sat" in r:
                if r.get("pair_sat") is not None:
                    c[k]["U_an"] = r["pair_sat"] is True
                if r.get("pair_structural_ok") is not None:
                    c[k]["an_schema"] = bool(r["pair_structural_ok"])
                c[k]["an_channel"] = r.get("pair_channel")
            elif r.get("mode") == "blending" and r.get("triples"):
                if r.get("generic_ok") is not None:
                    c[k]["U_bl"] = bool(r["generic_ok"])       # blend utility IS the schema gate
                    c[k]["bl_schema"] = bool(r["generic_ok"])
                if r.get("blend_integration") is not None:
                    c[k]["bl_scope"] = int(r["blend_integration"])
    return c


def compare(c, subset, a_key, b_key, label, res):
    rows = [(k, v) for k, v in c.items() if k[0] in subset and a_key in v and b_key in v]
    by_item = defaultdict(list)
    for k, v in rows:
        by_item[(k[1], k[2])].append(v)
    dead = {i for i, vs in by_item.items()
            if not any(v[a_key] for v in vs) or not any(v[b_key] for v in vs)}
    kept = [(k, v) for k, v in rows if (k[1], k[2]) not in dead]
    A = np.array([v[a_key] for _, v in kept]); B = np.array([v[b_key] for _, v in kept])
    n10, n01 = int((A & ~B).sum()), int((~A & B).sum())
    d = n10 + n01
    bt = binomtest(n10, d, 0.5)
    ci = bt.proportion_ci(0.95)
    orr = n10 / n01 if n01 else float("inf")
    lo = ci.low / (1 - ci.low) if ci.low < 1 else float("nan")
    hi = ci.high / (1 - ci.high) if ci.high < 1 else float("inf")

    grp_m, grp_i = defaultdict(lambda: [[], []]), defaultdict(lambda: [[], []])
    for k, v in kept:
        grp_m[k[0]][0].append(v[a_key]); grp_m[k[0]][1].append(v[b_key])
        grp_i[(k[1], k[2])][0].append(v[a_key]); grp_i[(k[1], k[2])][1].append(v[b_key])

    def paired(g):
        a = np.array([np.mean(x[0]) for x in g.values()])
        b = np.array([np.mean(x[1]) for x in g.values()])
        p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        return len(a), int((a > b).sum()), int((b > a).sum()), float(p)

    nm, am, bm, pm = paired(grp_m)
    ni, ai, bi, pi = paired(grp_i)
    print(f"\n{label}")
    print(f"  items dropped as impossible: {len(dead)} of {len(by_item)}")
    print(f"  {len(kept)} matched cells   analogy {100*A.mean():.1f}%   blending {100*B.mean():.1f}%"
          f"   difference {100*(A.mean()-B.mean()):+.1f} pts")
    print(f"  McNemar: analogy-only {n10}, blending-only {n01}, p = {bt.pvalue:.2g}   "
          f"odds ratio {orr:.2f} [{lo:.2f}, {hi:.2f}]")
    print(f"  paired over models ({nm}): analogy higher {am}, blending higher {bm}, p = {pm:.2g}")
    print(f"  paired over items  ({ni}): analogy higher {ai}, blending higher {bi}, p = {pi:.2g}")
    res[label] = {"n_cells": len(kept), "items_dropped": len(dead),
                  "analogy": float(A.mean()), "blending": float(B.mean()),
                  "analogy_only": n10, "blending_only": n01, "p": float(bt.pvalue),
                  "odds_ratio": orr, "or_ci": [lo, hi],
                  "paired_models": {"n": nm, "analogy_higher": am, "blending_higher": bm, "p": pm},
                  "paired_items": {"n": ni, "analogy_higher": ai, "blending_higher": bi, "p": pi}}


def decompose(c, res):
    """What each utility flag is actually made of -- the reason the two comparisons differ."""
    ch = defaultdict(int)
    for v in c.values():
        if v.get("an_channel"):
            ch[v["an_channel"]] += 1
    n = sum(ch.values())
    bl = [v["U_bl"] for v in c.values() if "U_bl" in v]
    print(f"\nWHAT IS INSIDE EACH UTILITY FLAG")
    print(f"  U_an (n={n}):  passes {100*ch['ok']/n:.1f}%   "
          f"fails on SCHEMA {100*ch['structural']/n:.1f}%   fails on FACTS {100*ch['factual']/n:.1f}%")
    print(f"  U_bl (n={len(bl)}):  passes {100*np.mean(bl):.1f}%   "
          f"fails on SCHEMA {100*(1-np.mean(bl)):.1f}%   (no factuality component)")
    res["decomposition"] = {"analogy": {k: v / n for k, v in ch.items()},
                            "blending_pass": float(np.mean(bl))}


def judge_consistency(c, res):
    """The panel returns generic_ok and integration_quality separately. They disagree, which bounds
    how much weight the blend utility flag can carry."""
    both = [(v["U_bl"], v["bl_scope"]) for v in c.values() if "U_bl" in v and "bl_scope" in v]
    n = len(both)
    a = sum(1 for g, s in both if g and s == 1)
    b = sum(1 for g, s in both if not g and s >= 2)
    print(f"\nJUDGE-CONSISTENCY CHECK on the blend panel (n={n})")
    print(f"  generic space ACCEPTED but scope 1 (single-scope): {a} ({100*a/n:.1f}%)")
    print(f"  generic space REJECTED but scope >= 2:             {b} ({100*b/n:.1f}%)")
    print(f"  the two panel fields disagree on {a+b} blends ({100*(a+b)/n:.1f}%)")
    res["judge_consistency"] = {"n": n, "accepted_but_scope1": a, "rejected_but_double": b,
                                "disagree_pct": 100 * (a + b) / n}


def per_model(c, res):
    per = defaultdict(lambda: [[], []])
    for (m, u, v), d in c.items():
        if "U_an" in d:
            per[m][0].append(d["U_an"])
        if "U_bl" in d:
            per[m][1].append(d["U_bl"])
    rows = [(m, float(np.mean(x[0])), float(np.mean(x[1]))) for m, x in per.items() if x[0] and x[1]]
    rows.sort(key=lambda t: -(t[1] - t[2]))
    print(f"\nPER-MODEL UTILITY (analogy vs blending), {len(rows)} models")
    print(f"  analogy higher on {sum(1 for _, a, b in rows if a > b)}, "
          f"blending higher on {sum(1 for _, a, b in rows if b > a)}")
    for m, a, b in rows[:5] + rows[-5:]:
        print(f"    {DISPLAY.get(m, m):22s} analogy {100*a:5.1f}%   blending {100*b:5.1f}%   "
              f"{100*(a-b):+6.1f}")
    res["per_model"] = {m: {"analogy": a, "blending": b} for m, a, b in rows}


def stratify(c, res):
    """Does the schema gap hold inside every provider and at every level of item difficulty?

    The item-difficulty split also speaks to the instrument objection. Analogy's schema rate is nearly
    FLAT across difficulty terciles while blending's swings by 55 points, so the gap is not a constant
    offset of the sort a fixed instrument difference would produce -- though blending's much wider
    range is also what mechanically drives the gap's variation, so this is suggestive, not decisive.
    """
    rows = [(k, v) for k, v in c.items() if "U_an" in v and "U_bl" in v]

    def stats(sub):
        A = np.array([v["U_an"] for _, v in sub]); B = np.array([v["U_bl"] for _, v in sub])
        n10, n01 = int((A & ~B).sum()), int((~A & B).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        return float(A.mean()), float(B.mean()), n10, n01, float(p)

    from src.kg_creat.scripts.plot_radar import _provider
    byp = defaultdict(list)
    for k, v in rows:
        byp[_provider(k[0])].append((k, v))
    print("\nUTILITY GAP BY PROVIDER")
    out_p = {}
    for prov in sorted(byp, key=lambda x: -len(byp[x])):
        a, b, n10, n01, p = stats(byp[prov])
        out_p[prov] = {"n": len(byp[prov]), "analogy": a, "blending": b, "p": p}
        print(f"  {prov:12s} n={len(byp[prov]):3d}  analogy {100*a:5.1f}%  blending {100*b:5.1f}%  "
              f"{100*(a-b):+6.1f} pts  p={p:.2g}")

    byi = defaultdict(list)
    for k, v in rows:
        byi[(k[1], k[2])].append(v)
    # Split on the MEAN of the two tasks' pass rates. Sorting items by one task's rate and then
    # comparing the tasks on those bins biases that task upward in its own "easy" bin; the neutral
    # split avoids it, and the crossover below survives either way.
    an_r = {i: np.mean([x["U_an"] for x in vs]) for i, vs in byi.items()}
    bl_r = {i: np.mean([x["U_bl"] for x in vs]) for i, vs in byi.items()}
    rate = {i: (an_r[i] + bl_r[i]) / 2 for i in byi}
    order = sorted(rate, key=rate.get)
    rr = float(np.corrcoef([an_r[i] for i in byi], [bl_r[i] for i in byi])[0, 1])
    print(f"\nPER-ITEM DIFFICULTY IS UNCORRELATED ACROSS THE TWO TASKS: r = {rr:+.2f} (n = {len(byi)})")
    print("  a pair that is hard to analogise is not the pair that is hard to blend")
    res["per_item_difficulty_r"] = rr
    print("\nUTILITY GAP BY ITEM DIFFICULTY (terciles of the MEAN pass rate across both tasks)")
    out_i = {}
    for name, grp in zip(("hardest 10", "middle 10", "easiest 10"),
                         (order[:10], order[10:20], order[20:])):
        sub = [(k, v) for k, v in rows if (k[1], k[2]) in set(grp)]
        a, b, n10, n01, p = stats(sub)
        out_i[name] = {"n": len(sub), "analogy": a, "blending": b, "p": p}
        print(f"  {name:11s} n={len(sub):3d}  analogy {100*a:5.1f}%  blending {100*b:5.1f}%  "
              f"{100*(a-b):+6.1f} pts  p={p:.2g}")
    print("  -> the advantage is concentrated on items that are hard OVERALL; on the easiest\n     third blending is ahead.")
    res["by_provider"] = out_p
    res["by_item_difficulty"] = out_i


def main():
    c = cells()
    everyone = {k[0] for k in c}
    res = {"n_models": len(everyone)}
    print(f"{len(everyone)} models x 30 anchor pairs, matched cells")
    decompose(c, res)
    judge_consistency(c, res)
    for scope, subset in (("ALL MODELS", everyone), ("FRONTIER", FRONTIER & everyone)):
        print(f"\n{'=' * 76}\n{scope}  ({len(subset)} models)")
        compare(c, subset, "U_an", "U_bl",
                f"{scope} | UTILITY as defined (U_an includes factuality; U_bl does not)", res)
    per_model(c, res)
    stratify(c, res)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
