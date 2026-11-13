"""Analyse the abstraction-bottleneck causal test (see build_abstraction_causal.py).

Reads data/kg_creat/abstraction_causal/scores/<condition>__<model>/path_scores.json and, for the
"previously failed" stratification, each model's paper-run blends (data/kg_creat/human_study/paper_scores).
Panel verdicts are recomputed from the stored per-judge votes: G = majority generic_ok, V = majority
coherent, Q = median scope (1 single-scope / 2 double-scope / 3 double-scope emergent).

Reports, per condition (control / hint / oracle), overall and per model:
  G                 valid generic space (under oracle this checks the GIVEN g, plus whether it was copied)
  V                 coherent blend
  success           G and V: a full valid blend, the benchmark's notion of a successful blend
  Q>=2, Q=3         double-scope / double-scope-emergent integration
  O(c')             emergent originality (pool includes every paper answer to the item)
and the primary causal contrasts, paired by (model, item) with exact McNemar tests:
  * on items the model FAILED (G = 0) in the paper run: success under oracle vs control, hint vs control
  * on all items: the same contrasts
plus the reference rate: V given a self-found valid g in the paper run.

    .venv/bin/python -m src.kg_creat.scripts.analyze_abstraction_causal
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path("data/kg_creat/abstraction_causal")
PAPER = Path("data/kg_creat/human_study/paper_scores")
CONDS = ("control", "hint", "oracle")


def maj(vs):
    vs = [bool(v) for v in vs if v is not None]
    return (sum(vs) * 2 > len(vs)) if vs else None


def med(vs):
    vs = sorted(v for v in vs if v is not None)
    return vs[len(vs) // 2] if vs else None


def verdicts(r):
    js = r.get("blend_judges") or []
    if not js:
        return None
    g, v = maj([j.get("generic_ok") for j in js]), maj([j.get("coherent") for j in js])
    q = med([j.get("scope") for j in js])
    return {"G": g, "V": v, "success": bool(g) and bool(v), "Q2": (q or 0) >= 2, "Q3": (q or 0) >= 3,
            "O": r.get("em_originality")}


def load_condition(cond):
    out = {}
    for md in sorted((ROOT / "scores").glob(f"{cond}__*")):
        model = md.name.split("__", 1)[1]
        for r in json.loads((md / "path_scores.json").read_text()):
            if r["mode"] == "blending":
                v = verdicts(r)
                out[(model, r["prompt_id"])] = v       # missing judges (unparsed answer) -> None = failure
    return out


def api_failures():
    """(model, prompt_id) cells where ANY condition's call failed at the API: excluded from every comparison
    (an API error is not a model failure). Unparsed answers from a successful call DO count as failures."""
    bad = set()
    for c in CONDS:
        for rp in (ROOT / c / "responses").glob("*/responses.json"):
            for r in json.loads(rp.read_text()):
                if r.get("api_error"):
                    bad.add((rp.parent.name, r["prompt_id"]))
    return bad


def paper_runs(models):
    out = {}
    for m in models:
        p = PAPER / m / "path_scores.json"
        if p.exists():
            for r in json.loads(p.read_text()):
                if r["mode"] == "blending":
                    out[(m, r["prompt_id"])] = verdicts(r)
    return out


def rate(xs):
    xs = list(xs)
    return (float(np.mean(xs)), len(xs)) if xs else (None, 0)


def mcnemar(a, b):
    """exact McNemar on paired booleans: returns (b_only, a_only, p) for 'b beats a'."""
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    n = a_only + b_only
    p = binomtest(b_only, n, 0.5).pvalue if n else 1.0
    return b_only, a_only, float(p)


def main():
    data = {c: load_condition(c) for c in CONDS}
    models = sorted({m for c in CONDS for (m, _) in data[c]})
    items = sorted({i for c in CONDS for (_, i) in data[c]}, key=lambda s: int(s[1:]))
    paper = paper_runs(models)
    bad = api_failures()
    get = lambda c, m, i, k: bool((data[c].get((m, i)) or {}).get(k))   # unparsed / missing = fail
    print(f"excluded (API failure in some condition): {sorted(bad)}")
    res = {"excluded_api_failures": [], "models": models, "n_items": len(items), "by_condition": {}, "by_model": {}, "contrasts": {}}
    res["excluded_api_failures"] = sorted(map(list, bad))

    print(f"{len(models)} models x {len(items)} items\n")
    print(f"{'condition':9s} {'G':>6s} {'V':>6s} {'succ':>6s} {'Q>=2':>6s} {'Q=3':>6s} {'O(c)':>6s}   (all items)")
    for c in CONDS:
        row = {k: rate(get(c, m, i, k) for m in models for i in items if (m, i) not in bad)[0]
               for k in ("G", "V", "success", "Q2", "Q3")}
        os_ = [data[c][(m, i)]["O"] for m in models for i in items
               if data[c].get((m, i)) and data[c][(m, i)]["O"] is not None and data[c][(m, i)]["V"]]
        row["O_among_coherent"] = float(np.mean(os_)) if os_ else None
        res["by_condition"][c] = row
        print(f"{c:9s} " + " ".join(f"{100 * row[k]:6.1f}" for k in ("G", "V", "success", "Q2", "Q3"))
              + f" {row['O_among_coherent'] or float('nan'):6.3f}")

    # reference: coherence given a self-found valid g, in the paper run, for these models
    ref = [v["V"] for (m, i), v in paper.items() if v and v["G"]]
    res["paper_V_given_selfG"] = rate(ref)
    print(f"\npaper run, these models: V | self-found valid g = {100 * np.mean(ref):.1f}% (n={len(ref)})")

    print("\n== per model: success (G and V) by condition; paper-run G rate")
    for m in models:
        pr = [paper[(m, i)]["G"] for i in items if paper.get((m, i))]
        res["by_model"][m] = {c: rate(get(c, m, i, "success") for i in items if (m, i) not in bad)[0] for c in CONDS}
        res["by_model"][m]["paper_G"] = rate(pr)[0]
        print(f"  {m[:38]:38s} paper G {100 * (res['by_model'][m]['paper_G'] or 0):5.1f} | "
              + " ".join(f"{c} {100 * res['by_model'][m][c]:5.1f}" for c in CONDS))

    print("\n== causal contrasts (paired by model x item; exact McNemar)")
    for label, keep in (("failed_in_paper", lambda m, i: paper.get((m, i)) and not paper[(m, i)]["G"]),
                        ("all_items", lambda m, i: True)):
        cells = [(m, i) for m in models for i in items if keep(m, i) and (m, i) not in bad]
        ctl = [get("control", m, i, "success") for m, i in cells]
        out = {"n_cells": len(cells), "control": float(np.mean(ctl)) if cells else None}
        for c in ("hint", "oracle"):
            x = [get(c, m, i, "success") for m, i in cells]
            b, a, p = mcnemar(ctl, x)
            out[c] = {"rate": float(np.mean(x)) if cells else None, "wins": b, "losses": a, "p": p}
        # quality of what is built, oracle vs control, among cells where each produced a valid blend
        for k in ("Q2", "Q3"):
            out[f"{k}_oracle"] = rate(get("oracle", m, i, k) for m, i in cells if get("oracle", m, i, "success"))[0]
            out[f"{k}_control"] = rate(get("control", m, i, k) for m, i in cells if get("control", m, i, "success"))[0]
        res["contrasts"][label] = out
        print(f"  [{label}] n={len(cells)}  control {100 * out['control']:.1f}%  "
              f"hint {100 * out['hint']['rate']:.1f}% (+{out['hint']['wins']}/-{out['hint']['losses']}, p={out['hint']['p']:.2g})  "
              f"oracle {100 * out['oracle']['rate']:.1f}% (+{out['oracle']['wins']}/-{out['oracle']['losses']}, p={out['oracle']['p']:.2g})")

    # elaboration quality on previously failed items, independent of the panel re-judging g; and conditional
    # on an accepted g (self-found under control/hint, supplied under oracle) -- is elaboration intact?
    failed = [(m, i) for m in models for i in items
              if paper.get((m, i)) and not paper[(m, i)]["G"] and (m, i) not in bad]
    res["elaboration_failed_items"] = {}
    print("\n== elaboration on previously failed items (V / Q>=2 / Q=3 regardless of G; then given accepted G)")
    for c in CONDS:
        acc = [x for x in failed if get(c, *x, "G")]
        e = {k: rate(get(c, m, i, k) for m, i in failed)[0] for k in ("V", "Q2", "Q3")}
        e.update({"V_given_G": rate(get(c, m, i, "V") for m, i in acc)[0],
                  "Q2_given_G": rate(get(c, m, i, "Q2") for m, i in acc)[0], "n_G": len(acc), "n": len(failed)})
        res["elaboration_failed_items"][c] = e
        print(f"  {c:8s} V {100 * e['V']:5.1f}  Q>=2 {100 * e['Q2']:5.1f}  Q=3 {100 * e['Q3']:5.1f}  |  "
              f"V|G {100 * e['V_given_G']:5.1f}  Q>=2|G {100 * e['Q2_given_G']:5.1f}  (n_G={len(acc)}/{len(failed)})")

    # did models actually use the given g?
    copied = []
    oracle_g = json.loads((ROOT / "oracle_g.json").read_text())
    for md in sorted((ROOT / "scores").glob("oracle__*")):
        resp = ROOT / "oracle" / "responses" / md.name.split("__", 1)[1] / "responses.json"
        if resp.exists():
            for r in json.loads(resp.read_text()):
                g = ((r.get("items") or [{}])[0] or {}).get("generic_space") or ""
                copied.append(g.strip().lower().rstrip(".") == oracle_g[r["prompt_id"]]["generic_space"].strip().lower().rstrip("."))
    res["oracle_g_copied_verbatim"] = rate(copied)
    print(f"\noracle: given g copied verbatim in {100 * np.mean(copied):.1f}% of answers (n={len(copied)})" if copied else "")

    (ROOT / "analysis.json").write_text(json.dumps(res, indent=1))
    print(f"\n-> {ROOT / 'analysis.json'}")


if __name__ == "__main__":
    main()
