"""Catalogue every frontier blend rejected at the generic-space gate, with the schema the model wrote
and the panel's stated reason -- and, on the SAME anchors, the schemas that were accepted.

The pairing is the point. A list of failures alone cannot distinguish "these anchors are impossible"
from "these models keep making the same mistake"; showing that other models found a workable generic
space for the very same inputs settles it.

The gate is the benchmark's biggest single failure channel (47% of frontier blends), and the claim
attached to it -- that the schema is instantiated by one input only -- is a claim about content, so it
has to be shown rather than summarised. This dumps all of them, grouped by anchor pair, hardest first.

It also asks whether the failures CONVERGE: for each item, the mean pairwise similarity among the
rejected generic spaces against the same among the accepted ones. If models fail the same way, the
rejected schemas will be more alike than the accepted ones -- a stronger claim than any single example.

    .venv_mlx/bin/python -m src.kg_creat.scripts.catalogue_generic_space_failures

Writes the full catalogue as JSON and a markdown section for the report.
"""
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.analyze_failure_modes import FRONTIER, SCORES
try:                                    # plot_radar pulls matplotlib + svgpath2mpl, absent in the MLX env
    from src.kg_creat.scripts.plot_radar import DISPLAY
except ModuleNotFoundError:             # display names only affect labels, never the data
    DISPLAY = {}

RESP = Path("data/kg_creat/kombine_test30/responses")
OUT_JSON = Path("data/kg_creat/kombine_test30/analysis/generic_space_failures.json")
OUT_MD = Path("docs/reports/2026-09-03_kg_creat_frontier_failures/generic_space_catalogue.md")
N_PAIRS_IN_FULL = 8          # anchor pairs written out case by case; the rest are tabulated


def _disp(m):
    return DISPLAY.get(m, m.split("_", 1)[-1])


def first_reason(judges, max_chars=320):
    """The panel's account of why the schema failed, trimmed to its generic-space clause."""
    for j in judges or []:
        if not isinstance(j, dict):
            continue
        e = (j.get("explanation") or "").strip()
        if not e:
            continue
        # judges often write "(1) GENERIC SPACE: ..." -- keep from there if present
        m = re.search(r"(GENERIC SPACE[:\s].*)", e, flags=re.S | re.I)
        e = m.group(1) if m else e
        e = re.sub(r"\s+", " ", e)
        return e[:max_chars] + ("…" if len(e) > max_chars else "")
    return ""


def main():
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)

    items = {}
    for f in sorted(RESP.glob("*/responses.json")):
        m = f.parent.name
        for r in json.loads(f.read_text()):
            if r.get("mode") == "blending" and r.get("items"):
                items[(m, r["u_label"], r["v_label"])] = r["items"][0]

    failed, passed = defaultdict(list), defaultdict(list)
    for f in sorted(SCORES.glob("*/path_scores.json")):
        m = f.parent.name
        if m not in FRONTIER:
            continue
        for r in json.loads(f.read_text()):
            if r.get("mode") != "blending" or r.get("generic_ok") is None:
                continue
            it = items.get((m, r["u_label"], r["v_label"])) or {}
            g = (it.get("generic_space") or "").strip()
            if not g:
                continue
            # `generic_ok` is the gate; scope is kept only for display alongside accepted blends.
            rec = {"model": m, "display": _disp(m), "g": g, "gate_ok": bool(r["generic_ok"]),
                   "concept": it.get("concept"),
                   "scope": int(r["blend_integration"]) if r.get("blend_integration") else None,
                   "reason": first_reason(r.get("blend_judges"))}
            (passed if rec["gate_ok"] else failed)[(r["u_label"], r["v_label"])].append(rec)

    pairs = sorted(set(failed) | set(passed),
                   key=lambda k: (-len(failed[k]), k))
    conv = {}
    for k in pairs:
        row = {}
        for label, group in (("failed", failed[k]), ("passed", passed[k])):
            if len(group) < 2:
                continue
            V = np.vstack([un(np.asarray(embed(r["g"]), float)) for r in group])
            S = V @ V.T
            iu = np.triu_indices(len(V), 1)
            row[label] = float(S[iu].mean())
        if len(row) == 2:
            conv[f"{k[0]} + {k[1]}"] = {**{f"{a}_mean_similarity": round(b, 3) for a, b in row.items()},
                                        "n_failed": len(failed[k]), "n_passed": len(passed[k])}
    d = [v["failed_mean_similarity"] - v["passed_mean_similarity"] for v in conv.values()]
    print(f"{sum(len(v) for v in failed.values())} frontier blends rejected at the generic-space gate, "
          f"across {len([k for k in pairs if failed[k]])} of {len(pairs)} anchor pairs\n")
    print("DO THE FAILURES CONVERGE? mean pairwise similarity of the rejected schemas vs the accepted ones")
    print(f"  rejected more alike on {sum(x > 0 for x in d)}/{len(d)} pairs; "
          f"mean difference {np.mean(d):+.3f}")

    solved = [k for k in pairs if passed[k]]
    print(f"\nANCHOR PAIRS WITH AT LEAST ONE ACCEPTED BLEND: {len(solved)}/{len(pairs)} "
          f"-- an anchor pair that defeats every frontier model is {'not observed' if len(solved) == len(pairs) else 'observed'}")

    print("\nHARDEST ANCHOR PAIRS (rejected / scored, frontier)")
    for k in pairs[:12]:
        n_f, n_p = len(failed[k]), len(passed[k])
        print(f"  {k[0] + ' + ' + k[1]:44s} {n_f:2d}/{n_f + n_p}")

    OUT_JSON.write_text(json.dumps(
        {"n_failed": sum(len(v) for v in failed.values()),
         "convergence": conv,
         "by_pair": {f"{k[0]} + {k[1]}": {"n_failed": len(failed[k]), "n_passed": len(passed[k]),
                                          "failures": failed[k], "passes": passed[k]}
                     for k in pairs}}, indent=1))

    L = ["# Generic-space failures: the full catalogue", "",
         f"Every frontier blend rejected at the generic-space gate — {sum(len(v) for v in failed.values())} "
         f"of {sum(len(v) for v in failed.values()) + sum(len(v) for v in passed.values())} — with the "
         f"schema the model wrote and the panel's reason, grouped by anchor pair, hardest first. "
         f"Generated by `catalogue_generic_space_failures.py`; nothing here is hand-picked.", ""]
    for k in pairs[:N_PAIRS_IN_FULL]:
        if not failed[k]:
            continue
        L += [f"## {k[0]} + {k[1]} — {len(failed[k])} of {len(failed[k]) + len(passed[k])} rejected", ""]
        for r in sorted(failed[k], key=lambda r: r["display"]):
            L += [f"**{r['display']}** → *{r['concept']}*  ",
                  f"g: “{r['g']}”  ",
                  f"panel: {r['reason']}", ""]
        if passed[k]:
            L += [f"**Accepted on the same anchors — {len(passed[k])} model(s) found a schema both "
                  f"inputs instantiate:**", ""]
            for r in sorted(passed[k], key=lambda r: (-(r["scope"] or 0), r["display"])):
                L += [f"**{r['display']}** → *{r['concept']}* (scope {r['scope']})  ",
                      f"g: “{r['g']}”", ""]
        else:
            L += ["*No frontier model was accepted on this pair.*", ""]
    L += ["## Remaining pairs", "", "| anchors | rejected / scored |", "|---|--:|"]
    for k in pairs[N_PAIRS_IN_FULL:]:
        L.append(f"| {k[0]} + {k[1]} | {len(failed[k])} / {len(failed[k]) + len(passed[k])} |")
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
