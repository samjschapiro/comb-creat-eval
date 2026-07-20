"""Roll path-level score records up to the per-constraint ideation/execution result.

Per mode: ``R_emit`` (mean novelty of emitted paths -- ideation), ``R_valid`` (mean novelty
of satisfying paths), ``sat_rate`` (execution) with a failure-channel breakdown. For Regime
A, also the **within-bundle deltas** vs the baseline (``dR_emit``, ``dsat``) that place each
constraint type in the (novelty, satisfaction) 2x2. Diversity ``D`` is deferred (secondary).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


def _clean(xs):
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _mean(xs):
    xs = _clean(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def _sat_rate(recs):
    """Fraction satisfying, over paths with a determined sat (None = unjudged, excluded)."""
    vals = [1.0 if r["sat"] else 0.0 for r in recs if r.get("sat") is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _by(recs, key):
    out = defaultdict(list)
    for r in recs:
        out[r[key]].append(r)
    return out


def aggregate(recs: list[dict]) -> dict:
    per_mode = {}
    for mode, rs in _by(recs, "mode").items():
        per_mode[mode] = {
            "n_paths": len(rs),
            "R_emit": _mean([r["R"] for r in rs]),
            "R_valid": _mean([r["R"] for r in rs if r.get("sat") is True]),
            "sat_rate": _sat_rate(rs),
            "wf_rate": _mean([1.0 if r["well_formed"] else 0.0 for r in rs]),
            "channels": dict(Counter(r.get("channel") for r in rs)),
        }

    # Within-bundle deltas vs baseline (Regime A only): the 2x2 coordinates.
    two_by_two = {}
    dacc = defaultdict(lambda: {"dR": [], "dsat": []})
    for _bundle, brs in _by([r for r in recs if r["regime"] == "A"], "bundle_id").items():
        by_mode = _by(brs, "mode")
        if "baseline" not in by_mode:
            continue
        base_R = _mean([r["R"] for r in by_mode["baseline"]])
        base_sat = _sat_rate(by_mode["baseline"])
        for mode, mrs in by_mode.items():
            if mode == "baseline":
                continue
            mR, msat = _mean([r["R"] for r in mrs]), _sat_rate(mrs)
            if base_R is not None and mR is not None:
                dacc[mode]["dR"].append(mR - base_R)
            if base_sat is not None and msat is not None:
                dacc[mode]["dsat"].append(msat - base_sat)
    for mode, d in dacc.items():
        two_by_two[mode] = {
            "mean_dR_emit": _mean(d["dR"]),
            "mean_dsat": _mean(d["dsat"]),
            "n_bundles": max(len(d["dR"]), len(d["dsat"])),
        }

    return {"per_mode": per_mode, "two_by_two": two_by_two, "n_paths_total": len(recs)}
