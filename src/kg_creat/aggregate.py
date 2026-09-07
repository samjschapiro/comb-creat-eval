"""Roll path-level score records up to the per-constraint ideation/execution result.

Per mode: ``R_emit`` (mean novelty of emitted paths -- ideation), ``R_valid`` (mean novelty
of satisfying paths), ``sat_rate`` (execution) with a failure-channel breakdown. For Regime
A, also the **within-bundle deltas** vs the baseline (``dR_emit``, ``dsat``) that place each
constraint type in the (novelty, satisfaction) 2x2. Diversity ``D`` is deferred (secondary).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict


# Emergent-creativity field renames (paper, 2026-08-30): "coherence" -> UTILITY (J^utl) and
# "validity"/"scope" -> INTEGRATION QUALITY (J^qua). Records scored before the rename carry the old
# keys, so normalize on read and existing path_scores.json keep working without re-judging.
_RENAMED = {
    "invention_coherent": "invention_utility",     # J^utl_an
    "invention_valid": "invention_integration",    # J^qua_an
    "blend_coherent": "blend_utility",             # J^utl_bl
    "scope": "blend_integration",                  # J^qua_bl in {1,2,3}
}


def normalize_scored_records(recs: list[dict]) -> list[dict]:
    """Upgrade pre-rename score records in place: old emergent-creativity keys -> the paper's names.

    Additive and idempotent -- old keys are left alone, so readers not yet migrated still work.
    """
    for r in recs:
        for old, new in _RENAMED.items():
            if old in r and new not in r:
                r[new] = r[old]
    return recs


def _clean(xs):
    return [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def _mean(xs):
    xs = _clean(xs)
    return round(sum(xs) / len(xs), 4) if xs else None


def _sat_rate(recs):
    """Fraction satisfying, over paths with a determined sat (None = unjudged, excluded)."""
    vals = [1.0 if r["sat"] else 0.0 for r in recs if r.get("sat") is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _creativity(recs):
    """Mean per-path creativity: ``E[R(P) · U(P)]`` over emitted paths (design.md §Scoring).

    Creativity needs BOTH novelty and utility, and utility is judge-gated
    (``U = (∏(1+α_t·n_t))·1[valid ∧ factual]``), so a path that violates its constraint
    contributes zero however remote it is. Taking the expectation over *all* emitted paths —
    rather than averaging novelty within the satisfying subset — is what keeps cells with very
    different pass rates comparable: conditioning on success would score a cell that succeeds
    5 % of the time on a small, heavily-selected survivor set.

    Reported at ``α = 0`` (utility is the bare satisfaction indicator). Because every cell here
    carries exactly one constraint of one type, a uniform α rescales all constrained cells
    identically and cannot reorder them — the constraint-type ranking is α-free. Only a
    per-type α could change it, and choosing those weights is exactly the researcher degree of
    freedom the pre-registration exists to constrain.
    """
    vals = [(r["R"] if r["sat"] else 0.0) for r in recs
            if r.get("sat") is not None and r.get("R") is not None
            and not (isinstance(r["R"], float) and math.isnan(r["R"]))]
    return round(sum(vals) / len(vals), 4) if vals else None


def _by(recs, key):
    out = defaultdict(list)
    for r in recs:
        out[r[key]].append(r)
    return out


def aggregate(recs: list[dict]) -> dict:
    normalize_scored_records(recs)
    per_mode = {}
    for mode, rs in _by(recs, "mode").items():
        # an "item" is one combination: a path (association), an analogy PAIR, or a fusion blend
        # (one per prompt, path-level sat). Verified genuine = judge-passed items (utility), the count
        # that isn't inflated by padded near-dupes.
        if mode == "analogy":
            items = [r for r in rs if "pair_sat" in r]                 # pair heads
            verified = sum(1 for r in items if r.get("pair_sat") is True)
            # analogy UTILITY is a PAIR verdict (structural AND factual) on the even head, not per-path
            # sat -- alias pair_sat/pair_channel onto sat/channel so the shared metric helpers read it.
            metric = [{**r, "sat": r.get("pair_sat"), "channel": r.get("pair_channel")} for r in items]
        else:  # association + fusion blending: one path per item, per-path sat
            items = rs
            verified = sum(1 for r in items if r.get("sat") is True)
            metric = rs
        per_mode[mode] = {
            "n_paths": len(rs),
            "n_items": len(items),
            "verified_genuine": verified,
            "R_emit": _mean([r["R"] for r in rs]),
            "R_valid": _mean([r["R"] for r in metric if r.get("sat") is True]),
            "sat_rate": _sat_rate(metric),
            "creativity": _creativity(metric),
            "wf_rate": _mean([1.0 if r["well_formed"] else 0.0 for r in rs]),
            "channels": dict(Counter(r.get("channel") for r in metric)),
            # emergent creativity: mean # of verified emergent inferences per item (heads carry it)
            "emergent_mean": _mean([r["emergent_count"] for r in rs if "emergent_count" in r]),
            # originality: mean item-specific inverse frequency per artifact (heads carry it)
            "originality_mean": _mean([r["originality"] for r in rs
                                       if r.get("originality") is not None]),
        }

    # Within-bundle deltas vs baseline (Regime A only): the 2x2 coordinates.
    two_by_two = {}
    dacc = defaultdict(lambda: {"dR": [], "dsat": [], "dcreat": []})
    for _bundle, brs in _by([r for r in recs if r["regime"] == "A"], "bundle_id").items():
        by_mode = _by(brs, "mode")
        if "baseline" not in by_mode:
            continue
        base_R = _mean([r["R"] for r in by_mode["baseline"]])
        base_sat = _sat_rate(by_mode["baseline"])
        base_creat = _creativity(by_mode["baseline"])
        for mode, mrs in by_mode.items():
            if mode == "baseline":
                continue
            mR, msat, mcreat = _mean([r["R"] for r in mrs]), _sat_rate(mrs), _creativity(mrs)
            if base_R is not None and mR is not None:
                dacc[mode]["dR"].append(mR - base_R)
            if base_sat is not None and msat is not None:
                dacc[mode]["dsat"].append(msat - base_sat)
            if base_creat is not None and mcreat is not None:
                dacc[mode]["dcreat"].append(mcreat - base_creat)
    for mode, d in dacc.items():
        two_by_two[mode] = {
            "mean_dR_emit": _mean(d["dR"]),
            "mean_dsat": _mean(d["dsat"]),
            "mean_dcreativity": _mean(d["dcreat"]),
            "n_bundles": max(len(d["dR"]), len(d["dsat"])),
        }

    return {"per_mode": per_mode, "two_by_two": two_by_two, "n_paths_total": len(recs)}
