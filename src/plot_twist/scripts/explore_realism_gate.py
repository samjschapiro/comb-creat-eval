"""Prototype several realism-GATED composite metrics and check face validity.

Motivation: surprise/coherence earned via genuinely impossible (sci-fi/supernatural)
elements is "cheating" the plot-twist task -- a fantastical reveal is trivially
surprising. So we gate per-story surprise & coherence by realism before aggregating,
and look for a defensible composite where expert humans outrank weak LLMs.

We reuse `div` per source from the already-computed tc.json (gating S/Coh does NOT
change diversity, which is over reveal embeddings), and recompute the gated S/Coh
means over annotations.json applying the SAME keep_story (human strong-only) filter.

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/explore_realism_gate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.plot_twist.sets import twist_types, keep_story

ANN = "data/plot_twist/annotations/annotations.json"
REAL = "data/plot_twist/realism/realism_scores.json"
TC = "data/plot_twist/tc/tc.json"
MANIFEST = "configs/plot_twist/pd_manifest.json"

# Models we EXPECT a defensible metric to rank low (small / weak generators).
WEAK = [
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemma-3-12b-it",
    "google/gemma-3-4b-it",
    "mistralai/mistral-small-3.1-24b-instruct",
    "amazon/nova-micro-v1",
    "openai/gpt-4o-mini",
    "qwen/qwen3-14b",
]


def num(r, k):
    v = r.get("scores", {}).get(k)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def eqz(rows, facets):
    """Equal-weight mean of z-scored facets, added as 'score' (in place copy)."""
    zs = {}
    for k in facets:
        v = np.array([d[k] for d in rows], dtype=float)
        zs[k] = (float(np.nanmean(v)), float(np.nanstd(v)) or 1.0)
    for d in rows:
        d["score"] = float(np.mean([(d[k] - zs[k][0]) / zs[k][1] for k in facets]))
    return rows


def main():
    ann = json.loads(Path(ANN).read_text())
    realism = json.loads(Path(REAL).read_text())
    tc = json.loads(Path(TC).read_text())
    div_by_src = {d["source"]: d["div"] for d in tc}
    raw_real_by_src = {d["source"]: d["mean_realism"] for d in tc}
    kept_sources = set(div_by_src)  # the 72 sources the production pipeline kept

    types = twist_types(MANIFEST)

    # group annotations by source, same filter as make_tc_barplot (strong_only humans),
    # restricted to sources the production run kept (so div lines up).
    groups = {}
    for r in ann:
        if not r.get("reveal"):
            continue
        if num(r, "surprise") is None or num(r, "coherence") is None:
            continue
        if r["id"] not in realism:
            continue
        if not keep_story(r["id"], types, strong_only=True):
            continue
        if r["source"] not in kept_sources:
            continue
        groups.setdefault(r["source"], []).append(r)

    # per-source facet bundles under raw + several gates
    def facets_for(rs):
        S = np.array([num(r, "surprise") for r in rs])
        C = np.array([num(r, "coherence") for r in rs])
        R = np.array([realism[r["id"]] for r in rs])
        g5 = (R >= 5).astype(float)        # hard gate: fully realistic only
        g4 = (R >= 4).astype(float)        # threshold gate: realistic-ish
        soft = R / 5.0                     # soft linear gate
        return {
            "mean_surprise": float(S.mean()),
            "mean_coherence": float(C.mean()),
            "mean_realism": float(R.mean()),
            # hard gate R==5
            "S_g5": float((S * g5).mean()),
            "C_g5": float((C * g5).mean()),
            # threshold gate R>=4
            "S_g4": float((S * g4).mean()),
            "C_g4": float((C * g4).mean()),
            # soft gate R/5
            "S_soft": float((S * soft).mean()),
            "C_soft": float((C * soft).mean()),
            # per-story product, hard-gated  (structural-tc style, but gated)
            "SC_g5": float((S * C * g5).mean()),
        }

    base = {}
    for src, rs in groups.items():
        f = facets_for(rs)
        f["source"] = src
        f["n"] = len(rs)
        f["div"] = div_by_src[src]
        f["mean_realism_raw"] = raw_real_by_src[src]
        base[src] = f

    rows0 = list(base.values())

    VARIANTS = {
        "V0 current  eq-z{S,Coh,Div,Real}":
            (lambda d: d, ["mean_surprise", "mean_coherence", "div", "mean_realism"]),
        "V1 hardgate eq-z{S·1[R=5],Coh·1[R=5],Div}":
            (lambda d: d, ["S_g5", "C_g5", "div"]),
        "V2 hardgate+Real eq-z{gS,gCoh,Div,Real}":
            (lambda d: d, ["S_g5", "C_g5", "div", "mean_realism"]),
        "V3 softgate eq-z{S·R/5,Coh·R/5,Div}":
            (lambda d: d, ["S_soft", "C_soft", "div"]),
        "V4 thresh>=4 eq-z{S·1[R>=4],Coh·1[R>=4],Div}":
            (lambda d: d, ["S_g4", "C_g4", "div"]),
        "V5 gatedprod eq-z{mean(S·Coh·1[R=5]),Div}":
            (lambda d: d, ["SC_g5", "div"]),
    }

    rank_track = {}  # source -> {variant: rank}
    for name, (_, facets) in VARIANTS.items():
        rows = [dict(d) for d in rows0]
        eqz(rows, facets)
        rows.sort(key=lambda d: -d["score"])
        for i, d in enumerate(rows):
            rank_track.setdefault(d["source"], {})[name] = (i + 1, d["score"])

    n = len(rows0)
    print(f"\n{n} sources.  Tracking human rank and worst (max) weak-model rank per variant.")
    print("=" * 100)
    for name in VARIANTS:
        hr, hs = rank_track["human"][name]
        weak_ranks = [(s.split("/")[-1], rank_track[s][name][0]) for s in WEAK if s in rank_track]
        worst_weak = max(r for _, r in weak_ranks) if weak_ranks else None
        best_weak = min(r for _, r in weak_ranks) if weak_ranks else None
        # how many non-human sources outrank humans
        above_human = sum(1 for s, rk in rank_track.items()
                          if s != "human" and rk[name][0] < hr)
        print(f"\n{name}")
        print(f"   human rank = {hr}/{n}  (z={hs:+.3f})   |   {above_human} models above human")
        print(f"   weak models rank in [{best_weak}..{worst_weak}] (lower=better)")
        wk = "  ".join(f"{nm}#{rk}" for nm, rk in sorted(weak_ranks, key=lambda x: x[1]))
        print(f"     {wk}")

    # full top-10 + human + weak for the two most promising variants
    for name in ["V1 hardgate eq-z{S·1[R=5],Coh·1[R=5],Div}",
                 "V2 hardgate+Real eq-z{gS,gCoh,Div,Real}"]:
        _, facets = VARIANTS[name]
        rows = [dict(d) for d in rows0]
        eqz(rows, facets)
        rows.sort(key=lambda d: -d["score"])
        print(f"\n\n### {name} -- top 12 + humans + weak ###")
        print(f"{'rank':>4} {'source':<40}{'score':>8}{'S_g5':>7}{'C_g5':>7}{'div':>7}{'Real':>6}")
        marks = {d["source"]: i + 1 for i, d in enumerate(rows)}
        show = rows[:12] + [d for d in rows if d["source"] == "human"] + \
               [d for d in rows if d["source"] in WEAK]
        seen = set()
        for d in show:
            if d["source"] in seen:
                continue
            seen.add(d["source"])
            tag = "  <-- HUMAN" if d["source"] == "human" else ("  (weak)" if d["source"] in WEAK else "")
            print(f"{marks[d['source']]:>4} {d['source'].split('/')[-1]:<40}{d['score']:>8.3f}"
                  f"{d['S_g5']:>7.2f}{d['C_g5']:>7.2f}{d['div']:>7.3f}{d['mean_realism']:>6.2f}{tag}")


if __name__ == "__main__":
    main()
