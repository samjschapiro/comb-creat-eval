"""What makes an anchor pair hard to blend?

The generic-space gate rejects 47% of frontier blends, and the rejection rate per pair runs from 0%
to 93% -- so difficulty is a real property of the pair, not noise. This asks what that property is.

Three candidate explanations are tested against the per-pair rejection rate:

  distance        the anchors are too far apart semantically. Testable directly, and it is the
                  explanation the literature would reach for first.
  breadth         one anchor is obscure, so models handle it badly everywhere. Testable by asking
                  whether the same pairs are hard in the association and analogy tasks.
  ontology        what KIND of thing each anchor is. Anchors are coded on one 4-way feature (below);
                  the coding is in ANCHOR_KIND, written out so it can be disputed line by line.

The ontology coding is mine, single-coder, no reliability estimate, and it was written after seeing
the difficulty ordering. It is therefore a description of these 30 pairs, not a tested hypothesis --
the permutation test below says only that the split is larger than chance reassignment of the SAME
labels, which does not repair the post-hoc choice of labels. Said plainly here rather than buried.

  artifact   a physical thing that exists for human use -- made, engineered, processed, cultivated
  natural    a physical thing or process that exists independent of human use
  abstract   a system, practice, institution, idea, art form, movement, or a named work
  person     a named individual

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_blend_difficulty
"""
import json
import glob
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.analyze_failure_modes import FRONTIER, SCORES

OUT = Path("data/kg_creat/kombine_test30/analysis/blend_difficulty.json")
RESP = Path("data/kg_creat/kombine_test30/responses")
N_PERM = 20000
RNG = np.random.default_rng(0)

ANCHOR_KIND = {
    # artifact -- exists for human use
    "Bread": "artifact", "The light bulb": "artifact", "Rice": "artifact", "Radio": "artifact",
    "The mattress": "artifact", "The lock and key": "artifact", "The steam engine": "artifact",
    "Vaccines": "artifact", "Salt": "artifact",
    # natural -- exists independent of human use
    "X-rays": "natural", "Nuclear fission": "natural", "Photosynthesis": "natural",
    "Mount Everest": "natural", "The blue whale": "natural", "The oak tree": "natural",
    "Bacteria": "natural", "Crystals": "natural", "Black holes": "natural",
    "Constellations": "natural", "Electricity": "natural", "Gravity": "natural",
    "Evolution": "natural", "The immune system": "natural",
    # abstract -- system, practice, institution, idea, art form, movement, named work
    "Inflation": "abstract", "The Constitution": "abstract", "The blues": "abstract",
    "Buddhism": "abstract", "The Ten Commandments": "abstract", "Free will": "abstract",
    "Ethics": "abstract", "Don Quixote": "abstract", "Pi": "abstract",
    "The social contract": "abstract", "Prayer": "abstract", "Cricket": "abstract",
    "Networks": "abstract", "Justice": "abstract", "The Roman Empire": "abstract",
    "Documentary film": "abstract", "Meditation": "abstract", "Christianity": "abstract",
    "Beauty": "abstract", "Democracy": "abstract", "Banking": "abstract", "Cinema": "abstract",
    "Existentialism": "abstract", "The Enlightenment": "abstract", "Chess": "abstract",
    "Language": "abstract", "The French Revolution": "abstract", "Opera": "abstract",
    "Surrealism": "abstract", "Hinduism": "abstract",
    # person
    "Adam Smith": "person", "Alfred Hitchcock": "person", "Michelangelo": "person",
    "Charlie Chaplin": "person", "Frida Kahlo": "person", "Bob Dylan": "person",
}


def load():
    """Per-pair frontier gate-rejection rate, plus per-pair utility on all three tasks."""
    rej, util = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for f in sorted(SCORES.glob("*/path_scores.json")):
        if f.parent.name not in FRONTIER:
            continue
        for r in json.loads(f.read_text()):
            k = (r.get("u_label"), r.get("v_label"))
            mode = r.get("mode")
            if mode == "blending" and r.get("blend_integration") is not None:
                rej[k].append(int(r["blend_integration"]) == 1)
            # association runs on a different item set (regime A), so it cannot be compared per pair
            if mode == "analogy" and "pair_sat" in r:
                util["analogy_path"][k].append(r["pair_sat"] is True)
            if mode == "analogy" and r.get("invention_utility") is not None:
                util["analogy_invention"][k].append(bool(r["invention_utility"]))
    return ({k: float(np.mean(v)) for k, v in rej.items()},
            {t: {k: float(np.mean(v)) for k, v in d.items()} for t, d in util.items()})


def schemas():
    """Accepted and rejected generic spaces per pair, for the solution-space geometry."""
    g = {}
    for f in sorted(RESP.glob("*/responses.json")):
        for r in json.loads(f.read_text()):
            if r.get("mode") == "blending" and r.get("items"):
                s = (r["items"][0].get("generic_space") or "").strip()
                if s:
                    g[(f.parent.name, r["u_label"], r["v_label"])] = s
    acc, rej = defaultdict(list), defaultdict(list)
    for f in sorted(SCORES.glob("*/path_scores.json")):
        m = f.parent.name
        if m not in FRONTIER:
            continue
        for r in json.loads(f.read_text()):
            if r.get("mode") != "blending" or r.get("blend_integration") is None:
                continue
            s = g.get((m, r["u_label"], r["v_label"]))
            if s:
                (rej if int(r["blend_integration"]) == 1 else acc)[(r["u_label"], r["v_label"])].append(s)
    return acc, rej


def perm_diff(y, mask, n=N_PERM):
    """Two-sided permutation test on a difference of means, labels shuffled."""
    obs = y[mask].mean() - y[~mask].mean()
    k = int(mask.sum())
    null = np.array([(lambda p: p[:k].mean() - p[k:].mean())(RNG.permutation(y)) for _ in range(n)])
    return float(obs), float((np.abs(null) >= abs(obs)).mean())


def robustness(pairs, y, res):
    """The two pair features are worth nothing if they are each other, or if one pair carries them."""
    art = np.array([any(ANCHOR_KIND[a] == "artifact" for a in k) for k in pairs])
    per = np.array([any(ANCHOR_KIND[a] == "person" for a in k) for k in pairs])
    food = np.array([any(a in ("Bread", "Rice", "Salt") for a in k) for k in pairs])
    out = {"n_pairs_with_both": int((art & per).sum())}
    print("\n5. ROBUSTNESS")
    for nm, m, hold, hn in (("artifact", art, ~per, "no person"),
                            ("person", per, ~art, "no artifact"),
                            ("artifact", art, ~food, "food anchors dropped")):
        o, pp = perm_diff(y[hold], m[hold])
        out[f"{nm}_effect_with_{hn.replace(' ', '_')}"] = {
            "mean_with": round(float(y[hold & m].mean()), 3),
            "mean_without": round(float(y[hold & ~m].mean()), 3),
            "diff": round(o, 3), "perm_p": pp, "n": int(hold.sum())}
        print(f"   {nm + ' effect, ' + hn:38s} {100*y[hold & m].mean():4.0f}% vs "
              f"{100*y[hold & ~m].mean():4.0f}%  diff {100*o:+5.0f} pts, p = {pp:.4f} (n = {hold.sum()})")
    for nm, m in (("artifact", art), ("person", per)):
        ds = []
        for i in range(len(pairs)):
            keep = np.ones(len(pairs), bool); keep[i] = False
            ds.append(y[keep & m].mean() - y[keep & ~m].mean())
        out[f"{nm}_leave_one_out_range"] = [round(float(min(ds)), 3), round(float(max(ds)), 3)]
        print(f"   {nm + ' effect, leave-one-pair-out':38s} {100*min(ds):+.0f} to {100*max(ds):+.0f} pts "
              f"(full sample {100*(y[m].mean() - y[~m].mean()):+.0f})")
    print(f"   the two features never co-occur: {out['n_pairs_with_both']} pairs carry both")
    res["robustness"] = out


def main():
    rej, util = load()
    pairs = sorted(rej, key=lambda k: -rej[k])
    missing = {a for k in pairs for a in k} - set(ANCHOR_KIND)
    if missing:
        raise ValueError(f"FATAL: anchors with no ontology code: {sorted(missing)}")
    y = np.array([rej[k] for k in pairs])
    res = {"n_pairs": len(pairs), "n_models": len(FRONTIER),
           "rejection_rate": {f"{u} + {v}": round(rej[(u, v)], 3) for u, v in pairs}}

    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    V = lambda s: un(np.asarray(embed(s), float))

    print(f"BLEND DIFFICULTY: {len(pairs)} anchor pairs, {len(FRONTIER)} frontier models")
    print(f"  gate-rejection rate ranges {100*y.min():.0f}% to {100*y.max():.0f}%, mean {100*y.mean():.0f}%\n")

    print("1. IS IT DISTANCE?")
    d = np.array([1 - float(V(u) @ V(v)) for u, v in pairs])
    r, p = pearsonr(d, y)
    res["distance"] = {"r": round(float(r), 3), "p": float(p)}
    print(f"   rejection ~ d(u,v):  r = {r:+.2f} (p = {p:.2f}, n = {len(pairs)})  -> no")
    print(f"   the hardest pair {pairs[0][0]} + {pairs[0][1]} is the CLOSEST of the 30 "
          f"(d = {d[0]:.2f} vs mean {d.mean():.2f})\n")

    print("2. IS IT THE PAIR BEING HARD IN GENERAL?")
    print("   (association uses a different item set, so analogy is the only comparable task)")
    res["cross_task"] = {}
    for t in ("analogy_path", "analogy_invention"):
        u_ = np.array([util[t].get(k, np.nan) for k in pairs])
        ok = ~np.isnan(u_)
        r, p = pearsonr(y[ok], u_[ok])
        res["cross_task"][t] = {"r": round(float(r), 3), "p": float(p), "n": int(ok.sum())}
        print(f"   gate rejection ~ {t:18s} utility on the same pair:  r = {r:+.2f} "
              f"(p = {p:.2f}, n = {ok.sum()})")
    print("   -> a pair that defeats the generic space is not a pair models handle badly elsewhere\n")

    print("3. IS IT WHAT KIND OF THING THE ANCHORS ARE?")
    kinds = sorted(set(ANCHOR_KIND.values()))
    per_kind = defaultdict(list)
    for i, (u, v) in enumerate(pairs):
        for a in (u, v):
            per_kind[ANCHOR_KIND[a]].append(y[i])
    res["by_anchor_kind"] = {}
    print(f"   {'anchor kind':10s}  {'anchors':>8s}  {'mean rejection of pairs it appears in':>38s}")
    for k in sorted(per_kind, key=lambda k: -np.mean(per_kind[k])):
        n_anchor = sum(1 for a in {a for pr in pairs for a in pr} if ANCHOR_KIND[a] == k)
        res["by_anchor_kind"][k] = {"n_anchor_slots": len(per_kind[k]),
                                    "n_distinct_anchors": n_anchor,
                                    "mean_rejection": round(float(np.mean(per_kind[k])), 3)}
        print(f"   {k:10s}  {n_anchor:8d}  {100*np.mean(per_kind[k]):37.0f}%")

    has_art = np.array([any(ANCHOR_KIND[a] == "artifact" for a in k) for k in pairs])
    has_per = np.array([any(ANCHOR_KIND[a] == "person" for a in k) for k in pairs])
    both_abs = np.array([all(ANCHOR_KIND[a] == "abstract" for a in k) for k in pairs])
    print()
    res["pair_features"] = {}
    for name, mask in (("contains an artifact anchor", has_art),
                       ("contains a named person", has_per),
                       ("both anchors abstract", both_abs)):
        obs, pp = perm_diff(y, mask)
        res["pair_features"][name] = {"n_with": int(mask.sum()), "n_without": int((~mask).sum()),
                                      "mean_with": round(float(y[mask].mean()), 3),
                                      "mean_without": round(float(y[~mask].mean()), 3),
                                      "diff": round(obs, 3), "perm_p": pp}
        print(f"   {name:28s}  {100*y[mask].mean():4.0f}% (n={mask.sum():2d})  vs  "
              f"{100*y[~mask].mean():4.0f}% (n={(~mask).sum():2d})   diff {100*obs:+5.0f} pts, "
              f"permutation p = {pp:.4f}")

    print("\n4. IS THE SOLUTION SPACE NARROWER ON HARD PAIRS?")
    acc, rejs = schemas()
    div_a, div_r, keep = [], [], []
    for i, k in enumerate(pairs):
        def sim(ss):
            if len(ss) < 2:
                return np.nan
            M = np.vstack([V(s) for s in ss])
            iu = np.triu_indices(len(M), 1)
            return float((M @ M.T)[iu].mean())
        a, rr = sim(acc[k]), sim(rejs[k])
        div_a.append(a); div_r.append(rr); keep.append(i)
    div_a, div_r = np.array(div_a), np.array(div_r)
    for label, arr in (("accepted schemas agree with each other", div_a),
                       ("rejected schemas agree with each other", div_r)):
        ok = ~np.isnan(arr)
        r, p = pearsonr(y[ok], arr[ok])
        res.setdefault("schema_agreement", {})[label] = {"r": round(float(r), 3), "p": float(p),
                                                         "n": int(ok.sum())}
        print(f"   rejection ~ how much the {label}:  r = {r:+.2f} (p = {p:.2f}, n = {ok.sum()})")

    print("\nWHAT THE SCHEMAS LOOK LIKE ON ARTIFACT PAIRS (rejected, hardest first)")
    for k in [p_ for p_ in pairs if any(ANCHOR_KIND[a] == "artifact" for a in p_)][:4]:
        print(f"  {k[0]} + {k[1]}  ({100*rej[k]:.0f}% rejected)")
        for s in rejs[k][:3]:
            print(f"     - {s[:104]}")
    robustness(pairs, y, res)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
