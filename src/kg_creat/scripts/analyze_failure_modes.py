"""How models fail on each Kombine task, by channel, per model.

The leaderboard says how much a model scores; this says what goes wrong. Every scored artifact carries
the gate that rejected it, so the breakdown is a count, not a judgement call:

  association / analogy path   ``channel`` in {ok, factual, structural} -- a hallucinated triple in the
                               path, or a malformed path (discontinuous, revisits a node)
  analogy invention h          ``invention_utility`` (was the mapping actually applied?) and
                               ``invention_integration`` (does the invention hold together?)
  blending                     ``channel`` "semantic" = the 3-judge panel rejected the generic space:
                               a schema only one input instantiates. ``blend_integration`` is scope
                               1/2/3 and ``blend_utility`` is coherence, both scored past that gate.

Reports the pool as a whole and a FRONTIER subset (the recent flagships), since the two answer
different questions: what the field does, versus what the best available models still get wrong.

    .venv/bin/python -m src.kg_creat.scripts.analyze_failure_modes
"""
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCORES = Path("data/kg_creat/kombine_test30/scores")
OUT = Path("data/kg_creat/kombine_test30/analysis/failure_modes.json")
# the recent flagships, named so the subset is auditable rather than "the good ones"
FRONTIER = {"openai_gpt-5", "openai_gpt-5-2", "openai_gpt-5-6-sol", "anthropic_claude-opus-4-5",
            "anthropic_claude-opus-4-6", "anthropic_claude-opus-5", "anthropic_claude-fable-5",
            "anthropic_claude-sonnet-5", "google_gemini-3-1-pro-preview", "google_gemini-3-7-flash",
            "google_gemini-3-flash-preview", "x-ai_grok-4-5", "x-ai_grok-4-6", "deepseek_deepseek-r1",
            "z-ai_glm-4-6"}


def load():
    per = defaultdict(list)
    for f in sorted(SCORES.glob("*/path_scores.json")):
        model = f.parent.name
        for r in json.loads(f.read_text()):
            per[model].append(r)
    return per


def rates(recs, subset):
    """Channel shares per task, plus the two invention gates, over the models in `subset`."""
    out = {}
    paths = {"association": [], "analogy": [], "blending": []}
    inv, blend = [], []
    for model in subset:
        for r in recs[model]:
            mode = r.get("mode")
            if mode == "baseline" and r.get("triples"):
                paths["association"].append(r)
            elif mode == "analogy":
                if "pair_sat" in r:
                    paths["analogy"].append(r)
                    if r.get("invention_utility") is not None:
                        inv.append(r)
            elif mode == "blending" and r.get("triples"):
                paths["blending"].append(r)
                if r.get("blend_integration") is not None:
                    blend.append(r)
    for task, rs in paths.items():
        c = Counter(r.get("channel") for r in rs)
        n = len(rs)
        out[task] = {"n": n, **{k: round(100 * v / n, 1) for k, v in c.items()}}
    out["analogy_invention"] = {
        "n": len(inv),
        "mapping_not_applied_pct": round(100 * np.mean([not r["invention_utility"] for r in inv]), 1),
        "incoherent_pct": round(100 * np.mean([not r.get("invention_integration") for r in inv]), 1)}
    scope = Counter(int(r["blend_integration"]) for r in blend)
    passed = [r for r in blend if r.get("sat") is True]
    out["blending_detail"] = {
        "n": len(blend),
        "scope_1_pct": round(100 * scope[1] / len(blend), 1),
        "scope_2_pct": round(100 * scope[2] / len(blend), 1),
        "scope_3_pct": round(100 * scope[3] / len(blend), 1),
        "coherent_given_passed_pct": round(100 * np.mean([bool(r.get("blend_utility")) for r in passed]), 1),
        "scope3_given_passed_pct": round(100 * np.mean([int(r["blend_integration"]) == 3 for r in passed]), 1)}
    return out


def per_model(recs, subset, task, field):
    out = {}
    for model in sorted(subset):
        if task == "blending":
            rs = [r for r in recs[model] if r.get("mode") == "blending" and r.get("triples")]
            if rs:
                out[model] = round(100 * np.mean([r.get("channel") == "semantic" for r in rs]), 1)
        else:
            rs = [r for r in recs[model] if r.get("mode") == "analogy" and r.get(field) is not None]
            if rs:
                out[model] = round(100 * np.mean([not r[field] for r in rs]), 1)
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def gate_probes(recs):
    """Two checks on the gates themselves, because a gate that fails 47% of artifacts is worth
    auditing before it is believed.

    BLEND: the panel rejects a generic space for being "unequally instantiated". If that were a
    proximity fact, a rejected `g` would sit measurably closer to one input than the other. It does
    not -- so the gate is detecting instantiation, which the embedding does not see, and cannot be
    replaced by a cheap cosine. What the geometry DOES show is a position effect: `g` leans toward the
    second-listed anchor whether or not the blend passes.

    ANALOGY: incoherence is not explained by how long the projection is or how remote the source
    concept is. The one visible predictor is a degenerate source -- a phi that is already an entity in
    the aligned paths, which leaves nothing to project.
    """
    from scipy.stats import mannwhitneyu, wilcoxon
    from src.kg_creat.embed import get_embedder
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    cache = {}

    def V(s):
        if s not in cache:
            cache[s] = un(np.asarray(embed(str(s)), float))
        return cache[s]

    resp = Path("data/kg_creat/kombine_test30/responses")
    g_by, an_by = {}, {}
    for f in sorted(resp.glob("*/responses.json")):
        m = f.parent.name
        for r in json.loads(f.read_text()):
            if not r.get("items"):
                continue
            k = (m, r.get("u_label"), r.get("v_label"))
            if r.get("mode") == "blending":
                g = (r["items"][0].get("generic_space") or "").strip()
                if g:
                    g_by[k] = g
            elif r.get("mode") == "analogy":
                an_by[k] = (r["items"][0], r.get("paths") or [])

    asym, signed, scope = [], [], []
    for model, rs in recs.items():
        for r in rs:
            if r.get("mode") != "blending" or r.get("blend_integration") is None:
                continue
            g = g_by.get((model, r["u_label"], r["v_label"]))
            if not g:
                continue
            du, dv = 1 - float(V(r["u_label"]) @ V(g)), 1 - float(V(r["v_label"]) @ V(g))
            asym.append(abs(du - dv)); signed.append(du - dv); scope.append(int(r["blend_integration"]))
    asym, signed, scope = np.array(asym), np.array(signed), np.array(scope)
    s1, s23 = asym[scope == 1], asym[scope >= 2]
    p_asym = mannwhitneyu(s1, s23).pvalue
    out = {"blend_asymmetry": {
        "scope1_mean": round(float(s1.mean()), 4), "scope23_mean": round(float(s23.mean()), 4),
        "n_scope1": int(len(s1)), "n_scope23": int(len(s23)), "mannwhitney_p": float(p_asym),
        "second_anchor_closer_pct": round(100 * float(np.mean(signed > 0)), 1),
        "signed_mean": round(float(signed.mean()), 4), "wilcoxon_p": float(wilcoxon(signed).pvalue)}}
    print(f"\nBLEND GATE PROBE (n = {len(asym)} blends with a generic space)")
    print(f"  |d(u,g) - d(v,g)|: rejected {s1.mean():.4f} vs accepted {s23.mean():.4f}  "
          f"Mann-Whitney p = {p_asym:.2f}  -> the gate is NOT distance asymmetry")
    print(f"  position effect: g sits closer to the SECOND anchor in "
          f"{100*np.mean(signed > 0):.1f}% of blends (Wilcoxon p = {wilcoxon(signed).pvalue:.1e})")

    n_proj, phi_in, coh, mok = [], [], [], []
    for model, rs in recs.items():
        for r in rs:
            if r.get("mode") != "analogy" or r.get("invention_utility") is None:
                continue
            k = (model, r["u_label"], r["v_label"])
            if k not in an_by:
                continue
            it, paths = an_by[k]
            ents = {str(e).lower() for p_ in paths for tp in p_ if len(tp) == 3 for e in (tp[0], tp[2])}
            n_proj.append(len(it.get("projection") or []))
            phi_in.append(str(it.get("projected") or "").lower() in ents)
            coh.append(bool(r.get("invention_integration"))); mok.append(bool(r["invention_utility"]))
    n_proj, phi_in, coh, mok = map(np.array, (n_proj, phi_in, coh, mok))
    out["analogy_incoherence"] = {
        "n": int(len(coh)),
        "proj_len_coherent": round(float(n_proj[coh].mean()), 2),
        "proj_len_incoherent": round(float(n_proj[~coh].mean()), 2),
        "proj_len_p": float(mannwhitneyu(n_proj[coh], n_proj[~coh]).pvalue),
        "coherent_pct_when_phi_in_paths": round(100 * float(coh[phi_in].mean()), 1),
        "n_phi_in_paths": int(phi_in.sum()),
        "coherent_pct_otherwise": round(100 * float(coh[~phi_in].mean()), 1),
        "coherent_pct_when_mapping_applied": round(100 * float(coh[mok].mean()), 1),
        "coherent_pct_when_not": round(100 * float(coh[~mok].mean()), 1)}
    a = out["analogy_incoherence"]
    print(f"\nANALOGY INVENTION PROBE (n = {a['n']})")
    print(f"  projection length: coherent {a['proj_len_coherent']} vs incoherent "
          f"{a['proj_len_incoherent']} (p = {a['proj_len_p']:.2f}) -> not it")
    print(f"  degenerate source (phi already an entity in the paths, n = {a['n_phi_in_paths']}): "
          f"coherent {a['coherent_pct_when_phi_in_paths']}% vs {a['coherent_pct_otherwise']}% otherwise")
    return out


def main():
    recs = load()
    everyone = set(recs)
    frontier = FRONTIER & everyone
    missing = FRONTIER - everyone
    if missing:
        raise ValueError(f"FATAL: frontier list names models with no scores: {sorted(missing)}")
    res = {"n_models_scored": len(everyone), "n_frontier": len(frontier),
           "frontier": sorted(frontier), "all_models": rates(recs, everyone),
           "frontier_models": rates(recs, frontier)}

    for label, block in (("ALL SCORED MODELS", res["all_models"]), ("FRONTIER ONLY", res["frontier_models"])):
        print(f"\n{label}  ({len(everyone) if label.startswith('ALL') else len(frontier)} models)")
        for task in ("association", "analogy", "blending"):
            b = block[task]
            chans = "  ".join(f"{k} {v}%" for k, v in b.items() if k != "n")
            print(f"  {task:12s} n={b['n']:5d}   {chans}")
        i, d = block["analogy_invention"], block["blending_detail"]
        print(f"  analogy invention  n={i['n']:4d}   mapping not applied {i['mapping_not_applied_pct']}%"
              f"   incoherent {i['incoherent_pct']}%")
        print(f"  blending detail    n={d['n']:4d}   scope1 {d['scope_1_pct']}%  scope2 {d['scope_2_pct']}%"
              f"  scope3 {d['scope_3_pct']}%   | past the gate: coherent {d['coherent_given_passed_pct']}%,"
              f" scope-3 {d['scope3_given_passed_pct']}%")

    res["per_model_generic_space_failure"] = per_model(recs, frontier, "blending", None)
    res["per_model_mapping_not_applied"] = per_model(recs, frontier, "analogy", "invention_utility")
    print("\nGENERIC-SPACE FAILURE RATE, frontier models (blending)")
    for m, v in res["per_model_generic_space_failure"].items():
        print(f"  {m:36s} {v:5.1f}%")
    print("\nMAPPING-NOT-APPLIED RATE, frontier models (analogy invention)")
    for m, v in res["per_model_mapping_not_applied"].items():
        print(f"  {m:36s} {v:5.1f}%")

    res["gate_probes"] = gate_probes(recs)

    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
