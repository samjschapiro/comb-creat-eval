"""Rebuttal (reviewer cGKy): "The panel includes evaluated models (self-preference?)".

The subjective-dimension panel is anthropic/claude-haiku-4.5, openai/o3 and openai/gpt-5.4 (Haiku 4.5 is
also the single factuality judge F). None of the three is among the 35 evaluated models, but the pool
holds same-family models (10 Anthropic, 7 OpenAI), so the question is family self-preference.
Using only stored per-judge votes (no API calls), from data/kg_creat/human_study/paper_scores:

  1. leave-one-judge-out leaderboards: the 35-model ranking re-derived from two judges (AND = strict,
     OR = lenient; scope min / max) and from each single judge, vs the paper's panel ranking.
  2. own-family bias (difference-in-differences): per judged unit, d = judge's vote minus the mean vote
     of the OTHER-family judge(s) on the same unit; DiD = mean d on the judge's own-family models minus
     mean d on third-party models (neither Anthropic nor OpenAI). Cluster bootstrap over models.
  3. family-excluded panel: each model's answers judged only by judges from OTHER families (Anthropic
     models by o3 + gpt-5.4, OpenAI models by Haiku, everyone else by the full panel); rank changes.
  4. factuality: Haiku 4.5 (paper) vs the earlier gpt-oss-120b verdicts (factual_gptoss) on the same
     triples. gpt-oss is itself an OpenAI model, so the DiD is reported for both families.

Writes data/kg_creat/kombine_test30/analysis/rebuttal_judge_self_preference.json.

    .venv/bin/python -m src.kg_creat.scripts.rebuttal_judge_self_preference
"""
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

from src.kg_creat.scripts.compute_composite import TASK_DIMS, _mean, _ok, artifact_dims
from src.kg_creat.scripts.score import finalize_sat

SCORES = Path("data/kg_creat/human_study/paper_scores")
OUT = Path("data/kg_creat/kombine_test30/analysis/rebuttal_judge_self_preference.json")
HAIKU, O3, G54 = "anthropic/claude-haiku-4.5", "openai/o3", "openai/gpt-5.4"
JUDGES = [HAIKU, O3, G54]
JFAM = {HAIKU: "anthropic", O3: "openai", G54: "openai"}
N_BOOT = 5000
SEED = 0


def family(model_dir):
    p = model_dir.split("_")[0]
    return p if p in ("anthropic", "openai") else "third_party"


# ------------------------------------------------------------------ panel re-aggregation
def agg_bool(votes, rule):
    v = [bool(x) for x in votes if x is not None]
    if not v:
        return None
    if rule == "majority":
        return sum(v) > len(v) / 2
    return all(v) if rule == "strict" else any(v)


def agg_scope(votes, rule):
    v = [x for x in votes if x is not None]
    if not v:
        return None
    if rule == "majority":
        return Counter(v).most_common(1)[0][0]
    return min(v) if rule == "strict" else max(v)


def reaggregate(recs, judges_for, rule):
    """Recompute the panel-derived fields from stored votes. judges_for(model_rec) -> list of judge ids."""
    recs = copy.deepcopy(recs)
    for r in recs:
        if r.get("mode") == "blending" and r.get("blend_judges"):
            js = [j for j in r["blend_judges"] if j.get("model") in judges_for]
            r["semantic_sat"] = agg_bool([j.get("generic_ok") for j in js], rule)
            r["blend_utility"] = agg_bool([j.get("coherent") for j in js], rule)
            r["blend_integration"] = agg_scope([j.get("scope") for j in js], rule)
            finalize_sat(r)
        elif r.get("mode") == "analogy" and r.get("invention_judges") and "pair_sat" in r:
            js = [j for j in r["invention_judges"] if j.get("model") in judges_for]
            r["invention_utility"] = agg_bool([j.get("coherent") for j in js], rule)
            r["invention_integration"] = agg_bool([j.get("valid") for j in js], rule)
    return recs


def overall(recs):
    raw = artifact_dims(recs)
    per = [100 * _mean([raw[t][k] for k in ds if _ok(raw[t].get(k))]) for t, ds in TASK_DIMS.items()]
    return _mean(per)


def rank_compare(base, alt, models):
    b, a = np.array([base[m] for m in models]), np.array([alt[m] for m in models])
    rb, ra = (-b).argsort().argsort(), (-a).argsort().argsort()
    return {"spearman": float(spearmanr(b, a)[0]), "kendall": float(kendalltau(b, a)[0]),
            "top1_same": bool(np.argmax(b) == np.argmax(a)),
            "top5_overlap": len(set(np.argsort(-b)[:5]) & set(np.argsort(-a)[:5])),
            "max_rank_shift": int(np.abs(rb - ra).max()),
            "rank_shift": {m: int(rb[i] - ra[i]) for i, m in enumerate(models)}}   # + = moved up


# ------------------------------------------------------------------ judged units for the DiD
def units(recs_by_model):
    """One row per judged unit: (model, family, dim, {judge: vote})."""
    rows = []
    for m, recs in recs_by_model.items():
        fam = family(m)
        seen = set()
        for r in recs:
            if r.get("mode") == "blending" and r.get("blend_judges"):
                js = {j["model"]: j for j in r["blend_judges"]}
                for dim, f in (("blend.generic_ok", lambda j: j.get("generic_ok")),
                               ("blend.coherent", lambda j: j.get("coherent")),
                               ("blend.double_scope", lambda j: None if j.get("scope") is None else j["scope"] >= 2),
                               ("blend.emergent", lambda j: None if j.get("scope") is None else j["scope"] == 3)):
                    rows.append((m, fam, dim, {k: f(v) for k, v in js.items()}))
            elif r.get("mode") == "analogy" and r.get("invention_judges"):
                key = (r["prompt_id"], r["sample_idx"], r.get("pair_idx"))
                if key in seen:
                    continue
                seen.add(key)
                js = {j["model"]: j for j in r["invention_judges"]}
                rows.append((m, fam, "analogy.valid", {k: v.get("valid") for k, v in js.items()}))
                rows.append((m, fam, "analogy.coherent", {k: v.get("coherent") for k, v in js.items()}))
    return rows


def did(rows, judge, dims, rng):
    """Cluster-bootstrapped (over models) DiD for one judge, pooled over `dims`."""
    others = [j for j in JUDGES if JFAM[j] != JFAM[judge]]
    own = JFAM[judge]
    per_model = defaultdict(list)            # model -> list of d
    fam_of = {}
    for m, fam, dim, votes in rows:
        if dim not in dims:
            continue
        v, o = votes.get(judge), [votes.get(j) for j in others]
        o = [float(x) for x in o if x is not None]
        if v is None or not o:
            continue
        per_model[m].append(float(v) - float(np.mean(o)))
        fam_of[m] = fam
    own_m = [m for m in per_model if fam_of[m] == own]
    tp_m = [m for m in per_model if fam_of[m] == "third_party"]
    oth_m = [m for m in per_model if fam_of[m] != own]

    def stat(ms_own, ms_ref):
        a = np.concatenate([per_model[m] for m in ms_own]); b = np.concatenate([per_model[m] for m in ms_ref])
        return a.mean() - b.mean(), a.mean(), b.mean()

    est, d_own, d_tp = stat(own_m, tp_m)
    est_all, _, d_oth = stat(own_m, oth_m)
    boots = [stat(list(rng.choice(own_m, len(own_m))), list(rng.choice(tp_m, len(tp_m))))[0] for _ in range(N_BOOT)]
    return {"judge": judge, "own_family": own, "dims": dims, "n_own_models": len(own_m), "n_third_party_models": len(tp_m),
            "mean_d_own": float(d_own), "mean_d_third_party": float(d_tp), "did_vs_third_party": float(est),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_two_sided_boot": float(2 * min(np.mean(np.array(boots) <= 0), np.mean(np.array(boots) >= 0))),
            "did_vs_all_other": float(est_all)}


def pass_rates(rows):
    t = defaultdict(lambda: defaultdict(list))
    for m, fam, dim, votes in rows:
        for j in JUDGES:
            if votes.get(j) is not None:
                t[(dim, j)][fam].append(float(votes[j]))
    return {f"{dim} | {j}": {fam: round(100 * float(np.mean(v)), 1) for fam, v in sorted(d.items())}
            for (dim, j), d in sorted(t.items())}


# ------------------------------------------------------------------ factuality: haiku vs gpt-oss
def factuality(recs_by_model, rng):
    per_model = {}
    for m, recs in recs_by_model.items():
        h, g = [], []
        for r in recs:
            if r.get("mode") not in ("baseline", "analogy"):
                continue
            a, b = r.get("factual"), r.get("factual_gptoss")
            if not a or not b or len(a) != len(b):
                continue
            for x, y in zip(a, b):
                if x is not None and y is not None:
                    h.append(float(x)); g.append(float(y))
        if h:
            per_model[m] = (np.array(h), np.array(g))
    fam_of = {m: family(m) for m in per_model}
    rates = {}
    for fam in ("anthropic", "openai", "third_party"):
        ms = [m for m in per_model if fam_of[m] == fam]
        H = np.concatenate([per_model[m][0] for m in ms]); G = np.concatenate([per_model[m][1] for m in ms])
        rates[fam] = {"n_models": len(ms), "n_triples": int(len(H)), "haiku_true_pct": round(100 * H.mean(), 1),
                      "gptoss_true_pct": round(100 * G.mean(), 1), "agreement_pct": round(100 * (H == G).mean(), 1)}

    def d_of(ms):
        return np.mean(np.concatenate([per_model[m][0] - per_model[m][1] for m in ms]))
    out = {"rates": rates}
    for fam in ("anthropic", "openai"):
        own = [m for m in per_model if fam_of[m] == fam]
        tp = [m for m in per_model if fam_of[m] == "third_party"]
        est = d_of(own) - d_of(tp)
        boots = [d_of(list(rng.choice(own, len(own)))) - d_of(list(rng.choice(tp, len(tp)))) for _ in range(N_BOOT)]
        out[f"did_{fam}_vs_third_party"] = {
            "meaning": "(haiku - gptoss) on this family's models minus on third-party models; + = Haiku relatively "
                       "more lenient on this family (for openai: equivalently gpt-oss relatively harsher on its own family)",
            "estimate_pp": round(100 * float(est), 2),
            "ci95_pp": [round(100 * float(np.percentile(boots, 2.5)), 2), round(100 * float(np.percentile(boots, 97.5)), 2)]}
    return out


def main():
    rng = np.random.default_rng(SEED)
    models = sorted(d.name for d in SCORES.iterdir()
                    if (d / "path_scores.json").exists() and not d.name.startswith("human"))
    recs = {m: json.loads((SCORES / m / "path_scores.json").read_text()) for m in models}
    paper = json.loads((SCORES / "composite.json").read_text())["per_model"]
    base = {m: paper[m]["overall"] for m in models}
    fams = Counter(family(m) for m in models)
    print(f"{len(models)} models: {dict(fams)}; judges {JUDGES}; none of the judges is an evaluated model: "
          f"{not any(j.split('/')[1].replace('.', '-') in m for j in JUDGES for m in models)}")

    # sanity: re-aggregating the full panel from stored votes reproduces the paper composite
    rep = {m: overall(reaggregate(recs[m], JUDGES, "majority")) for m in models}
    diff = max(abs(rep[m] - base[m]) for m in models)
    print(f"full-panel re-aggregation reproduces paper composite: max |diff| = {diff:.2e}")
    res = {"n_models": len(models), "families": dict(fams), "panel": JUDGES, "reproduction_max_abs_diff": diff,
           "leave_one_out": {}}

    # 1. leave-one-judge-out and single-judge leaderboards
    print(f"\n1. leaderboards without a judge   {'spearman':>8s} {'kendall':>8s} {'top1':>5s} {'top5':>5s} {'maxΔ':>5s}")
    configs = {}
    for drop in JUDGES:
        keep = [j for j in JUDGES if j != drop]
        for rule in ("strict", "lenient"):
            configs[f"drop {drop.split('/')[1]} ({rule})"] = (keep, rule)
    for j in JUDGES:
        configs[f"only {j.split('/')[1]}"] = ([j], "majority")
    for name, (keep, rule) in configs.items():
        alt = {m: overall(reaggregate(recs[m], keep, rule)) for m in models}
        c = rank_compare(base, alt, models)
        # if the dropped judge favoured its own family, dropping it for EVERY model moves that family down
        c["mean_rank_shift_by_family"] = {f: round(float(np.mean([c["rank_shift"][m] for m in models if family(m) == f])), 2)
                                          for f in fams}
        res["leave_one_out"][name] = c
        print(f"   {name:31s} {c['spearman']:8.3f} {c['kendall']:8.3f} {str(c['top1_same']):>5s} {c['top5_overlap']:>5d} {c['max_rank_shift']:>5d}"
              f"   mean rank shift: anthropic {c['mean_rank_shift_by_family']['anthropic']:+.2f}, openai {c['mean_rank_shift_by_family']['openai']:+.2f}, third-party {c['mean_rank_shift_by_family']['third_party']:+.2f}")

    # 2. own-family DiD
    rows = units(recs)
    res["pass_rates_pct"] = pass_rates(rows)
    all_dims = sorted({r[2] for r in rows})
    res["did"] = {}
    print("\n2. own-family bias (DiD vs third-party models; + = judge relatively more lenient on its own family)")
    for judge in JUDGES:
        for dims in [all_dims] + [[d] for d in all_dims]:
            key = f"{judge.split('/')[1]} | {'ALL' if len(dims) > 1 else dims[0]}"
            r = did(rows, judge, dims, rng)
            res["did"][key] = r
            print(f"   {key:36s} DiD {100 * r['did_vs_third_party']:+6.2f} pp  95% CI [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}]"
                  f"  (own {100 * r['mean_d_own']:+.1f}, third-party {100 * r['mean_d_third_party']:+.1f})")

    # 3. family-excluded panel
    def judges_excluding_family(m):
        fam = family(m)
        return [j for j in JUDGES if JFAM[j] != fam] if fam in ("anthropic", "openai") else JUDGES
    res["family_excluded"] = {}
    print("\n3. family-excluded panel (each model judged only by other-family judges)")
    for rule in ("strict", "lenient"):
        alt = {m: overall(reaggregate(recs[m], judges_excluding_family(m), rule if family(m) == "anthropic" else "majority"))
               for m in models}
        c = rank_compare(base, alt, models)
        fam_shift = {f: float(np.mean([c["rank_shift"][m] for m in models if family(m) == f])) for f in fams}
        c["mean_rank_shift_by_family"] = fam_shift
        c["overall_delta_by_family"] = {f: float(np.mean([alt[m] - base[m] for m in models if family(m) == f])) for f in fams}
        res["family_excluded"][rule] = c
        print(f"   tie rule for Anthropic models = {rule:8s} Spearman {c['spearman']:.3f}, Kendall {c['kendall']:.3f}, "
              f"top-1 same {c['top1_same']}, top-5 overlap {c['top5_overlap']}, max |rank shift| {c['max_rank_shift']}; "
              f"mean rank shift by family {({k: round(v, 2) for k, v in fam_shift.items()})}")

    # 4. factuality
    res["factuality"] = factuality(recs, rng)
    print("\n4. factuality, Haiku 4.5 vs gpt-oss-120b on the same triples")
    for fam, v in res["factuality"]["rates"].items():
        print(f"   {fam:12s} {v['n_models']:2d} models {v['n_triples']:6d} triples: Haiku true {v['haiku_true_pct']}%, "
              f"gpt-oss true {v['gptoss_true_pct']}%, agree {v['agreement_pct']}%")
    for k in ("did_anthropic_vs_third_party", "did_openai_vs_third_party"):
        v = res["factuality"][k]
        print(f"   {k}: {v['estimate_pp']:+.2f} pp, 95% CI {v['ci95_pp']}")

    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
