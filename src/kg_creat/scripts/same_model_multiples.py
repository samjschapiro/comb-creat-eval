"""Within-model baseline for inventive multiples (rebuttal, reviewer cGKy Q4: "how often do two samples from the
same model at T = 0.9 form multiples?").

Blending only, free: the abstraction causal test's CONTROL condition (data/kg_creat/abstraction_causal/control)
is a second T = 0.9 draw of the paper's blending prompt (same specs, byte-identical prompts) for six models on all
30 items. So each (model, item) has two independent blends: the paper run and the control. We compare, with the
paper's multiples criterion (name-free "relation object" properties, anchor echo removed, one-to-one matching at
cosine >= 0.674, tau = 2):
  within-model   paper draw vs control draw of the SAME model on the same item          (6 x 30 pairs)
  cross-model    two DIFFERENT models among the same six, same item (both draws pooled, one draw per model
                 per pair, every draw combination)                                        (15 x 30 x 4 pairs)
  same-provider  cross-model pairs whose models share a provider (here: gpt-5.2 / gpt-4o-mini)
  reference      the paper's 35-model blending rate (all cross-model pairs)
with an item-bootstrap 95% CI on the within / cross ratio.

    .venv_mlx/bin/python -m src.kg_creat.scripts.same_model_multiples
"""
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.analyze_inventive_multiples import _provider, shared_properties, slot_texts

PAPER = Path("data/kg_creat/kombine_test30/responses")
CTRL = Path("data/kg_creat/abstraction_causal/control/responses")
SWEEP = Path("data/kg_creat/kombine_test30/analysis/multiples_theta_sweep.json")
OUT = Path("data/kg_creat/kombine_test30/analysis/same_model_multiples.json")
THETA, TAU = 0.674, 2


def blends(path):
    out = {}
    for r in json.loads(path.read_text()):
        if r.get("mode") != "blending" or not r.get("items") or r.get("api_error"):
            continue
        it = r["items"][0]
        tags = it.get("tags") or []
        tri = (it.get("paths") or [[]])[0]
        st = [list(t) + [tags[i] if i < len(tags) else ""] for i, t in enumerate(tri)]
        out[r["prompt_id"]] = (r["u_label"], r["v_label"], st, it.get("concept"))
    return out


def main():
    models = sorted(d.name for d in CTRL.iterdir() if (d / "responses.json").exists())
    draws = {}                                   # (model, draw) -> {pid: (u, v, struct, name)}
    for m in models:
        draws[(m, "paper")] = blends(PAPER / m / "responses.json")
        draws[(m, "control")] = blends(CTRL / m / "responses.json")
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    cache = {}

    def mat(rec):
        u, v, st, _ = rec
        txt = [t for t, _ in slot_texts("blending", st, u, v, echo_filter=True)]
        for t in txt:
            if t not in cache:
                cache[t] = un(np.asarray(embed(t), float))
        return np.vstack([cache[t] for t in txt]) if txt else np.zeros((0, 384))

    M = {(m, d, pid): mat(rec) for (m, d), recs in draws.items() for pid, rec in recs.items()}
    items = sorted({pid for (_m, _d, pid) in M}, key=lambda s: int(s[1:]))

    rows = []                                    # one row per compared pair
    for pid in items:
        for m in models:                         # within-model
            a, b = (m, "paper", pid), (m, "control", pid)
            if a in M and b in M:
                rows.append({"kind": "within", "item": pid, "m1": m, "m2": m, "shared": shared_properties(M[a], M[b], tau=THETA),
                             "kmin": min(len(M[a]), len(M[b]))})
        for m1, m2 in itertools.combinations(models, 2):     # cross-model, every draw combination
            for d1, d2 in itertools.product(("paper", "control"), repeat=2):
                a, b = (m1, d1, pid), (m2, d2, pid)
                if a in M and b in M:
                    rows.append({"kind": "same_provider" if _provider(m1) == _provider(m2) else "cross", "item": pid,
                                 "m1": m1, "m2": m2, "shared": shared_properties(M[a], M[b], tau=THETA),
                                 "kmin": min(len(M[a]), len(M[b]))})

    def stats(rs):
        return {"n_pairs": len(rs), "tau1_pct": 100 * float(np.mean([r["shared"] >= 1 for r in rs])),
                "tau2_pct": 100 * float(np.mean([r["shared"] >= TAU for r in rs])),
                "tau3_pct": 100 * float(np.mean([r["shared"] >= 3 for r in rs])),
                "mean_shared": float(np.mean([r["shared"] for r in rs])),
                "per_property": float(np.mean([r["shared"] / r["kmin"] for r in rs if r["kmin"]]))}
    within = [r for r in rows if r["kind"] == "within"]
    cross = [r for r in rows if r["kind"] in ("cross", "same_provider")]
    samep = [r for r in rows if r["kind"] == "same_provider"]
    res = {"models": models, "theta": THETA, "tau": TAU, "within": stats(within), "cross_model": stats(cross),
           "same_provider_pairs": stats(samep) if samep else None,
           "per_model_within_tau2": {m: 100 * float(np.mean([r["shared"] >= TAU for r in within if r["m1"] == m])) for m in models},
           "per_model_within_tau1": {m: 100 * float(np.mean([r["shared"] >= 1 for r in within if r["m1"] == m])) for m in models}}
    ref = next((r for r in json.loads(SWEEP.read_text()) if abs(r["theta"] - THETA) < 1e-9), None) if SWEEP.exists() else None
    res["reference_35_models_blend_tau2_pct"] = ref["blend_tau2"] if ref else None

    # item bootstrap of the within / cross ratio (tau = 2, and tau = 1)
    rng = np.random.default_rng(0)
    by_item = defaultdict(lambda: {"w": [], "c": []})
    for r in within:
        by_item[r["item"]]["w"].append(r["shared"])
    for r in cross:
        by_item[r["item"]]["c"].append(r["shared"])
    its = sorted(by_item)
    for k in (1, 2):
        bs = []
        for _ in range(5000):
            pick = [its[i] for i in rng.integers(0, len(its), len(its))]
            w = np.mean([x >= k for i in pick for x in by_item[i]["w"]])
            c = np.mean([x >= k for i in pick for x in by_item[i]["c"]])
            if c > 0:
                bs.append(w / c)
        point = res["within"][f"tau{k}_pct"] / res["cross_model"][f"tau{k}_pct"] if res["cross_model"][f"tau{k}_pct"] else None
        res[f"within_over_cross_tau{k}"] = {"ratio": point, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    w, c = res["within"], res["cross_model"]
    print(f"{len(models)} models x {len(items)} items  (theta {THETA}, tau {TAU})")
    print(f"  within-model  n={w['n_pairs']:4d}  tau>=1 {w['tau1_pct']:5.1f}%  tau>=2 {w['tau2_pct']:5.1f}%  tau>=3 {w['tau3_pct']:4.1f}%  "
          f"mean shared {w['mean_shared']:.2f}  per-property {w['per_property']:.3f}")
    print(f"  cross-model   n={c['n_pairs']:4d}  tau>=1 {c['tau1_pct']:5.1f}%  tau>=2 {c['tau2_pct']:5.1f}%  tau>=3 {c['tau3_pct']:4.1f}%  "
          f"mean shared {c['mean_shared']:.2f}  per-property {c['per_property']:.3f}")
    if samep:
        s = res["same_provider_pairs"]
        print(f"  same-provider n={s['n_pairs']:4d}  tau>=2 {s['tau2_pct']:5.1f}%")
    print(f"  reference: all 35 models, blending tau>=2 {res['reference_35_models_blend_tau2_pct']}")
    for k in (1, 2):
        x = res[f"within_over_cross_tau{k}"]
        print(f"  within / cross at tau>={k}: {x['ratio']:.1f}x  [{x['ci'][0]:.1f}, {x['ci'][1]:.1f}]")
    print("  per model, within tau>=2 (tau>=1):", ", ".join(f"{m.split('_', 1)[1]} {res['per_model_within_tau2'][m]:.0f}% "
                                                        f"({res['per_model_within_tau1'][m]:.0f}%)" for m in models))
    ex = sorted(within, key=lambda r: -r["shared"])[:6]
    res["examples"] = []
    for r in ex:
        a, b = draws[(r["m1"], "paper")][r["item"]], draws[(r["m1"], "control")][r["item"]]
        res["examples"].append({"model": r["m1"], "item": r["item"], "inputs": a[:2], "shared": r["shared"],
                                "paper": {"name": a[3], "props": [t for t, _ in slot_texts("blending", a[2], a[0], a[1])]},
                                "control": {"name": b[3], "props": [t for t, _ in slot_texts("blending", b[2], b[0], b[1])]}})
        print(f"  e.g. {r['m1'].split('_', 1)[1]} on {a[0]} + {a[1]}: {a[3]!r} vs {b[3]!r}, {r['shared']} shared")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
