"""theta sweep for inventive multiples (rebuttal, reviewer cGKy Q4): every headline finding at each property-match
cosine theta, on the paper's pipeline (analyze_inventive_multiples: name-free "relation object" properties,
anchor echo removed, one-to-one maximum matching). Read-only with respect to inventive_multiples.json.

At each theta and tau = 2 (and tau = 3 for the collapse):
  rate of multiple pairs (overall / blending / analogy) and inventions in >= 1 multiple;
  blending / analogy: share of INVENTIONS in >= 1 multiple (the paper's 14.3x), pair rate at fixed tau, and per property (mean shared/kmin, averaged within item then over items);
  same-provider / cross-provider rate ratio;
  a cross-item null (pairs answering DIFFERENT anchors, same task; 30k per task, seed 0): % at tau >= 1 and >= 2.

    .venv_mlx/bin/python -m src.kg_creat.scripts.multiples_theta_sweep
"""
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.analyze_inventive_multiples import (NPZ, _provider, load_records, shared_properties,
                                                             slot_texts)

THETAS = (0.55, 0.60, 0.62, 0.65, 0.674, 0.70, 0.72, 0.75)
OUT = Path("data/kg_creat/kombine_test30/analysis/multiples_theta_sweep.json")
N_NULL = 30000


def main():
    d = np.load(NPZ, allow_pickle=True)
    names, tk, us, vs, mo = d["names"], d["tasks"], d["u"], d["v"], d["models"]
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    _con_txt, _con_emb, struct, _rels = load_records()
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    vec = {}
    S = {}
    for i in range(len(names)):
        st = slot_texts(str(tk[i]), struct.get((str(tk[i]), str(us[i]), str(vs[i]), str(mo[i])), []),
                        str(us[i]), str(vs[i]), echo_filter=True)
        for t, _ in st:
            if t not in vec:
                vec[t] = un(np.asarray(embed(t), float))
        S[i] = np.vstack([vec[t] for t, _ in st]) if st else np.zeros((0, 384))
    groups = defaultdict(list)
    for i in range(len(names)):
        groups[(str(tk[i]), str(us[i]), str(vs[i]))].append(i)
    pairs = [(task, (u, v), a, b, _provider(mo[a]) == _provider(mo[b]), min(len(S[a]), len(S[b])))
             for (task, u, v), idx in groups.items() for a, b in itertools.combinations(idx, 2)]
    rng = np.random.default_rng(0)
    null = []
    for task in ("analogy", "blending"):
        idx = [i for i in range(len(names)) if str(tk[i]) == task]
        while sum(1 for x in null if x[0] == task) < N_NULL:
            a, b = rng.choice(idx, 2, replace=False)
            if (str(us[a]), str(vs[a])) != (str(us[b]), str(vs[b])):
                null.append((task, int(a), int(b)))
    print(f"{len(names)} inventions, {len(pairs)} co-response pairs, {len(null)} null pairs")

    rows = []
    for th in THETAS:
        sh = [shared_properties(S[a], S[b], tau=th) for _t, _i, a, b, _s, _k in pairs]
        P = [dict(task=p[0], item=p[1], a=p[2], b=p[3], same=p[4], kmin=p[5], sh=s) for p, s in zip(pairs, sh)]
        pct = lambda xs, k=2: 100 * float(np.mean([x["sh"] >= k for x in xs])) if xs else float("nan")
        bl, an = [x for x in P if x["task"] == "blending"], [x for x in P if x["task"] == "analogy"]
        same, cross = [x for x in P if x["same"]], [x for x in P if not x["same"]]

        def per_prop(xs):
            by = defaultdict(list)
            for x in xs:
                if x["kmin"]:
                    by[x["item"]].append(x["sh"] / x["kmin"])
            return float(np.mean([np.mean(v) for v in by.values()]))
        inv_in = set()
        for x in P:
            if x["sh"] >= 2:
                inv_in |= {x["a"], x["b"]}
        nsh = [(t, shared_properties(S[a], S[b], tau=th)) for t, a, b in null]
        r = {"theta": th, "rate_tau2": pct(P), "blend_tau2": pct(bl), "analogy_tau2": pct(an),
             "inventions_in_multiple_pct": 100 * len(inv_in) / len(names),
             "inv_blend_pct": 100 * float(np.mean([i in inv_in for i in range(len(names)) if str(tk[i]) == "blending"])),
             "inv_analogy_pct": 100 * float(np.mean([i in inv_in for i in range(len(names)) if str(tk[i]) == "analogy"])),
             "blend_over_analogy_fixed_tau": pct(bl) / pct(an) if pct(an) else None,
             "blend_over_analogy_per_property": per_prop(bl) / per_prop(an),
             "same_pct": pct(same), "cross_pct": pct(cross), "same_over_cross": pct(same) / pct(cross),
             "rate_tau3": pct(P, 3), "tau3_over_tau2": pct(P, 3) / pct(P),
             "null_tau1_pct": 100 * float(np.mean([s >= 1 for _t, s in nsh])),
             "null_tau2_pct": 100 * float(np.mean([s >= 2 for _t, s in nsh]))}
        r["blend_over_analogy_inventions"] = r["inv_blend_pct"] / r["inv_analogy_pct"] if r["inv_analogy_pct"] else None
        rows.append(r)
        print(f"theta {th:.3f}: tau2 {r['rate_tau2']:.2f}% (bl {r['blend_tau2']:.2f} / an {r['analogy_tau2']:.2f}) "
              f"inv {r['inventions_in_multiple_pct']:.1f}% (bl {r['inv_blend_pct']:.1f} / an {r['inv_analogy_pct']:.1f} = {r['blend_over_analogy_inventions'] or float('nan'):.1f}x) | bl/an fixed {r['blend_over_analogy_fixed_tau'] or float('nan'):.1f}x "
              f"per-prop {r['blend_over_analogy_per_property']:.1f}x | same/cross {r['same_over_cross']:.2f}x "
              f"({r['same_pct']:.2f}/{r['cross_pct']:.2f}) | tau3 {r['rate_tau3']:.3f}% | null tau1 {r['null_tau1_pct']:.2f}% "
              f"tau2 {r['null_tau2_pct']:.3f}%")
    OUT.write_text(json.dumps(rows, indent=1))
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
