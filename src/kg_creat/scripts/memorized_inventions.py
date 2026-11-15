"""Memorised inventions: invented concepts that are already documented, in context, in pretraining text.

Companion to the inventive-multiples analysis (analyze_inventive_multiples.py) and the corpus contamination
check (corpus_novelty.py, whose infini-gram query/cache layer this reuses). Rebuttal, reviewer En17 Q1.

An analogy invention h or blend c' is a MEMORISED INVENTION when, in the 4.6T-token OLMo 2 pretraining mix:
  (1) attested   its name occurs verbatim >= 10 times;
  (2) in context the name co-occurs (within 100 tokens) with at least one of the item's two inputs;
  (3) described  >= 2 of the invention's property objects co-occur (within 100 tokens) with the name.
i.e. the concept exists in text, near its inputs, with the properties the model gave it.

Cross-item null (as in the multiples analysis): the same test with a DIFFERENT item's inputs (2) and a
different item's invention's property objects (3), same task, seed 0. The null rate is what (2)+(3) pass
by chance co-occurrence at 4.6T tokens; a SPECIFIC memorised invention passes with its own context and
fails with the mismatched one (removes stock words that co-occur with anything).

Property objects: objects of the invention's own triples (analogy: projection images; blending: triples
whose subject is the concept), excluding anchor echo (object = an input) and the name itself.
"Documented" follows corpus_novelty.documented (nonzero on the 4.6T index, exact zero accepted, sampled
zero -> exact 9.2B Dolma sample -> else inconclusive, counted as not documented).

Grouped by task (analogy vs blending, paired over the 30 shared anchor pairs), provider (model-label
permutation), and input properties (domains, prominence, anchor co-occurrence lift, anchor semantic
distance); related to utility, emergent originality and membership in an inventive multiple.

    .venv/bin/python -m src.kg_creat.scripts.memorized_inventions
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from src.kg_creat.scripts import corpus_novelty as C

OUT = Path("data/kg_creat/corpus_novelty/memorized_inventions.json")
MIN_COUNT = 10
C.WORKERS = 10


def _n(s):
    return " ".join(str(s).lower().split()).strip(".,;:'\"")


def inventions():
    """One record per (model, task, item) invention with its name and property objects."""
    out = []
    for p in sorted(C.RESP.glob("*/responses.json")):
        for r in json.loads(p.read_text()):
            it = (r.get("items") or [None])[0]
            if not it or r["mode"] not in ("analogy", "blending"):
                continue
            u, v = r["u_label"], r["v_label"]
            if r["mode"] == "analogy":
                name = it.get("invention")
                trip = [pr["image"] for pr in (it.get("projection") or []) if isinstance(pr, dict) and len(pr.get("image", [])) == 3]
            else:
                name = it.get("concept")
                trip = [t for t in ((it.get("paths") or [[]])[0]) if len(t) == 3 and _n(t[0]) == _n(name or "")]
            if not name or not str(name).strip():
                continue
            bad = {_n(u), _n(v), _n(name)}
            objs = sorted({str(t[2]).strip() for t in trip if str(t[2]).strip() and _n(t[2]) not in bad})
            out.append({"model": p.parent.name, "task": r["mode"], "prompt_id": r["prompt_id"], "u": u, "v": v,
                        "domain_u": r.get("domain_u"), "domain_v": r.get("domain_v"), "name": str(name).strip(), "objects": objs})
    return out


def f(x, spec=".1%"):
    """format, tolerating None (a task with no memorised inventions)."""
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def provider(m):
    return m.split("_")[0]


def main():
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.CACHE_D.update(C._load_cache())
    inv = inventions()
    for x in inv:
        x["count"] = C.query(C.BIG, C.clause(x["name"]))["count"]
    print(f"{len(inv)} inventions; names counted (cached): {sum(x['count'] is not None for x in inv)}")

    # null pairing: each invention gets a different item's anchors and a different item's invention's objects
    rng = random.Random(0)
    by_task = defaultdict(list)
    for x in inv:
        by_task[x["task"]].append(x)
    for x in inv:
        others = [y for y in by_task[x["task"]] if y["prompt_id"] != x["prompt_id"]]
        y = rng.choice(others)
        z = rng.choice([o for o in others if o["objects"]] or others)
        x["null_u"], x["null_v"], x["null_objects"], x["null_item"] = y["u"], y["v"], z["objects"], y["prompt_id"]

    att = [x for x in inv if (x["count"] or 0) >= MIN_COUNT]
    if os.environ.get("SMOKE"):                     # quick end-to-end check on a handful of inventions
        att = att[:int(os.environ["SMOKE"])]
    print(f"attested >= {MIN_COUNT}: {len(att)}")
    print("context queries (own + null)")
    C.run_all([(C.BIG, C.cooc(x["name"], a), 100) for x in att for a in (x["u"], x["v"], x["null_u"], x["null_v"])])
    fall = [(C.SMALL, C.cooc(x["name"], a), 100) for x in att for a in (x["u"], x["v"], x["null_u"], x["null_v"])
            if C.query(C.BIG, C.cooc(x["name"], a)).get("count") == 0 and C.query(C.BIG, C.cooc(x["name"], a)).get("approx")]
    C.run_all(fall)
    for x in att:
        x["ctx_u"], x["ctx_v"] = C.documented(x["name"], x["u"]), C.documented(x["name"], x["v"])
        x["null_ctx_u"], x["null_ctx_v"] = C.documented(x["name"], x["null_u"]), C.documented(x["name"], x["null_v"])
        x["in_context"] = bool(x["ctx_u"]) or bool(x["ctx_v"])
        x["null_in_context"] = bool(x["null_ctx_u"]) or bool(x["null_ctx_v"])

    print("property queries (own + null), only where the context step passed")
    jobs = []
    for x in att:
        if x["in_context"]:
            jobs += [(C.BIG, C.cooc(x["name"], o), 100) for o in x["objects"]]
        if x["null_in_context"]:
            jobs += [(C.BIG, C.cooc(x["name"], o), 100) for o in x["null_objects"]]
    C.run_all(jobs)
    C.run_all([j for j in jobs if C.query(*j).get("count") == 0 and C.query(*j).get("approx")
               for j in [(C.SMALL, j[1], 100)]])
    for x in inv:
        a = x in att
        x["attested"] = a
        own = [C.documented(x["name"], o) for o in x["objects"]] if a and x.get("in_context") else []
        nul = [C.documented(x["name"], o) for o in x["null_objects"]] if a and x.get("null_in_context") else []
        x["props_documented"], x["n_objects"] = sum(bool(d) for d in own), len(x["objects"])
        x["null_props_documented"] = sum(bool(d) for d in nul)
        x["memorized"] = bool(a and x.get("in_context") and x["props_documented"] >= 2)
        x["memorized_strict"] = bool(x["memorized"] and x.get("ctx_u") and x.get("ctx_v"))
        x["null_memorized"] = bool(a and x.get("null_in_context") and x["null_props_documented"] >= 2)
        x["specific"] = x["memorized"] and not x["null_memorized"]

    attach_outcomes(inv)
    res = analyse(inv)
    if os.environ.get("SMOKE"):
        return
    OUT.write_text(json.dumps({"inventions": inv, **res}, indent=1, default=str))
    print(f"\n-> {OUT}")


def attach_outcomes(inv):
    """utility pass, emergent originality (paper scores) and membership in an inventive multiple (tau = 2)."""
    sc = {}
    for p in C.PAPER.glob("*/path_scores.json"):
        for r in json.loads(p.read_text()):
            if r["mode"] == "analogy" and "pair_sat" in r:
                sc[(p.parent.name, r["prompt_id"])] = (r.get("pair_sat") is True, r.get("em_originality"))
            elif r["mode"] == "blending" and r.get("triples"):
                sc[(p.parent.name, r["prompt_id"])] = (r.get("sat") is True, r.get("em_originality"))
    mult = json.loads(C.MULT.read_text())
    inm = {(m["task"], m["u"], m["v"], m[s]["model"]) for m in mult["multiples"] for s in ("a", "b")}
    for x in inv:
        ok, eo = sc.get((x["model"], x["prompt_id"]), (None, None))
        x["passed"], x["em_originality"] = ok, eo
        x["in_multiple"] = (x["task"], x["u"], x["v"], x["model"]) in inm


def rate(xs, k):
    xs = list(xs)
    return float(np.mean([bool(x[k]) for x in xs])) if xs else None


def analyse(inv):
    res = {}
    T = ("analogy", "blending")
    # ---- headline + null
    head = {}
    for t in T + ("all",):
        xs = [x for x in inv if t == "all" or x["task"] == t]
        head[t] = {"n": len(xs), "attested": rate(xs, "attested"),
                   "in_context": float(np.mean([bool(x.get("in_context")) for x in xs])),
                   "null_in_context": float(np.mean([bool(x.get("null_in_context")) for x in xs])),
                   "memorized": rate(xs, "memorized"), "memorized_strict": rate(xs, "memorized_strict"),
                   "null_memorized": rate(xs, "null_memorized"), "specific": rate(xs, "specific"),
                   "n_memorized": sum(x["memorized"] for x in xs), "n_specific": sum(x["specific"] for x in xs)}
    res["headline"] = head
    print("\n== headline (rates over all inventions)")
    for t, h in head.items():
        print(f"  {t:9s} n={h['n']:4d} attested {h['attested']:.1%}  in-context {h['in_context']:.1%} (null {h['null_in_context']:.1%})  "
              f"MEMORISED {h['memorized']:.1%} (null {h['null_memorized']:.1%}; strict {h['memorized_strict']:.1%})  specific {h['specific']:.1%} [{h['n_specific']}]")

    # ---- task, paired over shared anchor pairs
    items = defaultdict(dict)
    for t in T:
        for (u, v), xs in group(inv, t, lambda x: (x["u"], x["v"])).items():
            items[(u, v)][t] = {k: rate(xs, k) for k in ("memorized", "specific", "null_memorized")}
    shared = [k for k in items if all(t in items[k] for t in T)]
    tk = {}
    for k in ("memorized", "specific"):
        a = [items[i]["analogy"][k] for i in shared]
        b = [items[i]["blending"][k] for i in shared]
        p = wilcoxon(b, a).pvalue if any(x != y for x, y in zip(a, b)) else 1.0
        tk[k] = {"analogy": float(np.mean(a)), "blending": float(np.mean(b)), "n_items": len(shared), "wilcoxon_p": float(p),
                 "items_blend_higher": sum(y > x for x, y in zip(a, b)), "items_analogy_higher": sum(x > y for x, y in zip(a, b))}
    res["task"] = tk
    print("\n== task (paired over anchor pairs)")
    for k, v in tk.items():
        print(f"  {k:10s} analogy {v['analogy']:.1%} vs blending {v['blending']:.1%}  (blend higher on {v['items_blend_higher']}, "
              f"analogy on {v['items_analogy_higher']} of {v['n_items']}; Wilcoxon p={v['wilcoxon_p']:.2g})")

    # ---- provider: rate per provider, permutation over model labels
    models = sorted({x["model"] for x in inv})
    prov = {}
    for t in T + ("all",):
        xs = [x for x in inv if t == "all" or x["task"] == t]
        bym = {m: rate([x for x in xs if x["model"] == m], "memorized") for m in models}
        provs = sorted({provider(m) for m in models})
        pr = {p: float(np.mean([bym[m] for m in models if provider(m) == p])) for p in provs}

        def spread(lbl):
            g = defaultdict(list)
            for m, p in zip(models, lbl):
                g[p].append(bym[m])
            return float(np.var([np.mean(v) for v in g.values() if len(v) >= 2]))
        lbl = [provider(m) for m in models]
        obs = spread(lbl)
        rng = np.random.default_rng(0)
        null = [spread(list(rng.permutation(lbl))) for _ in range(5000)]
        prov[t] = {"by_provider": pr, "by_model": bym, "n_models": {p: sum(provider(m) == p for m in models) for p in provs},
                   "between_provider_var_p": float((1 + sum(n >= obs for n in null)) / 5001)}
    res["provider"] = prov
    print("\n== provider (memorised rate; model-label permutation test of between-provider spread, providers with >= 2 models)")
    for t, v in prov.items():
        s = ", ".join(f"{p} {r:.1%}" for p, r in sorted(v["by_provider"].items(), key=lambda kv: -kv[1]))
        print(f"  {t:9s} p={v['between_provider_var_p']:.3f}  {s}")
    top = sorted(prov["all"]["by_model"].items(), key=lambda kv: -kv[1])
    print("  models, most -> least:", ", ".join(f"{m.split('_', 1)[1]} {r:.0%}" for m, r in top[:6]), "...",
          ", ".join(f"{m.split('_', 1)[1]} {r:.0%}" for m, r in top[-4:]))

    # ---- input properties, per item
    pool = {r["label"]: r for r in json.loads(C.POOL.read_text())["rows"]}
    cn = json.loads((C.OUT / "corpus_novelty.json").read_text())["anchors"]
    mi = {(r["task"], r["u"], r["v"]): r for r in json.loads(C.MULT.read_text())["per_item"]}
    rows = []
    for t in T:
        for (u, v), xs in group(inv, t, lambda x: (x["u"], x["v"])).items():
            a = cn.get(f"{u} || {v}", {})
            rows.append({"task": t, "u": u, "v": v, "domains": tuple(sorted((xs[0]["domain_u"], xs[0]["domain_v"]))),
                         "memorized": rate(xs, "memorized"), "specific": rate(xs, "specific"),
                         "prominence": float(np.mean([np.log1p((pool.get(e) or {}).get("sitelinks") or 0) for e in (u, v)])),
                         "log_lift": float(np.log1p(a.get("lift100") or 0)), "log_cooc": float(np.log1p(a.get("cooc100") or 0)),
                         "anchor_distance": (mi.get((t, u, v)) or {}).get("anchor_label_cos_distance"),
                         "multiples_tau2": (mi.get((t, u, v)) or {}).get("rate_tau2_pct")})
    inp = {}
    for t in T:
        rs = [r for r in rows if r["task"] == t]
        for x in ("prominence", "log_lift", "log_cooc", "anchor_distance", "multiples_tau2"):
            ok = [r for r in rs if r[x] is not None]
            xs, ys = [r[x] for r in ok], [r["memorized"] for r in ok]
            if len(set(ys)) < 2 or len(set(xs)) < 2:       # undefined: no item has a memorised invention
                inp[f"{t}|{x}"] = {"n": len(ok), "spearman": None, "p_perm": None}
                continue
            inp[f"{t}|{x}"] = {"n": len(ok), "spearman": float(spearmanr(xs, ys).statistic), "p_perm": C.perm_p(xs, ys, 5000)}
    dom = defaultdict(list)
    for x in inv:
        for d in {x["domain_u"], x["domain_v"]}:
            dom[d].append(x["memorized"])
    inp["by_domain"] = {d: {"rate": float(np.mean(v)), "n": len(v)} for d, v in sorted(dom.items(), key=lambda kv: -np.mean(kv[1]))}
    res["inputs"] = inp
    res["per_item"] = rows
    print("\n== input properties (per item; Spearman with memorised rate, permutation p)")
    for k, v in inp.items():
        if k != "by_domain":
            print(f"  {k:28s} n={v['n']:2d} rho={f(v['spearman'], '+.2f')} p={f(v['p_perm'], '.3f')}")
    print("  by domain (an invention counts for both of its inputs' domains):",
          ", ".join(f"{d} {v['rate']:.1%} (n={v['n']})" for d, v in inp["by_domain"].items()))

    # ---- outcomes
    oc = {}
    for t in T:
        xs = [x for x in inv if x["task"] == t]
        mem, non = [x for x in xs if x["memorized"]], [x for x in xs if not x["memorized"]]
        within = []
        for _k, g in group(inv, t, lambda x: x["prompt_id"]).items():
            a = [x["em_originality"] for x in g if x["memorized"] and x["em_originality"] is not None]
            b = [x["em_originality"] for x in g if not x["memorized"] and x["em_originality"] is not None]
            if a and b:
                within.append(np.mean(a) - np.mean(b))
        oc[t] = {"pass_memorized": rate(mem, "passed"), "pass_other": rate(non, "passed"),
                 "em_orig_memorized": float(np.mean([x["em_originality"] for x in mem if x["em_originality"] is not None])) if mem else None,
                 "em_orig_other": float(np.mean([x["em_originality"] for x in non if x["em_originality"] is not None])),
                 "em_orig_within_item_diff": float(np.mean(within)) if within else None, "n_items_within": len(within),
                 "within_item_p": float(wilcoxon(within).pvalue) if len(within) > 5 else None,
                 "multiple_memorized": rate(mem, "in_multiple"), "multiple_other": rate(non, "in_multiple"),
                 "share_of_multiple_members_memorized": rate([x for x in xs if x["in_multiple"]], "memorized")}
    res["outcomes"] = oc
    print("\n== outcomes: memorised vs other inventions")
    for t, v in oc.items():
        print(f"  {t:9s} pass {f(v['pass_memorized'])} vs {f(v['pass_other'])} | em_orig {f(v['em_orig_memorized'], '.3f')} vs "
              f"{f(v['em_orig_other'], '.3f')} (within item {f(v['em_orig_within_item_diff'], '+.3f')}, n={v['n_items_within']}, "
              f"p={f(v['within_item_p'], '.2g')}) | in a multiple {f(v['multiple_memorized'])} vs {f(v['multiple_other'])}; "
              f"{f(v['share_of_multiple_members_memorized'])} of multiple members are memorised")

    ex = sorted([x for x in inv if x["specific"]], key=lambda x: -x["count"])
    res["examples_specific"] = [{k: x[k] for k in ("task", "model", "u", "v", "name", "count", "objects", "props_documented")} for x in ex[:40]]
    print("\n== specific memorised inventions, most frequent names first")
    for x in ex[:20]:
        print(f"  [{x['task'][:5]}] {x['u']} + {x['v']}: {x['name']!r} ({x['count']:,}) <- {x['model'].split('_', 1)[1]}; "
              f"props {x['props_documented']}/{x['n_objects']}: {x['objects'][:4]}")
    return res


def group(inv, task, key):
    g = defaultdict(list)
    for x in inv:
        if x["task"] == task:
            g[key(x)].append(x)
    return g


if __name__ == "__main__":
    main()
