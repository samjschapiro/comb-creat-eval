"""Do humans produce inventive multiples? The paper's tau-inventive-multiple test, unchanged, applied to human
inventions: properties = "relation object" with the invention's name dropped and anchor echoes removed; two
inventions on the same item are a tau-multiple when a one-to-one pairing of their properties at cosine >= COS_SLOT
has size >= tau (analyze_inventive_multiples.slot_texts / shared_properties).

Inventions: analogy (the projected image triples) and blending (the blend structure), on the shared non-control
items. Humans = kept sessions; LLMs = the 35 scored models. Three pair types on the same items: human-human,
human-LLM, LLM-LLM. A pair rate does not depend on how many inventions an item has, so the three are comparable
even though an item has a few humans and ~35 models. Also reported among utility-passed inventions only.

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_human_multiples
"""
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.analyze_inventive_multiples import COS_SLOT, matched_pairs, shared_properties, slot_texts

HUMAN = "data/kg_creat/human_study/score_input/human_kept/responses.json"
LLM = Path("data/kg_creat/kombine_test30/responses")
SCORES = Path("data/kg_creat/human_study/combined_scores")
OUT = Path("data/kg_creat/human_study/analysis/human_multiples.json")
TAUS = (1, 2, 3)


def struct(mode, it):
    if mode == "blending":
        tags = it.get("tags") or []
        return [list(t) + [tags[i] if i < len(tags) else ""] for i, t in enumerate((it.get("paths") or [[]])[0])]
    return [{"source": p.get("source"), "image": p.get("image")} for p in (it.get("projection") or [])]


def utility(path, key_of):
    u = {}
    for r in json.loads(path.read_text()):
        if r["mode"] == "blending":
            u[key_of(r)] = bool(r.get("blend_utility"))
        elif r["mode"] == "analogy" and "pair_sat" in r:
            u[key_of(r)] = bool(r.get("invention_utility"))
    return u


def main():
    hum = [r for r in json.load(open(HUMAN)) if not r.get("is_control") and r["mode"] in ("analogy", "blending")]
    shared = {r["prompt_id"] for r in hum}
    hu = utility(SCORES / "human_kept" / "path_scores.json", lambda r: (r.get("sample_idx"), r["prompt_id"]))
    inv = []   # (group, unit, task, item, name, slots, passed)
    for r in hum:
        if r.get("items"):
            it = r["items"][0]
            inv.append(("human", f"h{r['sample_idx']}", r["mode"], r["prompt_id"],
                        it.get("concept") if r["mode"] == "blending" else it.get("invention"),
                        slot_texts(r["mode"], struct(r["mode"], it), r["u_label"], r["v_label"]),
                        hu.get((r["sample_idx"], r["prompt_id"]), False)))
    for d in sorted(SCORES.iterdir()):
        if not d.is_dir() or d.name.startswith("human") or not (LLM / d.name / "responses.json").exists():
            continue
        lu = utility(d / "path_scores.json", lambda r: r["prompt_id"])
        for r in json.load(open(LLM / d.name / "responses.json")):
            if r["mode"] in ("analogy", "blending") and r["prompt_id"] in shared and r.get("items"):
                it = r["items"][0]
                inv.append(("llm", d.name, r["mode"], r["prompt_id"],
                            it.get("concept") if r["mode"] == "blending" else it.get("invention"),
                            slot_texts(r["mode"], struct(r["mode"], it), r["u_label"], r["v_label"]),
                            lu.get(r["prompt_id"], False)))
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    vec = {t: un(np.asarray(embed(t), float)) for t in sorted({t for x in inv for t, _ in x[5]})}
    M = [np.vstack([vec[t] for t, _ in x[5]]) if x[5] else np.zeros((0, 384)) for x in inv]

    by_item = defaultdict(list)
    for i, x in enumerate(inv):
        by_item[(x[2], x[3])].append(i)
    pairs = []
    for (task, item), idx in by_item.items():
        for a, b in itertools.combinations(idx, 2):
            if inv[a][1] == inv[b][1]:
                continue
            kind = "-".join(sorted((inv[a][0], inv[b][0])))
            pairs.append((task, kind, a, b, shared_properties(M[a], M[b]), inv[a][6] and inv[b][6],
                          min(len(inv[a][5]), len(inv[b][5]))))

    res = {"n_inventions": {}, "rates": {}, "examples": {}}
    for task in ("analogy", "blending"):
        res["n_inventions"][task] = {g: sum(1 for x in inv if x[2] == task and x[0] == g) for g in ("human", "llm")}
        res["n_inventions"][task]["mean_properties"] = {g: float(np.mean([len(x[5]) for x in inv if x[2] == task and x[0] == g]))
                                                        for g in ("human", "llm")}
        res["rates"][task] = {}
        for kind in ("human-human", "human-llm", "llm-llm"):
            for only_passed in (False, True):
                P = [p for p in pairs if p[0] == task and p[1] == kind and (p[5] or not only_passed)]
                res["rates"][task][f"{kind}{' (passed)' if only_passed else ''}"] = {
                    "pairs": len(P), **{f"tau{t}": {"count": sum(p[4] >= t for p in P),
                                                    "pct": 100 * float(np.mean([p[4] >= t for p in P])) if P else None}
                                       for t in TAUS},
                    # fair to property count: the tau rate among pairs that both carry >= tau properties, and
                    # the share of the smaller invention's properties the other re-uses
                    **{f"eligible_tau{t}": {"pairs": sum(p[6] >= t for p in P),
                                            "pct": 100 * float(np.mean([p[4] >= t for p in P if p[6] >= t]))
                                            if any(p[6] >= t for p in P) else None} for t in (1, 2)},
                    "per_property_pct": 100 * float(np.mean([p[4] / p[6] for p in P if p[6]])) if any(p[6] for p in P) else None}
        # human inventions with at least one tau=2 partner, and who the partner is
        hi = [i for i, x in enumerate(inv) if x[0] == "human" and x[2] == task]
        part = defaultdict(set)
        for p in pairs:
            if p[0] == task and p[4] >= 2:
                part[p[2]].add(inv[p[3]][0]); part[p[3]].add(inv[p[2]][0])
        res["n_inventions"][task]["human_with_tau2_partner"] = {
            "human": sum("human" in part[i] for i in hi), "llm": sum("llm" in part[i] for i in hi), "of": len(hi)}
        ex = sorted([p for p in pairs if p[0] == task and p[1] != "llm-llm" and p[4] >= 1], key=lambda p: -p[4])
        res["examples"][task] = []
        for p in ex[:12]:
            a, b = (p[2], p[3]) if inv[p[2]][0] == "human" else (p[3], p[2])
            res["examples"][task].append({
                "kind": p[1], "item": inv[a][3], "shared": p[4], "passed_both": p[5],
                "a": {"unit": inv[a][1], "name": inv[a][4], "props": [t for t, _ in inv[a][5]]},
                "b": {"unit": inv[b][1], "name": inv[b][4], "props": [t for t, _ in inv[b][5]]},
                "matched": [(inv[a][5][i][0], inv[b][5][j][0], round(c, 3)) for i, j, c in matched_pairs(M[a], M[b])]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for task in res["rates"]:
        n = res["n_inventions"][task]
        print(f"\n== {task}: human {n['human']} / LLM {n['llm']} inventions; mean properties "
              f"H {n['mean_properties']['human']:.2f} L {n['mean_properties']['llm']:.2f}; "
              f"human inventions with a tau=2 partner: human {n['human_with_tau2_partner']['human']}, "
              f"LLM {n['human_with_tau2_partner']['llm']} of {n['human_with_tau2_partner']['of']}")
        for k, v in res["rates"][task].items():
            el = "  ".join(f"elig.tau{t} {v[f'eligible_tau{t}']['pct']:.2f}% of {v[f'eligible_tau{t}']['pairs']}"
                           for t in (1, 2) if v[f'eligible_tau{t}']['pct'] is not None)
            print(f"  {k:26s} pairs {v['pairs']:6d}  " + "  ".join(
                f"tau{t} {v[f'tau{t}']['count']:5d} ({v[f'tau{t}']['pct']:.2f}%)" if v['pairs'] else f"tau{t} -" for t in TAUS)
                  + f"  | {el}  per-prop {v['per_property_pct'] or 0:.2f}%")
        for e in res["examples"][task][:10]:
            print(f"   [{e['kind']}, {e['item']}, shared {e['shared']}, both passed {e['passed_both']}] "
                  f"{e['a']['name']!r} vs {e['b']['unit']} {e['b']['name']!r}: {e['matched']}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
