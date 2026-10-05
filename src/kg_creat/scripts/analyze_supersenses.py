"""Human vs. LLM concepts and relations by WordNet supersense (established coarse semantic categories).

Supersenses are WordNet's lexicographer classes -- 26 for nouns (person, location, artifact, cognition, feeling, ...)
and 15 for verbs (creation, cognition, motion, possession, ...) -- the standard coarse semantic labels of
supersense tagging (Ciaramita & Johnson 2003; Schneider & Smith 2015).

  concept   the noun supersense of the whole phrase if WordNet has it (e.g. "industrial revolution"), else of its
            head noun (last word); the first (most frequent) sense. Phrases WordNet does not know at all
            (mostly proper names and technical compounds) form their own category, "not in WordNet".
  relation  the verb supersense of its first content verb (auxiliaries and copulas skipped unless they are the
            only verb); a relation with no verb ("like", "inside", "via") is "no verb".

Units: humans = kept participants, LLMs = the 35 models; the 86 shared items. For each category, each group's
share of element uses, and a 95% bootstrap CI for the human - LLM difference resampling participants and models.
Writes data/kg_creat/human_study/analysis/supersenses.json.

    NLTK_DATA=data/nltk_data .venv/bin/python -m src.kg_creat.scripts.analyze_supersenses
"""
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from nltk.corpus import wordnet as wn

HUMAN = "data/kg_creat/human_study/score_input/human_kept/responses.json"
LLM = Path("data/kg_creat/kombine_test30/responses")
SCORED = "data/kg_creat/kombine_test30/scores"
OUT = Path("data/kg_creat/human_study/analysis/supersenses.json")
TASK = {"baseline": "association", "analogy": "analogy", "blending": "blending"}
AUX = {"is", "are", "was", "were", "be", "been", "being", "am", "can", "could", "will", "would", "shall", "should",
       "may", "might", "must", "do", "does", "did", "not", "also", "often", "a", "an", "the"}
BOOT, SEED = 2000, 0


def concept_sense(s):
    syn = wn.synsets(s.replace(" ", "_"), pos=wn.NOUN)
    if not syn:
        words = re.findall(r"[a-z][a-z'-]*", s)
        if not words:
            return "not in WordNet"
        head = wn.morphy(words[-1], wn.NOUN) or words[-1]
        syn = wn.synsets(head, pos=wn.NOUN)
    return syn[0].lexname().replace("noun.", "") if syn else "not in WordNet"


def relation_sense(s):
    words = re.findall(r"[a-z][a-z'-]*", s)
    verbs = [w for w in words if wn.synsets(wn.morphy(w, wn.VERB) or w, pos=wn.VERB)]
    main = [w for w in verbs if w not in AUX]
    w = (main or verbs or [None])[0]
    if w is None:
        return "no verb"
    return wn.synsets(wn.morphy(w, wn.VERB) or w, pos=wn.VERB)[0].lexname().replace("verb.", "")


def elements():
    from src.kg_creat.scripts.score import _artifact_elements
    hum = json.load(open(HUMAN))
    shared = {r["prompt_id"] for r in hum if not r.get("is_control")}
    scored = {Path(p).parent.name for p in glob.glob(f"{SCORED}/*/summary.json")}
    rows = []

    def add(group, unit, r):
        if r["prompt_id"] not in shared:
            return
        for it in r.get("items") or []:
            for kind, s in _artifact_elements(it.get("paths") or [], r.get("u_label"), r.get("v_label")):
                if s:
                    rows.append((group, unit, TASK[r["mode"]], kind, s))
    for r in hum:
        if not r.get("is_control"):
            add("human", f"h{r['sample_idx']}", r)
    for d in sorted(LLM.iterdir()):
        if d.name in scored:
            for r in json.load(open(d / "responses.json")):
                add("llm", d.name, r)
    return rows


def shares(rows, cats):
    c = Counter(r for r in rows)
    n = sum(c.values()) or 1
    return np.array([c[k] / n for k in cats])


def main():
    rng = np.random.default_rng(SEED)
    rows = elements()
    cache = {}
    sense = lambda kind, s: cache.setdefault((kind, s), concept_sense(s) if kind == "c" else relation_sense(s))
    res = {}
    for task in ("association", "analogy", "blending"):
        res[task] = {}
        for kind, name in (("c", "concepts"), ("r", "relations")):
            by_unit = {"human": defaultdict(list), "llm": defaultdict(list)}
            for g, u, t, k, s in rows:
                if t == task and k == kind:
                    by_unit[g][u].append(sense(k, s))
            allc = Counter(x for g in by_unit for u in by_unit[g] for x in by_unit[g][u])
            cats = [c for c, _ in allc.most_common()]
            flat = {g: [x for u in by_unit[g] for x in by_unit[g][u]] for g in by_unit}
            sh = {g: shares(flat[g], cats) for g in flat}
            units = {g: list(by_unit[g]) for g in by_unit}
            boots = []
            for _ in range(BOOT):
                b = {g: [x for u in rng.choice(units[g], len(units[g])) for x in by_unit[g][u]] for g in by_unit}
                boots.append(shares(b["human"], cats) - shares(b["llm"], cats))
            lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
            ex = {g: defaultdict(Counter) for g in flat}
            for g, u, t, k, s in rows:
                if t == task and k == kind:
                    ex[g][sense(k, s)][s] += 1
            res[task][name] = [{"category": c, "human": float(sh["human"][i]), "llm": float(sh["llm"][i]),
                                "diff": float(sh["human"][i] - sh["llm"][i]), "ci95": [float(lo[i]), float(hi[i])],
                                "examples_human": [w for w, _ in ex["human"][c].most_common(6)],
                                "examples_llm": [w for w, _ in ex["llm"][c].most_common(6)]}
                               for i, c in enumerate(cats)]
            res[task][f"n_{name}"] = {g: len(flat[g]) for g in flat}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for task in res:
        for name in ("concepts", "relations"):
            print(f"\n== {task} {name}  (n human {res[task][f'n_{name}']['human']}, LLM {res[task][f'n_{name}']['llm']})")
            top = sorted(res[task][name], key=lambda r: -abs(r["diff"]))[:9]
            for r in top:
                sig = "*" if (r["ci95"][0] > 0 or r["ci95"][1] < 0) else " "
                print(f"  {sig} {r['category']:15s} human {100*r['human']:5.1f}%  LLM {100*r['llm']:5.1f}%  diff {100*r['diff']:+5.1f} "
                      f"[{100*r['ci95'][0]:+.1f},{100*r['ci95'][1]:+.1f}]  H: {r['examples_human'][:3]}  L: {r['examples_llm'][:3]}")


if __name__ == "__main__":
    main()
