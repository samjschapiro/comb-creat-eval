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

Significance. A participant's or model's elements are not independent draws, so every test treats the participant
or model as the unit:
  omnibus     per task and element kind, do the two groups' category distributions differ at all? Statistic: total
              variation distance between the pooled distributions; null by permuting the group labels of whole
              units (N_PERM permutations).
  category    the same permutation null for each category's share difference (two-sided), then Benjamini-Hochberg
              FDR within each task x kind family.
  bootstrap   two-sided p from the unit bootstrap above, BH-corrected in the same families. Label permutation
              assumes the groups are exchangeable, which the very unequal unit sizes (a participant gives ~10
              elements per task, a model hundreds) strain; the bootstrap does not. `significant` = both q < .05,
              and that is what the paper reports.
Also "all": the three tasks pooled with equal weight. Writes data/kg_creat/human_study/analysis/supersenses.json.

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
N_PERM = 10000


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


def pooled(rows, sense, rng):
    """Across tasks: each group's share per category is the MEAN of its three per-task shares, so every task
    weighs the same (LLM association alone has ~10x the elements of the other tasks). Bootstrap as above."""
    tasks = ("association", "analogy", "blending")
    out = {}
    for kind, name in (("c", "concepts"), ("r", "relations")):
        by = {g: {t: defaultdict(list) for t in tasks} for g in ("human", "llm")}
        for g, u, t, k, s in rows:
            if k == kind:
                by[g][t][u].append(sense(k, s))
        cats = [c for c, _ in Counter(x for g in by for t in tasks for u in by[g][t] for x in by[g][t][u]).most_common()]

        def mean_share(g, pick):
            return np.mean([shares([x for u in pick(g, t) for x in by[g][t][u]], cats) for t in tasks], axis=0)
        sh = {g: mean_share(g, lambda g, t: list(by[g][t])) for g in by}
        units = {g: sorted({u for t in tasks for u in by[g][t]}) for g in by}
        boots = []
        for _ in range(BOOT):
            draw = {g: rng.choice(units[g], len(units[g])) for g in by}
            boots.append(mean_share("human", lambda g, t: [u for u in draw[g] if u in by[g][t]])
                         - mean_share("llm", lambda g, t: [u for u in draw[g] if u in by[g][t]]))
        lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
        out[name] = [{"category": c, "human": float(sh["human"][i]), "llm": float(sh["llm"][i]),
                      "diff": float(sh["human"][i] - sh["llm"][i]), "ci95": [float(lo[i]), float(hi[i])],
                      "examples_human": [], "examples_llm": []} for i, c in enumerate(cats)]
    return out


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def permutation(by_unit, cats, rng):
    """Unit-level permutation test: (omnibus p on total variation distance, per-category two-sided p)."""
    units = [(g, u) for g in ("human", "llm") for u in by_unit[g]]
    ix = {c: i for i, c in enumerate(cats)}
    C = np.zeros((len(units), len(cats)))
    for k, (g, u) in enumerate(units):
        for x in by_unit[g][u]:
            C[k, ix[x]] += 1
    lab = np.array([g == "human" for g, _ in units])

    def diff(mask):
        a, b = C[mask].sum(0), C[~mask].sum(0)
        return a / a.sum() - b / b.sum()
    d0 = diff(lab)
    tvd0 = 0.5 * np.abs(d0).sum()
    ge, tvd_ge = np.zeros(len(cats)), 0
    for _ in range(N_PERM):
        d = diff(rng.permutation(lab))
        ge += np.abs(d) >= np.abs(d0) - 1e-12
        tvd_ge += 0.5 * np.abs(d).sum() >= tvd0 - 1e-12
    return float(tvd0), (tvd_ge + 1) / (N_PERM + 1), (ge + 1) / (N_PERM + 1)


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
            B = np.array(boots)
            lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
            p_boot = np.minimum(1, 2 * np.minimum((B <= 0).mean(0), (B >= 0).mean(0)) + 1 / BOOT)
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
            tvd, p_omni, p_cat = permutation(by_unit, cats, rng)
            q = bh(p_cat)
            qb = bh(p_boot)
            for i, r in enumerate(res[task][name]):
                r["p"], r["q"] = float(p_cat[i]), float(q[i])
                r["p_boot"], r["q_boot"] = float(p_boot[i]), float(qb[i])
                r["significant"] = bool(q[i] < .05 and qb[i] < .05)
            res[task][f"omnibus_{name}"] = {"tvd": tvd, "p": float(p_omni), "n_perm": N_PERM,
                                            "units": {g: len(by_unit[g]) for g in by_unit}}
    res["all"] = pooled(rows, sense, rng)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)
    for task in res:
        for name in ("concepts", "relations"):
            if f"n_{name}" not in res[task]:
                res[task][f"n_{name}"] = {"human": 0, "llm": 0}
            om = res[task].get(f"omnibus_{name}", {})
            print(f"\n== {task} {name}  (n human {res[task][f'n_{name}']['human']}, LLM {res[task][f'n_{name}']['llm']}; "
                  f"omnibus TVD {om.get('tvd', float('nan')):.3f}, p {om.get('p', float('nan')):.4f})")
            top = sorted(res[task][name], key=lambda r: -abs(r["diff"]))[:12]
            for r in top:
                sig = "**" if r.get("significant") else "  "
                print(f"  q_perm {r.get('q', float('nan')):.4f} q_boot {r.get('q_boot', float('nan')):.4f}", end="")
                print(f"  {sig} {r['category']:15s} human {100*r['human']:5.1f}%  LLM {100*r['llm']:5.1f}%  diff {100*r['diff']:+5.1f} "
                      f"[{100*r['ci95'][0]:+.1f},{100*r['ci95'][1]:+.1f}]  H: {r['examples_human'][:3]}  L: {r['examples_llm'][:3]}")


if __name__ == "__main__":
    main()
