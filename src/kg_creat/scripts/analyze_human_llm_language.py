"""What kinds of concepts and relations do humans use, compared with LLMs, on each task? (qualitative contrast)

Elements are an answer's non-anchor concepts and its relations (score._artifact_elements over the answer's paths;
for a blend, its structure triples). Humans = kept sessions; LLMs = the 35 scored models' answers; the 86 items
both answered. Three views per task, with element and concept counts written alongside:

  features      words per concept / relation; how common the words are (wordfreq Zipf, higher = more common);
                share of concepts containing a rare word (Zipf < 3); association chain length
  distinctive   words over-used by each group in concepts and in relations: log-odds ratio with an informative
                Dirichlet prior (Monroe, Colaresi & Quinn 2008), z-scored, so frequent words are not favoured
                merely for being frequent nor rare words for being rare
  clusters      k-means (k = 30) over the MiniLM embeddings of every concept occurrence; clusters where humans or
                LLMs are over-represented relative to their overall share, with the most frequent members

Two stages (embedder and sklearn live in different environments):

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_human_llm_language --stage embed
    .venv/bin/python -m src.kg_creat.scripts.analyze_human_llm_language --stage analyze
"""
import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HUMAN = "data/kg_creat/human_study/score_input/human_kept/responses.json"
LLM = Path("data/kg_creat/kombine_test30/responses")
SCORED = "data/kg_creat/kombine_test30/scores"
OUT = Path("data/kg_creat/human_study/analysis")
CACHE = OUT / "language_concepts.npz"
TASK = {"baseline": "association", "analogy": "analogy", "blending": "blending"}
STOP = set("a an the of to in on for and or by with from as at is are be been its it their his her into".split())
K_CLUSTERS, SEED = 30, 0


def norm(s):
    return " ".join(str(s).strip().lower().split())


def elements():
    """[(group, task, item, kind ('c' concept / 'r' relation), surface)] and association chain lengths."""
    from src.kg_creat.scripts.score import _artifact_elements
    hum = json.load(open(HUMAN))
    shared = {r["prompt_id"] for r in hum if not r.get("is_control")}
    scored = {Path(p).parent.name for p in glob.glob(f"{SCORED}/*/summary.json")}
    rows, hops = [], defaultdict(list)

    def add(group, r):
        if r["prompt_id"] not in shared:
            return
        for it in r.get("items") or []:
            paths = it.get("paths") or []
            if r["mode"] == "baseline":
                hops[group] += [len(p) for p in paths if p]
            for kind, s in _artifact_elements(paths, r.get("u_label"), r.get("v_label")):
                if s:
                    rows.append((group, TASK[r["mode"]], r["prompt_id"], kind, s))
    for r in hum:
        if not r.get("is_control"):
            add("human", r)
    for d in sorted(LLM.iterdir()):
        if d.name in scored:
            for r in json.load(open(d / "responses.json")):
                add("llm", r)
    return rows, hops


def stage_embed():
    from src.kg_creat.embed import get_embedder
    rows, _ = elements()
    concepts = sorted({s for _g, _t, _i, k, s in rows if k == "c"})
    embed = get_embedder()
    X = np.stack([embed(s) for s in concepts]).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, X=X, concepts=np.array(concepts, dtype=object))
    print(f"embedded {len(concepts)} distinct concepts -> {CACHE}")


def words(s):
    return [w for w in re.findall(r"[a-z][a-z'-]*", s.lower()) if w not in STOP]


def log_odds(c1, c2, prior_scale=1.0):
    """Monroe et al. (2008) log-odds with informative Dirichlet prior; returns {word: z} (positive = group 1)."""
    prior = Counter()
    for c in (c1, c2):
        prior.update(c)
    a0 = sum(prior.values())
    alpha = {w: prior_scale * prior[w] / a0 * len(prior) for w in prior}
    A = sum(alpha.values())
    n1, n2 = sum(c1.values()), sum(c2.values())
    z = {}
    for w in prior:
        aw = alpha[w]
        d = (np.log((c1[w] + aw) / (n1 + A - c1[w] - aw)) - np.log((c2[w] + aw) / (n2 + A - c2[w] - aw)))
        var = 1 / (c1[w] + aw) + 1 / (c2[w] + aw)
        z[w] = float(d / np.sqrt(var))
    return z


def stage_analyze():
    from sklearn.cluster import KMeans
    from wordfreq import zipf_frequency
    rows, hops = elements()
    zipf = lambda w: zipf_frequency(w, "en")
    res = {"n": {}, "features": {}, "distinctive": {}, "clusters": {}}
    for task in ("association", "analogy", "blending"):
        T = [r for r in rows if r[1] == task]
        res["n"][task] = {g: {k: sum(1 for r in T if r[0] == g and r[3] == k) for k in "cr"} for g in ("human", "llm")}
        feats = {}
        for g in ("human", "llm"):
            C = [r[4] for r in T if r[0] == g and r[3] == "c"]
            R = [r[4] for r in T if r[0] == g and r[3] == "r"]
            cw = [words(s) for s in C]
            rw = [words(s) for s in R]
            feats[g] = {
                "words_per_concept": float(np.mean([len(s.split()) for s in C])),
                "words_per_relation": float(np.mean([len(s.split()) for s in R])),
                "concept_word_zipf": float(np.mean([zipf(w) for ws in cw for w in ws])),
                "relation_word_zipf": float(np.mean([zipf(w) for ws in rw for w in ws])),
                "concepts_with_rare_word": float(np.mean([any(zipf(w) < 3 for w in ws) for ws in cw if ws])),
                "distinct_concepts_per_100": float(100 * len(set(C)) / len(C)),
            }
            if task == "association":
                feats[g]["chain_length"] = float(np.mean(hops[g]))
        res["features"][task] = feats
        dist = {}
        for kind, name in (("c", "concept_words"), ("r", "relation_words")):
            ch = Counter(w for r in T if r[0] == "human" and r[3] == kind for w in words(r[4]))
            cl = Counter(w for r in T if r[0] == "llm" and r[3] == kind for w in words(r[4]))
            z = log_odds(ch, cl)
            top = lambda sign: [(w, round(z[w], 1), ch[w], cl[w]) for w in sorted(z, key=lambda w: sign * z[w], reverse=True)[:20]]
            dist[name] = {"human": top(1), "llm": top(-1)}
        res["distinctive"][task] = dist

    # concept clusters per task over the shared embedding of every distinct concept
    cache = np.load(CACHE, allow_pickle=True)
    vec = {s: v for s, v in zip(cache["concepts"], cache["X"])}
    for task in ("association", "analogy", "blending"):
        occ = [(r[0], r[4]) for r in rows if r[1] == task and r[3] == "c" and r[4] in vec]
        uniq = sorted({s for _g, s in occ})
        km = KMeans(n_clusters=K_CLUSTERS, random_state=SEED, n_init=10).fit(np.stack([vec[s] for s in uniq]))
        lab = dict(zip(uniq, km.labels_))
        base = np.mean([g == "human" for g, _ in occ])
        cl = []
        for k in range(K_CLUSTERS):
            members = [(g, s) for g, s in occ if lab[s] == k]
            hs = np.mean([g == "human" for g, _ in members])
            top_h = [s for s, _ in Counter(s for g, s in members if g == "human").most_common(8)]
            top_l = [s for s, _ in Counter(s for g, s in members if g == "llm").most_common(8)]
            cl.append({"cluster": k, "n": len(members), "human_share": float(hs), "enrichment": float(hs / base),
                       "top_human": top_h, "top_llm": top_l})
        cl.sort(key=lambda c: -c["enrichment"])
        res["clusters"][task] = {"human_share_overall": float(base), "clusters": cl}

    json.dump(res, open(OUT / "language_contrast.json", "w"), indent=1)
    for task in ("association", "analogy", "blending"):
        print(f"\n===== {task}  (concept occurrences human {res['n'][task]['human']['c']}, LLM {res['n'][task]['llm']['c']})")
        f = res["features"][task]
        for k in f["human"]:
            print(f"  {k:26s} human {f['human'][k]:7.2f}   LLM {f['llm'][k]:7.2f}")
        for name in ("concept_words", "relation_words"):
            d = res["distinctive"][task][name]
            print(f"  {name} over-used by humans: " + ", ".join(f"{w}({z})" for w, z, *_ in d["human"][:14]))
            print(f"  {name} over-used by LLMs:   " + ", ".join(f"{w}({-z})" for w, z, *_ in d["llm"][:14]))
        c = res["clusters"][task]
        print(f"  clusters (human share overall {c['human_share_overall']:.3f}):")
        for x in c["clusters"][:4]:
            print(f"    HUMAN x{x['enrichment']:.1f} (n {x['n']}): {x['top_human'][:6]}")
        for x in c["clusters"][-4:]:
            print(f"    LLM   x{x['enrichment']:.2f} (n {x['n']}): {x['top_llm'][:6]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["embed", "analyze"], required=True)
    stage_embed() if ap.parse_args().stage == "embed" else stage_analyze()
