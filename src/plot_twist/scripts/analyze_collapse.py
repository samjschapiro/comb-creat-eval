"""Mode-collapse analysis of LLM plot twists, contrasted with the human population.

For each source (every LLM, and the 18 STRONG human twists) we quantify how
narrowly it samples the space of twists, from the annotations:

  - unique_name_ratio : distinct protagonist names / stories  (1.0 = never repeats)
  - top_name_share    : share of stories using the single most common name
  - self_dup_rate     : fraction of a source's reveals with a near-twin (cos>0.6)
                        among its OWN reveals (a repeated twist)
  - mean_self_sim     : mean pairwise cosine SIMILARITY of a source's reveals
                        (collapse = high; = 1 - Div)
  - arch_entropy      : normalized entropy over K global twist-archetypes
                        (1 = uses all archetypes evenly, 0 = piles into one)

Archetypes are KMeans clusters of all reveal embeddings; we also report which
archetypes are LLM-overused vs human (the "LLM cliche" twists), and the
single-model vs pooled-LLM-population contrast.

Usage: python src/plot_twist/scripts/analyze_collapse.py
"""

from __future__ import annotations

import collections
import json
import re

import numpy as np
from sklearn.cluster import KMeans

from src.plot_twist.sets import twist_types

ANN = "data/plot_twist/annotations/annotations.json"
MANIFEST = "configs/plot_twist/pd_manifest.json"
K = 14
STOP = set(
    "The A An And Or But Of To In On At For With As By From Into Over After Before When Then "
    "While Where Who Which That This These Those It Its He She They Them His Her Their You Your "
    "I We Our Not No So If About Out Up Down Mr Mrs Ms Dr Sir Madam One Two Three Both Each Every "
    "Some All More Most Now Later Here There What Why How Even Still Just Only Because Although "
    "Though However Meanwhile Suddenly Finally Eventually Instead Perhaps Maybe Yet Upon Despite "
    "Within Without Among Between Through During Until Against Toward Behind Across Around Above "
    "Below Inside Outside Twist Setup Reveal Story Individual Protagonist Character Reader".split()
)


def protagonist(text: str):
    c = collections.Counter(w for w in re.findall(r"\b[A-Z][a-z]+\b", text) if w not in STOP)
    return c.most_common(1)[0][0] if c else None


def norm_entropy(counts):
    p = np.array([c for c in counts if c > 0], dtype=float)
    p /= p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def main():
    recs = json.loads(open(ANN).read())
    types = twist_types(MANIFEST)
    items = []
    for r in recs:
        if not r.get("reveal"):
            continue
        if r["source"] == "human":
            if types.get(r["id"]) != "STRONG":
                continue
            src = "Expert humans"
        else:
            src = r["source"]
        items.append({"src": src, "reveal": r["reveal"], "setup": r.get("setup") or ""})

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    E = np.asarray(enc.encode([it["reveal"] for it in items], normalize_embeddings=True), dtype=np.float32)
    for it, e in zip(items, E):
        it["e"] = e

    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(E)
    for it, l in zip(items, km.labels_):
        it["arch"] = int(l)

    by = collections.defaultdict(list)
    for it in items:
        by[it["src"]].append(it)

    rows = []
    for src, its in by.items():
        names = [protagonist(it["setup"] + " " + it["reveal"]) for it in its]
        names = [n for n in names if n]
        Es = np.array([it["e"] for it in its])
        S = Es @ Es.T
        n = len(Es)
        mean_sim = float((S.sum() - np.trace(S)) / (n * (n - 1)))
        np.fill_diagonal(S, 0.0)
        self_dup = float((S > 0.6).any(1).mean())
        ac = collections.Counter(it["arch"] for it in its)
        rows.append({
            "src": src, "n": n,
            "uniq_name_ratio": round(len(set(names)) / len(names), 3) if names else None,
            "top_name_share": round(collections.Counter(names).most_common(1)[0][1] / len(names), 3) if names else None,
            "top_name": collections.Counter(names).most_common(1)[0][0] if names else None,
            "self_dup": round(self_dup, 3),
            "mean_self_sim": round(mean_sim, 3),
            "arch_entropy": round(norm_entropy([ac.get(k, 0) for k in range(K)]), 3),
            "dom_arch": ac.most_common(1)[0][0], "dom_arch_share": round(ac.most_common(1)[0][1] / n, 3),
        })
    rows.sort(key=lambda d: d["arch_entropy"])

    print(f"{'source':<34}{'n':>4}{'uniqNm':>8}{'topNm':>7}{'selfDup':>8}{'selfSim':>8}{'archEnt':>8}  topName/domArch")
    for d in rows:
        human = " <-- HUMAN" if d["src"] == "Expert humans" else ""
        print(f"{d['src'].split('/')[-1]:<34}{d['n']:>4}{d['uniq_name_ratio']:>8}{d['top_name_share']:>7}"
              f"{d['self_dup']:>8}{d['mean_self_sim']:>8}{d['arch_entropy']:>8}  {d['top_name']}/a{d['dom_arch']}{human}")

    # archetype labels (exemplar reveal nearest centroid) + human vs LLM share
    print("\n=== twist archetypes (KMeans clusters of reveals) ===")
    human_set = set(id(it) for it in by["Expert humans"])
    for k in range(K):
        members = [it for it in items if it["arch"] == k]
        cen = km.cluster_centers_[k]
        ex = max(members, key=lambda it: float(it["e"] @ cen))
        h = sum(1 for it in members if id(it) in human_set)
        llm = len(members) - h
        h_rate = h / len(by["Expert humans"])
        llm_rate = llm / (len(items) - len(by["Expert humans"]))
        tag = "LLM-OVERUSED" if llm_rate > 2 * max(h_rate, 1e-9) else ("human-leaning" if h_rate > llm_rate else "")
        print(f"  a{k:<2} n={len(members):<4} human={h:<2}({h_rate*100:.0f}%) llm%={llm_rate*100:.0f}%  {tag}")
        print(f"       e.g.: {ex['reveal'][:150]}")

    # single-model vs pooled-LLM-population diversity (archetype entropy)
    llm_items = [it for it in items if it["src"] != "Expert humans"]
    pooled_ent = norm_entropy([sum(1 for it in llm_items if it["arch"] == k) for k in range(K)])
    med_model_ent = float(np.median([d["arch_entropy"] for d in rows if d["src"] != "Expert humans"]))
    hum_ent = next(d["arch_entropy"] for d in rows if d["src"] == "Expert humans")
    print(f"\narchetype entropy: human={hum_ent:.3f}  median single-model={med_model_ent:.3f}  pooled-LLM-population={pooled_ent:.3f}")
    json.dump(rows, open("data/plot_twist/collapse.json", "w"), indent=2)
    print("saved: data/plot_twist/collapse.json")


if __name__ == "__main__":
    main()
