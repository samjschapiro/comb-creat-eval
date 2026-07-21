"""Derive relation CLASSES and per-bundle constraint targets from baseline responses.

Nothing here is hand-specified. We take every relation the models actually emitted in the
no-constraint (baseline) pass, cluster them in embedding space, and call each cluster a
relation *class* named by its most frequent members. Per bundle we then pick constraint
targets **against the models' own default behaviour**, so each constraint bites by construction:

    exclusion  -> the class most used on that bundle   (removes its default route)
    inclusion  -> a class rarely used on that bundle   (forces a route it wouldn't take)
    ordering   -> the REVERSE of the natural order of the two most co-occurring classes

Reproducible: "the K clusters of relations models produce on these prompts, and the
most/least-used of them per bundle."
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def collect(responses_dir) -> tuple[Counter, dict, dict]:
    """-> (global relation counts, {bundle: Counter(relation)}, {bundle: [relation-sequences]})"""
    glob_counts: Counter = Counter()
    per_bundle: dict[str, Counter] = defaultdict(Counter)
    per_bundle_seqs: dict[str, list] = defaultdict(list)
    for md in Path(responses_dir).glob("*/responses.json"):
        for r in json.loads(md.read_text()):
            if r.get("mode") != "baseline":
                continue
            for path in r.get("paths") or []:
                rels = [str(t[1]).strip().lower() for t in path if len(t) == 3]
                if not rels:
                    continue
                glob_counts.update(rels)
                per_bundle[r["bundle_id"]].update(rels)
                per_bundle_seqs[r["bundle_id"]].append(rels)
    return glob_counts, per_bundle, per_bundle_seqs


def derive_classes(counts: Counter, embed, k: int = 8, top_n: int = 150, seed: int = 0) -> list[dict]:
    """Cluster the *high-mass* emitted relations into k classes; name each by its top members.

    Open-vocabulary output has a huge singleton tail (~1 use each), which chains
    average-linkage clustering into one blob plus noise. So we cluster only the ``top_n`` most
    frequent relations (the mass) with k-means, which yields balanced, usable classes.
    """
    from scipy.cluster.vq import kmeans2

    rels = [r for r, _ in counts.most_common(top_n)]
    if len(rels) <= k:
        return [{"id": i, "name": r, "members": [r], "count": counts[r], "share": 0.0}
                for i, r in enumerate(rels)]
    X = np.vstack([embed(r) for r in rels]).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    _centroids, labels = kmeans2(X, k, minit="++", seed=seed, missing="warn")

    total = sum(counts[r] for r in rels)
    classes = []
    for cid in sorted(set(labels)):
        members = [r for r, l in zip(rels, labels) if l == cid]
        if not members:
            continue
        members.sort(key=lambda r: -counts[r])
        n = sum(counts[m] for m in members)
        classes.append({"id": int(cid), "name": members[0], "members": members,
                        "count": n, "share": round(n / total, 3)})
    classes.sort(key=lambda c: -c["count"])
    for i, c in enumerate(classes):
        c["id"] = i
    return classes


NAME_PROMPT = """These relationship phrases were all used by language models to connect entities, and
were grouped together as one cluster:

{members}

Give a SHORT label (1-3 words) naming the KIND of relationship this cluster represents — e.g.
"membership", "location or origin", "causal influence", "creation or authorship".
Return valid JSON only: {{ "name": "string" }}"""


async def name_classes(classes: list[dict], client, model: str, max_members: int = 12) -> list[dict]:
    """Have an LLM name each cluster from its members (the top member is a misleading label:
    the 'birthplace of' cluster actually holds located-in / headquartered-in / founded)."""
    import asyncio

    from src.dat_eval.llm import call_llm_async
    from src.kg_creat.judge import _extract_json

    async def one(c):
        prompt = NAME_PROMPT.format(members=", ".join(c["members"][:max_members]))
        raw = await call_llm_async(client, messages=[{"role": "user", "content": prompt}],
                                   model=model, temperature=0.0, max_tokens=300)
        obj = _extract_json(raw) if raw else None
        c["name"] = (obj or {}).get("name") or c["name"]
        return c

    named = list(await asyncio.gather(*[one(c) for c in classes]))
    # Disambiguate collisions (the LLM can hand two different clusters the same label, e.g.
    # 'affiliation' for both the influence cluster and the agency cluster).
    seen: Counter = Counter(c["name"].strip().lower() for c in named)
    used: Counter = Counter()
    for c in named:
        key = c["name"].strip().lower()
        if seen[key] > 1:
            used[key] += 1
            c["name"] = f"{c['name']} ({c['members'][0]})"
    return named


def _class_of(rel: str, classes: list[dict]) -> int | None:
    for c in classes:
        if rel in c["members"]:
            return c["id"]
    return None


def derive_targets(per_bundle: dict, per_bundle_seqs: dict, classes: list[dict],
                   min_share: float = 0.08) -> dict:
    """Per bundle: exclusion = most-used class, inclusion = least-used, ordering = reversed order.

    ``min_share`` keeps a constraint target from being an unusable shard: an inclusion target must
    hold a real share of the corpus, or "every path must include an X-type relation" is impossible
    rather than merely hard (which would floor the cell and look like a finding).
    """
    usable = [c for c in classes if c.get("share", 0) >= min_share]
    if not usable:
        usable = classes[:3]
    targets = {}
    for bid, counter in per_bundle.items():
        cls_use: Counter = Counter()
        for rel, n in counter.items():
            cid = _class_of(rel, classes)
            if cid is not None:
                cls_use[cid] += n
        if not cls_use:
            continue
        excl = max(cls_use, key=lambda c: cls_use[c])
        # inclusion (common): a substantial class this bundle leans on least (0 counts allowed)
        incl = min((c["id"] for c in usable if c["id"] != excl),
                   key=lambda cid: cls_use.get(cid, 0), default=None)
        # inclusion (rare): a niche class -- a deliberately harder difficulty tier, since
        # requiring a rarely-used kind of relation forces a route models almost never take.
        rare_pool = [c for c in classes if c.get("share", 0) < min_share and len(c["members"]) >= 2]
        incl_rare = min((c["id"] for c in rare_pool),
                        key=lambda cid: cls_use.get(cid, 0), default=None)

        # ordering: the most co-occurring class pair, constraint = reverse of the natural order
        pair_counts: Counter = Counter()
        for seq in per_bundle_seqs[bid]:
            cids = [_class_of(r, classes) for r in seq]
            cids = [c for c in cids if c is not None]
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    if cids[i] != cids[j]:
                        pair_counts[(cids[i], cids[j])] += 1
        order = None
        if pair_counts:
            (a, b), _ = pair_counts.most_common(1)[0]
            order = (b, a)  # REVERSE the models' natural order so the constraint bites
        targets[bid] = {"exclusion": excl, "inclusion": incl, "inclusion_rare": incl_rare,
                        "ordering": order, "class_usage": dict(cls_use)}
    return targets
