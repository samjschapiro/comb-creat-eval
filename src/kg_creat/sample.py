"""Matched-bundle sampler for the constraint benchmark.

Regime A (constraints I-IV / A-D): for each hub-sharing endpoint pair ``(u, v)`` at a
fixed hop count ``h``, enumerate the feasible route set on ``G_c`` and search for a
*biting-but-satisfiable* parameter of each constraint type. A constraint is

    satisfiable  iff >= 1 feasible route satisfies it   (a valid answer exists), and
    biting       iff >= 1 feasible route violates it     (it removes obvious routes).

The unit is an **endpoint bundle**: {baseline + one prompt per available constraint} on a
fixed ``(u, v, h)``, so within-bundle deltas isolate the constraint *type* from entity
difficulty. Every constraint records its biting stats (feasible-fraction covariate).

Regime B (analogy / blending) uses curated cross-domain seeds (see ``sample_regime_b``);
its pivot-first automation is a later enhancement.

Operates on the graph-side :class:`~src.kg_creat.graph.LabeledWalk` route set (ground
truth on ``G_c``) -- distinct from ``scoring.py``, which scores the model's emitted path.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter

from src.kg_creat.graph import KnowledgeGraph, LabeledWalk

_TAXONOMIC = ("P279", "P31")  # subclass-of / instance-of: the classic shortcut exclusion blocks
# Relations whose *targets* are class-like (types / occupations / fields) -> poor endpoints.
_CLASSLIKE_RELS = {"P31", "P279", "P106", "P101"}


# ----------------------------------------------------------------------------- endpoints

def _undirected_degree(gc: KnowledgeGraph) -> dict[str, int]:
    return {n: len(gc.adjacency(n)) for n in gc.nodes()}


def _class_nodes(gc: KnowledgeGraph) -> set[str]:
    """Nodes that are the *target* of a class-like relation (types, occupations, fields).

    A node is class-like if something is instance-of / subclass-of / has-occupation /
    has-field-of-work pointing to it -- e.g. 'human', 'physicist', 'philosopher',
    'physics'. These are categories, not concept endpoints, so we exclude them from
    endpoint selection (they remain valid interiors / categorical types).
    """
    classes: set[str] = set()
    for n in gc.nodes():
        for _neighbor, rel, forward in gc.adjacency(n):
            if rel in _CLASSLIKE_RELS and not forward:  # forward=False => n is the tail
                classes.add(n)
                break
    return classes


def _prominent_nodes(gc: KnowledgeGraph, min_degree: int) -> list[str]:
    """Non-class nodes with degree >= min_degree, sorted by degree (real concept endpoints)."""
    deg = _undirected_degree(gc)
    classes = _class_nodes(gc)
    return sorted((n for n in gc.nodes() if deg[n] >= min_degree and n not in classes), key=lambda n: -deg[n])


def endpoint_pairs(gc: KnowledgeGraph, h: int, min_routes: int, min_degree: int, max_paths: int):
    """Yield (u, v, routes) for prominent, non-class node pairs with >= min_routes h-hop routes."""
    prominent = _prominent_nodes(gc, min_degree)
    for u, v in itertools.combinations(prominent, 2):
        routes = gc.enumerate_paths(u, v, hop_count=h, max_paths=max_paths)
        if len(routes) >= min_routes:
            yield u, v, routes


# ------------------------------------------------------------ biting-constraint finders

def _relation_presence(routes: list[LabeledWalk]) -> Counter:
    """relation id -> number of routes that contain it."""
    pres: Counter = Counter()
    for w in routes:
        pres.update(w.relation_set())
    return pres


def find_exclusion(gc: KnowledgeGraph, routes: list[LabeledWalk]) -> dict | None:
    """Forbid a relation present in some (but not all) routes; prefer the taxonomic shortcut."""
    n = len(routes)
    cands = [(r, c) for r, c in _relation_presence(routes).items() if 0 < c < n]
    if not cands:
        return None
    r, c = max(cands, key=lambda rc: (rc[0] in _TAXONOMIC, rc[1]))  # taxonomic first, else most-removed
    return {"type": "exclusion", "relation": r, "relation_label": gc.relation_label(r),
            "n_satisfy": n - c, "n_violate": c, "biting_fraction": round(c / n, 3)}


def find_inclusion(gc: KnowledgeGraph, routes: list[LabeledWalk], target_frac: float = 0.34) -> dict | None:
    """Require a *content* relation present in ~a third of routes.

    Excludes taxonomic relations (instance-of / subclass-of) so inclusion demands a
    meaningful content link and stays complementary to exclusion (which blocks the
    taxonomic shortcut) -- the two never collapse onto the same relation.
    """
    n = len(routes)
    cands = [(r, c) for r, c in _relation_presence(routes).items() if 0 < c < n and r not in _TAXONOMIC]
    if not cands:
        return None
    r, c = min(cands, key=lambda rc: abs(rc[1] - n * target_frac))  # closest to target frequency
    return {"type": "inclusion", "relation": r, "relation_label": gc.relation_label(r),
            "n_satisfy": c, "n_violate": n - c, "biting_fraction": round((n - c) / n, 3)}


def find_categorical(gc: KnowledgeGraph, routes: list[LabeledWalk]) -> dict | None:
    """Require an interior entity of a type present in some (but not all) routes; most-contrastive."""
    n = len(routes)
    pres: Counter = Counter()
    for w in routes:
        tset: set[str] = set()
        for node in w.nodes[1:-1]:  # interior only
            tset.update(gc.types(node))
        pres.update(tset)
    cands = [(t, c) for t, c in pres.items() if 0 < c < n]
    if not cands:
        return None
    t, c = min(cands, key=lambda tc: abs(tc[1] - n / 2))  # most contrastive
    return {"type": "categorical", "entity_type": t, "type_label": gc.type_label(t), "sign": "include",
            "n_satisfy": c, "n_violate": n - c, "biting_fraction": round((n - c) / n, 3)}


def find_ordering(gc: KnowledgeGraph, routes: list[LabeledWalk]) -> dict | None:
    """Require relation a before b, chosen where co-present routes genuinely differ in order."""
    n = len(routes)
    rels_per = [w.relations for w in routes]
    vocab = {r for rr in rels_per for r in rr}
    best = None
    for a, b in itertools.permutations(vocab, 2):
        copresent = [rr for rr in rels_per if a in rr and b in rr]
        if len(copresent) < 2:
            continue
        ab = sum(1 for rr in copresent if rr.index(a) < rr.index(b))
        ba = len(copresent) - ab
        if ab >= 1 and ba >= 1:  # genuine order contrast among co-present routes
            score = (len(copresent), min(ab, ba))
            if best is None or score > best[0]:
                best = (score, a, b, ab)
    if best is None:
        return None
    _, a, b, ab = best
    return {"type": "ordering", "before": a, "after": b,
            "before_label": gc.relation_label(a), "after_label": gc.relation_label(b),
            "n_satisfy": ab, "n_violate": n - ab, "biting_fraction": round((n - ab) / n, 3)}


_FINDERS = {"exclusion": find_exclusion, "inclusion": find_inclusion,
            "categorical": find_categorical, "ordering": find_ordering}


# --------------------------------------------------------------------------- assembly

def build_bundle(gc: KnowledgeGraph, u: str, v: str, h: int, k: int,
                 routes: list[LabeledWalk], bundle_id: str) -> dict:
    """Assemble a matched bundle: baseline + every constraint type that bites-and-satisfies."""
    cells = {"baseline": {"mode": "baseline", "constraint": None}}
    for name, finder in _FINDERS.items():
        found = finder(gc, routes)
        if found is not None:
            cells[name] = {"mode": name, "constraint": found}
    return {
        "bundle_id": bundle_id, "regime": "A",
        "u": u, "v": v, "u_label": gc.label(u), "v_label": gc.label(v),
        "h": h, "k": k, "n_routes_baseline": len(routes),
        "n_constraints": len(cells) - 1, "cells": cells,
    }


def sample_matched_bundles(gc: KnowledgeGraph, h: int = 3, k: int = 5, min_routes: int = 4,
                           min_degree: int = 3, max_bundles: int = 10, max_paths: int = 64,
                           require_constraints: int = 3, max_per_source: int = 2) -> list[dict]:
    """Sample up to ``max_bundles`` matched bundles, best (most constraint cells) first.

    A bundle is kept only if it supports >= ``require_constraints`` of the four A-D types,
    so within-bundle deltas compare like with like. ``max_per_source`` caps how many
    bundles share the same source entity, for endpoint diversity.
    """
    qualifying = []
    for u, v, routes in endpoint_pairs(gc, h, min_routes, min_degree, max_paths):
        b = build_bundle(gc, u, v, h, k, routes, bundle_id="")
        if b["n_constraints"] >= require_constraints:
            qualifying.append(b)
    qualifying.sort(key=lambda b: (b["n_constraints"], b["n_routes_baseline"]), reverse=True)

    selected, per_source = [], Counter()
    for b in qualifying:
        if per_source[b["u"]] >= max_per_source:
            continue
        per_source[b["u"]] += 1
        b["bundle_id"] = f"A{len(selected)}"
        selected.append(b)
        if len(selected) >= max_bundles:
            break
    print(f"[sample] {len(qualifying)} qualifying bundles; kept {len(selected)} "
          f"(<= {max_per_source} per source, <= {max_bundles} total)")
    return selected


def sample_random_bundles(gc: KnowledgeGraph, n_bundles: int = 40, k: int = 5, seed: int = 0,
                          min_degree: int = 3, max_per_source: int = 3,
                          entity_domains: dict[str, str] | None = None) -> list[dict]:
    """Sample Regime-A endpoints as RANDOM pairs of arbitrary entities -- like the analogy task.

    Deliberately drops the matched-bundle machinery (route-existence and constraint-biting
    verification). Verifying a path exists between ``u`` and ``v`` selects for the unsurprising
    pairs and defeats the point of combinatorial creativity, which is to connect things that are
    *not* obviously connected; and biting no longer needs a graph guarantee because Pass-2 targets
    are derived from each model's own baseline behaviour on the pair. So we draw random pairs from
    the graph's real (non-class) entities and let baseline success be a *measured* quantity, not an
    engineered precondition -- the same posture as ``sample_regime_b``.

    The only floor kept is a light per-node one (``min_degree``, class nodes excluded) so each
    endpoint is a real, linkable entity the model and judge can engage with -- identical to the
    analogy sampler. It is not a constraint on the *pair*.
    """
    rng = random.Random(seed)
    ed = entity_domains or {}
    pool = _prominent_nodes(gc, min_degree)
    if len(pool) < 2:
        raise ValueError("FATAL: not enough prominent nodes in G_c for random Regime-A sampling")

    seen, per_source, bundles = set(), Counter(), []
    max_unique = len(pool) * (len(pool) - 1) // 2
    while len(bundles) < min(n_bundles, max_unique):
        u, v = rng.sample(pool, 2)
        key = frozenset((u, v))
        if key in seen or per_source[u] >= max_per_source:
            continue
        seen.add(key)
        per_source[u] += 1
        du, dv = ed.get(u), ed.get(v)
        bundles.append({
            "bundle_id": f"A{len(bundles)}", "regime": "A", "u": u, "v": v,
            "u_label": gc.label(u), "v_label": gc.label(v), "h": None, "k": k,
            "domain_u": du, "domain_v": dv, "cross_domain": (du != dv) if (du and dv) else None,
            # Only the baseline cell; constrained cells are added by make_pass2 from baseline behaviour.
            "cells": {"baseline": {"constraint": None}},
        })
    print(f"[sample] {len(bundles)} random Regime-A endpoint pairs from a pool of {len(pool)} "
          f"entities (min_degree={min_degree}, <= {max_per_source} per source)")
    return bundles


# ------------------------------------------------------------------------- Regime B

def sample_regime_b(gc: KnowledgeGraph, n_analogy: int, n_blend: int, seed: int = 0,
                    min_degree: int = 3, entity_domains: dict[str, str] | None = None) -> list[dict]:
    """Randomly sample the semantic tier's stimuli from G_c (seeded, reproducible).

    Analogy draws entity PAIRS; blending draws single ANCHORS (the model supplies both directions).

    The task is deliberately hard: can a model find a valid analogy (or blend) between two
    *arbitrary*, non-obviously-related entities? Hand-curating canonical pairs (atom/solar-system)
    would trivialize it -- it would test analogies we already know exist. So we draw random pairs
    from the graph's real (non-class) entities; the difficulty is the point. Each pair is tagged
    with its endpoints' domains (a study variable) so success can be analyzed within- vs cross-domain.
    """
    rng = random.Random(seed)
    ed = entity_domains or {}
    pool = _prominent_nodes(gc, min_degree)
    if len(pool) < 2:
        raise ValueError("FATAL: not enough prominent nodes in G_c for Regime-B sampling")

    def rand_pairs(n: int) -> list[tuple[str, str]]:
        seen, pairs = set(), []
        max_unique = len(pool) * (len(pool) - 1) // 2
        while len(pairs) < min(n, max_unique):
            u, v = rng.sample(pool, 2)
            key = frozenset((u, v))
            if key not in seen:
                seen.add(key)
                pairs.append((u, v))
        return pairs

    def spec(prefix, i, mode, u, v, constraint):
        du, dv = ed.get(u), ed.get(v)
        return {"bundle_id": f"{prefix}{i}", "regime": "B", "mode": mode, "u": u, "v": v,
                "u_label": gc.label(u), "v_label": gc.label(v) if v else None, "constraint": constraint,
                "domain_u": du, "domain_v": dv,
                "cross_domain": (du != dv) if (du and dv) else None}

    specs = [spec("E", i, "analogy", u, v, None) for i, (u, v) in enumerate(rand_pairs(n_analogy))]
    # Blending is single-stimulus: one anchor, and the model must find BOTH directions itself.
    # Its second domain is therefore an outcome to be measured, not a design variable we set.
    anchors = rng.sample(pool, min(n_blend, len(pool)))
    specs += [spec("F", i, "blending", u, None, {"type": "blending"}) for i, u in enumerate(anchors)]
    return specs
