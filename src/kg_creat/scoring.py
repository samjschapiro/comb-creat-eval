"""Constraint checking + novelty scoring over emitted KG paths (kg_creat).

Operates on a model-emitted path as an ordered list of ``(head, relation, tail)``
triples — the output format CREATE uses — so this plugs directly onto a CREATE-style
parser downstream (see ``docs/tracks/kg_creat/methods.md``). This module implements the
**syntactic** constraints (taxonomy I–IV: exclusion / inclusion / ordering / metapath /
waypoint / hub-avoidance / categorical / depth / budget / disjointness), all checkable
*exactly* on the emitted path, plus the within-path novelty ``R``.

The **semantic** constraints — analogy (V) and blending (VI) — require a judge
(is the isomorphism / fusion genuine), so they are *not* implemented in this exact checker;
see ``docs/tracks/kg_creat/constraints.md`` §Operationalizing analogy and blending.

Conventions (per methods.md):
- **Relation labels** are matched against a controlled vocabulary — exact after normalization.
- **Named entities** (waypoint / hub) are matched CREATE-style — normalized substring, with
  optional aliases.
- Constraint satisfaction here is **deterministic and judge-free**; only factuality (a
  separate step) is judged. This is what keeps the execution axis clean for I–IV.

Novelty is computed downstream over the parsed path, so the embedding **unit**
(``concept`` = entities ``(c,c')`` vs ``triple`` = ``(c,r,c')`` sentences) is a post-hoc
ablation, not a pre-commit — persist the full triples and both are recoverable.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np


def _norm(s: str) -> str:
    """Normalize a label for matching: string, stripped, lowercased."""
    return str(s).strip().lower()


def _entity_matches(name: str, emitted: str, aliases: dict[str, list[str]] | None = None) -> bool:
    """Whether an emitted entity string refers to `name` (normalized substring + aliases).

    CREATE-style leniency: a match holds if a candidate label is a substring of the emitted
    string or vice versa (handles qualifiers like "Michael Jordan (basketball)").
    """
    aliases = aliases or {}
    cands = {_norm(name)} | {_norm(a) for a in aliases.get(name, [])}
    e = _norm(emitted)
    return any(c in e or e in c for c in cands if c)


# --- emitted-path object ---


@dataclass
class EmittedPath:
    """A model-emitted path as an ordered list of ``(head, relation, tail)`` triples."""

    triples: list[tuple[str, str, str]]

    @property
    def relations(self) -> list[str]:
        """Normalized relation label per hop."""
        return [_norm(t[1]) for t in self.triples]

    @property
    def entities(self) -> list[str]:
        """Ordered entity spine ``[e0, e1, ..., eh]`` — head of the first triple + each tail.

        Assumes structural continuity (checked by :func:`well_formed`); on a discontinuous
        path this still returns head0 + tails, which the continuity check will reject.
        """
        if not self.triples:
            return []
        return [_norm(self.triples[0][0])] + [_norm(t[2]) for t in self.triples]

    @property
    def interior(self) -> list[str]:
        """Interior entities ``{e1, ..., e_{h-1}}`` — endpoints excluded (they are fixed)."""
        ents = self.entities
        return ents[1:-1] if len(ents) > 2 else []

    @property
    def hop_count(self) -> int:
        return len(self.triples)

    def triple_sentences(self) -> list[str]:
        """Each triple verbalized as ``"head rel tail"`` (the (c,r,c') novelty unit)."""
        return [f"{_norm(h)} {_norm(r)} {_norm(t)}" for h, r, t in self.triples]


# --- structural well-formedness (Level 0 of the leveled utility) ---


def is_continuous(path: EmittedPath) -> bool:
    """Each triple's tail refers to the next triple's head (normalized substring, CREATE-style)."""
    for a, b in zip(path.triples, path.triples[1:]):
        tail_a, head_b = _norm(a[2]), _norm(b[0])
        if not (tail_a in head_b or head_b in tail_a):
            return False
    return True


def well_formed(
    path: EmittedPath,
    u: str,
    v: str,
    h: int | None = None,
    aliases: dict[str, list[str]] | None = None,
) -> tuple[bool, str]:
    """Level-0 coherence: right endpoints, continuous, node-distinct (variable length).

    Path length is NOT constrained by default (``h=None``), matching CREATE's variable-length
    connection paths. Pass an integer ``h`` only if a task variant fixes an exact hop count.

    Returns ``(ok, reason)`` where reason is ``"ok"`` or a failure channel
    (``empty`` / ``malformed_triple`` / ``wrong_start`` / ``wrong_end`` / ``wrong_hops`` /
    ``discontinuous`` / ``revisits_node``).
    """
    if not path.triples:
        return False, "empty"
    for t in path.triples:
        if len(t) != 3 or any(x is None for x in t):
            return False, "malformed_triple"
    ents = path.entities
    if not _entity_matches(u, ents[0], aliases):
        return False, "wrong_start"
    if not _entity_matches(v, ents[-1], aliases):
        return False, "wrong_end"
    if h is not None and path.hop_count != h:
        return False, "wrong_hops"
    if not is_continuous(path):
        return False, "discontinuous"
    if len(set(ents)) != len(ents):
        return False, "revisits_node"
    return True, "ok"


# --- syntactic constraint predicates (exact on the emitted path) ---
# Relation-level: exclusion / inclusion / ordering / metapath / budget.


def exclusion(path: EmittedPath, forbidden: set[str] | list[str]) -> bool:
    """No forbidden relation label appears (avoid-the-obvious)."""
    X = {_norm(x) for x in forbidden}
    return all(r not in X for r in path.relations)


def inclusion(path: EmittedPath, required: set[str] | list[str]) -> bool:
    """Every required relation label appears at least once."""
    I = {_norm(x) for x in required}
    return I.issubset(set(path.relations))


def ordering(path: EmittedPath, before: str, after: str) -> bool:
    """First occurrence of `before` precedes first occurrence of `after` (both present)."""
    rels = path.relations
    a, b = _norm(before), _norm(after)
    if a not in rels or b not in rels:
        return False
    return rels.index(a) < rels.index(b)


def metapath(path: EmittedPath, template: list[str]) -> bool:
    """Relation sequence matches a template of labels / ``"*"`` wildcards (analogy floor)."""
    if len(template) != path.hop_count:
        return False
    return all(m == "*" or _norm(m) == r for m, r in zip(template, path.relations))


def budget(path: EmittedPath, cost: dict[str, float], limit: float) -> bool:
    """Total relation cost within budget (exclusion is the special case ``cost=inf``)."""
    cost = {_norm(k): v for k, v in cost.items()}
    return sum(cost.get(r, 0.0) for r in path.relations) <= limit


# Entity-level: waypoint / hub-avoidance / categorical.


def waypoint(
    path: EmittedPath,
    required: list[str],
    aliases: dict[str, list[str]] | None = None,
    mode: str = "all",
) -> bool:
    """Path passes through the required waypoint entities (``mode`` = ``all`` or ``any``)."""
    interior = path.interior

    def present(w: str) -> bool:
        return any(_entity_matches(w, e, aliases) for e in interior)

    hits = [present(w) for w in required]
    return all(hits) if mode == "all" else any(hits)


def hub_avoidance(
    path: EmittedPath,
    hubs: list[str],
    aliases: dict[str, list[str]] | None = None,
) -> bool:
    """Path avoids all named hub entities (interior only)."""
    return not any(_entity_matches(hb, e, aliases) for hb in hubs for e in path.interior)


def categorical(
    path: EmittedPath,
    type_: str,
    type_map: dict[str, set[str] | list[str]],
    sign: str = "include",
) -> bool:
    """Path passes through (``include``) / avoids (``exclude``) an interior entity of type `type_`.

    ``type_map`` maps a normalized entity label to its set of type labels. This is the one
    syntactic constraint needing external metadata (KG types); if that metadata is unavailable
    the check is delegated to the judge (not handled here).
    """
    T = _norm(type_)

    def has_T(e: str) -> bool:
        return T in {_norm(x) for x in type_map.get(e, set())}

    hit = any(has_T(e) for e in path.interior)
    return hit if sign == "include" else not hit


def depth(path: EmittedPath, geodesic: int, delta: int = 1) -> bool:
    """Path is at least ``geodesic + delta`` hops (non-obviousness / elaboration)."""
    return path.hop_count >= geodesic + delta


# Set-level: disjointness across the k returned paths.


def disjointness(paths: list[EmittedPath], mode: str = "node") -> bool:
    """The k paths are pairwise distinct — interior-node-disjoint (``node``) or distinct
    relation sets (``relation``)."""
    if mode == "node":
        sets = [set(p.interior) for p in paths]
        return all(sets[i].isdisjoint(sets[j]) for i, j in itertools.combinations(range(len(sets)), 2))
    if mode == "relation":
        sets = [frozenset(p.relations) for p in paths]
        return all(sets[i] != sets[j] for i, j in itertools.combinations(range(len(sets)), 2))
    raise ValueError(f"mode must be 'node' or 'relation', got {mode!r}")


# --- within-path novelty R ---


def novelty_R(
    path: EmittedPath,
    embed: Callable[[str], np.ndarray],
    unit: str = "triple",
    drop_endpoints: bool = False,
) -> float:
    """Within-path novelty = mean pairwise cosine distance over the path's items (DAT primitive).

    Args:
        path: the emitted path.
        embed: callable mapping a string to an embedding vector (SBERT at runtime; any
            deterministic embedding works for testing).
        unit: ``"concept"`` embeds the entities ``(c,c')``; ``"triple"`` embeds the
            ``"head rel tail"`` sentences ``(c,r,c')``. This is the post-hoc ablation axis.
        drop_endpoints: for ``unit="concept"``, exclude the fixed ``u,v`` endpoints so novelty
            is driven by the intermediates (no effect for ``unit="triple"``).

    Returns:
        Mean pairwise cosine distance in ``[0, 2]`` (typically ``[0, 1]`` for normalized
        embeddings), or ``nan`` if fewer than 2 items.
    """
    if unit == "concept":
        items = path.entities
        if drop_endpoints and len(items) > 2:
            items = items[1:-1]
    elif unit == "triple":
        items = path.triple_sentences()
    else:
        raise ValueError(f"unit must be 'concept' or 'triple', got {unit!r}")

    if len(items) < 2:
        return float("nan")

    vecs = np.array([np.asarray(embed(x), dtype=float) for x in items])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normed = vecs / np.where(norms == 0.0, 1.0, norms)
    dists = [
        1.0 - float(np.dot(normed[i], normed[j]))
        for i, j in itertools.combinations(range(len(items)), 2)
    ]
    return float(np.mean(dists))
