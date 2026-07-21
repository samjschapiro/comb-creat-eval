"""Structure-mapping validity for the semantic tier (analogy + blending).

Both tasks ask for two relational structures that are *parallel* -- the same relation sequence
carrying different entities (Gentner's structure-mapping floor). They differ only in what is
pinned:

    analogy   two stimuli (u, v)  -> one structure emanating from each; the structures must be
                                     entirely disjoint (two distinct systems, not one).
    blending  one stimulus (u)    -> two structures emanating outward from the SAME anchor into
                                     different domains; they must share the anchor and nothing else.

Keeping the predicates here (rather than in a plotting script) means the scorer and the figures
apply the identical definition of a valid structure.
"""

from __future__ import annotations


def _norm(x) -> str:
    return str(x).strip().lower()


def entities(triples: list) -> set[str]:
    return {_norm(triples[0][0])} | {_norm(t[2]) for t in triples}


def relations(triples: list) -> list[str]:
    return [_norm(t[1]) for t in triples]


def node_distinct(triples: list) -> bool:
    """A structure must not revisit an entity (rejects circular / self-referential 'analogies')."""
    if not triples:
        return False
    ents = [_norm(triples[0][0])] + [_norm(t[2]) for t in triples]
    return len(set(ents)) == len(ents)


def continuous(triples: list) -> bool:
    return bool(triples) and all(_norm(a[2]) == _norm(b[0]) for a, b in zip(triples, triples[1:]))


def relations_match(path_a: list, path_b: list) -> bool:
    """The mapping floor: the SAME relation at every position, not a paraphrase."""
    return bool(path_a) and bool(path_b) and relations(path_a) == relations(path_b)


def structures_disjoint(path_a: list, path_b: list) -> bool:
    """Analogy: the two structures may share no entity at all."""
    if not path_a or not path_b:
        return False
    return not (entities(path_a) & entities(path_b))


def blend_structural_ok(path_a: list, path_b: list, anchor: str) -> tuple[bool, str]:
    """Blending: two branches emanating from one anchor, sharing the anchor and nothing else.

    Returns ``(ok, reason)``; ``reason`` is the first failing gate, so failures are attributable
    rather than collapsed into a single boolean.
    """
    if not path_a or not path_b:
        return False, "missing_branch"
    a0, b0 = _norm(path_a[0][0]), _norm(path_b[0][0])
    if not (a0 == b0 == _norm(anchor)):
        return False, "branch_not_anchored"
    if not (continuous(path_a) and continuous(path_b)):
        return False, "discontinuous"
    if not (node_distinct(path_a) and node_distinct(path_b)):
        return False, "revisits_node"
    shared = entities(path_a) & entities(path_b)
    if shared != {_norm(anchor)}:
        return False, "branches_overlap"  # the branches must diverge, not retrace each other
    if not relations_match(path_a, path_b):
        return False, "relations_differ"
    return True, "ok"


def analogy_structural_ok(path_a: list, path_b: list) -> tuple[bool, str]:
    """Analogy's counterpart to :func:`blend_structural_ok`, same first-failing-gate contract."""
    if not path_a or not path_b:
        return False, "missing_structure"
    if not (continuous(path_a) and continuous(path_b)):
        return False, "discontinuous"
    if not (node_distinct(path_a) and node_distinct(path_b)):
        return False, "revisits_node"
    if not structures_disjoint(path_a, path_b):
        return False, "structures_overlap"
    if not relations_match(path_a, path_b):
        return False, "relations_differ"
    return True, "ok"


def branch_tips(path_a: list, path_b: list) -> tuple[str, str]:
    """The outward ends of the two branches -- how far the blend travelled from its anchor."""
    return str(path_a[-1][2]), str(path_b[-1][2])
