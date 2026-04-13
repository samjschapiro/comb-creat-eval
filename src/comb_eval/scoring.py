"""Intra-response diversity scoring for combinatorial creativity evaluation.

For each prompt the model returns k candidate paths. We validate each path
independently (structural validity + constraint satisfaction) and compute
diversity over the *valid subset*. If fewer than 2 valid paths remain the
prompt contributes NaN to the model's diversity aggregate — it is simply
dropped, not penalized, so capability and creativity stay decoupled.

Two diversity measures are computed in parallel:

- **Edge-set Jaccard distance**: 1 - |A ∩ B| / |A ∪ B| on the set of
  unordered node-pair edges. Captures structural overlap and ignores order.
- **Edge-label-sequence edit distance** (normalized Levenshtein): captures
  sequential novelty in edge-label traversal, closer in spirit to
  Hivemind's text-level similarity.

Both are averaged over all C(n, 2) pairs of valid paths.

`solve_rate` (fraction of returned paths that are valid) is reported
separately — a capability signal that must not contaminate diversity.
"""

import itertools
import math
from collections import Counter
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from src.comb_eval.graph import get_edge_label, label_to_node
from src.comb_eval.llm import LLMResponse
from src.comb_eval.prompts import EvalPrompt


# --- per-path validation ---


@dataclass
class PathValidation:
    """Validity breakdown for a single candidate path in a k-paths response."""

    valid: bool
    error_type: str  # 'none', 'empty', 'hallucinated_node', 'hallucinated_edge',
                     #  'wrong_endpoints', 'wrong_hops', 'include_violation', 'exclude_violation'
    node_ids: list[int] = field(default_factory=list)
    edge_labels: list[str] = field(default_factory=list)


def _validate_path(
    G: nx.Graph,
    prompt: EvalPrompt,
    path_labels: list[str],
    l2n: dict[str, int],
) -> PathValidation:
    """Validate a single candidate path against the prompt's constraints."""
    if not path_labels:
        return PathValidation(valid=False, error_type="empty")

    # Check for hallucinated nodes
    for nl in path_labels:
        if nl not in l2n:
            return PathValidation(valid=False, error_type="hallucinated_node")

    node_ids = [l2n[nl] for nl in path_labels]

    # Check edges exist
    actual_edge_labels: list[str] = []
    for i in range(len(node_ids) - 1):
        u, v = node_ids[i], node_ids[i + 1]
        if u == v or not G.has_edge(u, v):
            return PathValidation(valid=False, error_type="hallucinated_edge")
        actual_edge_labels.append(get_edge_label(G, u, v))

    # Endpoint + hop checks
    if path_labels[0] != prompt.start or path_labels[-1] != prompt.end:
        return PathValidation(
            valid=False, error_type="wrong_endpoints",
            node_ids=node_ids, edge_labels=actual_edge_labels,
        )
    if len(node_ids) - 1 != prompt.hop_count:
        return PathValidation(
            valid=False, error_type="wrong_hops",
            node_ids=node_ids, edge_labels=actual_edge_labels,
        )

    # Constraint checks
    label_set = set(actual_edge_labels)
    if not set(prompt.include_labels).issubset(label_set):
        return PathValidation(
            valid=False, error_type="include_violation",
            node_ids=node_ids, edge_labels=actual_edge_labels,
        )
    if set(prompt.exclude_labels) & label_set:
        return PathValidation(
            valid=False, error_type="exclude_violation",
            node_ids=node_ids, edge_labels=actual_edge_labels,
        )

    return PathValidation(
        valid=True, error_type="none",
        node_ids=node_ids, edge_labels=actual_edge_labels,
    )


# --- pairwise diversity primitives ---


def _edge_set(node_ids: list[int]) -> set[frozenset]:
    """Unordered edges as frozenset node-pairs — order-agnostic structural signature."""
    return {frozenset((node_ids[i], node_ids[i + 1])) for i in range(len(node_ids) - 1)}


def _jaccard_distance(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


def _levenshtein(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n]


def _norm_edit_distance(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 0.0
    return _levenshtein(a, b) / max(len(a), len(b))


# --- per-prompt diversity scoring ---


@dataclass
class DiversityScore:
    """Intra-response diversity score for one prompt.

    Attributes:
        n_returned: How many paths the model returned (may differ from k).
        n_valid: How many of the returned paths are valid.
        solve_rate: n_valid / n_returned (NaN if n_returned == 0). Capability signal.
        mean_jaccard: Mean pairwise edge-set Jaccard distance over valid paths.
            NaN if n_valid < 2.
        mean_edit_distance: Mean pairwise normalized Levenshtein on edge-label
            sequences over valid paths. NaN if n_valid < 2.
        error_distribution: Counter of per-path error types (incl. 'none').
        parse_success: Whether the response JSON parsed at all.
    """

    n_returned: int
    n_valid: int
    solve_rate: float
    mean_jaccard: float
    mean_edit_distance: float
    error_distribution: dict[str, int]
    parse_success: bool


def score_response(
    G: nx.Graph,
    prompt: EvalPrompt,
    response: LLMResponse,
) -> DiversityScore:
    """Score a single k-paths response for intra-response diversity."""
    if not response.parse_success or not response.paths:
        return DiversityScore(
            n_returned=0, n_valid=0,
            solve_rate=float("nan"),
            mean_jaccard=float("nan"),
            mean_edit_distance=float("nan"),
            error_distribution={"parse_failure": 1},
            parse_success=False,
        )

    l2n = label_to_node(G)

    validations = [_validate_path(G, prompt, p, l2n) for p in response.paths]
    valid = [v for v in validations if v.valid]
    n_returned = len(validations)
    n_valid = len(valid)

    err_counts = Counter(v.error_type for v in validations)

    solve_rate = n_valid / n_returned if n_returned > 0 else float("nan")

    if n_valid < 2:
        return DiversityScore(
            n_returned=n_returned, n_valid=n_valid,
            solve_rate=solve_rate,
            mean_jaccard=float("nan"),
            mean_edit_distance=float("nan"),
            error_distribution=dict(err_counts),
            parse_success=True,
        )

    edge_sets = [_edge_set(v.node_ids) for v in valid]
    label_seqs = [v.edge_labels for v in valid]

    jaccard_dists = [
        _jaccard_distance(edge_sets[i], edge_sets[j])
        for i, j in itertools.combinations(range(n_valid), 2)
    ]
    edit_dists = [
        _norm_edit_distance(label_seqs[i], label_seqs[j])
        for i, j in itertools.combinations(range(n_valid), 2)
    ]

    return DiversityScore(
        n_returned=n_returned,
        n_valid=n_valid,
        solve_rate=solve_rate,
        mean_jaccard=float(np.mean(jaccard_dists)),
        mean_edit_distance=float(np.mean(edit_dists)),
        error_distribution=dict(err_counts),
        parse_success=True,
    )


def score_eval_set(
    G: nx.Graph,
    prompts: list[EvalPrompt],
    responses: list[LLMResponse],
) -> list[DiversityScore]:
    """Score a full evaluation set.

    Raises:
        ValueError: If prompts and responses have different lengths.
    """
    if len(prompts) != len(responses):
        raise ValueError(
            f"Mismatch: {len(prompts)} prompts but {len(responses)} responses"
        )
    return [score_response(G, p, r) for p, r in zip(prompts, responses)]


# --- aggregation ---


def _nanmean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def aggregate_scores(scores: list[DiversityScore]) -> dict:
    """Compute aggregate statistics over a set of scores.

    Returns a dict with:
        - n_prompts: total prompts scored.
        - n_scorable: prompts with n_valid >= 2 (contributed to diversity).
        - mean_jaccard: mean pairwise Jaccard distance across scorable prompts.
        - mean_edit_distance: mean pairwise normalized edit distance, scorable.
        - mean_solve_rate: mean solve_rate across prompts where n_returned > 0.
        - parse_success_rate: fraction of prompts where the JSON parsed.
        - error_distribution: total per-path error counts across all prompts.
    """
    if not scores:
        return {"n_prompts": 0}

    jaccards = [s.mean_jaccard for s in scores]
    edits = [s.mean_edit_distance for s in scores]
    solves = [s.solve_rate for s in scores]

    n_scorable = sum(1 for s in scores if s.n_valid >= 2)

    error_totals: Counter = Counter()
    for s in scores:
        error_totals.update(s.error_distribution)

    return {
        "n_prompts": len(scores),
        "n_scorable": n_scorable,
        "mean_jaccard": _nanmean(jaccards),
        "mean_edit_distance": _nanmean(edits),
        "mean_solve_rate": _nanmean(solves),
        "parse_success_rate": sum(1 for s in scores if s.parse_success) / len(scores),
        "error_distribution": dict(error_totals),
    }
