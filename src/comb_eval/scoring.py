"""Scoring module for combinatorial creativity evaluation.

Implements the Creativity = Utility x Novelty metric from arXiv:2509.21043.
Utility is binary-gated by constraint satisfaction; Novelty combines path
length and edge-label surprise.
"""

import math
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.comb_eval.graph import get_edge_label, label_to_node
from src.comb_eval.llm import LLMResponse
from src.comb_eval.prompts import EvalPrompt


@dataclass
class PathScore:
    """Score breakdown for a single path response.

    Attributes:
        creativity: Final creativity score (utility * novelty).
        utility: Utility score (0 or positive). Zero if any constraint violated.
        novelty: Novelty score combining hop count and edge label surprise.
        valid_path: Whether the path is structurally valid on the graph.
        correct_start: Whether path starts at the required node.
        correct_end: Whether path ends at the required node.
        correct_hops: Whether path has the required number of hops.
        includes_satisfied: Whether all inclusion constraints are met.
        excludes_satisfied: Whether no exclusion constraints are violated.
        error_type: Type of error if any ('none', 'parse_failure', 'hallucinated_node',
                     'hallucinated_edge', 'invalid_path', 'wrong_endpoints', 'wrong_hops',
                     'constraint_violation').
        hop_count: Actual number of hops in the response path.
        surprise: Average edge label surprise score.
    """

    creativity: float
    utility: float
    novelty: float
    valid_path: bool
    correct_start: bool
    correct_end: bool
    correct_hops: bool
    includes_satisfied: bool
    excludes_satisfied: bool
    error_type: str
    hop_count: int
    surprise: float


def compute_edge_label_distribution(G: nx.Graph) -> dict[str, float]:
    """Compute the empirical edge label frequency distribution.

    Returns:
        Dict mapping each edge label to its frequency (sums to 1).
    """
    label_counts = Counter()
    for _, _, data in G.edges(data=True):
        label_counts[data["label"]] += 1
    total = sum(label_counts.values())
    return {label: count / total for label, count in label_counts.items()}


def compute_surprise(edge_labels: list[str], label_dist: dict[str, float]) -> float:
    """Compute average surprise of edge labels in a path.

    Surprise for a label l = -log(frequency(l)).
    Average surprise S(P) = (1/k) * sum(-log(w_l)) for k edges.

    Args:
        edge_labels: List of edge labels used in the path.
        label_dist: Empirical label frequency distribution.

    Returns:
        Average surprise score. Higher = rarer labels used.
    """
    if not edge_labels:
        return 0.0

    surprises = []
    for label in edge_labels:
        freq = label_dist.get(label, 1e-10)  # small epsilon for unseen labels
        surprises.append(-math.log(freq))

    return sum(surprises) / len(surprises)


def compute_novelty(
    hop_count: int,
    edge_labels: list[str],
    label_dist: dict[str, float],
    alpha_h: float = 1.0,
    alpha_r: float = 1.0,
) -> float:
    """Compute the novelty score for a path.

    N(P) = alpha_h * h + alpha_r * S(P)

    where h = hop count (longer paths explore more distant connections)
    and S(P) = average edge label surprise.

    Args:
        hop_count: Number of hops in the path.
        edge_labels: Edge labels used in the path.
        label_dist: Empirical edge label frequency distribution.
        alpha_h: Weight for hop count component.
        alpha_r: Weight for surprise component.

    Returns:
        Novelty score.
    """
    surprise = compute_surprise(edge_labels, label_dist)
    return alpha_h * hop_count + alpha_r * surprise


def compute_utility(
    prompt: EvalPrompt,
    constraints_met: bool,
    alpha_I: float = 1.0,
    alpha_X: float = 1.0,
) -> float:
    """Compute the utility score for a response.

    U(P; x) = [1 + alpha_I * |I|] * [1 + alpha_X * |X|] * indicator(constraints)

    Utility is zero if any constraint is violated (binary gate), but scales
    multiplicatively with the number of constraints when all are satisfied.

    Args:
        prompt: The eval prompt with constraint specifications.
        constraints_met: Whether all constraints (endpoints, hops, include/exclude) are met.
        alpha_I: Weight scaling for inclusion constraints.
        alpha_X: Weight scaling for exclusion constraints.

    Returns:
        Utility score (0 if constraints violated, positive otherwise).
    """
    if not constraints_met:
        return 0.0

    n_include = len(prompt.include_labels)
    n_exclude = len(prompt.exclude_labels)

    return (1 + alpha_I * n_include) * (1 + alpha_X * n_exclude)


def score_response(
    G: nx.Graph,
    prompt: EvalPrompt,
    response: LLMResponse,
    label_dist: dict[str, float],
    alpha_h: float = 1.0,
    alpha_r: float = 1.0,
    alpha_I: float = 1.0,
    alpha_X: float = 1.0,
) -> PathScore:
    """Score a single LLM response against its prompt.

    Validates the response path against the graph and prompt constraints,
    then computes creativity = utility * novelty.

    Args:
        G: The graph.
        prompt: The evaluation prompt.
        response: The parsed LLM response.
        label_dist: Empirical edge label frequency distribution.
        alpha_h: Novelty weight for hop count.
        alpha_r: Novelty weight for surprise.
        alpha_I: Utility weight for inclusion constraints.
        alpha_X: Utility weight for exclusion constraints.

    Returns:
        PathScore with full breakdown.
    """
    # Handle parse failures
    if not response.parse_success or not response.path:
        return PathScore(
            creativity=0.0, utility=0.0, novelty=0.0,
            valid_path=False, correct_start=False, correct_end=False,
            correct_hops=False, includes_satisfied=False, excludes_satisfied=False,
            error_type="parse_failure", hop_count=0, surprise=0.0,
        )

    # Build label-to-node mapping
    l2n = label_to_node(G)

    # Check for hallucinated nodes
    for node_label in response.path:
        if node_label not in l2n:
            return PathScore(
                creativity=0.0, utility=0.0, novelty=0.0,
                valid_path=False, correct_start=False, correct_end=False,
                correct_hops=False, includes_satisfied=False, excludes_satisfied=False,
                error_type="hallucinated_node", hop_count=len(response.path) - 1, surprise=0.0,
            )

    # Convert path to node IDs and validate edges
    node_ids = [l2n[label] for label in response.path]
    actual_edge_labels = []

    for i in range(len(node_ids) - 1):
        u, v = node_ids[i], node_ids[i + 1]
        if not G.has_edge(u, v):
            return PathScore(
                creativity=0.0, utility=0.0, novelty=0.0,
                valid_path=False, correct_start=False, correct_end=False,
                correct_hops=False, includes_satisfied=False, excludes_satisfied=False,
                error_type="hallucinated_edge" if (u != v) else "invalid_path",
                hop_count=len(response.path) - 1, surprise=0.0,
            )
        actual_edge_labels.append(get_edge_label(G, u, v))

    # Path is structurally valid on the graph
    valid_path = True
    actual_hops = len(response.path) - 1

    # Check endpoint constraints
    correct_start = response.path[0] == prompt.start
    correct_end = response.path[-1] == prompt.end
    correct_hops = actual_hops == prompt.hop_count

    # Check inclusion/exclusion constraints
    label_set = set(actual_edge_labels)
    includes_satisfied = set(prompt.include_labels).issubset(label_set)
    excludes_satisfied = len(set(prompt.exclude_labels) & label_set) == 0

    all_constraints_met = (
        correct_start and correct_end and correct_hops
        and includes_satisfied and excludes_satisfied
    )

    # Determine error type
    if all_constraints_met:
        error_type = "none"
    elif not correct_start or not correct_end:
        error_type = "wrong_endpoints"
    elif not correct_hops:
        error_type = "wrong_hops"
    else:
        error_type = "constraint_violation"

    # Compute scores
    surprise = compute_surprise(actual_edge_labels, label_dist)
    novelty = compute_novelty(actual_hops, actual_edge_labels, label_dist, alpha_h, alpha_r)
    utility = compute_utility(prompt, all_constraints_met, alpha_I, alpha_X)
    creativity = utility * novelty

    return PathScore(
        creativity=creativity,
        utility=utility,
        novelty=novelty,
        valid_path=valid_path,
        correct_start=correct_start,
        correct_end=correct_end,
        correct_hops=correct_hops,
        includes_satisfied=includes_satisfied,
        excludes_satisfied=excludes_satisfied,
        error_type=error_type,
        hop_count=actual_hops,
        surprise=surprise,
    )


def score_eval_set(
    G: nx.Graph,
    prompts: list[EvalPrompt],
    responses: list[LLMResponse],
    alpha_h: float = 1.0,
    alpha_r: float = 1.0,
    alpha_I: float = 1.0,
    alpha_X: float = 1.0,
) -> list[PathScore]:
    """Score a full evaluation set.

    Args:
        G: The graph.
        prompts: List of evaluation prompts.
        responses: List of LLM responses (same length as prompts).
        alpha_h: Novelty weight for hop count.
        alpha_r: Novelty weight for surprise.
        alpha_I: Utility weight for inclusion constraints.
        alpha_X: Utility weight for exclusion constraints.

    Returns:
        List of PathScore objects.

    Raises:
        ValueError: If prompts and responses have different lengths.
    """
    if len(prompts) != len(responses):
        raise ValueError(
            f"Mismatch: {len(prompts)} prompts but {len(responses)} responses"
        )

    label_dist = compute_edge_label_distribution(G)

    return [
        score_response(G, prompt, response, label_dist, alpha_h, alpha_r, alpha_I, alpha_X)
        for prompt, response in zip(prompts, responses)
    ]


def aggregate_scores(scores: list[PathScore]) -> dict:
    """Compute aggregate statistics over a set of scores.

    Returns a dict with:
        - mean_creativity: Average creativity score (the main metric).
        - mean_utility: Average utility.
        - mean_novelty: Average novelty (over all, including zeros).
        - mean_novelty_valid: Average novelty over paths with utility > 0.
        - constraint_satisfaction_rate: Fraction with utility > 0.
        - valid_path_rate: Fraction with structurally valid paths.
        - parse_success_rate: Fraction where response parsing succeeded.
        - error_distribution: Counts of each error type.
        - by_difficulty: Per-difficulty-level aggregate stats.
        - by_hop_count: Per-hop-count aggregate stats.
    """
    if not scores:
        return {"mean_creativity": 0.0, "n_prompts": 0}

    creativities = [s.creativity for s in scores]
    utilities = [s.utility for s in scores]
    novelties = [s.novelty for s in scores]
    valid_novelties = [s.novelty for s in scores if s.utility > 0]

    error_dist = Counter(s.error_type for s in scores)

    # Per-difficulty breakdown
    by_difficulty = {}
    for s in scores:
        # Infer difficulty from prompt_id or use a default grouping
        # We'll key by the number of constraints (include + exclude count)
        pass  # Filled in by the orchestration script which has prompt info

    result = {
        "n_prompts": len(scores),
        "mean_creativity": float(np.mean(creativities)),
        "mean_utility": float(np.mean(utilities)),
        "mean_novelty": float(np.mean(novelties)),
        "mean_novelty_valid": float(np.mean(valid_novelties)) if valid_novelties else 0.0,
        "constraint_satisfaction_rate": sum(1 for s in scores if s.utility > 0) / len(scores),
        "valid_path_rate": sum(1 for s in scores if s.valid_path) / len(scores),
        "parse_success_rate": sum(1 for s in scores if s.error_type != "parse_failure") / len(scores),
        "error_distribution": dict(error_dist),
    }

    return result
