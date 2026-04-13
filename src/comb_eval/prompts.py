"""Eval prompt generation for combinatorial creativity evaluation.

Generates path-finding problems at varying difficulty levels using BFS,
following the methodology from arXiv:2509.21043 Section 3. Each prompt
specifies a start node, end node, target hop count, and optional
inclusion/exclusion constraints on edge labels.
"""

import random
from collections import deque
from dataclasses import dataclass, field, asdict

import networkx as nx

from src.comb_eval.graph import get_node_label, get_edge_label


@dataclass
class EvalPrompt:
    """A single combinatorial creativity evaluation prompt.

    Attributes:
        start: Start node label (3-letter uppercase).
        end: End node label (3-letter uppercase).
        hop_count: Target number of hops in the path.
        include_labels: Edge labels that MUST appear in the path.
        exclude_labels: Edge labels that must NOT appear in the path.
        difficulty_level: Constraint difficulty level (1-5).
        base_path: The reference path used to generate this prompt (node labels).
        base_path_labels: Edge labels along the base path.
        prompt_id: Unique identifier for this prompt.
    """

    start: str
    end: str
    hop_count: int
    include_labels: list[str] = field(default_factory=list)
    exclude_labels: list[str] = field(default_factory=list)
    difficulty_level: int = 1
    base_path: list[str] = field(default_factory=list)
    base_path_labels: list[str] = field(default_factory=list)
    prompt_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def bfs_paths(
    G: nx.Graph,
    source: int,
    target_hop_count: int,
    max_paths: int = 100,
) -> list[list[int]]:
    """Find paths of exactly target_hop_count hops from source using BFS.

    Returns paths as lists of node IDs. Does not revisit nodes within a path.

    Args:
        G: The graph.
        source: Starting node ID.
        target_hop_count: Exact number of hops desired.
        max_paths: Maximum number of paths to return.

    Returns:
        List of paths (each path is a list of node IDs).
    """
    # BFS with path tracking: (current_node, path_so_far)
    queue = deque([(source, [source])])
    found_paths = []

    while queue and len(found_paths) < max_paths:
        current, path = queue.popleft()

        if len(path) - 1 == target_hop_count:
            found_paths.append(path)
            continue

        if len(path) - 1 > target_hop_count:
            continue

        for neighbor in G.neighbors(current):
            if neighbor not in path:  # no revisiting
                queue.append((neighbor, path + [neighbor]))

    return found_paths


def count_valid_paths_up_to_k(
    G: nx.Graph,
    source: int,
    target: int,
    hop_count: int,
    include_labels: set[str],
    exclude_labels: set[str],
    k: int,
) -> int:
    """Count distinct valid paths satisfying all constraints, up to k.

    Stops early once k valid paths are found (so returns min(true_count, k)).
    Paths are node-distinct (no revisits within a single path) but the k
    returned paths may share sub-structure — distinctness across paths is
    guaranteed by the BFS enumeration (each enqueued path is a unique
    node sequence).

    Args:
        G: The graph.
        source: Start node ID.
        target: End node ID.
        hop_count: Required number of hops.
        include_labels: Edge labels that must appear at least once.
        exclude_labels: Edge labels that must not appear at all.
        k: Maximum number of paths to count.

    Returns:
        min(true_count, k).
    """
    queue = deque([(source, [source], set())])
    found = 0

    while queue and found < k:
        current, path, collected = queue.popleft()

        if len(path) - 1 == hop_count:
            if current == target and include_labels.issubset(collected):
                found += 1
            continue

        if len(path) - 1 >= hop_count:
            continue

        for neighbor in G.neighbors(current):
            if neighbor in path:
                continue
            edge_label = get_edge_label(G, current, neighbor)
            if edge_label in exclude_labels:
                continue
            new_collected = collected | {edge_label}
            queue.append((neighbor, path + [neighbor], new_collected))

    return found


def generate_prompts_for_path(
    G: nx.Graph,
    path: list[int],
    max_level: int = 4,
    p_include: float = 0.5,
    min_valid_paths: int = 5,
    rng: random.Random | None = None,
) -> list[EvalPrompt]:
    """Generate prompts at multiple difficulty levels for a base path.

    Level 1: no constraints.
    Level l > 1: introduces (l-1) constraints, each with probability p_include
    of being an inclusion constraint (vs exclusion).

    Following Section 3 of arXiv:2509.21043:
    - Inclusion labels are drawn from labels present in the base path.
    - Exclusion labels are drawn from labels NOT present in the base path.
    - Each generated prompt is verified to have at least min_valid_paths
      distinct valid solutions (required for intra-response diversity scoring).

    Args:
        G: The graph.
        path: Base path as list of node IDs.
        max_level: Maximum difficulty level (1-indexed).
        p_include: Probability that each constraint is inclusion (vs exclusion).
        min_valid_paths: Minimum number of distinct valid paths required for
            a prompt to be kept (= k in the diversity-scoring design).
        rng: Seeded random instance.

    Returns:
        List of EvalPrompt objects, one per valid difficulty level.
    """
    if rng is None:
        rng = random.Random()

    hop_count = len(path) - 1
    source = path[0]
    target = path[-1]

    # Collect edge labels in the base path
    path_labels = []
    for i in range(hop_count):
        path_labels.append(get_edge_label(G, path[i], path[i + 1]))

    path_label_set = set(path_labels)
    # All lowercase letters not in the path
    all_labels = set("abcdefghijklmnopqrstuvwxyz")
    non_path_labels = all_labels - path_label_set

    # Node labels for reference
    node_labels = [get_node_label(G, n) for n in path]

    prompts = []

    for level in range(1, max_level + 1):
        if level == 1:
            if count_valid_paths_up_to_k(
                G, source, target, hop_count, set(), set(), min_valid_paths
            ) < min_valid_paths:
                continue
            prompt = EvalPrompt(
                start=node_labels[0],
                end=node_labels[-1],
                hop_count=hop_count,
                include_labels=[],
                exclude_labels=[],
                difficulty_level=1,
                base_path=node_labels,
                base_path_labels=path_labels,
            )
            prompts.append(prompt)
            continue

        # Generate (level - 1) constraints
        n_constraints = level - 1
        include_set = set()
        exclude_set = set()

        available_include = list(path_label_set - include_set)
        available_exclude = list(non_path_labels - exclude_set)

        for _ in range(n_constraints):
            is_include = rng.random() < p_include

            if is_include and available_include:
                label = rng.choice(available_include)
                include_set.add(label)
                available_include.remove(label)
            elif available_exclude:
                label = rng.choice(available_exclude)
                exclude_set.add(label)
                available_exclude.remove(label)
            elif available_include:
                label = rng.choice(available_include)
                include_set.add(label)
                available_include.remove(label)
            # else: no more labels available, skip this constraint

        if count_valid_paths_up_to_k(
            G, source, target, hop_count, include_set, exclude_set, min_valid_paths
        ) < min_valid_paths:
            continue  # Skip this difficulty if fewer than k distinct solutions exist

        prompt = EvalPrompt(
            start=node_labels[0],
            end=node_labels[-1],
            hop_count=hop_count,
            include_labels=sorted(include_set),
            exclude_labels=sorted(exclude_set),
            difficulty_level=level,
            base_path=node_labels,
            base_path_labels=path_labels,
        )
        prompts.append(prompt)

    return prompts


def generate_eval_set(
    G: nx.Graph,
    hop_counts: list[int],
    max_level: int = 4,
    prompts_per_hop: int = 10,
    p_include: float = 0.5,
    min_valid_paths: int = 5,
    seed: int = 42,
) -> list[EvalPrompt]:
    """Generate a full evaluation set across hop counts and difficulty levels.

    For each hop count, samples base paths and generates prompts at multiple
    difficulty levels. Every generated prompt is guaranteed to have at least
    `min_valid_paths` distinct valid solutions, so intra-response diversity
    scoring (k-paths-per-prompt) has headroom to vary across models.

    Args:
        G: The graph.
        hop_counts: List of target hop counts (e.g., [3, 4, 5]).
        max_level: Maximum difficulty level per path.
        prompts_per_hop: Number of base paths to sample per hop count.
        p_include: Probability of inclusion vs exclusion constraints.
        min_valid_paths: Minimum distinct valid paths required per prompt (= k).
        seed: Random seed.

    Returns:
        List of EvalPrompt objects with assigned prompt_ids.
    """
    rng = random.Random(seed)
    all_prompts = []
    prompt_counter = 0

    nodes = list(G.nodes())

    for hop_count in hop_counts:
        paths_found = 0
        attempts = 0
        max_attempts = prompts_per_hop * 50  # avoid infinite loop

        while paths_found < prompts_per_hop and attempts < max_attempts:
            attempts += 1
            source = rng.choice(nodes)
            paths = bfs_paths(G, source, hop_count, max_paths=5)

            if not paths:
                continue

            path = rng.choice(paths)
            prompts = generate_prompts_for_path(
                G, path,
                max_level=max_level,
                p_include=p_include,
                min_valid_paths=min_valid_paths,
                rng=rng,
            )

            if not prompts:
                continue  # base path yielded no prompts meeting the k threshold

            for p in prompts:
                p.prompt_id = f"h{hop_count}_l{p.difficulty_level}_{prompt_counter:04d}"
                prompt_counter += 1

            all_prompts.extend(prompts)
            paths_found += 1

    return all_prompts
