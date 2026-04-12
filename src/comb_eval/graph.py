"""Graph construction and manipulation for combinatorial creativity evaluation.

Builds a synthetic labeled graph following the framework from arXiv:2509.21043,
scaled down to fit in LLM context windows. Nodes are 3-letter uppercase labels,
edges carry lowercase single-letter labels.
"""

import json
import random
import string
from pathlib import Path

import networkx as nx
import numpy as np


def generate_node_labels(n_nodes: int, rng: random.Random) -> list[str]:
    """Generate unique 3-letter uppercase node labels.

    Samples n_nodes labels from the space of all 3-letter uppercase combinations
    (AAA through ZZZ = 17,576 possible). For small graphs we sample a subset;
    the original paper uses the full set.

    Args:
        n_nodes: Number of nodes to generate.
        rng: Seeded random.Random instance.

    Returns:
        Sorted list of unique 3-letter node labels.

    Raises:
        ValueError: If n_nodes exceeds the label space.
    """
    max_labels = 26**3  # 17,576
    if n_nodes > max_labels:
        raise ValueError(f"Cannot generate {n_nodes} unique 3-letter labels (max {max_labels})")

    # Generate all possible 3-letter combinations, then sample
    all_labels = [
        f"{a}{b}{c}"
        for a in string.ascii_uppercase
        for b in string.ascii_uppercase
        for c in string.ascii_uppercase
    ]
    sampled = rng.sample(all_labels, n_nodes)
    return sorted(sampled)


def build_graph(
    n_nodes: int,
    avg_degree: float,
    edge_label_weights: dict[str, float] | None = None,
    seed: int = 42,
) -> nx.Graph:
    """Build a random labeled graph for combinatorial creativity evaluation.

    Creates an Erdos-Renyi-style random graph with labeled edges. Each edge
    receives a lowercase letter label drawn from a (possibly non-uniform)
    distribution, matching the original CC paper's setup.

    Args:
        n_nodes: Number of nodes in the graph.
        avg_degree: Target average degree. Edge probability = avg_degree / (n_nodes - 1).
        edge_label_weights: Optional dict mapping lowercase letters to weights.
            If None, uses uniform distribution over a-z.
        seed: Random seed for reproducibility.

    Returns:
        networkx.Graph with:
            - Node attribute 'label': 3-letter uppercase string
            - Edge attribute 'label': single lowercase letter
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Generate node labels
    node_labels = generate_node_labels(n_nodes, rng)

    # Compute edge probability for target average degree
    edge_prob = avg_degree / (n_nodes - 1)
    if edge_prob > 1.0:
        raise ValueError(
            f"avg_degree={avg_degree} with n_nodes={n_nodes} requires edge_prob={edge_prob:.3f} > 1.0"
        )

    # Build edge label distribution
    if edge_label_weights is None:
        labels = list(string.ascii_lowercase)
        weights = np.ones(26) / 26
    else:
        labels = list(edge_label_weights.keys())
        raw_weights = np.array([edge_label_weights[l] for l in labels])
        weights = raw_weights / raw_weights.sum()

    # Create graph with labeled nodes
    G = nx.Graph()
    for i, label in enumerate(node_labels):
        G.add_node(i, label=label)

    # Add edges (Erdos-Renyi style) with random labels
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                edge_label = np_rng.choice(labels, p=weights)
                G.add_edge(i, j, label=edge_label)

    return G


def get_node_label(G: nx.Graph, node: int) -> str:
    """Get the 3-letter label for a node."""
    return G.nodes[node]["label"]


def get_edge_label(G: nx.Graph, u: int, v: int) -> str:
    """Get the lowercase letter label for an edge."""
    return G.edges[u, v]["label"]


def graph_to_edge_list(G: nx.Graph) -> list[dict]:
    """Convert graph to a list of edge dicts for serialization.

    Returns:
        List of dicts with keys: source, target, label (all using node labels).
    """
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": get_node_label(G, u),
            "target": get_node_label(G, v),
            "label": data["label"],
        })
    return edges


def graph_to_adjacency_text(G: nx.Graph) -> str:
    """Convert graph to a human-readable adjacency list string.

    Format:
        NODE: neighbor1(label1), neighbor2(label2), ...

    This is the representation presented to LLMs in the prompt.
    """
    lines = []
    for node in sorted(G.nodes()):
        node_label = get_node_label(G, node)
        neighbors = []
        for neighbor in sorted(G.neighbors(node)):
            edge_label = get_edge_label(G, node, neighbor)
            neighbor_label = get_node_label(G, neighbor)
            neighbors.append(f"{neighbor_label}({edge_label})")
        if neighbors:
            lines.append(f"{node_label}: {', '.join(neighbors)}")
    return "\n".join(lines)


def label_to_node(G: nx.Graph) -> dict[str, int]:
    """Build a mapping from node labels to node IDs."""
    return {G.nodes[n]["label"]: n for n in G.nodes()}


def save_graph(G: nx.Graph, path: str | Path) -> None:
    """Save graph to JSON file."""
    path = Path(path)
    data = {
        "nodes": [
            {"id": n, "label": G.nodes[n]["label"]}
            for n in sorted(G.nodes())
        ],
        "edges": [
            {
                "source": u,
                "target": v,
                "label": d["label"],
            }
            for u, v, d in G.edges(data=True)
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_graph(path: str | Path) -> nx.Graph:
    """Load graph from JSON file."""
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    G = nx.Graph()
    for node in data["nodes"]:
        G.add_node(node["id"], label=node["label"])
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], label=edge["label"])
    return G


def graph_stats(G: nx.Graph) -> dict:
    """Compute summary statistics for the graph."""
    degrees = [d for _, d in G.degree()]
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "avg_degree": np.mean(degrees),
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "is_connected": nx.is_connected(G),
        "n_components": nx.number_connected_components(G),
        "diameter": nx.diameter(G) if nx.is_connected(G) else None,
    }
