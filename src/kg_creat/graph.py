"""Common knowledge-graph abstraction for the kg_creat track.

Ports the Comb-Creat task substrate (constrained labeled-graph pathfinding,
arXiv:2509.21043) from a *synthetic* Erdos-Renyi graph onto *real* knowledge
graphs administered to frontier models at test time. The synthetic
`src/comb_eval/graph.py` is an undirected, single-letter-labelled `nx.Graph`;
a real KG is directed and multi-relational, so this module wraps an
`nx.MultiDiGraph` and is deliberately KG-agnostic.

The object this module produces is the **construction subgraph** `G_c = (V, E,
Sigma)` from the design spec: a typed slice used for (a) sampling endpoints with
known connectivity, (b) designing constraints that are *satisfiable* and
*biting*, and (c) as a factuality reference. It is NOT a closed answer space —
at eval time the model answers over the open KG and factuality is judged
separately (see docs/tracks/kg_creat/design.md).

Adding a new KG (Wikidata, Hetionet, PrimeKG, ...) is a *builder* that emits a
`KnowledgeGraph` via `from_triples`, not a new pipeline: everything downstream
(prompt construction, scoring, serialization) consumes this common interface.

Per repo conventions this module is implementation (HOW). Orchestration (which
KG, which seeds, where to write) lives in `src/kg_creat/scripts/`.
"""

import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

# Schema version for the serialized G_c JSON. Bump on any breaking change to
# the on-disk format so stale caches fail loudly rather than load silently.
GC_SCHEMA_VERSION = 1


@dataclass
class LabeledWalk:
    """A labeled walk through the KG: P = (e_0, r_1, e_1, ..., r_h, e_h).

    A walk traverses the *undirected projection* of the KG (a meaningful path
    may use an edge against its stored direction), but every step records the
    true directed triple so factuality and rendering stay exact.

    Attributes:
        nodes: Entity ids along the walk, length h + 1.
        relations: Relation ids, length h. relations[i] is the edge between
            nodes[i] and nodes[i + 1].
        forward: Orientation of each step, length h. If forward[i] is True the
            stored triple is (nodes[i], relations[i], nodes[i + 1]); if False it
            is (nodes[i + 1], relations[i], nodes[i]).
    """

    nodes: list[str]
    relations: list[str]
    forward: list[bool]

    @property
    def hop_count(self) -> int:
        return len(self.relations)

    def triples(self) -> list[tuple[str, str, str]]:
        """Return the directed (head, rel, tail) triples in stored orientation."""
        out = []
        for i, rel in enumerate(self.relations):
            a, b = self.nodes[i], self.nodes[i + 1]
            head, tail = (a, b) if self.forward[i] else (b, a)
            out.append((head, rel, tail))
        return out

    def relation_set(self) -> set[str]:
        return set(self.relations)


class KnowledgeGraph:
    """A typed construction subgraph G_c = (V, E, Sigma) over one KG.

    Thin, KG-agnostic wrapper around an `nx.MultiDiGraph`:
        - node id: stable string (Wikidata QID, Hetionet `Compound::DB00...`, ...)
        - node attr `label`: human-readable entity name (embedded for novelty R)
        - node attr `types`: list of type ids for categorical constraints (may be empty)
        - edge key: relation id `r in Sigma` (so parallel relations between a
          pair of entities are distinct edges)

    Relation and type ids are mapped to human labels by graph-level tables
    (`relation_labels`, `type_labels`) rather than repeated per edge/node.
    """

    def __init__(
        self,
        name: str,
        graph: nx.MultiDiGraph,
        relation_labels: dict[str, str],
        type_labels: dict[str, str] | None = None,
        build_meta: dict | None = None,
    ):
        self.name = name
        self.graph = graph
        self.relation_labels = dict(relation_labels)
        self.type_labels = dict(type_labels or {})
        self.build_meta = dict(build_meta or {})

    # --- construction ---

    @classmethod
    def from_triples(
        cls,
        name: str,
        triples: list[tuple[str, str, str]],
        labels: dict[str, str],
        relation_labels: dict[str, str],
        node_types: dict[str, list[str]] | None = None,
        type_labels: dict[str, str] | None = None,
        build_meta: dict | None = None,
    ) -> "KnowledgeGraph":
        """Build a KnowledgeGraph from a flat triple list (the common builder entry).

        Args:
            name: KG identifier (e.g. "wikidata", "hetionet").
            triples: (head_id, rel_id, tail_id) directed triples.
            labels: entity_id -> human label, for every entity appearing in triples.
            relation_labels: rel_id -> human label, for every relation used.
            node_types: optional entity_id -> list of type ids (categorical constraints).
            type_labels: optional type_id -> human label.
            build_meta: provenance dict recorded in the serialized graph.

        Raises:
            ValueError: if any entity or relation in `triples` lacks a label
                (fail fast — silent unlabeled nodes corrupt novelty/rendering).
        """
        node_types = node_types or {}
        G = nx.MultiDiGraph()

        for head, rel, tail in triples:
            for ent in (head, tail):
                if ent not in labels:
                    raise ValueError(f"FATAL: entity {ent!r} has no label in `labels`")
            if rel not in relation_labels:
                raise ValueError(f"FATAL: relation {rel!r} has no label in `relation_labels`")
            for ent in (head, tail):
                if ent not in G:
                    G.add_node(ent, label=labels[ent], types=list(node_types.get(ent, [])))
            # key=rel collapses duplicate identical triples but keeps parallel
            # relations between the same pair distinct.
            G.add_edge(head, tail, key=rel)

        return cls(
            name=name,
            graph=G,
            relation_labels=relation_labels,
            type_labels=type_labels,
            build_meta=build_meta,
        )

    # --- basic accessors ---

    def __len__(self) -> int:
        return self.graph.number_of_nodes()

    def __contains__(self, node: str) -> bool:
        return node in self.graph

    def nodes(self) -> list[str]:
        return list(self.graph.nodes())

    def label(self, node: str) -> str:
        """Human label for an entity."""
        return self.graph.nodes[node]["label"]

    def types(self, node: str) -> list[str]:
        """Type ids of an entity (for categorical constraints)."""
        return self.graph.nodes[node].get("types", [])

    def relation_label(self, rel: str) -> str:
        """Human label for a relation id (falls back to the id if unmapped)."""
        return self.relation_labels.get(rel, rel)

    def type_label(self, type_id: str) -> str:
        """Human label for a type id (falls back to the id if unmapped)."""
        return self.type_labels.get(type_id, type_id)

    def relation_vocabulary(self) -> list[str]:
        """Sorted list of relation ids actually present on edges."""
        return sorted({key for _, _, key in self.graph.edges(keys=True)})

    def has_triple(self, head: str, rel: str, tail: str) -> bool:
        """Whether the directed triple (head, rel, tail) is in G_c.

        A local factuality *reference* only — G_c is not the closed answer
        space, so a False here does not mean the triple is non-factual.
        """
        return self.graph.has_edge(head, tail, key=rel)

    # --- typed adjacency over the undirected projection ---

    def adjacency(self, node: str) -> list[tuple[str, str, bool]]:
        """Neighbors of `node` over the undirected projection.

        Returns (neighbor_id, rel_id, forward) triples, where forward=True means
        the stored edge is (node, rel, neighbor) and forward=False means it is
        (neighbor, rel, node). Both orientations are returned so a walk may
        traverse an edge against its stored direction while still recording the
        true triple.
        """
        out: list[tuple[str, str, bool]] = []
        for _, nbr, key in self.graph.out_edges(node, keys=True):
            out.append((nbr, key, True))
        for pred, _, key in self.graph.in_edges(node, keys=True):
            out.append((pred, key, False))
        return out

    # --- relation-frequency table ---

    def relation_frequency(self) -> Counter:
        """Counter of relation id -> number of edges carrying it.

        The design's per-KG relation-frequency table: documents that
        relation-*type* frequency is a curation artifact (so it is demoted to a
        baseline, not used for novelty) and lets the sampler pick *biting*
        constraints over relations the construction subgraph actually contains.
        """
        return Counter(key for _, _, key in self.graph.edges(keys=True))

    # --- path enumeration (powers the satisfiable-and-biting sampler) ---

    def enumerate_paths(
        self,
        source: str,
        target: str,
        hop_count: int,
        max_paths: int = 100,
        forbidden_relations: frozenset[str] | None = None,
        required_relations: frozenset[str] | None = None,
    ) -> list[LabeledWalk]:
        """Enumerate node-distinct labeled walks source -> target of exactly hop_count hops.

        Traverses the undirected projection (so direction does not block a
        meaningful path) without revisiting nodes, recording each step's true
        directed triple. Mirrors `comb_eval.prompts.bfs_paths` but typed and
        directed-aware. Used to verify a sampled (u, v, K, h) prompt is
        satisfiable (>= some valid paths) and that constraints bite.

        Args:
            source: start entity id.
            target: end entity id.
            hop_count: exact number of hops.
            max_paths: stop after this many walks are found.
            forbidden_relations: relation ids no step may use (exclusion constraint).
            required_relations: relation ids that must all appear (inclusion constraint).

        Returns:
            List of LabeledWalk, each ending at `target` with `hop_count` hops.
        """
        if source not in self.graph or target not in self.graph:
            return []
        forbidden = forbidden_relations or frozenset()
        required = required_relations or frozenset()

        # queue items: (current, nodes_so_far, rels_so_far, forward_so_far, rel_set)
        queue: deque = deque([(source, [source], [], [], set())])
        found: list[LabeledWalk] = []

        while queue and len(found) < max_paths:
            current, nodes, rels, fwd, relset = queue.popleft()
            hops = len(rels)

            if hops == hop_count:
                if current == target and required.issubset(relset):
                    found.append(LabeledWalk(nodes=nodes, relations=rels, forward=fwd))
                continue
            if hops > hop_count:
                continue

            for nbr, rel, forward in self.adjacency(current):
                if nbr in nodes:  # node-distinct walks only
                    continue
                if rel in forbidden:
                    continue
                queue.append(
                    (
                        nbr,
                        nodes + [nbr],
                        rels + [rel],
                        fwd + [forward],
                        relset | {rel},
                    )
                )

        return found

    def count_paths_up_to_k(
        self,
        source: str,
        target: str,
        hop_count: int,
        k: int,
        forbidden_relations: frozenset[str] | None = None,
        required_relations: frozenset[str] | None = None,
    ) -> int:
        """min(number of valid walks, k) — early-stopping satisfiability check.

        Mirrors `comb_eval.prompts.count_valid_paths_up_to_k`: used to gate a
        sampled prompt on having at least `k` distinct valid solutions (headroom
        for the k-path diversity term) before it is shown to a model.
        """
        return len(
            self.enumerate_paths(
                source,
                target,
                hop_count,
                max_paths=k,
                forbidden_relations=forbidden_relations,
                required_relations=required_relations,
            )
        )

    # --- subgraph extraction ---

    def subgraph_around(self, seeds: list[str], radius: int) -> "KnowledgeGraph":
        """Construction subgraph: all nodes within `radius` undirected hops of `seeds`.

        Args:
            seeds: entity ids to expand from (must exist in the graph).
            radius: number of undirected hops to expand.

        Returns:
            A new KnowledgeGraph induced on the reachable nodes, carrying over
            the relation/type label tables and recording the seeds/radius in
            build_meta.

        Raises:
            ValueError: if any seed is absent (fail fast).
        """
        missing = [s for s in seeds if s not in self.graph]
        if missing:
            raise ValueError(f"FATAL: seed entities not in graph: {missing}")

        reached: set[str] = set(seeds)
        frontier: set[str] = set(seeds)
        for _ in range(radius):
            nxt: set[str] = set()
            for node in frontier:
                for nbr, _, _ in self.adjacency(node):
                    if nbr not in reached:
                        nxt.add(nbr)
            reached |= nxt
            frontier = nxt
            if not frontier:
                break

        sub = self.graph.subgraph(reached).copy()
        meta = dict(self.build_meta)
        meta["subgraph_seeds"] = list(seeds)
        meta["subgraph_radius"] = radius
        return KnowledgeGraph(
            name=self.name,
            graph=sub,
            relation_labels=self.relation_labels,
            type_labels=self.type_labels,
            build_meta=meta,
        )

    # --- rendering (factuality reference / debugging) ---

    def adjacency_text(
        self,
        nodes: list[str] | None = None,
        max_neighbors: int | None = None,
    ) -> str:
        """Human-readable adjacency listing (outgoing edges).

        Format per line: `Label [id]: rel_label -> NeighborLabel [id]; ...`.
        Intended as a factuality reference / debugging view, not necessarily the
        model-facing prompt (a real G_c is too large to dump wholesale).

        Args:
            nodes: restrict to these entity ids (default: all, sorted by label).
            max_neighbors: cap neighbors listed per node (None = no cap).
        """
        if nodes is None:
            nodes = sorted(self.graph.nodes(), key=lambda n: self.label(n))
        lines = []
        for node in nodes:
            edges = list(self.graph.out_edges(node, keys=True))
            if max_neighbors is not None:
                edges = edges[:max_neighbors]
            parts = [
                f"{self.relation_label(key)} -> {self.label(nbr)} [{nbr}]"
                for _, nbr, key in edges
            ]
            lines.append(f"{self.label(node)} [{node}]: {'; '.join(parts)}")
        return "\n".join(lines)

    # --- summary statistics ---

    def stats(self) -> dict:
        """Summary statistics for the construction subgraph."""
        G = self.graph
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        rel_freq = self.relation_frequency()
        # Weakly-connected components on the underlying undirected structure.
        n_components = nx.number_weakly_connected_components(G) if n_nodes else 0
        n_typed = sum(1 for n in G.nodes() if self.types(n))
        return {
            "name": self.name,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "n_relation_types": len(rel_freq),
            "n_weakly_connected_components": n_components,
            "avg_out_degree": (n_edges / n_nodes) if n_nodes else 0.0,
            "n_nodes_with_types": n_typed,
            "top_relations": rel_freq.most_common(10),
        }

    # --- serialization ---

    def to_json_dict(self) -> dict:
        """Serialize to a JSON-able dict (the cached G_c format)."""
        return {
            "schema_version": GC_SCHEMA_VERSION,
            "name": self.name,
            "directed": True,
            "relation_labels": self.relation_labels,
            "type_labels": self.type_labels,
            "build_meta": self.build_meta,
            "nodes": [
                {
                    "id": n,
                    "label": self.label(n),
                    "types": self.types(n),
                }
                for n in sorted(self.graph.nodes())
            ],
            "edges": [
                {"source": u, "rel": key, "target": v}
                for u, v, key in self.graph.edges(keys=True)
            ],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "KnowledgeGraph":
        """Reconstruct a KnowledgeGraph from a serialized dict (fail fast on schema drift)."""
        version = data.get("schema_version")
        if version != GC_SCHEMA_VERSION:
            raise ValueError(
                f"FATAL: G_c schema_version {version} != expected {GC_SCHEMA_VERSION}. "
                "Rebuild the construction subgraph."
            )
        G = nx.MultiDiGraph()
        for node in data["nodes"]:
            G.add_node(node["id"], label=node["label"], types=list(node.get("types", [])))
        for edge in data["edges"]:
            G.add_edge(edge["source"], edge["target"], key=edge["rel"])
        return cls(
            name=data["name"],
            graph=G,
            relation_labels=data["relation_labels"],
            type_labels=data.get("type_labels", {}),
            build_meta=data.get("build_meta", {}),
        )

    def save(self, path: str | Path) -> None:
        """Write the construction subgraph to a JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGraph":
        """Load a construction subgraph from a JSON file.

        Raises:
            FileNotFoundError: if the cache file does not exist (fail fast — never
                silently rebuild or fall back).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"G_c cache not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_json_dict(data)
