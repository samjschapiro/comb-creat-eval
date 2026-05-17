"""Per-prompt-cluster archive of past on-policy responses, FAISS-backed.

Maintains, for each prompt cluster c, a FAISS HNSW index over SBERT
embeddings of responses that passed the appropriateness gate during
training. Used to compute k-NN novelty for new candidate responses.

Admission rule: a response y is admitted to cluster c's archive iff
    A(y, x) > τ   AND   N(y, x; archive_c) > ρ
where ρ is set per cluster (e.g. running median of admitted novelties).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ClusterArchive:
    """FAISS HNSW archive for one prompt cluster."""

    dim: int
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    max_size: int | None = 5000
    # Skip the novelty (rho) gate for the first warmup_admissions entries.
    # Without this, the empty-archive bootstrap value would pin the median
    # at the artificial default and block all subsequent admissions.
    warmup_admissions: int = 8

    _index: object = field(default=None, init=False, repr=False)
    _embeddings: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _admitted_novelties: list[float] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        import faiss
        self._index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self.hnsw_ef_construction

    def add(
        self,
        embedding: np.ndarray,
        novelty_at_admission: float,
        record_novelty: bool = True,
    ) -> None:
        """Add an embedding to the archive.

        Args:
            embedding: SBERT vector, shape (dim,), L2-normalized.
            novelty_at_admission: k-NN novelty score at admission time.
            record_novelty: If False, do not record this novelty value in
                the running median used for rho. Set False for admissions
                where the novelty score came from the empty-archive
                bootstrap (otherwise the artificial 1.0 contaminates the
                median forever).
        """
        if embedding.ndim != 1 or embedding.shape[0] != self.dim:
            raise ValueError(f"FATAL: expected shape ({self.dim},), got {embedding.shape}")
        # During warmup, the archive is too small for novelty values to be
        # representative — early entries are artificially distant because
        # there's almost nothing to compare against. Recording those values
        # in the running median would inflate rho permanently and block
        # post-warmup admissions. Skip recording during warmup entirely.
        in_warmup = len(self._embeddings) < self.warmup_admissions
        should_record = record_novelty and not in_warmup

        if self.max_size is not None and len(self._embeddings) >= self.max_size:
            # Random eviction; novelty-thresholded admission is the primary cap.
            evict_idx = int(np.random.randint(0, len(self._embeddings)))
            self._embeddings[evict_idx] = embedding
            if should_record and evict_idx < len(self._admitted_novelties):
                self._admitted_novelties[evict_idx] = novelty_at_admission
            self._rebuild_index()
            return
        self._embeddings.append(embedding.astype(np.float32))
        if should_record:
            self._admitted_novelties.append(float(novelty_at_admission))
        self._index.add(embedding.reshape(1, -1).astype(np.float32))

    def knn_distances(self, embedding: np.ndarray, k: int) -> np.ndarray:
        """Return cosine distances to the k nearest archive members.

        Embeddings are assumed L2-normalized; inner product == cosine sim.
        Returns 1 - sim as cosine distance. Empty archive returns empty array.
        """
        if len(self._embeddings) == 0:
            return np.array([], dtype=np.float32)
        k_eff = min(k, len(self._embeddings))
        sims, _ = self._index.search(embedding.reshape(1, -1).astype(np.float32), k_eff)
        return 1.0 - sims[0]

    def admission_threshold(self, mode: str | float) -> float:
        # During warmup: accept anything that passes the appropriateness
        # gate. This builds a real archive before the rho gate kicks in.
        if len(self._embeddings) < self.warmup_admissions:
            return -float("inf")
        if mode == "median":
            if not self._admitted_novelties:
                return 0.0
            return float(np.median(self._admitted_novelties))
        if isinstance(mode, (int, float)):
            return float(mode)
        if mode is None:
            return 0.0
        raise ValueError(f"FATAL: unknown admission threshold mode: {mode}")

    def _rebuild_index(self) -> None:
        import faiss
        self._index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        self._index.hnsw.efConstruction = self.hnsw_ef_construction
        if self._embeddings:
            self._index.add(np.stack(self._embeddings).astype(np.float32))

    def __len__(self) -> int:
        return len(self._embeddings)


class ClusterArchiveSet:
    """Set of per-cluster archives keyed by cluster id."""

    def __init__(self, n_clusters: int, dim: int, **archive_kwargs):
        self.n_clusters = n_clusters
        self.dim = dim
        self._archives: dict[int, ClusterArchive] = {
            c: ClusterArchive(dim=dim, **archive_kwargs) for c in range(n_clusters)
        }

    def __getitem__(self, cluster_id: int) -> ClusterArchive:
        return self._archives[cluster_id]

    def save(self, path: Path) -> None:
        raise NotImplementedError("persistence not yet implemented")

    @classmethod
    def load(cls, path: Path) -> "ClusterArchiveSet":
        raise NotImplementedError("persistence not yet implemented")
