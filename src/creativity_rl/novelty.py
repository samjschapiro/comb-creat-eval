"""k-NN novelty score against a cluster archive.

novelty(y, x; A) = mean of cosine distances from φ(y) to its k nearest
neighbors in the cluster archive A_{c(x)}. Empty archive returns the
maximum-possible novelty (1.0 under cosine), which gives the first few
responses for each cluster a maximal novelty signal and is the standard
NS bootstrapping behavior.
"""

from __future__ import annotations

import numpy as np

from src.creativity_rl.archive import ClusterArchive


def knn_novelty(
    embedding: np.ndarray,
    archive: ClusterArchive,
    k: int,
) -> float:
    """Mean cosine distance to k nearest archive members.

    Args:
        embedding: SBERT embedding of the response, L2-normalized.
        archive: cluster archive for the response's prompt cluster.
        k: number of nearest neighbors.

    Returns:
        Mean cosine distance in [0, 2]. Empty archive returns 1.0
        (the expected cosine distance between two random unit vectors
        in high dim is ~1.0; this is the bootstrapping default).
    """
    distances = archive.knn_distances(embedding, k)
    if distances.size == 0:
        return 1.0
    return float(np.mean(distances))
