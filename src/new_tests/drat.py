"""Divergent Remote Association Test (DRAT) implementation.

DRAT presents two distant anchors and asks for 10 words that connect them.
Scoring:

  Utility(w | A, B) = max(cos(w, A), cos(w, B))
  Survivors S = {w in W | Utility(w | A, B) > tau}
  DRAT(W | A, B) = (100 / (k(k-1))) * sum_{i != j} d(w_i, w_j)
                 = 100 * mean pairwise cosine distance over S

where tau is the 90th percentile of Utility(w | A, B) for random nouns.

See docs/tracks/new_tests/drat_design.md for the full design rationale.
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

from src.dat_eval.cdat import SBERTEmbeddings, validate_words_sbert


def drat_prompt(anchor_a: str, anchor_b: str, style: str = "default") -> str:
    """Generate the DRAT prompt for an anchor pair.

    Args:
        style: "default" — standard "connects" framing.
               "analogical" — "metaphorically applied to both" framing per the
                 design doc's analogy-literature motivation.
    """
    if style == "default":
        bridge_clause = f'each of which connects "{anchor_a}" and "{anchor_b}"'
    elif style == "analogical":
        bridge_clause = (
            f'each of which could be metaphorically applied to both '
            f'"{anchor_a}" and "{anchor_b}"'
        )
    else:
        raise ValueError(f"unknown prompt style: {style!r}")

    return (
        f'Please give 10 words that are as different from each other as possible, '
        f'in all meanings and uses of the words, and {bridge_clause}.\n\n'
        f'Only use single nouns. Do not use proper nouns. '
        f'Do not use the anchor words themselves or variations of them.\n\n'
        f'Respond with ONLY a JSON array of exactly 10 words, like: '
        f'["word1", "word2", "word3", "word4", "word5", "word6", "word7", "word8", "word9", "word10"]'
    )


def compute_tau(
    anchor_a: str,
    anchor_b: str,
    noun_pool: list[str],
    embeddings: SBERTEmbeddings,
    percentile: float = 90.0,
) -> dict:
    """Compute the per-pair utility threshold from a random-noun null distribution.

    For each random noun n in the pool, compute Utility(n | A, B). Return the
    `percentile`th percentile as tau. Higher percentile = stricter gate.

    Args:
        anchor_a, anchor_b: The two anchor strings.
        noun_pool: List of random nouns to compute the null distribution.
        embeddings: SBERT model.
        percentile: Quantile (0-100) of the null to use as tau.

    Returns:
        Dict with tau, noun_utilities (list), noun_pool_size, percentile.
    """
    all_to_encode = [anchor_a, anchor_b] + list(noun_pool)
    vectors = embeddings.encode_batch(all_to_encode)
    a_vec = vectors[0]
    b_vec = vectors[1]
    noun_vecs = vectors[2:]

    utilities = []
    for v in noun_vecs:
        cos_a = float(1.0 - cosine_distance(v, a_vec))
        cos_b = float(1.0 - cosine_distance(v, b_vec))
        utilities.append(max(cos_a, cos_b))

    tau = float(np.percentile(utilities, percentile))
    return {
        "tau": tau,
        "noun_utilities": utilities,
        "noun_pool_size": len(noun_pool),
        "percentile": percentile,
    }


def score_drat(
    words: list[str],
    anchor_a: str,
    anchor_b: str,
    embeddings: SBERTEmbeddings,
    tau: float,
    n_min: int = 5,
) -> dict:
    """Score a model response with the DRAT metric.

    Args:
        words: Raw list of words from the model.
        anchor_a, anchor_b: The two anchors for this item.
        embeddings: SBERT model.
        tau: Per-pair utility threshold (from compute_tau).
        n_min: Minimum number of survivors required to score.

    Returns:
        Dict with drat (float), n_valid, n_survivors, survivors, scored_words,
        utilities, mean_pairwise_distance, tau, sufficient (bool), reason (if not).
    """
    valid = validate_words_sbert(words)

    if len(valid) < n_min:
        return {
            "drat": 0.0,
            "n_valid": len(valid),
            "n_survivors": 0,
            "survivors": [],
            "scored_words": valid,
            "utilities": [],
            "tau": tau,
            "sufficient": False,
            "reason": f"only {len(valid)} valid words; need >= {n_min}",
        }

    # Use up to first 10 valid words (matches DAT/CDAT convention; gate runs after)
    scored_words = valid[:10]

    all_to_encode = [anchor_a, anchor_b] + scored_words
    vectors = embeddings.encode_batch(all_to_encode)
    a_vec = vectors[0]
    b_vec = vectors[1]
    word_vecs = vectors[2:]

    # Per-word utility: max(cos(w, A), cos(w, B))
    utilities = []
    for v in word_vecs:
        cos_a = float(1.0 - cosine_distance(v, a_vec))
        cos_b = float(1.0 - cosine_distance(v, b_vec))
        utilities.append(max(cos_a, cos_b))

    # Survivor set: words with utility above tau
    survivor_indices = [i for i, u in enumerate(utilities) if u > tau]
    survivors = [scored_words[i] for i in survivor_indices]

    if len(survivors) < n_min:
        return {
            "drat": 0.0,
            "n_valid": len(valid),
            "n_survivors": len(survivors),
            "survivors": survivors,
            "scored_words": scored_words,
            "utilities": utilities,
            "tau": tau,
            "sufficient": False,
            "reason": f"only {len(survivors)} survivors above tau={tau:.3f}; need >= {n_min}",
        }

    # Mean pairwise cosine distance over survivors, scaled by 100
    survivor_vecs = [word_vecs[i] for i in survivor_indices]
    k = len(survivor_vecs)
    distances = []
    for i in range(k):
        for j in range(i + 1, k):
            d = float(cosine_distance(survivor_vecs[i], survivor_vecs[j]))
            distances.append(d)
    mean_distance = float(np.mean(distances))
    drat = 100.0 * mean_distance

    return {
        "drat": drat,
        "n_valid": len(valid),
        "n_survivors": k,
        "survivors": survivors,
        "scored_words": scored_words,
        "utilities": utilities,
        "tau": tau,
        "sufficient": True,
        "mean_pairwise_distance": mean_distance,
    }
