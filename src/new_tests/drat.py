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


def _anchor_list_str(anchors: list[str], joiner: str = "and") -> str:
    """Format anchor list for prompt: '"A" and "B"' or '"A", "B", and "C"'."""
    quoted = [f'"{a}"' for a in anchors]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} {joiner} {quoted[1]}"
    return f"{', '.join(quoted[:-1])}, {joiner} {quoted[-1]}"


def drat_prompt(anchors_or_a, anchor_b: str | None = None, style: str = "default") -> str:
    """Generate the DRAT prompt.

    Two call signatures supported for backward compatibility:
      - drat_prompt(anchor_a, anchor_b, style="...")  — original 2-anchor.
      - drat_prompt(["A", "B", "C", ...], style="...") — N-anchor.
    """
    if isinstance(anchors_or_a, list):
        anchors = anchors_or_a
    else:
        anchors = [anchors_or_a]
        if anchor_b is not None:
            anchors.append(anchor_b)

    if len(anchors) < 2:
        raise ValueError(f"drat_prompt needs >= 2 anchors, got {len(anchors)}")

    anchor_str = _anchor_list_str(anchors)

    if style == "default":
        bridge_clause = (
            f'each of which connects {anchor_str}'
            if len(anchors) == 2
            else f'each of which connects all of {anchor_str}'
        )
    elif style == "analogical":
        bridge_clause = (
            f'each of which could be metaphorically applied to both {anchor_str}'
            if len(anchors) == 2
            else f'each of which could be metaphorically applied to all of {anchor_str}'
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
    anchors_or_a,
    anchor_b_or_pool,
    noun_pool_or_embeddings=None,
    embeddings: SBERTEmbeddings | None = None,
    percentile: float = 90.0,
) -> dict:
    """Compute the per-anchor-group utility threshold from a random-noun null.

    Two call signatures supported:
      compute_tau(anchor_a, anchor_b, noun_pool, embeddings, percentile=...)
      compute_tau([A, B, C, ...], noun_pool, embeddings, percentile=...)
    """
    if isinstance(anchors_or_a, list):
        anchors = anchors_or_a
        noun_pool = anchor_b_or_pool
        emb = noun_pool_or_embeddings
    else:
        anchors = [anchors_or_a, anchor_b_or_pool]
        noun_pool = noun_pool_or_embeddings
        emb = embeddings

    all_to_encode = list(anchors) + list(noun_pool)
    vectors = emb.encode_batch(all_to_encode)
    n_anchors = len(anchors)
    anchor_vecs = vectors[:n_anchors]
    noun_vecs = vectors[n_anchors:]

    utilities = []
    for v in noun_vecs:
        sims = [float(1.0 - cosine_distance(v, av)) for av in anchor_vecs]
        utilities.append(max(sims))  # max-utility (anchor in any of the n)

    tau = float(np.percentile(utilities, percentile))
    return {
        "tau": tau,
        "noun_utilities": utilities,
        "noun_pool_size": len(noun_pool),
        "percentile": percentile,
        "n_anchors": n_anchors,
    }


def score_drat(
    words: list[str],
    anchors_or_a,
    anchor_b_or_emb=None,
    embeddings_or_tau=None,
    tau_or_n_min=None,
    n_min: int | None = None,
) -> dict:
    """Score a model response with the DRAT metric.

    Two call signatures supported (preserves backward-compat with 2-anchor):
      score_drat(words, anchor_a, anchor_b, embeddings, tau, n_min=...)
      score_drat(words, [A, B, C, ...], embeddings, tau, n_min=...)
    """
    # Disambiguate: list -> N-anchor, str -> 2-anchor backward compat
    if isinstance(anchors_or_a, list):
        anchors = anchors_or_a
        embeddings = anchor_b_or_emb
        tau = embeddings_or_tau
        if n_min is None:
            n_min = tau_or_n_min if tau_or_n_min is not None else 5
    else:
        anchors = [anchors_or_a, anchor_b_or_emb]
        embeddings = embeddings_or_tau
        tau = tau_or_n_min
        if n_min is None:
            n_min = 5

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
            "n_anchors": len(anchors),
            "sufficient": False,
            "reason": f"only {len(valid)} valid words; need >= {n_min}",
        }

    # Use up to first 10 valid words (matches DAT/CDAT convention; gate runs after)
    scored_words = valid[:10]

    all_to_encode = list(anchors) + scored_words
    vectors = embeddings.encode_batch(all_to_encode)
    n_anchors = len(anchors)
    anchor_vecs = vectors[:n_anchors]
    word_vecs = vectors[n_anchors:]

    # Per-word utility: max(cos(w, A_i)) over all anchors A_i
    utilities = []
    for v in word_vecs:
        sims = [float(1.0 - cosine_distance(v, av)) for av in anchor_vecs]
        utilities.append(max(sims))

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
            "n_anchors": n_anchors,
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
        "n_anchors": n_anchors,
        "sufficient": True,
        "mean_pairwise_distance": mean_distance,
    }
