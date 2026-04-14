"""Word-Circle Construction Eval.

Inspired by the Circle Construction task in Nagarajan, Wu, Ding, Raghunathan
(ICML 2025), "Roll the dice & look before you leap" (arXiv:2504.15266).

The model is given a seed word and must produce an ordered chain of N new
words such that consecutive words are associated AND the final word closes
back to the seed. The closing requirement is STRUCTURAL, not instructive —
it's a property of the geometry the model must plan, not a rule we ask it
to remember. This avoids the "constraint-following measures capability, not
creativity" failure mode of lexical/semantic C-PACE.

Verification (all automatic via FastText cosine):
  - Edge coherence: every consecutive pair (including w_N → w_0) must have
    cosine similarity >= tau_edge.
  - Closure: cosine(w_N, w_0) >= tau_closure. (Same cosine check as any
    edge, reported separately because closure is the planning-hard edge.)
  - Diversity: mean pairwise FastText distance across all N new words.

Thresholds tau_edge and tau_closure are post-hoc scoring knobs — they do NOT
appear in the model's prompt. The model sees natural-language instructions
("associate with the previous word", "close the circle back to the seed")
and we verify the resulting circle at scoring time.

Cross-track import note: reuses `FastTextEmbeddings` from `src.dat_eval.pace`
for consistent embedding infrastructure.
"""

import itertools
import json
import math
import re
from dataclasses import dataclass, field

import numpy as np

from src.dat_eval.pace import FastTextEmbeddings, DEFAULT_SEEDS


# --- prompting ---


def circle_prompt(seed: str, n_words: int) -> str:
    """Natural-language prompt asking for an n_words-long associative circle
    that closes back to the seed.
    """
    return (
        f'Starting with the word "{seed}", construct a CIRCULAR associative '
        f"chain of exactly {n_words} DIFFERENT words.\n"
        f"\n"
        f"Rules:\n"
        f'1. The FIRST word must directly associate with "{seed}".\n'
        f"2. Each subsequent word must associate with the word immediately "
        f"before it in the chain.\n"
        f'3. The FINAL (the {n_words}-th) word must associate naturally back '
        f'with "{seed}" — this CLOSES the circle by meaningful association, '
        f"NOT by repeating the seed.\n"
        f"4. All {n_words} words must be distinct from each other AND NONE "
        f'of them may be "{seed}" itself. The circle is closed by '
        f'ASSOCIATION; do not repeat "{seed}" as any of the {n_words} words.\n'
        f"5. Do not use proper nouns (names, brands, etc.).\n"
        f"\n"
        f"Think about how to plan the full chain so the last word naturally "
        f'connects back to "{seed}" without being "{seed}" itself. For each '
        f"word, provide a brief reason explaining its association with the "
        f"previous word (or with the seed, for the last word — explain how "
        f'it closes the circle).\n'
        f"\n"
        f"Return ONLY a JSON object in this exact format:\n"
        f'{{"circle": ['
        f'{{"word": "", "reason": ""}}, '
        f"... exactly {n_words} entries ...]}}"
    )


# --- parsing ---


def _extract_outermost_object(raw: str) -> dict | None:
    """Best-effort extraction of the outermost JSON object containing 'circle'."""
    # strip code fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fence.group(1) if fence else raw

    start = candidate.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(candidate)):
            c = candidate[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = candidate[start:i + 1]
                    try:
                        data = json.loads(blob)
                        if isinstance(data, dict) and "circle" in data:
                            return data
                    except json.JSONDecodeError:
                        pass
                    break
        start = candidate.find("{", start + 1)
    return None


def parse_circle_response(raw: str | None, seed: str, n_words: int) -> list[str]:
    """Parse a circle response into a list of word strings (lowercased).

    Returns the model's N new words (not including the seed). If parsing
    produces fewer than 2 valid entries, returns whatever was extracted.
    Fall-back regex scrapes alphabetic tokens if JSON parsing fails.
    """
    if not raw:
        return []

    data = _extract_outermost_object(raw)
    if data is not None:
        results = data.get("circle", [])
        if isinstance(results, list):
            words = []
            for entry in results:
                if not isinstance(entry, dict):
                    continue
                w = str(entry.get("word", "")).strip().lower()
                if w:
                    words.append(w)
            if words:
                return words[:n_words]

    # Fallback — scrape alphabetic tokens, skipping schema keywords + seed
    skip = {"circle", "word", "reason", "results", seed.lower()}
    tokens = re.findall(r"\b[a-zA-Z]+\b", raw)
    out = []
    for t in tokens:
        tl = t.lower()
        if tl not in skip:
            out.append(tl)
        if len(out) >= n_words:
            break
    return out


# --- FastText-based verification and scoring ---


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


@dataclass
class CircleScore:
    """Scoring breakdown for a single circle attempt."""

    seed: str
    words: list[str]
    n_returned: int
    n_in_vocab: int
    edge_cosines: list[float]           # length N: w0→w1, w1→w2, ..., w_{N-1}→w_N
    closure_cosine: float               # w_N → w_0
    # Post-hoc / threshold-agnostic scores
    mean_edge_cosine: float
    min_edge_cosine: float
    pairwise_diversity: float           # mean pairwise distance across all N words
    pace_internal_score: float          # PACE-style: mean dist from pos i to all prior
    # Basic structural checks
    distinct: bool                       # all N words distinct from each other
    excludes_seed: bool                  # none of the N words is the seed itself


def _pairwise_diversity(vectors: list[np.ndarray]) -> float:
    """Mean pairwise cosine distance across a set of vectors."""
    n = len(vectors)
    if n < 2:
        return float("nan")
    dists = []
    for i, j in itertools.combinations(range(n), 2):
        c = _cosine(vectors[i], vectors[j])
        dists.append(1.0 - c)
    return float(np.mean(dists))


def _pace_internal_score(vectors: list[np.ndarray]) -> float:
    """PACE-style internal chain score: for each position i (>=1), mean
    cosine distance from position i to all preceding positions; then
    average across positions.
    """
    n = len(vectors)
    if n < 2:
        return float("nan")
    per_pos = []
    for i in range(1, n):
        ds = [1.0 - _cosine(vectors[i], vectors[j]) for j in range(i)]
        per_pos.append(float(np.mean(ds)))
    return float(np.mean(per_pos))


def score_circle(
    raw_words: list[str],
    seed: str,
    embeddings: FastTextEmbeddings,
) -> CircleScore:
    """Score one circle attempt. Stores vectors and similarities so the
    caller can threshold them post-hoc at any tau.
    """
    # Normalize; drop OOV words (zero vectors would give NaN cosines).
    seed_vec = embeddings.encode(seed.lower())

    word_vecs: list[np.ndarray] = []
    kept_words: list[str] = []
    for w in raw_words:
        v = embeddings.encode(w.lower())
        if np.linalg.norm(v) > 0:
            word_vecs.append(v)
            kept_words.append(w.lower())

    n_returned = len(raw_words)
    n_in_vocab = len(kept_words)

    edge_cosines: list[float] = []
    closure_cosine = float("nan")
    mean_edge = float("nan")
    min_edge = float("nan")
    pairwise_div = float("nan")
    pace_internal = float("nan")

    if n_in_vocab >= 1 and np.linalg.norm(seed_vec) > 0:
        # Edge 0: seed -> word_0 (first model output)
        all_vecs = [seed_vec] + word_vecs  # length n_in_vocab + 1
        # Consecutive edges: 0-1, 1-2, ..., (n-1)-n
        edge_cosines = [
            _cosine(all_vecs[i], all_vecs[i + 1])
            for i in range(len(all_vecs) - 1)
        ]
        closure_cosine = _cosine(all_vecs[-1], seed_vec)

        if edge_cosines:
            mean_edge = float(np.mean(edge_cosines))
            min_edge = float(np.min(edge_cosines))

        pairwise_div = _pairwise_diversity(word_vecs)
        pace_internal = _pace_internal_score(word_vecs)

    distinct = len(set(kept_words)) == len(kept_words)
    excludes_seed = seed.lower() not in set(kept_words)

    return CircleScore(
        seed=seed,
        words=kept_words,
        n_returned=n_returned,
        n_in_vocab=n_in_vocab,
        edge_cosines=edge_cosines,
        closure_cosine=closure_cosine,
        mean_edge_cosine=mean_edge,
        min_edge_cosine=min_edge,
        pairwise_diversity=pairwise_div,
        pace_internal_score=pace_internal,
        distinct=distinct,
        excludes_seed=excludes_seed,
    )


# --- aggregation at a chosen (tau_edge, tau_closure) pair ---


def evaluate_at_thresholds(
    score: CircleScore,
    tau_edge: float,
    tau_closure: float,
    expected_n: int,
) -> dict:
    """Apply thresholds to a CircleScore, returning boolean verdicts.

    Returns a dict with:
      edge_coherence_rate: fraction of consecutive edges above tau_edge
                            (not counting closure)
      all_edges_ok: bool, every consecutive edge >= tau_edge
      closure_ok: closure_cosine >= tau_closure
      valid_circle: all_edges_ok AND closure_ok AND distinct AND excludes_seed
                    AND n_in_vocab == expected_n
    """
    edges = score.edge_cosines
    # First len(edges) - 0 edges are seed->w1, w1->w2, ..., but we already
    # lumped closure separately (stored as closure_cosine). Here "edges" are
    # the intra-chain-and-entry edges, excluding the closure edge. For the
    # Nagarajan formulation we care that EVERY edge including closure is
    # valid, so check both.
    if not edges:
        return {
            "edge_coherence_rate": float("nan"),
            "all_edges_ok": False,
            "closure_ok": False,
            "valid_circle": False,
        }
    above = [c >= tau_edge for c in edges]
    edge_coherence_rate = float(np.mean(above))
    all_edges_ok = all(above)
    closure_ok = (
        not math.isnan(score.closure_cosine)
        and score.closure_cosine >= tau_closure
    )
    valid = (
        all_edges_ok
        and closure_ok
        and score.distinct
        and score.excludes_seed
        and score.n_in_vocab == expected_n
    )
    return {
        "edge_coherence_rate": edge_coherence_rate,
        "all_edges_ok": all_edges_ok,
        "closure_ok": closure_ok,
        "valid_circle": valid,
    }


# --- scoring multi-trial cross-chain diversity (Hivemind-style) ---


def cross_trial_diversity(trials: list[CircleScore]) -> float:
    """Mean pairwise FastText distance between the WORD SETS of multiple
    circle trials for the same seed. Captures how varied a model's outputs
    are when sampled multiple times on the same prompt — a creativity-
    distinct-from-capability signal inspired by Hivemind.

    Uses centroid vectors of each circle (average of word vectors) for
    pairwise comparison.
    """
    # We don't have access to the raw vectors here (they're not serialized
    # in CircleScore). Instead use word-level overlap as a cheap proxy.
    # Cosine-vector centroid diversity can be computed at scoring time
    # from re-encoded words.
    if len(trials) < 2:
        return float("nan")
    dists = []
    for a, b in itertools.combinations(trials, 2):
        set_a, set_b = set(a.words), set(b.words)
        if not set_a or not set_b:
            continue
        jaccard = len(set_a & set_b) / max(1, len(set_a | set_b))
        dists.append(1.0 - jaccard)
    return float(np.mean(dists)) if dists else float("nan")


__all__ = [
    "DEFAULT_SEEDS",
    "FastTextEmbeddings",
    "CircleScore",
    "circle_prompt",
    "parse_circle_response",
    "score_circle",
    "evaluate_at_thresholds",
    "cross_trial_diversity",
]
