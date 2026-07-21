"""Predict-the-twist: can a model infer a story's withheld plot twist from its setup?

Each TwistBench story is annotated with a `setup` (what the reader is led to believe)
and a `reveal` (the withheld twist). We give a predictor model the setup alone, ask it
to predict the reveal, and score the guess by embedding similarity to the real reveal.

This module holds the HOW (prompt + scoring). Orchestration lives in
scripts/run_predict.py.
"""

import re

# --- Stage 1: locate the twist boundary so we can feed the real pre-twist prose ---

BOUNDARY_SYSTEM = "You identify where the plot twist begins in a short story."

BOUNDARY_USER_TEMPLATE = (
    "Below is a complete short story built around a plot twist — a late reveal that "
    "makes the reader reinterpret earlier events.\n\n"
    "Identify the exact sentence where the twist/reveal BEGINS (the first sentence "
    "that delivers, or starts delivering, the surprise). Copy the FIRST 12 WORDS of "
    "that sentence VERBATIM, exactly as they appear in the story — no quotes, no "
    "commentary, no changes.\n\n"
    'Story:\n"""{story}"""\n\n'
    "Output only those verbatim words."
)


def build_boundary_messages(story: str) -> list[dict]:
    """Chat messages asking a model to return the verbatim opening of the twist sentence."""
    return [
        {"role": "system", "content": BOUNDARY_SYSTEM},
        {"role": "user", "content": BOUNDARY_USER_TEMPLATE.format(story=story)},
    ]


def split_at_twist(story: str, locator: str, min_pre_words: int = 20) -> str:
    """Return the story text up to (excluding) the twist, located via `locator`.

    `locator` is the verbatim opening of the twist sentence. We match it in the story
    tolerant of whitespace/newline differences, backing off to fewer leading words if
    the tail mismatches. Raises loudly (no silent fallback) if the twist can't be
    located or the pre-twist prose is implausibly short.
    """
    words = locator.split()
    if len(words) < 3:
        raise ValueError(f"locator too short to trust: {locator!r}")
    words = words[:12]
    match = None
    for n in range(len(words), 2, -1):  # back off from 12 words down to 3
        pattern = r"\s+".join(re.escape(w) for w in words[:n])
        match = re.search(pattern, story)
        if match:
            break
    if not match:
        raise ValueError(f"twist locator not found in story: {locator!r}")
    pre = story[: match.start()].strip()
    if len(pre.split()) < min_pre_words:
        raise ValueError(
            f"pre-twist prose only {len(pre.split())} words; locator likely at story start"
        )
    return pre


# --- Stage 2: predict the withheld twist from the pre-twist prose ---

PREDICT_SYSTEM = (
    "You are given the beginning of a short story that is built around a single plot "
    "twist. The story is cut off just before the twist — the late reveal that makes "
    "the reader reinterpret everything before it. Predict that twist."
)

PREDICT_USER_TEMPLATE = (
    "Story so far (it cuts off just before the twist):\n\n{pre_reveal}\n\n"
    "Predict the plot twist / reveal this story is building toward. "
    "Describe only the reveal, in 1-3 sentences. Do not retell the story."
)


def build_predict_messages(pre_reveal: str) -> list[dict]:
    """Chat messages asking a predictor to guess the withheld twist from the story so far."""
    return [
        {"role": "system", "content": PREDICT_SYSTEM},
        {"role": "user", "content": PREDICT_USER_TEMPLATE.format(pre_reveal=pre_reveal)},
    ]


def cosine_similarity_scores(predictions: list[str], reveals: list[str], embed_model: str):
    """Cosine similarity between each predicted twist and its true reveal.

    Embeddings are L2-normalized, so cosine == dot product. Returns a list of floats
    aligned with the inputs. Requires sentence-transformers.
    """
    if len(predictions) != len(reveals):
        raise ValueError(
            f"FATAL: {len(predictions)} predictions != {len(reveals)} reveals"
        )
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model)
    pred_emb = np.asarray(
        model.encode(predictions, normalize_embeddings=True, show_progress_bar=False)
    )
    reveal_emb = np.asarray(
        model.encode(reveals, normalize_embeddings=True, show_progress_bar=False)
    )
    return (pred_emb * reveal_emb).sum(axis=1).tolist()
