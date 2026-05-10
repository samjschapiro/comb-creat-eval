"""Canonical NoveltyBench distinctness classifier.

Loads the fine-tuned DeBERTa-v3-large equivalence classifier from
[yimingzhang/deberta-v3-large-generation-similarity](https://huggingface.co/yimingzhang/deberta-v3-large-generation-similarity).

Label convention (verified by probing on canonical pairs):
- LABEL_0 = DISTINCT (functionally different)
- LABEL_1 = EQUIVALENT (functionally same)

The model card lacks a usage example or label-space documentation, but
the convention above is consistent across:
- Cross-domain story / non-story pairs → ~0.92 LABEL_0
- Same-plot-with-renamed-characters pairs → ~0.73 LABEL_1
- Tokenizer falls back to the base microsoft/deberta-v3-large tokenizer
  since the classifier repo does not ship its own tokenizer files.
"""

from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CLASSIFIER_REPO = "yimingzhang/deberta-v3-large-generation-similarity"
TOKENIZER_REPO = "microsoft/deberta-v3-large"

# Label convention — DO NOT change without re-probing the classifier.
DISTINCT = 0
EQUIVALENT = 1


_lock = threading.Lock()
_loaded: dict[str, object] = {}


def _load(device: str = "cpu"):
    """Load and cache (tokenizer, model). Idempotent across calls."""
    with _lock:
        if "tok" in _loaded:
            return _loaded["tok"], _loaded["model"]
        tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
        model = (
            AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_REPO)
            .to(device)
            .eval()
        )
        _loaded["tok"] = tok
        _loaded["model"] = model
        _loaded["device"] = device
        return tok, model


def is_distinct(a: str, b: str, threshold: float = 0.5, device: str = "cpu") -> bool:
    """Returns True iff the classifier judges (a, b) DISTINCT.

    threshold applies to P(LABEL_0). Default 0.5 = argmax. Push higher to
    bias toward EQUIVALENT (more conservative classes / fewer novelty
    points), lower to bias toward DISTINCT.
    """
    tok, model = _load(device)
    inp = tok(a, b, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inp).logits
    p_distinct = float(torch.softmax(logits, dim=-1)[0, DISTINCT].item())
    return p_distinct >= threshold


def batch_distinct(
    pairs: Sequence[tuple[str, str]],
    threshold: float = 0.5,
    device: str = "cpu",
    batch_size: int = 16,
) -> list[bool]:
    """Vectorised version. pairs is a sequence of (a, b) tuples."""
    if not pairs:
        return []
    tok, model = _load(device)
    out: list[bool] = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        a_batch = [a for a, _ in chunk]
        b_batch = [b for _, b in chunk]
        inp = tok(
            a_batch,
            b_batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(device)
        with torch.no_grad():
            logits = model(**inp).logits
        probs = torch.softmax(logits, dim=-1)[:, DISTINCT].cpu().numpy()
        out.extend(bool(p >= threshold) for p in probs)
    return out
