"""Divergent Semantic Integration (DSI) scorer.

Reference: Johnson, Kaufman, Baker, et al. (2022), "Divergent semantic
integration (DSI): Extracting creativity from narratives with distributional
semantic modeling," Behavior Research Methods. DSI scores a text by the average
*semantic distance* among its words in contextual-embedding (BERT) space: higher
DSI = more far-flung concepts integrated = (per that literature) more creative.

This is the embedding/semantic-distance metric H4 predicts is BLIND to plot
twists (it tracks exploratory creativity, not transformational): see
docs/HYPOTHESES.md H4.

Method here:
  1. Split the text into word chunks that fit BERT's 512-token window (context is
     preserved within a chunk).
  2. Run BERT (output_hidden_states); take a chosen hidden layer; aggregate
     word-piece sub-tokens to one vector per word (mean).
  3. DSI = mean pairwise cosine DISTANCE (1 - cosine similarity) over the words.

Length adaptation (the original DSI targets short narratives; our stories are
900--22k words): we collect word vectors across the whole text, then if there are
more than `max_words` we take a seeded random sample before the O(n^2) pairwise
step. DSI is a mean, so the sample is an unbiased estimate; report `max_words`.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer

_WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass
class DSIConfig:
    model_name: str = "bert-base-uncased"
    layer: int = -2          # hidden_states index (-1 last, -2 second-to-last); 0 = embeddings
    chunk_words: int = 300   # words per BERT pass (stays under 512 word-pieces)
    max_words: int = 600     # cap for the O(n^2) pairwise step; seeded subsample above this
    seed: int = 0
    content_only: bool = False  # if True, drop a small stopword set (a "content-word" DSI)


_STOP = set(
    "the a an and or but of to in on at for with as by from is are was were be been being "
    "this that these those it its he she they them his her their you your i we our not no".split()
)


class DSIScorer:
    """Loads BERT once; reuse across many texts."""

    def __init__(self, cfg: DSIConfig | None = None):
        self.cfg = cfg or DSIConfig()
        self.tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        self.model = AutoModel.from_pretrained(
            self.cfg.model_name, output_hidden_states=True
        ).eval()

    @torch.no_grad()
    def _word_vectors(self, text: str) -> torch.Tensor:
        words = _WORD_RE.findall(text.lower())
        if self.cfg.content_only:
            words = [w for w in words if w not in _STOP]
        vecs: list[torch.Tensor] = []
        for i in range(0, len(words), self.cfg.chunk_words):
            chunk = words[i : i + self.cfg.chunk_words]
            enc = self.tok(
                chunk, is_split_into_words=True, return_tensors="pt",
                truncation=True, max_length=512,
            )
            hs = self.model(**enc).hidden_states[self.cfg.layer][0]  # (seq, hidden)
            agg: dict[int, list[torch.Tensor]] = defaultdict(list)
            for tok_idx, wid in enumerate(enc.word_ids(0)):
                if wid is not None:
                    agg[wid].append(hs[tok_idx])
            for wid in sorted(agg):
                vecs.append(torch.stack(agg[wid]).mean(0))
        return torch.stack(vecs) if vecs else torch.empty(0)

    @staticmethod
    def _mean_pairwise_distance(E: torch.Tensor) -> float:
        """Mean pairwise cosine distance (1 - cos) over the rows of E."""
        m = E.shape[0]
        if m < 2:
            return float("nan")
        En = E / E.norm(dim=1, keepdim=True).clamp_min(1e-8)
        dist = 1.0 - (En @ En.T)
        return float((dist.sum() - dist.diagonal().sum()) / (m * (m - 1)))

    def score(self, text: str) -> tuple[float, int]:
        """Whole-text DSI (the original definition). Returns (dsi, n_words_used).
        For long texts the O(n^2) step is capped by a seeded subsample -- but note
        the DSI paper (Johnson 2022, Study 5) shows DSI--creativity validity DEGRADES
        with length, so prefer `score_chunked` for long stories."""
        E = self._word_vectors(text)
        n = E.shape[0]
        if n < 2:
            return float("nan"), n
        if n > self.cfg.max_words:
            rng = random.Random(self.cfg.seed)
            E = E[sorted(rng.sample(range(n), self.cfg.max_words))]
        return self._mean_pairwise_distance(E), min(n, self.cfg.max_words)

    def score_chunked(self, text: str, chunk_words: int = 100) -> dict:
        """Length-robust DSI for long texts: split into consecutive `chunk_words`
        windows (each within DSI's validated short-narrative range), compute DSI per
        chunk, and average. Returns {mean, std, n_chunks, chunk_words}. The mean of
        fixed-size-chunk DSIs is, by construction, ~invariant to total story length."""
        E = self._word_vectors(text)
        n = E.shape[0]
        per = [
            self._mean_pairwise_distance(E[i : i + chunk_words])
            for i in range(0, n - chunk_words + 1, chunk_words)
        ]
        if not per:  # text shorter than one chunk -> score the whole thing
            per = [self._mean_pairwise_distance(E)]
        t = torch.tensor(per)
        return {
            "mean": float(t.mean()), "std": float(t.std(unbiased=False)),
            "n_chunks": len(per), "chunk_words": chunk_words,
        }
