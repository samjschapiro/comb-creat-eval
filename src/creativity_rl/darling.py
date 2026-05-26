"""Faithful reimplementation of the DARLING reward (Meta, arXiv 2509.02534).

Ported from facebookresearch/darling
verl/verl/utils/reward_score/diversity_rewards.py, dropping the verl /
asyncio harness. Logic and constants are kept identical:

- Pairwise semantic-equivalence between the n rollouts of one prompt.
- Short-response unigram shortcut (max len <= 5 words).
- Otherwise a similarity classifier; we use the public deberta path
  `yimingzhang/deberta-v3-large-generation-similarity`, input encoded as
  [CLS] s1 [SEP] s2 [SEP] with token_type_ids 0 then 1, equivalent iff
  softmax class-1 probability > 0.102 (their threshold).
- Union-find clusters responses into semantic-equivalence partitions.
- Per-response diversity = (group_size - own_partition_size) / group_size.

Full DARLING reward (multiplicative, as in their wildchat darling.sh):
    r(x, y_i | group) = quality(x, y_i) * diversity(y_i | group)
GRPO advantage is group-mean-centered, NOT std-normalized
(algorithm.norm_adv_by_std_in_grpo=False in their config).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

_SIM_MODEL_ID = "yimingzhang/deberta-v3-large-generation-similarity"
_EQUIV_THRESHOLD = 0.102  # their constant


def _maybe_test_equality(r0: str, r1: str) -> bool | None:
    """Cheap shortcut for very short responses (their maybe_test_equality)."""
    u0 = r0.strip().lower().split()
    u1 = r1.strip().lower().split()
    m = max(len(u0), len(u1))
    if m <= 5:
        common = set(u0) & set(u1)
        return len(common) * 2 >= m
    return None


class SimilarityClassifier:
    """Public deberta generation-similarity classifier, their encoding."""

    def __init__(self, device: str = "cuda"):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(_SIM_MODEL_ID)
            .to(device)
            .eval()
        )

    @torch.inference_mode()
    def equivalent(self, s1: str, s2: str) -> bool:
        shortcut = _maybe_test_equality(s1, s2)
        if shortcut is not None:
            return shortcut
        ids = [self.tok.cls_token_id]
        for s in (s1, s2):
            ids.extend(
                self.tok.encode(
                    s, truncation=True, max_length=128, add_special_tokens=False
                )
            )
            ids.append(self.tok.sep_token_id)
        prompt_len = ids.index(self.tok.sep_token_id) + 1
        tt = [0] * prompt_len + [1] * (len(ids) - prompt_len)
        iid = torch.tensor(ids, device=self.device, dtype=torch.int64).unsqueeze(0)
        tid = torch.tensor(tt, device=self.device, dtype=torch.int64).unsqueeze(0)
        out = self.model(input_ids=iid, token_type_ids=tid)
        score = out["logits"].softmax(-1)[0, 1].item()
        return score > _EQUIV_THRESHOLD


def _partition_distinctness(responses: list[str], clf: SimilarityClassifier) -> np.ndarray:
    """Union-find partition diversity, verbatim DARLING `partition()`
    (verl/utils/reward_score/partition_reward_vllm_serve_modernbert.py):
    for a response in a cluster of size s out of n,
        reward = (n - s) / (n - 1)
    then floor every reward at 0.1 (their `max(r, 0.1)` -- diversity is
    never 0, so a homogeneous group still passes quality through at 0.1
    instead of zeroing the multiplicative reward and the gradient).
    """
    n = len(responses)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if clf.equivalent(responses[i], responses[j]):
                union(i, j)

    sizes: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1

    def _div(i: int) -> float:
        s = sizes[find(i)]
        r = (n - s) / (n - 1) if n > 1 else 0.0
        return max(r, 0.1)  # DARLING floor: diversity is never 0

    return np.array([_div(i) for i in range(n)], dtype=np.float32)


@dataclass
class DarlingReward:
    """TRL-compatible reward: quality * partition-diversity, per prompt group.

    quality_scorer.score(prompts, completions) -> np.ndarray of RM scores
    (Athene-RM-8B in the faithful setup). Diversity is computed within the
    group of completions that share a prompt (TRL passes one prompt's K
    generations together; we group by the prompt string).
    """

    quality_scorer: object
    clf: SimilarityClassifier
    _last: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.__name__ = "darling"

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        from src.creativity_rl.reward_callable import (
            _flatten_completion,
            _flatten_prompt,
        )

        fp = [_flatten_prompt(p) for p in prompts]
        fc = [_flatten_completion(c) for c in completions]
        quality = np.asarray(self.quality_scorer.score(fp, fc), dtype=np.float32)

        groups: dict[str, list[int]] = {}
        for i, p in enumerate(fp):
            groups.setdefault(p, []).append(i)

        diversity = np.zeros(len(fc), dtype=np.float32)
        for idxs in groups.values():
            d = _partition_distinctness([fc[i] for i in idxs], self.clf)
            for k, i in enumerate(idxs):
                diversity[i] = d[k]

        reward = quality * diversity  # multiplicative, as in darling.sh
        self._last = {
            "reward/mean": float(reward.mean()),
            "reward/std": float(reward.std()),
            "quality/mean": float(quality.mean()),
            "diversity/mean": float(diversity.mean()),
            "diversity/frac_unique": float((diversity > 0).mean()),
        }
        return [float(r) for r in reward]

    def telemetry(self) -> dict:
        return dict(self._last)
