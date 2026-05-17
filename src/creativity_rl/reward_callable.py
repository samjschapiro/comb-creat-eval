"""TRL-compatible reward function wrapping MCNS = novelty * gate(appropriateness).

TRL's GRPOTrainer expects a callable with signature:

    reward_func(prompts: list[str], completions: list[str], **kwargs) -> list[float]

Extra dataset columns are passed through as kwargs. We attach a
`cluster_id` column to the training dataset so the reward function can
route each response to the correct per-cluster archive.

Side effects: after computing reward, qualifying responses (A > tau and
N > admission_threshold) are admitted to their cluster's archive. This
state mutation is deliberate; the archive must grow with training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.creativity_rl.archive import ClusterArchiveSet
from src.creativity_rl.novelty import knn_novelty
from src.creativity_rl.reward import MCNSReward
from src.creativity_rl.scoring import AppropriatenessScorer, SBERTEmbedder


@dataclass
class MCNSRewardFunction:
    scorer: AppropriatenessScorer
    embedder: SBERTEmbedder
    archive_set: ClusterArchiveSet
    reward: MCNSReward
    k_nearest: int
    admission_mode: str | float = "median"

    # Telemetry for logging.
    _last_appropriateness: np.ndarray | None = None
    _last_novelty: np.ndarray | None = None
    _last_reward: np.ndarray | None = None
    _last_pass_rate: float = 0.0
    _step: int = 0

    def __post_init__(self) -> None:
        # TRL >=0.16 inspects reward_funcs[i].__name__ for logging.
        self.__name__ = "mcns_reward"

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        **kwargs,
    ) -> list[float]:
        if "cluster_id" not in kwargs:
            raise ValueError(
                "FATAL: reward function requires 'cluster_id' in kwargs. "
                "Add a cluster_id column to the training dataset."
            )
        cluster_ids = list(kwargs["cluster_id"])
        if len(cluster_ids) != len(completions):
            raise ValueError(
                f"FATAL: cluster_id length {len(cluster_ids)} != "
                f"completions length {len(completions)}"
            )

        # TRL passes conversational prompts and completions as
        # list-of-message-dicts when the model uses a chat template. Flatten
        # to plain strings: the RM and SBERT need raw text inputs.
        flat_prompts = [_flatten_prompt(p) for p in prompts]
        flat_completions = [_flatten_completion(c) for c in completions]

        appropriateness = self.scorer.score(flat_prompts, flat_completions)
        embeddings = self.embedder.encode(flat_completions)

        # Compute novelty AND remember whether the archive was empty at
        # query time, so we don't record the artificial bootstrap value
        # in the running-median rho.
        novelty = np.zeros(len(completions), dtype=np.float32)
        archive_was_empty = np.zeros(len(completions), dtype=bool)
        for i, (emb, cid) in enumerate(zip(embeddings, cluster_ids)):
            archive = self.archive_set[int(cid)]
            archive_was_empty[i] = len(archive) == 0
            novelty[i] = knn_novelty(emb, archive, self.k_nearest)

        rewards = self.reward(appropriateness, novelty)

        # Archive admission: A > tau AND N > rho_c (or during warmup, any
        # passing response is admitted).
        passes_gate = appropriateness > self.reward.tau
        for i, (emb, cid, passes, n, was_empty) in enumerate(
            zip(embeddings, cluster_ids, passes_gate, novelty, archive_was_empty)
        ):
            if not passes:
                continue
            archive = self.archive_set[int(cid)]
            rho = archive.admission_threshold(self.admission_mode)
            if n > rho:
                archive.add(
                    emb,
                    novelty_at_admission=float(n),
                    record_novelty=not was_empty,
                )

        self._last_appropriateness = appropriateness
        self._last_novelty = novelty
        self._last_reward = rewards
        self._last_pass_rate = float(passes_gate.mean())
        self._step += 1

        return [float(r) for r in rewards]

    def telemetry(self) -> dict:
        """Last-call summary stats for logging."""
        if self._last_reward is None:
            return {}
        archive_sizes = [len(self.archive_set[c]) for c in range(self.archive_set.n_clusters)]
        return {
            "reward/mean": float(self._last_reward.mean()),
            "reward/std": float(self._last_reward.std()),
            "reward/nonzero_frac": float((self._last_reward > 0).mean()),
            "appropriateness/mean": float(self._last_appropriateness.mean()),
            "appropriateness/pass_rate": self._last_pass_rate,
            "novelty/mean": float(self._last_novelty.mean()),
            "novelty/std": float(self._last_novelty.std()),
            "archive/total_size": int(sum(archive_sizes)),
            "archive/mean_size": float(np.mean(archive_sizes)),
            "archive/max_size": int(max(archive_sizes)),
        }


def _flatten_prompt(p) -> str:
    """Extract plain user text from a TRL conversational prompt."""
    if isinstance(p, str):
        return p
    if isinstance(p, list):
        for msg in p:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        raise ValueError(f"FATAL: no user message in prompt: {p}")
    raise TypeError(f"FATAL: unexpected prompt type: {type(p)}")


def _flatten_completion(c) -> str:
    """Extract plain text from a TRL conversational completion.

    TRL emits either a raw string (text-format prompts) or
    [{"role": "assistant", "content": "..."}] (conversational prompts).
    """
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # Take the last assistant turn (usually the only one).
        for msg in reversed(c):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        # Fallback: concatenate any content fields.
        return "".join(m.get("content", "") for m in c if isinstance(m, dict))
    if isinstance(c, dict):
        return c.get("content", "")
    raise TypeError(f"FATAL: unexpected completion type: {type(c)}")
