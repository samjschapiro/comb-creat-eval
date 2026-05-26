"""DMPO (Distribution-Matching Policy Optimization).

Faithful TRL implementation of arXiv:2605.19461 (Li et al., 2026). Adds a
group-level forward-KL approximation to GRPO via an MSE term:

    L_DM = (1/G) Σ_i (p(o_i|O) - q_θ(o_i|O))²

where, for each prompt group O = {o_1, ..., o_K}:
    p(o_i|O)   = softmax(r_i / α)_i          (target Boltzmann; detached)
    q_θ(o_i|O) = softmax(φ(o_i))_i           (policy distribution)
    φ(o_i)     = (1/|o_i|) Σ_t log π_θ(o_{i,t} | o_{i,<t}, x)

Full objective: L = L_GRPO + λ · L_DM.

Implementation notes
--------------------
1. *Reward stash.* TRL's GRPOTrainer only exposes the z-scored advantages
   to compute_loss, not raw rewards. We wrap each reward function so the
   most recent batch's raw rewards are readable from the trainer.
2. *Zero extra forward.* We override `_get_per_token_logps` to capture
   the per-token log-probs already computed by parent's compute_loss,
   preserving the autograd graph. The DMPO term piggy-backs on that
   single forward pass.
3. *Group structure.* GRPOTrainer batches rollouts contiguously per
   prompt: positions [0:K] are prompt 0's K samples, [K:2K] prompt 1's,
   etc. We reshape (B*K,) -> (B, K) and softmax along K.
"""

from __future__ import annotations

from typing import Any

import torch

try:
    from trl import GRPOTrainer
except Exception:  # pragma: no cover -- importing trl is heavy; offline tests stub it
    GRPOTrainer = object  # type: ignore[misc,assignment]


class _RewardStash:
    """Wrap a reward function so DMPOTrainer can read the most recent
    batch's raw rewards. Preserves __name__ for TRL's metric logging."""

    def __init__(self, fn):
        self._fn = fn
        self.last: list[float] | None = None
        self.__name__ = getattr(fn, "__name__", "reward")

    def __call__(self, prompts=None, completions=None, **kwargs):
        out = self._fn(prompts=prompts, completions=completions, **kwargs)
        try:
            self.last = [float(x) for x in out]
        except (TypeError, ValueError):
            self.last = None
        return out


def compute_dmpo_term(
    per_token_logps: torch.Tensor,   # (N, T)  N = B*K
    completion_mask: torch.Tensor,   # (N, T)
    rewards: torch.Tensor,           # (N,)
    num_generations: int,            # K
    alpha: float,                    # Boltzmann temperature
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pure-function DMPO MSE term. Returns (dm_loss, telemetry dict).
    Kept standalone so it can be unit-tested without TRL."""
    N, _ = per_token_logps.shape
    K = num_generations
    assert N % K == 0, f"batch {N} not divisible by K={K}"
    assert rewards.shape == (N,), f"rewards shape {tuple(rewards.shape)} != ({N},)"
    B = N // K

    comp_lens = completion_mask.sum(dim=-1).clamp(min=1).float()
    phi = (per_token_logps * completion_mask).sum(dim=-1) / comp_lens  # (N,)

    phi_g = phi.view(B, K)
    rewards_g = rewards.view(B, K).to(phi.dtype)

    p = torch.softmax(rewards_g / alpha, dim=-1).detach()
    q = torch.softmax(phi_g, dim=-1)

    dm_loss = ((p - q) ** 2).mean()
    q_d = q.detach()
    telem = {
        "dm/loss": float(dm_loss.detach()),
        "dm/p_max_mean": float(p.max(dim=-1).values.mean()),
        "dm/q_max_mean": float(q_d.max(dim=-1).values.mean()),
        # Effective entropy of the policy distribution over the group:
        # high = mode-covering (good), low = mode-seeking (collapse).
        "dm/q_entropy_mean": float(
            -(q_d.clamp(min=1e-12).log() * q_d).sum(dim=-1).mean()
        ),
    }
    return dm_loss, telem


class DMPOTrainer(GRPOTrainer):  # type: ignore[misc]
    """GRPOTrainer + DMPO distribution-matching term (arXiv:2605.19461).

    Args:
        dm_lambda: Weight on the distribution-matching term (paper: 2.0).
        dm_alpha: Boltzmann temperature on rewards (paper: 1/15).
    """

    def __init__(self, *args, dm_lambda: float = 2.0, dm_alpha: float = 1 / 15.0, **kwargs):
        # Wrap reward_funcs so we can read raw rewards in compute_loss.
        rf = kwargs.get("reward_funcs", None)
        stashes: list[_RewardStash] = []
        if rf is not None:
            if callable(rf):
                wrapped = _RewardStash(rf)
                stashes = [wrapped]
                kwargs["reward_funcs"] = wrapped
            elif isinstance(rf, (list, tuple)):
                new_list: list[Any] = []
                for f in rf:
                    if callable(f):
                        w = _RewardStash(f)
                        stashes.append(w)
                        new_list.append(w)
                    else:
                        new_list.append(f)
                kwargs["reward_funcs"] = new_list

        super().__init__(*args, **kwargs)
        self._reward_stashes = stashes
        self.dm_lambda = float(dm_lambda)
        self.dm_alpha = float(dm_alpha)
        self._last_per_token_logps: torch.Tensor | None = None

    # ---- hooks ----------------------------------------------------------------

    def _get_per_token_logps(self, *args, **kwargs):
        """Override to stash per-token logps (still autograd-attached) so the
        DMPO term in compute_loss can reuse them without a second forward."""
        out = super()._get_per_token_logps(*args, **kwargs)
        self._last_per_token_logps = out
        return out

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Clear the stash so we only use logps re-captured by parent's call
        # to _get_per_token_logps below; never reuse a stale tensor from a
        # prior batch (e.g., from _generate_and_score_completions).
        self._last_per_token_logps = None
        grpo_loss = super().compute_loss(
            model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
        )

        per_token_logps = self._last_per_token_logps
        if per_token_logps is None:
            return grpo_loss

        completion_mask = inputs.get("completion_mask", None)
        if completion_mask is None:
            return grpo_loss

        # Sum raw rewards across reward functions (matches TRL's aggregation).
        rewards: torch.Tensor | None = None
        for s in self._reward_stashes:
            if s.last is None:
                continue
            r = torch.tensor(s.last, dtype=torch.float32, device=per_token_logps.device)
            rewards = r if rewards is None else rewards + r
        if rewards is None or rewards.numel() != per_token_logps.shape[0]:
            return grpo_loss

        try:
            dm_loss, telem = compute_dmpo_term(
                per_token_logps=per_token_logps,
                completion_mask=completion_mask,
                rewards=rewards,
                num_generations=self.num_generations,
                alpha=self.dm_alpha,
            )
        except AssertionError:
            return grpo_loss

        # Log telemetry where TRL is already aggregating metrics.
        if hasattr(self, "_metrics") and isinstance(self._metrics, dict):
            mode = "train" if getattr(model, "training", True) else "eval"
            bucket = self._metrics.setdefault(mode, {})
            for k, v in telem.items():
                bucket.setdefault(k, []).append(v)
            bucket.setdefault("dm/lambda", []).append(self.dm_lambda)

        return grpo_loss + self.dm_lambda * dm_loss
