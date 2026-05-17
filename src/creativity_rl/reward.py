"""MCNS reward: novelty conditional on appropriateness gate.

    r_MCNS(y, x) = N(y, x; A_t) * 1[A(y, x) > τ]    (hard gate)
    r_MCNS(y, x) = N(y, x; A_t) * σ((A(y, x) - τ)/T)  (soft gate)

The reward is the *only* optimization signal. Appropriateness contributes
nothing positive — it gates novelty. This is the MCNS commitment from
Lehman & Stanley (2010); see docs/memos/mcns_dpo.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MCNSReward:
    tau: float
    mode: str = "hard"          # "hard" | "soft"
    soft_temperature: float | None = None

    def __post_init__(self) -> None:
        if self.mode == "soft" and self.soft_temperature is None:
            raise ValueError("FATAL: soft_temperature required when mode='soft'")
        if self.mode not in ("hard", "soft"):
            raise ValueError(f"FATAL: unknown mode {self.mode}")

    def __call__(self, appropriateness: np.ndarray, novelty: np.ndarray) -> np.ndarray:
        """Compute r_MCNS for a batch of (a_i, n_i) pairs.

        Args:
            appropriateness: shape (B,), A(y_i, x_i) scores.
            novelty:         shape (B,), N(y_i, x_i; A_t) scores.

        Returns:
            Shape (B,) reward values.
        """
        if appropriateness.shape != novelty.shape:
            raise ValueError(
                f"FATAL: shape mismatch: appropriateness {appropriateness.shape} "
                f"vs novelty {novelty.shape}"
            )
        if self.mode == "hard":
            gate = (appropriateness > self.tau).astype(np.float32)
        else:
            gate = _sigmoid((appropriateness - self.tau) / self.soft_temperature)
        return novelty.astype(np.float32) * gate


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))
