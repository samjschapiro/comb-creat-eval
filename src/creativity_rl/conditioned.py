"""Conditioned divergence: prompt formatting and target/pair selection.

The model is conditioned on a prompt x and a set S of answers already
given, and asked to produce a good answer that is substantively
different from S. This module has no model dependencies so its logic
can be validated without a GPU.

Two consumers:
  - supervised training: needs only the selected target (the appropriate
    candidate farthest from S).
  - preference fallback (DPO): needs a (chosen, rejected) pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def format_conditioned_prompt(x: str, S: list[str]) -> str:
    """Build the user message: the task, the answers already given, and
    an instruction to produce a good answer different from them.

    If S is empty the task is returned unchanged (the conditioned
    behavior is undefined with nothing to differ from). This is explicit,
    not a silent fallback: callers building training data must supply a
    non empty S.
    """
    if not S:
        return x
    listed = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(S))
    return (
        f"{x}\n\n"
        f"Here are answers that have already been given:\n\n"
        f"{listed}\n\n"
        f"Write a new answer that is high quality and substantively "
        f"different from all of the answers above."
    )


@dataclass
class Selection:
    chosen_idx: int
    rejected_idx: int | None  # None when no contrastive negative exists


def _validate(appropriateness: np.ndarray, novelty: np.ndarray, n: int) -> None:
    if appropriateness.shape != (n,) or novelty.shape != (n,):
        raise ValueError(
            f"FATAL: expected shape ({n},) for appropriateness and novelty, "
            f"got {appropriateness.shape} and {novelty.shape}"
        )


def select_sft_target(
    appropriateness: np.ndarray,
    novelty: np.ndarray,
    tau: float,
) -> int | None:
    """Index of the supervised target: appropriate candidate farthest
    from S. Returns None if no candidate passes the appropriateness gate.
    """
    n = len(appropriateness)
    _validate(appropriateness, novelty, n)
    feasible = np.where(appropriateness > tau)[0]
    if feasible.size == 0:
        return None
    return int(feasible[np.argmax(novelty[feasible])])


def select_pair(
    appropriateness: np.ndarray,
    novelty: np.ndarray,
    tau: float,
) -> Selection | None:
    """(chosen, rejected) for the preference fallback.

    - both pass the gate: chosen = farthest from S, rejected = closest.
      Dropped if every feasible candidate has identical novelty (no
      contrast to learn).
    - exactly one passes: it is chosen; rejected = an infeasible
      candidate if one exists (teaches the appropriateness gate), else
      no pair.
    - none pass: no pair.
    """
    n = len(appropriateness)
    _validate(appropriateness, novelty, n)
    feasible = np.where(appropriateness > tau)[0]
    infeasible = np.where(appropriateness <= tau)[0]

    if feasible.size >= 2:
        chosen = int(feasible[np.argmax(novelty[feasible])])
        rejected = int(feasible[np.argmin(novelty[feasible])])
        if novelty[chosen] == novelty[rejected]:
            return None
        return Selection(chosen_idx=chosen, rejected_idx=rejected)

    if feasible.size == 1:
        chosen = int(feasible[0])
        if infeasible.size > 0:
            # Closest infeasible is the most informative negative.
            rejected = int(infeasible[np.argmax(novelty[infeasible])])
            return Selection(chosen_idx=chosen, rejected_idx=rejected)
        return Selection(chosen_idx=chosen, rejected_idx=None)

    return None
