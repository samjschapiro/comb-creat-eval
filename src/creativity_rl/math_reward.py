"""Rule-based math reward for DMPO/GRPO replication (arXiv:2605.19461).

Follows the paper's composite reward structure for verifiable tasks
(Sec. 4.3): R = r_format + r_correctness, where
    r_format       = 0.1 if the answer is in the expected box/Answer format
    r_correctness  = 1.0 if the extracted answer matches the ground truth

No reward model is loaded -- the verifier is pure regex + numerical
equality, so the only GPU memory cost is the policy itself.
"""

from __future__ import annotations

import re

# DMPO paper §4.3: format = "Chain-of-Thought reasoning followed by
# `Answer: [Solution]`". Primary path matches that. GSM8K-style ####
# fallback is for the GSM8K dataset only. \boxed{} fallback handles
# models (like Qwen2.5-Math) that default to that even when prompted
# for "Answer:". If nothing matches, fall back to last numeric token.
_ANSWER = re.compile(
    r"answer\s*[:=]\s*\$?\\?(?:boxed\{)?([^\n$}]+?)\}?\s*\$?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_HASHES = re.compile(r"####\s*([^\n]+)")
_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_answer(text: str | None) -> str | None:
    """Pull the model's final answer from a completion. Returns the
    normalized answer string, or None if no parseable answer found.
    Tries (in order): `Answer: X`, `#### X`, `\\boxed{X}`, last number."""
    if not text:
        return None
    for pat in (_ANSWER, _HASHES, _BOXED):
        matches = pat.findall(text)
        if matches:
            return _normalize(matches[-1])
    nums = re.findall(r"-?\d+(?:[\.,/]\d+)*", text)
    if nums:
        return _normalize(nums[-1])
    return None


def has_answer_format(text: str | None) -> bool:
    """Paper's r_fmt criterion: output contains an `Answer:` block. Strict
    on the keyword to faithfully match §4.3's spec, not just any parseable
    number. \\boxed{} alone does not earn the format reward."""
    if not text:
        return False
    return _ANSWER.search(text) is not None


def _normalize(s: str) -> str:
    s = s.strip().rstrip(".").strip("$").strip()
    s = s.replace(",", "").replace(" ", "")
    # Numeric canonicalization: 42.0 -> 42, 3.14 -> 3.14.
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return repr(f)
    except (ValueError, TypeError):
        return s


def answers_equal(a: str | None, b: str | None) -> bool:
    """Equal if string-equal after normalization OR numerically within 1e-6."""
    if a is None or b is None:
        return False
    a, b = _normalize(a), _normalize(b)
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return False


def _flatten_completion(c) -> str:
    """TRL passes completions as either strings or chat-template message lists."""
    if isinstance(c, str):
        return c
    if isinstance(c, list) and c and isinstance(c[0], dict):
        return "".join(m.get("content", "") for m in c)
    return str(c)


class MathRewardFn:
    """Composite math reward, paper-faithful (DMPO §4.3):
        R(o) = r_fmt + r_opt
        r_fmt = 0.1  iff `Answer: X` block present
        r_opt = 1.0  iff extracted answer matches ground truth (math binary)
                       = 0.0 otherwise (paper: `r_opt = QR if valid else 0`;
                       for math, QR is binary correctness)

    Ground truth comes in as the dataset's `ground_truth` column. TRL
    forwards extra dataset columns as kwargs when remove_unused_columns
    is False on the trainer config.
    """

    def __init__(self, format_weight: float = 0.1, correct_weight: float = 1.0):
        self.format_weight = format_weight
        self.correct_weight = correct_weight
        self.__name__ = "math_correctness"
        self._last_extracted: list[str | None] = []

    def __call__(
        self,
        prompts=None,
        completions=None,
        ground_truth=None,
        **kwargs,
    ) -> list[float]:
        gts = ground_truth if ground_truth is not None else kwargs.get("answer")
        rewards: list[float] = []
        self._last_extracted = []
        if completions is None:
            return rewards
        for i, c in enumerate(completions):
            text = _flatten_completion(c)
            self._last_extracted.append(extract_answer(text))
            r = 0.0
            # Format reward: strictly requires the `Answer:` keyword
            # block, per paper §4.3 ("...followed by Answer: [Solution]").
            if has_answer_format(text):
                r += self.format_weight
            # Correctness reward (binary for math).
            if gts is not None:
                gt = gts[i]
                a = self._last_extracted[-1]
                if a is not None and answers_equal(a, str(gt)):
                    r += self.correct_weight
            rewards.append(r)
        return rewards
