"""Offline unit tests for the DMPO implementation. No model load, no GPU.

Covers:
1. compute_dmpo_term math: p = softmax(r/α), q = softmax(φ), MSE shape and
   that the loss is exactly the analytical value.
2. Gradient routing: grad flows through per_token_logps -> q, NOT through
   rewards (which are the detached target).
3. The MSE collapses to ~0 when policy length-normalized log-probs are
   chosen to make q ≈ p.
4. _RewardStash captures the most recent batch's rewards.
5. MathRewardFn: format-only when no GT, full score when GT matches,
   format-only when GT mismatches, robust to chat-style completions.

    uv run python src/creativity_rl/scripts/test_dmpo_offline.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.creativity_rl.dmpo import _RewardStash, compute_dmpo_term
from src.creativity_rl.math_reward import (
    MathRewardFn,
    answers_equal,
    extract_answer,
    has_answer_format,
)


def _make_logps(B: int, K: int, T: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    per_token_logps = torch.randn(B * K, T, requires_grad=True)
    completion_mask = torch.ones(B * K, T)
    # Vary completion lengths so length-normalization is exercised.
    for i in range(B * K):
        completion_mask[i, T - (i % 3) :] = 0
    return per_token_logps, completion_mask


# ---- 1. compute_dmpo_term: shapes + analytical value ------------------------


def test_dmpo_shapes_and_value():
    B, K, T = 3, 4, 8
    alpha = 1 / 15
    logps, mask = _make_logps(B, K, T)
    rewards = torch.tensor([0.1, 0.9, 0.2, 0.8] * B, dtype=torch.float32)

    dm_loss, telem = compute_dmpo_term(logps, mask, rewards, K, alpha)

    # Recompute analytically
    comp_lens = mask.sum(dim=-1).clamp(min=1)
    phi = (logps * mask).sum(dim=-1) / comp_lens
    p = torch.softmax(rewards.view(B, K) / alpha, dim=-1)
    q = torch.softmax(phi.view(B, K), dim=-1)
    expected = ((p - q) ** 2).mean().item()

    assert abs(float(dm_loss) - expected) < 1e-6, (
        f"dm_loss {float(dm_loss):.6f} != expected {expected:.6f}"
    )
    assert "dm/loss" in telem and "dm/q_entropy_mean" in telem
    print(f"  ok: dm_loss={float(dm_loss):.6f}")


# ---- 2. Gradient routing ----------------------------------------------------


def test_grad_flows_through_logps_only():
    B, K, T = 2, 4, 6
    logps, mask = _make_logps(B, K, T)
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0] * B, requires_grad=False)

    dm_loss, _ = compute_dmpo_term(logps, mask, rewards, K, alpha=1.0)
    dm_loss.backward()

    assert logps.grad is not None and logps.grad.abs().sum() > 0, "no grad on logps"
    # rewards is detached as the target inside compute_dmpo_term, so it must
    # NOT have a grad even after backward.
    assert rewards.grad is None, "rewards should not receive grad"
    print(
        f"  ok: |grad logps|={logps.grad.abs().sum():.3f}, rewards.grad={rewards.grad}"
    )


# ---- 3. Loss -> 0 when policy already matches the target --------------------


def test_loss_near_zero_when_matched():
    B, K, T = 2, 4, 5
    alpha = 1.0
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0] * B, dtype=torch.float32)
    # With uniform mask, phi_i = mean_t(logps_{i,t}). To get phi_i = X, every
    # token logp in row i must be X (so the mean is X). Then softmax(phi) ==
    # softmax(rewards/alpha=1) and the MSE is exactly 0.
    target_phi = rewards.clone()
    mask = torch.ones(B * K, T)
    logps = target_phi.view(-1, 1).expand(-1, T).clone()
    logps.requires_grad_(True)

    dm_loss, _ = compute_dmpo_term(logps, mask, rewards, K, alpha)
    assert float(dm_loss) < 1e-10, f"expected ~0, got {float(dm_loss)}"
    print(f"  ok: matched-distributions dm_loss={float(dm_loss):.2e}")


# ---- 4. _RewardStash --------------------------------------------------------


def test_reward_stash_captures():
    calls = []

    def fn(prompts, completions, **kw):
        calls.append((prompts, completions))
        return [0.5 * len(c) for c in completions]

    stash = _RewardStash(fn)
    out = stash(prompts=["p1", "p2"], completions=["aa", "bbbb"])
    assert out == [1.0, 2.0]
    assert stash.last == [1.0, 2.0]
    # __name__ preserved (TRL metric logging uses it).
    assert stash.__name__ == "fn"
    print(f"  ok: stash.last={stash.last}")


# ---- 5. MathRewardFn --------------------------------------------------------


def test_math_reward_format_only_no_gt():
    """Paper §4.3: r_fmt = 0.1 iff `Answer:` keyword present. \\boxed{}
    alone does NOT count; the model must follow the prompted format."""
    fn = MathRewardFn()
    rewards = fn(
        completions=[
            "Reasoning... Answer: 42",            # +0.1 (format only, no GT)
            "I think the answer is \\boxed{42}.", # 0 (no `Answer:` keyword)
            "I am not sure what to put.",         # 0
        ],
    )
    assert abs(rewards[0] - 0.1) < 1e-9, rewards
    assert rewards[1] == 0.0 and rewards[2] == 0.0, rewards
    print(f"  ok: no-GT rewards={rewards}")


def test_math_reward_full_score_when_correct():
    """Paper §4.3: r_fmt and r_opt are independent additive terms. Format
    compliance is NOT a gate on correctness credit. The model can earn
    correctness via the \\boxed{} fallback even when it skips `Answer:`."""
    fn = MathRewardFn()
    rewards = fn(
        completions=[
            "...reasoning... Answer: 42",   # +1.1 (format + correct)
            "...reasoning... Answer: 99",   # +0.1 (format, wrong)
            "...reasoning... \\boxed{42}.", # +1.0 (correct, no format reward)
            "...reasoning... \\boxed{99}.", # 0.0 (neither)
        ],
        ground_truth=["42", "42", "42", "42"],
    )
    assert abs(rewards[0] - 1.1) < 1e-9, rewards
    assert abs(rewards[1] - 0.1) < 1e-9, rewards
    assert abs(rewards[2] - 1.0) < 1e-9, rewards
    assert rewards[3] == 0.0, rewards
    print(f"  ok: rewards={rewards}")


def test_math_reward_chat_completion_format():
    fn = MathRewardFn()
    chat = [{"role": "assistant", "content": "Reasoning.\nAnswer: 7"}]
    r = fn(completions=[chat], ground_truth=["7"])
    assert abs(r[0] - 1.1) < 1e-9, r
    print(f"  ok: chat-format reward={r}")


def test_extract_answer_priority_order():
    """Parser prefers `Answer: X` > `#### X` > `\\boxed{X}` > last number."""
    cases = [
        ("So x=42. Answer: 42", "42"),                       # `Answer:`
        ("Just \\boxed{99}.", "99"),                         # boxed fallback
        ("####  100", "100"),                                # GSM8K hashes
        ("Reasoning leads to \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),
        ("Answer: 3.14", "3.14"),
        ("Answer: \\boxed{7}", "7"),                         # nested boxed in Answer
        ("", None),
    ]
    for text, expected in cases:
        got = extract_answer(text)
        assert got == expected or (got and expected and answers_equal(got, expected)), (
            f"extract_answer({text!r}) -> {got!r}, expected {expected!r}"
        )
    print(f"  ok: extract_answer({len(cases)} cases)")


def test_has_answer_format_strict():
    """Format check must be strict: requires the `Answer:` keyword."""
    assert has_answer_format("Solution. Answer: 42") is True
    assert has_answer_format("Solution. answer = 42") is True   # case/`=` ok
    assert has_answer_format("Solution. \\boxed{42}") is False  # not enough
    assert has_answer_format("Just 42.") is False
    assert has_answer_format("") is False
    print("  ok: has_answer_format")


def test_answers_equal_numeric_tolerance():
    assert answers_equal("42", "42.0")
    assert answers_equal("3.14", "3.1400000")
    assert not answers_equal("42", "43")
    assert not answers_equal(None, "1")
    print("  ok: answers_equal")


# ---- main -------------------------------------------------------------------


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for t in tests:
        try:
            print(f"[{t.__name__}]")
            t()
        except Exception as e:
            fails += 1
            print(f"  FAIL: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - fails}/{len(tests)} passed")
    sys.exit(0 if fails == 0 else 1)
