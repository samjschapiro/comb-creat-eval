"""Decisive Athene-RM-8B sanity check: does our reproduced reward model
actually rank obviously-good responses above obviously-bad ones, with a
sane scale? If not, the DARLING quality signal is noise and every run is
poisoned. Also prints the raw (unrounded) LR from the latest wandb run.

    uv run python src/creativity_rl/scripts/diag_athene_sanity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

PAIRS = [
    (
        "Explain why the sky appears blue during the day.",
        # good
        "Sunlight contains all colors. As it passes through the atmosphere, "
        "shorter (blue) wavelengths scatter much more than longer (red) ones "
        "(Rayleigh scattering, ∝ 1/λ^4), so light reaching your eyes "
        "from all directions is dominated by blue.",
        # bad
        "idk honestly, it just is. ask a scientist or google it lol.",
    ),
    (
        "Write a haiku about autumn (5-7-5).",
        "Crisp leaves drift downward\nA cold wind hums through bare boughs\n"
        "Frost waits in the dark",
        "autumn is a season it happens every year leaves fall and its cold ok "
        "thats it",
    ),
    (
        "Give three concrete tips for a software engineering interview.",
        "1) Think aloud so the interviewer follows your reasoning. 2) Clarify "
        "requirements and edge cases before coding. 3) State complexity and "
        "test your solution on a small example before declaring done.",
        "just be confident and wing it, you'll probably be fine, no need to "
        "prepare much.",
    ),
    (
        "Summarize the plot of Romeo and Juliet in two sentences.",
        "Two young lovers from feuding Verona families, the Montagues and "
        "Capulets, secretly marry. A chain of miscommunications leads both to "
        "die by suicide, and their deaths finally reconcile the families.",
        "It's about some people in Italy and there is fighting and then it is "
        "sad at the end I think, not totally sure.",
    ),
]


def main() -> None:
    from src.creativity_rl.scoring import AppropriatenessScorer

    sc = AppropriatenessScorer(
        model_name="Nexusflow/Athene-RM-8B",
        device="cuda",
        load_in_4bit=False,
        max_length=1024,
    )
    print("is_athene:", sc._is_athene, flush=True)

    prompts, goods, bads = zip(*PAIRS)
    gq = sc.score(list(prompts), list(goods))
    bq = sc.score(list(prompts), list(bads))
    ok = 0
    for i, p in enumerate(prompts):
        win = gq[i] > bq[i]
        ok += int(win)
        print(
            f"\n[{i}] {p[:60]!r}\n"
            f"    good={gq[i]:+.4f}  bad={bq[i]:+.4f}  "
            f"good>bad? {'YES' if win else 'NO'}  (margin={gq[i]-bq[i]:+.4f})",
            flush=True,
        )
    print(
        f"\n=== Athene ranks good>bad on {ok}/{len(prompts)} pairs; "
        f"good mean={sum(gq)/len(gq):+.3f}  bad mean={sum(bq)/len(bq):+.3f} ===",
        flush=True,
    )

    # raw LR from the latest darling_h100 run
    try:
        import wandb

        api = wandb.Api()
        rs = sorted(
            api.runs("schapirolab/comb-creat-eval",
                     filters={"display_name": "darling_h100"}),
            key=lambda r: r.created_at,
        )
        r = rs[-1]
        h = r.history(samples=200, keys=["train/learning_rate"], pandas=True)
        lrs = h["train/learning_rate"].dropna().tolist()
        print(
            f"\nLR raw (run {r.id}): first={lrs[0]:.3e} "
            f"last={lrs[-1]:.3e} min={min(lrs):.3e} max={max(lrs):.3e}",
            flush=True,
        )
    except Exception as e:
        print("LR fetch failed:", e, flush=True)


if __name__ == "__main__":
    main()
