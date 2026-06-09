"""Smoke test for the fixed-rubric judge.

Scores a twist / predictable / random contrast triple (built-in, or from
stimuli_file) and checks the rubric orders them the way the theory predicts:

  - twist has the highest OVERALL,
  - twist > predictable on SURPRISE (a predictable ending re-reads nothing),
  - twist > random on COHERENCE (a random ending contradicts the setup).

This is a cheap correctness check on the rubric before any scale-up, NOT the
benchmark. Usage:

    uv run python src/plot_twist/scripts/run_rubric_smoke.py configs/plot_twist/rubric.yaml
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from src.utils import init_directory, load_config, save_config
from src.plot_twist.rubric_judge import RubricConfig, score_stories, save_scores


# A shared setup with three endings. Same first paragraph; the ending differs.
_SETUP = (
    "Mara spent the last Saturday of the month clearing out her mother's house. "
    "Everyone had always said she had her father's eyes, though she had never "
    "met him and her mother would never speak of him. In the bottom drawer of "
    "the writing desk, under the lining paper, she found a small brass key and, "
    "taped beneath it, a bundle of letters in a hand she did not recognize."
)

BUILTIN_TRIPLE = [
    {
        "id": "twist",
        "label": "twist",
        "story": _SETUP + " "
        "The letters were from a hospital in another city, dated the spring "
        "before Mara was born, addressed to her mother and beginning, each one, "
        "'Thank you for taking her.' Mara sat on the floor a long time. The eyes "
        "no one could place, the photographs that began when she was already "
        "walking, the way her mother had held her a little too carefully, like "
        "something borrowed -- all of it turned, quietly, into a different life.",
    },
    {
        "id": "predictable",
        "label": "predictable",
        "story": _SETUP + " "
        "They were old love letters between her mother and father, written the "
        "year they met. Mara read them slowly, crying a little, and understood "
        "her mother had loved him after all. She kept the bundle, locked the "
        "drawer again, and drove home in the early dark, missing them both.",
    },
    {
        "id": "random",
        "label": "random",
        "story": _SETUP + " "
        "The letters turned out to be a coded treasure map left by a pirate "
        "ancestor. By midnight Mara had bought a boat and a metal detector, and "
        "within a week she was diving off the coast of Belize, hauling up chests "
        "of Spanish gold and laughing into the spray.",
    },
]


def _fmt(x: float | None) -> str:
    return "  NA" if x is None else f"{x:4.1f}"


def main(config_path: str) -> None:
    cfg_dict = load_config(config_path)
    out = init_directory(cfg_dict["output_dir"], overwrite=True)
    save_config(cfg_dict, out)

    cfg = RubricConfig(
        judge_models=cfg_dict["judge_models"],
        temperature=cfg_dict.get("temperature", 0.0),
        max_tokens=cfg_dict.get("max_tokens", 400),
        concurrency=cfg_dict.get("concurrency", 16),
    )

    stimuli_file = cfg_dict.get("stimuli_file")
    if stimuli_file:
        stimuli = json.loads(Path(stimuli_file).read_text())
    else:
        stimuli = BUILTIN_TRIPLE

    scores = asyncio.run(score_stories(cfg, stimuli))
    save_scores(scores, cfg, out)

    by_id = {s.story_id: s for s in scores}
    print(f"\nrubric {cfg.rubric_version} | judges: {', '.join(cfg.judge_models)}\n")
    print(f"{'id':<12} {'present':>8} {'surpr':>6} {'inev':>6} {'prose':>6} {'overall':>8}")
    for s in scores:
        print(
            f"{s.story_id:<12} {str(s.twist_present):>8} "
            f"{_fmt(s.surprise)} {_fmt(s.coherence)} {_fmt(s.prose_quality)} "
            f"{_fmt(s.overall):>8}"
        )

    # Sanity checks (only when the built-in triple is used).
    if not stimuli_file and {"twist", "predictable", "random"} <= set(by_id):
        t, p, r = by_id["twist"], by_id["predictable"], by_id["random"]
        checks = {
            "twist overall highest": (
                t.overall is not None
                and (p.overall is None or t.overall >= p.overall)
                and (r.overall is None or t.overall >= r.overall)
            ),
            "twist surprise > predictable": (
                t.surprise is not None and p.surprise is not None and t.surprise > p.surprise
            ),
            "twist coherence > random": (
                t.coherence is not None
                and r.coherence is not None
                and t.coherence > r.coherence
            ),
        }
        print()
        for name, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"\n-> {sum(checks.values())}/{len(checks)} sanity checks passed")

    print(f"\nsaved: {out/'rubric_scores.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: run_rubric_smoke.py <config.yaml>")
        sys.exit(1)
    main(sys.argv[1])
