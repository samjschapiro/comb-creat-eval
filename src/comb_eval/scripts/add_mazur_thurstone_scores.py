"""Add Mazur Creative-Writing Thurstone-rating scores to benchmarks.json.

The lechmazur/writing repository switched its Canonical Leaderboard from the
V2 absolute 0--10 rubric to a Thurstone pairwise-comparison rating
(``pilot_2026_04_18_anchor_n20_fullroster_cap3_v1``). The previous V2
leaderboard is now archived but unchanged.

Adds one key per matched model:
    mazur_cw_thurstone: Thurstone log-odds-style rating (range ~ -6 to +4,
                       higher is better)

Usage:
    uv run python src/comb_eval/scripts/add_mazur_thurstone_scores.py
"""

import json
from pathlib import Path


# Transcribed from
#   https://github.com/lechmazur/writing/blob/main/README.md
# (fetched 2026-04-29). The Canonical Leaderboard has 23 models; only the
# ones that match a model in our eval pool are listed here.
MAZUR_THURSTONE_SCORES: dict[str, float] = {
    # Display name on Mazur board -> our OpenRouter key
    # gpt-5.4-medium  (medium reasoning is our default)
    "openai_gpt-5-4":                    +3.6311,
    # gpt-5.2-medium
    "openai_gpt-5-2":                    +3.0438,
    # claude-opus-4-6-16K  (16K reasoning budget)
    "anthropic_claude-opus-4-6":         +3.1530,
    # claude-sonnet-4-6-16K
    "anthropic_claude-sonnet-4-6":       +2.5355,
    # gemini-3.1-pro-preview
    "google_gemini-3-1-pro-preview":     -0.7073,
    # grok-4-1-fast-reasoning -- our pool has the original grok-4 (not 4.1
    # fast); kept here as a tentative match.
    "x-ai_grok-4":                       -1.4181,
}


def main():
    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"Mazur Thurstone leaderboard models we matched: {len(MAZUR_THURSTONE_SCORES)}")
    print(f"Models currently in benchmarks.json: {len(benchmarks)}")
    print()

    added = 0
    for key, score in MAZUR_THURSTONE_SCORES.items():
        if key not in benchmarks:
            benchmarks[key] = {}
        benchmarks[key]["mazur_cw_thurstone"] = score
        marker = " (new)" if "arena_overall" not in benchmarks[key] else ""
        print(f"  {key}: mazur_cw_thurstone={score:+.4f}{marker}")
        added += 1

    with open(bench_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nAdded mazur_cw_thurstone for {added} models. Saved to {bench_path}")


if __name__ == "__main__":
    main()
