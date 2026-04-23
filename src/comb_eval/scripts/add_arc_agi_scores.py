"""Add ARC-AGI v2 scores to benchmarks.json.

ARC-AGI v2 evaluates pattern-inference reasoning on visual grid puzzles
designed to be resistant to pure pattern-matching~\\citep{chollet2019measure}.
Higher is better. Scores are transcribed from the llm-stats.com
ARC-AGI-v2 leaderboard (https://llm-stats.com/benchmarks/arc-agi-v2),
which itself aggregates scores from the official ARC Prize leaderboard.

Adds one key to benchmarks.json per model:
    arc_agi_v2:  fraction of tasks solved (0..1), higher is better.

Usage:
    uv run python src/comb_eval/scripts/add_arc_agi_scores.py
"""

import json
from pathlib import Path


# Per-model ARC-AGI v2 scores, verbatim from the llm-stats leaderboard
# (fetched 2026-04-23). Keys are our OpenRouter filesystem keys.
ARC_AGI_V2_SCORES: dict[str, float] = {
    # Already in our pool via the original or previous-expansion evals
    "google_gemini-2-5-pro":             0.049,
    "openai_o3":                         0.065,
    "anthropic_claude-opus-4-5":         0.376,
    "anthropic_claude-sonnet-4-6":       0.583,
    "anthropic_claude-opus-4-6":         0.688,
    "openai_gpt-5-4":                    0.733,
    # Added by the ARC-AGI-coverage eval on 2026-04-23
    "openai_gpt-5-2":                    0.529,
    "google_gemini-3-flash-preview":     0.336,
    "x-ai_grok-4":                       0.159,
    "google_gemini-3-1-pro-preview":     0.771,
    # Present on the leaderboard but NOT in our eval pool (for reference):
    #   Claude Opus 4              0.086  -- dropped for cost
    #   GPT-5.2 Pro                0.542  -- dropped for cost
    #   Gemini 3 Pro (non-image)   0.311  -- not on OpenRouter
    #   Muse Spark                 0.425  -- not on OpenRouter
}


def main():
    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"ARC-AGI v2 models to add: {len(ARC_AGI_V2_SCORES)}")
    print(f"Models currently in benchmarks.json: {len(benchmarks)}")
    print()

    added = 0
    for key, score in ARC_AGI_V2_SCORES.items():
        if key not in benchmarks:
            benchmarks[key] = {}
        benchmarks[key]["arc_agi_v2"] = score
        marker = " (new)" if "arena_overall" not in benchmarks[key] else ""
        print(f"  {key}: arc_agi_v2={score}{marker}")
        added += 1

    with open(bench_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nAdded ARC-AGI v2 scores for {added} models.")
    print(f"Saved to {bench_path}")


if __name__ == "__main__":
    main()
