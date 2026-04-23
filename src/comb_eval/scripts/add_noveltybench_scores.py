"""Add NoveltyBench (Zhang et al. 2025, arXiv:2504.05228) scores to benchmarks.

NoveltyBench measures how many functionally-distinct responses a model produces
over k=10 samples (Distinct_10), plus a combined diversity+utility score
(Utility_10). Higher is better for both. Scores are transcribed from Table 1
of the paper (p. 6).

Adds two keys to benchmarks.json per model:
    noveltybench_distinct:  mean # of equivalence classes per prompt (1..10)
    noveltybench_utility:   user-utility score combining novelty and quality

Usage:
    uv run python src/comb_eval/scripts/add_noveltybench_scores.py
"""

import json
from pathlib import Path


# (distinct_10, utility_10) per model. Keys are our OpenRouter filesystem keys.
# Direct transcription from NoveltyBench paper Table 1 (arXiv:2504.05228v4, p. 6).
# Higher = more diverse / better.
NOVELTYBENCH_SCORES: dict[str, tuple[float, float]] = {
    # Anthropic
    "anthropic_claude-3-5-haiku":           (1.94, 2.50),
    "anthropic_claude-3-5-sonnet":          (1.76, 2.36),
    "anthropic_claude-3-opus":              (2.04, 2.67),
    # OpenAI
    "openai_gpt-4o-mini":                   (2.65, 3.11),
    "openai_gpt-4o":                        (2.88, 3.27),
    # Google (Gemini) — map to dated OpenRouter slugs where we eval'd
    "google_gemini-1-5-pro":                (1.85, 2.73),
    "google_gemini-2-0-flash-lite-001":     (2.83, 3.20),
    "google_gemini-2-0-flash-001":          (2.81, 3.17),
    "google_gemini-2-0-pro":                (2.25, 2.64),
    # Cohere (map to dated OpenRouter slugs — that's what we evaluated)
    "cohere_command-r7b-12-2024":           (3.58, 3.35),
    "cohere_command-r-08-2024":             (2.68, 2.98),
    "cohere_command-r-plus-08-2024":        (2.79, 3.08),
    # Google (Gemma 2)
    "google_gemma-2-2b-it":                 (5.66, 4.63),
    "google_gemma-2-9b-it":                 (3.25, 3.93),
    "google_gemma-2-27b-it":                (3.03, 3.77),
    # Meta (Llama 3)
    "meta-llama_llama-3-2-1b-instruct":     (6.74, 2.81),
    "meta-llama_llama-3-2-3b-instruct":     (5.10, 3.24),
    "meta-llama_llama-3-1-8b-instruct":     (5.24, 3.76),
    "meta-llama_llama-3-3-70b-instruct":    (2.49, 2.87),
    "meta-llama_llama-3-1-405b-instruct":   (3.20, 3.39),
}


def main():
    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"NoveltyBench models to add: {len(NOVELTYBENCH_SCORES)}")
    print(f"Models currently in benchmarks.json: {len(benchmarks)}")
    print()

    added = 0
    for key, (distinct, utility) in NOVELTYBENCH_SCORES.items():
        # Create entry if NoveltyBench-only model not yet present
        if key not in benchmarks:
            benchmarks[key] = {}
        benchmarks[key]["noveltybench_distinct"] = distinct
        benchmarks[key]["noveltybench_utility"] = utility
        marker = " (new)" if "arena_overall" not in benchmarks[key] else ""
        print(f"  {key}: distinct={distinct}, utility={utility}{marker}")
        added += 1

    with open(bench_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nAdded NoveltyBench scores for {added} models.")
    print(f"Saved to {bench_path}")


if __name__ == "__main__":
    main()
