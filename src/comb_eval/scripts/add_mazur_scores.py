"""Add Lech Mazur Creative Writing Benchmark V2 scores to benchmarks.json.

Uses the V2 leaderboard from https://github.com/lechmazur/writing (April 2025
update, 50 models graded by 7 newer grader LLMs on a 16-question rubric).
Only V2 is used — V3/V4 have smaller model sets and different rubrics, so
pooling would introduce version skew.

Scores are on a 0-10 mean rubric scale. Writes `mazur_cw_v2` field.

Usage:
    uv run python src/comb_eval/scripts/add_mazur_scores.py
"""

import json
from pathlib import Path


# V2 leaderboard scraped from README.md (as of April 2026).
MAZUR_V2_SCORES: dict[str, float] = {
    "GPT-5 (medium reasoning)": 8.60,
    "Kimi K2": 8.56,
    "Claude Opus 4.1 (no reasoning)": 8.47,
    "Claude Opus 4.1 Thinking 16K": 8.45,
    "o3-pro (medium reasoning)": 8.44,
    "o3 (medium reasoning)": 8.39,
    "Gemini 2.5 Pro": 8.38,
    "Claude Opus 4 Thinking 16K": 8.36,
    "Claude Opus 4 (no reasoning)": 8.31,
    "GPT-5 mini (medium reasoning)": 8.31,
    "Qwen 3 235B A22B": 8.30,
    "DeepSeek R1": 8.30,
    "Qwen 3 235B A22B 25-07 Think": 8.24,
    "DeepSeek R1 05/28": 8.19,
    "GPT-4o Mar 2025": 8.18,
    "Claude Sonnet 4 Thinking 16K": 8.14,
    "Claude 3.7 Sonnet Thinking 16K": 8.11,
    "Claude Sonnet 4 (no reasoning)": 8.09,
    "Gemini 2.5 Pro Preview 05-06": 8.09,
    "Gemini 2.5 Pro Exp 03-25": 8.05,
    "Claude 3.5 Sonnet 2024-10-22": 8.03,
    "Qwen QwQ-32B 16K": 8.02,
    "Baidu Ernie 4.5 300B A47B": 8.00,
    "Gemma 3 27B": 7.99,
    "Claude 3.7 Sonnet": 7.94,
    "Mistral Medium 3": 7.73,
    "GPT-OSS-120B": 7.71,
    "DeepSeek V3-0324": 7.70,
    "Grok 4": 7.69,
    "Gemini 2.5 Flash": 7.65,
    "Grok 3 Beta (no reasoning)": 7.64,
    "GPT-4.5 Preview": 7.56,
    "Qwen 3 30B A3B": 7.53,
    "o4-mini (medium reasoning)": 7.50,
    "Gemini 2.0 Flash Think Exp 01-21": 7.38,
    "Claude 3.5 Haiku": 7.35,
    "Grok 3 Mini Beta (low)": 7.35,
    "GLM-4.5": 7.34,
    "Qwen 2.5 Max": 7.29,
    "Gemini 2.0 Flash Exp": 7.15,
    "o1 (medium reasoning)": 7.02,
    "Mistral Large 2": 6.90,
    "GPT-4o mini": 6.72,
    "o1-mini": 6.49,
    "Grok 2 12-12": 6.36,
    "Microsoft Phi-4": 6.26,
    "Llama 4 Maverick": 6.20,
    "o3-mini (high reasoning)": 6.17,
    "o3-mini (medium reasoning)": 6.15,
    "Amazon Nova Pro": 6.05,
}


# Map OpenRouter IDs to Mazur V2 leaderboard names. None = not in V2 leaderboard.
# Where our model is a close cousin (same family, same scale) but not the exact
# variant Mazur tested, we use None — we'd rather have fewer data points than
# noisy ones.
OPENROUTER_KEY_TO_MAZUR_V2: dict[str, str | None] = {
    # --- Anthropic ---
    "anthropic_claude-opus-4-6": None,  # Mazur has 4.1, not 4.5/4.6
    "anthropic_claude-opus-4-5": None,
    "anthropic_claude-sonnet-4-6": None,
    "anthropic_claude-sonnet-4-5": None,  # only Sonnet 4 in V2
    "anthropic_claude-sonnet-4": "Claude Sonnet 4 (no reasoning)",
    "anthropic_claude-haiku-4-5": None,
    "anthropic_claude-3-5-haiku": "Claude 3.5 Haiku",
    "anthropic_claude-3-haiku": None,
    # --- OpenAI ---
    "openai_gpt-5-4": None,  # V2 only has GPT-5
    "openai_gpt-5-4-mini": None,
    "openai_gpt-5-4-nano": None,
    "openai_gpt-5": "GPT-5 (medium reasoning)",
    "openai_gpt-5-mini": "GPT-5 mini (medium reasoning)",
    "openai_gpt-5-nano": None,
    "openai_gpt-4o": "GPT-4o Mar 2025",
    "openai_gpt-4o-mini": "GPT-4o mini",
    "openai_gpt-4-1": None,  # not in V2
    "openai_gpt-4-1-mini": None,
    "openai_gpt-4-1-nano": None,
    "openai_gpt-4-turbo": None,
    "openai_gpt-3-5-turbo": None,
    "openai_o3": "o3 (medium reasoning)",
    "openai_o3-mini": "o3-mini (medium reasoning)",
    "openai_o4-mini": "o4-mini (medium reasoning)",
    # --- Google ---
    "google_gemini-2-5-pro": "Gemini 2.5 Pro",
    "google_gemini-2-5-flash": "Gemini 2.5 Flash",
    "google_gemini-2-0-flash-001": "Gemini 2.0 Flash Exp",
    "google_gemma-3-27b-it": "Gemma 3 27B",
    "google_gemma-2-27b-it": None,
    "google_gemma-2-9b-it": None,
    # --- Meta ---
    "meta-llama_llama-4-maverick": "Llama 4 Maverick",
    "meta-llama_llama-4-scout": None,
    "meta-llama_llama-3-3-70b-instruct": None,
    "meta-llama_llama-3-1-70b-instruct": None,
    "meta-llama_llama-3-1-8b-instruct": None,
    "meta-llama_llama-3-2-3b-instruct": None,
    # --- DeepSeek ---
    "deepseek_deepseek-r1": "DeepSeek R1",
    "deepseek_deepseek-chat-v3-0324": "DeepSeek V3-0324",
    "deepseek_deepseek-chat": None,  # different generation
    # --- Qwen ---
    "qwen_qwen3-235b-a22b": "Qwen 3 235B A22B",
    "qwen_qwen-2-5-72b-instruct": None,
    "qwen_qwen3-32b": None,  # V2 has 30B A3B, not 32B dense
    "qwen_qwen3-14b": None,
    "qwen_qwen3-8b": None,
    "qwen_qwq-32b": "Qwen QwQ-32B 16K",
    # --- Mistral ---
    "mistralai_mistral-large-2407": "Mistral Large 2",
    "mistralai_mistral-large-2411": "Mistral Large 2",  # same family
    "mistralai_mistral-nemo": None,
    "mistralai_mistral-7b-instruct-v0-1": None,
    "mistralai_mistral-small-24b-instruct-2501": None,
    # --- Cohere ---
    "cohere_command-a": None,
    "cohere_command-r-plus-08-2024": None,
    # --- Other ---
    "microsoft_phi-4": "Microsoft Phi-4",
    "nvidia_llama-3-1-nemotron-70b-instruct": None,
}


def main():
    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"Mazur V2 leaderboard: {len(MAZUR_V2_SCORES)} models")
    print(f"Our benchmark entries: {len(benchmarks)}")
    print()

    matched = 0
    unmatched = []

    for or_key, bench_entry in benchmarks.items():
        mz_name = OPENROUTER_KEY_TO_MAZUR_V2.get(or_key)
        if mz_name is None:
            unmatched.append(or_key)
            continue
        if mz_name not in MAZUR_V2_SCORES:
            print(f"  {or_key} (wanted='{mz_name}'): NOT FOUND in Mazur V2 leaderboard")
            unmatched.append(or_key)
            continue
        bench_entry["mazur_cw_v2"] = MAZUR_V2_SCORES[mz_name]
        print(f"  {or_key} -> {mz_name}  (mazur_cw_v2={MAZUR_V2_SCORES[mz_name]:.2f})")
        matched += 1

    with open(bench_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nMatched {matched}/{len(benchmarks)} models.")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for m in unmatched:
            print(f"  - {m}")
    print(f"\nSaved to {bench_path}")


if __name__ == "__main__":
    main()
