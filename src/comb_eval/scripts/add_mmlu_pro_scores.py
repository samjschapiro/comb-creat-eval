"""Add MMLU-Pro scores to benchmarks.json.

Pulls from data/dat_eval/mmlu_pro_raw.json (scraped from artificialanalysis.ai)
and matches models to our OpenRouter IDs. Writes `mmlu_pro` field (0-1 accuracy).

Matching strategy:
  1. Explicit Artificial-Analysis display name per OpenRouter ID (preferred — AA
     has many variants with suffixes like "(Non-reasoning)", "(Reasoning)",
     "(May '24)" that we need to disambiguate deliberately).
  2. For reasoning models we pick the "(Reasoning)" variant where the model is
     reasoning-capable by default (o3, DeepSeek R1, QwQ).
  3. For non-reasoning models we pick the "(Non-reasoning)" variant where AA
     offers both.

Usage:
    uv run python src/comb_eval/scripts/add_mmlu_pro_scores.py
"""

import json
from pathlib import Path


# Map each OpenRouter key in our eval set to the exact Artificial Analysis display
# name. `None` means no MMLU-Pro score exists for this model on AA (we tried).
# Where AA lacks the exact model, we leave it None rather than pick a close cousin.
OPENROUTER_KEY_TO_AA_NAME: dict[str, str | None] = {
    # --- Anthropic ---
    "anthropic_claude-opus-4-6": None,  # AA goes 4.1 Opus -> 4.5 Sonnet; no Opus 4.5/4.6 entry
    "anthropic_claude-opus-4-5": None,  # AA has no Opus 4.5 (only 4.5 Sonnet/Haiku); do not substitute Sonnet for Opus
    "anthropic_claude-sonnet-4-6": None,
    "anthropic_claude-sonnet-4-5": "Claude 4.5 Sonnet (Non-reasoning)",
    "anthropic_claude-sonnet-4": "Claude 4 Sonnet (Non-reasoning)",
    "anthropic_claude-haiku-4-5": "Claude 4.5 Haiku (Non-reasoning)",
    "anthropic_claude-3-5-haiku": None,  # AA may lack this
    "anthropic_claude-3-haiku": None,
    # --- OpenAI ---
    "openai_gpt-5-4": None,  # GPT-5.4 not on AA (they have 5.2)
    "openai_gpt-5-4-mini": None,
    "openai_gpt-5-4-nano": None,
    "openai_gpt-5": "GPT-5 (low)",  # closest to our GPT-5 medium
    "openai_gpt-5-mini": "GPT-5 mini (medium)",
    "openai_gpt-5-nano": "GPT-5 nano (medium)",
    "openai_gpt-4o": "GPT-4o (March 2025, chatgpt-4o-latest)",
    "openai_gpt-4o-mini": None,  # AA may or may not; try fallback
    "openai_gpt-4-1": "GPT-4.1",
    "openai_gpt-4-1-mini": "GPT-4.1 mini",
    "openai_gpt-4-1-nano": "GPT-4.1 nano",
    "openai_gpt-4-turbo": "GPT-4 Turbo",
    "openai_gpt-3-5-turbo": "GPT-3.5 Turbo",
    "openai_o3": "o3",
    "openai_o3-mini": "o3-mini",
    "openai_o4-mini": "o4-mini (high)",
    # --- Google ---
    "google_gemini-2-5-pro": "Gemini 2.5 Pro Preview (May' 25)",
    "google_gemini-2-5-flash": "Gemini 2.5 Flash (Non-reasoning)",
    "google_gemini-2-0-flash-001": "Gemini 2.0 Flash (Feb '25)",
    "google_gemma-3-27b-it": None,  # AA doesn't carry gemma-3-27b in current snapshot
    "google_gemma-2-27b-it": None,
    "google_gemma-2-9b-it": None,
    # --- Meta ---
    "meta-llama_llama-4-maverick": None,  # may not be on AA
    "meta-llama_llama-4-scout": None,
    "meta-llama_llama-3-3-70b-instruct": None,  # AA has 3.3 Nemotron, not plain
    "meta-llama_llama-3-1-70b-instruct": "Llama 3.1 Instruct 70B",
    "meta-llama_llama-3-1-8b-instruct": None,  # not in scraped list
    "meta-llama_llama-3-2-3b-instruct": "Llama 3.2 Instruct 3B",
    # --- DeepSeek ---
    "deepseek_deepseek-r1": "DeepSeek R1 (Jan '25)",
    "deepseek_deepseek-chat-v3-0324": "DeepSeek V3 0324",
    "deepseek_deepseek-chat": "DeepSeek V3.1 (Non-reasoning)",  # closest
    # --- Qwen ---
    "qwen_qwen3-235b-a22b": "Qwen3 235B A22B (Non-reasoning)",
    "qwen_qwen-2-5-72b-instruct": None,  # AA has Qwen2.5 32B/Max, not 72B
    "qwen_qwen3-32b": "Qwen3 32B (Non-reasoning)",
    "qwen_qwen3-14b": "Qwen3 14B (Non-reasoning)",
    "qwen_qwen3-8b": "Qwen3 8B (Non-reasoning)",
    "qwen_qwq-32b": "QwQ 32B",
    # --- Mistral ---
    "mistralai_mistral-large-2407": "Mistral Large 2 (Jul '24)",
    "mistralai_mistral-large-2411": "Mistral Large 2 (Jul '24)",  # same family, no Nov '24 AA entry
    "mistralai_mistral-nemo": None,
    "mistralai_mistral-7b-instruct-v0-1": "Mistral 7B Instruct",
    "mistralai_mistral-small-24b-instruct-2501": "Mistral Small 3",
    # --- Cohere ---
    "cohere_command-a": "Command A",
    "cohere_command-r-plus-08-2024": "Command-R+ (Apr '24)",
    # --- Other ---
    "microsoft_phi-4": "Phi-4",
    "nvidia_llama-3-1-nemotron-70b-instruct": "Llama 3.1 Nemotron Instruct 70B",
}


def main():
    raw_path = Path("data/dat_eval/mmlu_pro_raw.json")
    with open(raw_path) as f:
        aa_scores = json.load(f)

    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"Artificial Analysis MMLU-Pro entries: {len(aa_scores)}")
    print(f"Our benchmark entries: {len(benchmarks)}")
    print()

    matched = 0
    unmatched = []

    for or_key, bench_entry in benchmarks.items():
        aa_name = OPENROUTER_KEY_TO_AA_NAME.get(or_key)
        if aa_name is None:
            unmatched.append(or_key)
            continue
        if aa_name not in aa_scores:
            print(f"  {or_key} (wanted='{aa_name}'): NOT FOUND in AA scores")
            unmatched.append(or_key)
            continue
        bench_entry["mmlu_pro"] = aa_scores[aa_name]
        print(f"  {or_key} -> {aa_name}  (mmlu_pro={aa_scores[aa_name]:.3f})")
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
