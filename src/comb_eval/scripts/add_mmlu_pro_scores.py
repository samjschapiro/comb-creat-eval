"""Add MMLU-Pro scores to benchmarks.json using the TIGER-Lab leaderboard.

Source: results.csv from
    https://huggingface.co/datasets/TIGER-Lab/mmlu_pro_leaderboard_submission
which is the table powering https://huggingface.co/spaces/TIGER-Lab/MMLU-Pro.

We use this rather than artificialanalysis.ai because the TIGER-Lab leaderboard
is methodologically tied to the original MMLU-Pro paper~(Wang et al., 2024) and
covers a much broader pool of models. Where TIGER-Lab evaluates a model
themselves we use that row; otherwise we fall back to a self-reported number
on the same leaderboard. Models without any TIGER-Lab entry get no
``mmlu_pro`` field.

Local snapshot at ``data/dat_eval/mmlu_pro_raw.json`` is keyed by TIGER-Lab
model names verbatim. To refresh the snapshot, re-fetch the CSV.

Usage:
    uv run python src/comb_eval/scripts/add_mmlu_pro_scores.py
"""

import json
from pathlib import Path


# Map our OpenRouter model keys to the TIGER-Lab leaderboard's model-name
# strings. Order within each list = priority (first match wins). ``[]`` means
# no usable TIGER-Lab entry; the script will leave ``mmlu_pro`` unset.
OPENROUTER_KEY_TO_TIGERLAB: dict[str, list[str]] = {
    # --- Anthropic ---
    "anthropic_claude-3-5-sonnet":     ["Claude-3.5-Sonnet (2024-10-22)", "Claude-3.5-Sonnet (2024-06-20)"],
    "anthropic_claude-3-5-haiku":      ["Claude-3-5-Haiku-20241022"],
    "anthropic_claude-3-opus":         ["Claude-3-Opus"],
    "anthropic_claude-3-haiku":        ["Claude-3-Haiku-20240307"],
    "anthropic_claude-sonnet-4":       ["Claude-4-Sonnet"],
    "anthropic_claude-sonnet-4-5":     ["Claude-4.5-Sonnet(Thinking)"],
    "anthropic_claude-sonnet-4-6":     ["Claude-4.6-Sonnet(Thinking)"],
    "anthropic_claude-haiku-4-5":      [],
    "anthropic_claude-opus-4-5":       ["Claude-4.5-Opus(Thinking)"],
    "anthropic_claude-opus-4-6":       ["Claude-4.6-Opus(Thinking)"],
    # --- OpenAI ---
    "openai_gpt-4o":                   ["GPT-4o (2024-08-06)", "GPT-4o (2024-11-20)", "GPT-4o (2024-05-13)"],
    "openai_gpt-4o-mini":              ["GPT-4o-mini"],
    "openai_gpt-4-1":                  ["GPT-4.1"],
    "openai_gpt-4-1-mini":             [],
    "openai_gpt-4-1-nano":             [],
    "openai_gpt-4-turbo":              ["GPT-4-Turbo"],
    "openai_gpt-3-5-turbo":            [],
    "openai_o3":                       ["GPT-o3-high"],
    "openai_o3-mini":                  ["GPT-o3-mini"],
    "openai_o4-mini":                  [],
    "openai_gpt-5":                    ["GPT-5(high)"],
    "openai_gpt-5-mini":               [],
    "openai_gpt-5-nano":               [],
    "openai_gpt-5-4":                  ["GPT-5.4"],
    "openai_gpt-5-4-mini":             [],
    "openai_gpt-5-4-nano":             [],
    # --- Google ---
    "google_gemini-2-5-pro":           ["Gemini-2.5-Pro", "Gemini-2.5-Pro-Exp-03-25"],
    "google_gemini-2-5-flash":         [],
    "google_gemini-2-0-flash-001":     ["Gemini-2.0-Flash", "Gemini-2.0-Flash-exp"],
    "google_gemini-2-0-flash-lite-001":["Gemini-2.0-Flash-Lite"],
    "google_gemini-1-5-pro":           ["Gemini-1.5-Pro-002", "Gemini-1.5-Pro"],
    "google_gemini-2-0-pro":           ["Gemini-2.0-Pro"],
    "google_gemma-3-27b-it":           ["Gemma-3-27B-it"],
    "google_gemma-2-27b-it":           ["Gemma-2-27B-it"],
    "google_gemma-2-9b-it":            ["Gemma-2-9B-it"],
    "google_gemma-2-2b-it":            ["Gemma-2-2B-it"],
    # --- DeepSeek ---
    "deepseek_deepseek-r1":            ["DeepSeek-R1"],
    "deepseek_deepseek-chat-v3-0324":  ["Deepseek-V3-0324"],
    "deepseek_deepseek-chat":          ["Deepseek-V3"],
    # --- Qwen ---
    "qwen_qwen3-235b-a22b":            ["Qwen3-235B-A22B-Instruct-2507", "Qwen3-235B-A22B"],
    "qwen_qwq-32b":                    ["QwQ-32B"],
    "qwen_qwen-2-5-72b-instruct":      ["Qwen2.5-72B"],
    "qwen_qwen3-32b":                  [],
    "qwen_qwen3-14b":                  [],
    "qwen_qwen3-8b":                   [],
    # --- Mistral ---
    "mistralai_mistral-large-2407":    ["Mistral-Large-Instruct-2407"],
    "mistralai_mistral-large-2411":    ["Mistral-Large-Instruct-2411"],
    "mistralai_mistral-nemo":          ["Mistral-Nemo-Instruct-2407"],
    "mistralai_mistral-7b-instruct-v0-1": ["Mistral-7B-Instruct-v0.1"],
    "mistralai_mistral-small-24b-instruct-2501": ["Mistral-Small-instruct"],
    # --- Microsoft ---
    "microsoft_phi-4":                 ["Phi-4"],
    # --- Meta ---
    "meta-llama_llama-3-3-70b-instruct":   ["Llama-3.3-70B-Instruct"],
    "meta-llama_llama-3-1-70b-instruct":   ["Llama-3.1-70B-Instruct"],
    "meta-llama_llama-3-1-8b-instruct":    ["Llama-3.1-8B-Instruct"],
    "meta-llama_llama-3-1-405b-instruct":  ["Llama-3.1-405B-Instruct"],
    "meta-llama_llama-4-maverick":         ["Llama4-Maverick"],
    "meta-llama_llama-4-scout":            ["Llama4-Scout"],
    "meta-llama_llama-3-2-3b-instruct":    ["Llama-3.2-3B"],
    "meta-llama_llama-3-2-1b-instruct":    ["Llama-3.2-1B"],
    # --- Cohere ---
    "cohere_command-a":                [],
    "cohere_command-r-plus-08-2024":   [],
    "cohere_command-r-08-2024":        [],
    "cohere_command-r7b-12-2024":      [],
    # --- NVIDIA ---
    "nvidia_llama-3-1-nemotron-70b-instruct": ["Llama-3.1-Nemotron-70B-Instruct-HF"],
}


def main():
    raw_path = Path("data/dat_eval/mmlu_pro_raw.json")
    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(raw_path) as f:
        tigerlab = json.load(f)
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"TIGER-Lab leaderboard rows: {len(tigerlab)}")
    print(f"Our benchmark entries: {len(benchmarks)}")
    print()

    set_, cleared, missing = 0, 0, []
    for or_key in benchmarks:
        names = OPENROUTER_KEY_TO_TIGERLAB.get(or_key)
        if not names:
            if "mmlu_pro" in benchmarks[or_key]:
                del benchmarks[or_key]["mmlu_pro"]
                cleared += 1
            missing.append(or_key)
            continue
        match = next((n for n in names if n in tigerlab), None)
        if match is None:
            if "mmlu_pro" in benchmarks[or_key]:
                del benchmarks[or_key]["mmlu_pro"]
                cleared += 1
            print(f"  {or_key}: requested {names!r}, none found in TIGER-Lab")
            missing.append(or_key)
            continue
        benchmarks[or_key]["mmlu_pro"] = float(tigerlab[match])
        print(f"  {or_key} -> {match}  (mmlu_pro={tigerlab[match]:.4f})")
        set_ += 1

    with open(bench_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nSet {set_}, cleared {cleared}, no match for {len(missing)}.")
    print(f"Saved to {bench_path}")


if __name__ == "__main__":
    main()
