"""Add EQ-Bench Creative Writing scores to benchmarks.json.

Pulls from /tmp/eqbench_cw.json (fetched externally) and matches models to the
OpenRouter IDs used in our eval. Writes updated benchmarks.json with:
    eq_bench_cw:           ELO score (primary leaderboard rank)
    eq_bench_cw_rubric:    Absolute rubric score (0-20 scale)

Usage:
    uv run python src/comb_eval/scripts/add_eqbench_scores.py
"""

import json
from pathlib import Path


# Map our OpenRouter model IDs (as they appear in benchmarks.json keys, filesystem-safe)
# to EQ-Bench model name substrings. EQ-Bench has a single primary name per model;
# some substring-matching is needed because they use slightly different naming.
OPENROUTER_KEY_TO_EQBENCH = {
    # Anthropic
    "anthropic_claude-opus-4-6": "claude-opus-4-6",
    "anthropic_claude-opus-4-5": "claude-opus-4-5",
    "anthropic_claude-sonnet-4-6": "claude-sonnet-4-6",
    "anthropic_claude-sonnet-4-5": "claude-sonnet-4.5",
    "anthropic_claude-sonnet-4": "claude-sonnet-4",
    "anthropic_claude-haiku-4-5": "claude-haiku-4.5",
    "anthropic_claude-3-5-haiku": "claude-3.5-haiku",
    "anthropic_claude-3-haiku": "claude-3-haiku",
    # OpenAI
    "openai_gpt-5-4": "gpt-5.4",
    "openai_gpt-5-4-mini": "gpt-5.4-mini",
    "openai_gpt-5-4-nano": "gpt-5.4-nano",
    "openai_gpt-5": "gpt-5-",
    "openai_gpt-5-mini": "gpt-5-mini",
    "openai_gpt-5-nano": "gpt-5-nano",
    "openai_gpt-4o": "chatgpt-4o-latest",
    "openai_gpt-4o-mini": "gpt-4o-mini",
    "openai_gpt-4-1": "gpt-4.1",
    "openai_gpt-4-1-mini": "gpt-4.1-mini",
    "openai_gpt-4-1-nano": "gpt-4.1-nano",
    "openai_gpt-4-turbo": "gpt-4-turbo",
    "openai_gpt-3-5-turbo": "gpt-3.5-turbo",
    "openai_o3": "o3-2025",
    "openai_o3-mini": "o3-mini",
    "openai_o4-mini": "o4-mini",
    # Google
    "google_gemini-2-5-pro": "gemini-2.5-pro",
    "google_gemini-2-5-flash": "gemini-2.5-flash",
    "google_gemini-2-0-flash-001": "gemini-2.0-flash",
    "google_gemma-3-27b-it": "gemma-3-27b",
    "google_gemma-2-27b-it": "gemma-2-27b",
    "google_gemma-2-9b-it": "gemma-2-9b",
    # Meta
    "meta-llama_llama-4-maverick": "Llama-4-Maverick",
    "meta-llama_llama-4-scout": "Llama-4-Scout",
    "meta-llama_llama-3-3-70b-instruct": "llama-3.3-70b",
    "meta-llama_llama-3-1-70b-instruct": "llama-3.1-70b",
    "meta-llama_llama-3-1-8b-instruct": "llama-3.1-8b",
    # DeepSeek
    "deepseek_deepseek-r1": "deepseek-r1",
    "deepseek_deepseek-chat-v3-0324": "DeepSeek-V3-0324",
    "deepseek_deepseek-chat": "deepseek-chat",
    # Qwen
    "qwen_qwen3-235b-a22b": "Qwen3-235B",
    "qwen_qwen-2-5-72b-instruct": "Qwen2.5-72B",
    "qwen_qwen3-32b": "Qwen3-32B",
    "qwen_qwq-32b": "QwQ-32B",
    # Mistral
    "mistralai_mistral-large-2407": "Mistral-Large",
    "mistralai_mistral-large-2411": "mistral-large-2411",
    "mistralai_mistral-nemo": "Mistral-Nemo",
    # Cohere
    "cohere_command-a": "command-a",
    "cohere_command-r-plus-08-2024": "command-r-plus-08-2024",
    # Other
    "microsoft_phi-4": "phi-4",
    "nvidia_llama-3-1-nemotron-70b-instruct": "Llama-3.1-Nemotron-70B",
}


def match_model(search: str, eqbench_models: list[dict]) -> dict | None:
    """Find an EQ-Bench entry whose name contains `search` (case-insensitive).
    Prefers exact matches, then models that end with `search`, then substrings.
    """
    search_lower = search.lower()

    # Exact
    for m in eqbench_models:
        if m["model"].lower() == search_lower:
            return m

    # Ends with (e.g. search="gpt-4.1" should prefer a model ending in "gpt-4.1"
    # not one that has "gpt-4.1-mini" in the middle)
    for m in eqbench_models:
        name = m["model"].lower()
        if name.endswith(search_lower) or name.endswith(search_lower + "-preview"):
            return m

    # Contains with word-boundary-ish check: search is followed by a non-digit
    # to avoid matching "claude-sonnet-4" inside "claude-sonnet-4-6"
    import re
    pat = re.compile(re.escape(search_lower) + r"(?:$|[^0-9.])")
    for m in eqbench_models:
        if pat.search(m["model"].lower()):
            return m

    # Fallback: plain substring
    for m in eqbench_models:
        if search_lower in m["model"].lower():
            return m

    return None


def main():
    # Load EQ-Bench data
    eqbench_path = Path("/tmp/eqbench_cw.json")
    if not eqbench_path.exists():
        raise FileNotFoundError(
            "EQ-Bench data not found at /tmp/eqbench_cw.json. "
            "Fetch from https://eqbench.com/creative_writing.js first."
        )
    with open(eqbench_path) as f:
        eqbench = json.load(f)
    eqbench_models = eqbench["models"]

    # Load existing benchmarks
    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"EQ-Bench models: {len(eqbench_models)}")
    print(f"Our models in benchmarks.json: {len(benchmarks)}")
    print()

    matched = 0
    unmatched = []

    for or_key, bench_entry in benchmarks.items():
        search = OPENROUTER_KEY_TO_EQBENCH.get(or_key)
        if search is None:
            print(f"  {or_key}: NO MAPPING")
            unmatched.append(or_key)
            continue

        match = match_model(search, eqbench_models)
        if match is None:
            print(f"  {or_key} (search='{search}'): NO MATCH in EQ-Bench")
            unmatched.append(or_key)
            continue

        bench_entry["eq_bench_cw"] = match["elo_score"]
        bench_entry["eq_bench_cw_rubric"] = match["creative_writing_score"]
        print(f"  {or_key} -> {match['model']}  (elo={match['elo_score']}, rubric={match['creative_writing_score']})")
        matched += 1

    # Save
    with open(bench_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nMatched {matched}/{len(benchmarks)} models to EQ-Bench CW.")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for m in unmatched:
            print(f"  - {m}")
    print(f"\nSaved to {bench_path}")


if __name__ == "__main__":
    main()
