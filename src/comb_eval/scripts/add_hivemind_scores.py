"""Add Hivemind paper (arXiv:2510.22954) intra-model similarity scores to benchmarks.

The Hivemind paper reports intra-model similarity: for each model, how similar
its own responses are across multiple samples of the same prompt. Higher values
mean more homogeneous / less diverse output; lower values mean more diverse.

Adds two keys to benchmarks.json:
    hivemind_intra_sim:         weighted mean cosine similarity (0-1)
                                lower = more diverse, plausibly more "creative"
    hivemind_pct_sim_above_0.8: fraction of queries where mean similarity > 0.8

Usage:
    uv run python src/comb_eval/scripts/add_hivemind_scores.py
"""

import json
from pathlib import Path


# Map our OpenRouter keys to Hivemind model name substrings.
# Hivemind uses date-stamped names like `gpt-4o-2024-11-20`.
OPENROUTER_KEY_TO_HIVEMIND = {
    # Anthropic (Hivemind was written before Claude 4.x, so no Sonnet 4/Opus 4 coverage)
    "anthropic_claude-3-haiku": "claude-3-haiku-20240307",
    "anthropic_claude-3-5-haiku": "claude-3-5-haiku-20241022",
    # OpenAI
    "openai_gpt-4o": "gpt-4o-2024-11-20",
    "openai_gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "openai_gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "openai_gpt-3-5-turbo": "gpt-3.5-turbo",
    "openai_gpt-4-1": "gpt-4.1",
    "openai_gpt-4-1-mini": "gpt-4.1-mini",
    "openai_gpt-4-1-nano": "gpt-4.1-nano",
    "openai_o3-mini": "o3-mini",
    "openai_o4-mini": "o4-mini",
    # DeepSeek
    "deepseek_deepseek-chat-v3-0324": "deepseek-v3",
    "deepseek_deepseek-chat": "deepseek-v3",
    # Google (Hivemind has gemini-1.5/2.0, not 2.5)
    "google_gemma-2-27b-it": "gemma-2-27b-it",
    "google_gemma-2-9b-it": "gemma-2-9b-it",
    "google_gemini-2-0-flash-001": "gemini-2.0-flash",
    # Meta
    "meta-llama_llama-3-1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama_llama-3-1-8b-instruct": "llama-3.1-8b-instruct",
    "meta-llama_llama-3-3-70b-instruct": "llama-3.3-70b-instruct",
    "meta-llama_llama-3-2-1b-instruct": "llama-3.2-1b-instruct",
    "meta-llama_llama-3-2-3b-instruct": "llama-3.2-3b-instruct",
    # Qwen
    "qwen_qwen-2-5-72b-instruct": "qwen2.5-72b-instruct",
    "qwen_qwen3-32b": "qwen3-32b",
    "qwen_qwen3-14b": "qwen3-14b",
    "qwen_qwen3-8b": "qwen3-8b",
    # QwQ omitted — not in Hivemind data
    # Mistral
    "mistralai_mistral-nemo": "Mistral-Nemo",
    "mistralai_mistral-7b-instruct-v0-1": "mistral-7b-instruct-v0.1",
    "mistralai_mistral-small-24b-instruct-2501": "mistral-small-24b-instruct-2501",
    # Mistral-Large-2407 not in Hivemind (only -2411)
    "mistralai_mistral-large-2411": "mistral-large-instruct-2411",
    # NVIDIA
    "nvidia_llama-3-1-nemotron-70b-instruct": "Llama-3.1-Nemotron-70B",
    # Microsoft
    "microsoft_phi-4": "phi-4",
    # Cohere
    "cohere_command-r-plus-08-2024": "command-r-plus-08-2024",
}


def match_model(search: str, hivemind_models: list[dict]) -> dict | None:
    """Find a Hivemind entry whose model name contains `search` (case-insensitive)."""
    import re
    search_lower = search.lower()

    # Exact
    for m in hivemind_models:
        if m["model"].lower() == search_lower:
            return m

    # Ends with
    for m in hivemind_models:
        name = m["model"].lower()
        if name.endswith(search_lower):
            return m

    # Word-boundary contains
    pat = re.compile(re.escape(search_lower) + r"(?:$|[^0-9.])")
    for m in hivemind_models:
        if pat.search(m["model"].lower()):
            return m

    # Plain substring fallback
    for m in hivemind_models:
        if search_lower in m["model"].lower():
            return m
    return None


def main():
    hivemind_path = Path("/tmp/hivemind_repetition.json")
    if not hivemind_path.exists():
        raise FileNotFoundError(f"Hivemind data not found at {hivemind_path}")
    with open(hivemind_path) as f:
        hivemind = json.load(f)
    hivemind_models = hivemind["models"]

    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"Hivemind models: {len(hivemind_models)}")
    print(f"Our models in benchmarks.json: {len(benchmarks)}")
    print()

    matched = 0
    unmatched = []

    # Iterate over *all* mapped OpenRouter keys, not just existing benchmark
    # entries — this lets us add Hivemind-only models that weren't in the
    # original arena score fetch.
    all_keys = set(benchmarks.keys()) | set(OPENROUTER_KEY_TO_HIVEMIND.keys())

    for or_key in sorted(all_keys):
        search = OPENROUTER_KEY_TO_HIVEMIND.get(or_key)
        if search is None:
            unmatched.append((or_key, "no mapping"))
            continue

        match = match_model(search, hivemind_models)
        if match is None:
            unmatched.append((or_key, f"search='{search}' no match"))
            continue

        # Create entry if missing (for Hivemind-only models)
        if or_key not in benchmarks:
            benchmarks[or_key] = {}

        benchmarks[or_key]["hivemind_intra_sim"] = match["intra_model_mean_similarity"]
        benchmarks[or_key]["hivemind_pct_sim_above_0_8"] = match["pct_queries_similarity_above_0.8"]
        print(f"  {or_key} -> {match['model']}  (sim={match['intra_model_mean_similarity']:.3f})")
        matched += 1

    with open(bench_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nMatched {matched}/{len(benchmarks)} models to Hivemind data.")
    if unmatched:
        print(f"\nUnmatched ({len(unmatched)}):")
        for m, reason in unmatched:
            print(f"  {m}: {reason}")
    print(f"\nSaved to {bench_path}")


if __name__ == "__main__":
    main()
