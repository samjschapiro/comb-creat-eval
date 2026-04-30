"""Add Hivemind intra-model similarity scores to benchmarks.json.

Source: Table 6 of the Hivemind paper (Jiang et al., NeurIPS 2025;
arXiv:2510.22954, ``Artificial Hivemind: The Open-Ended Homogeneity of
Language Models''). Table 6 reports, for each of 79 models, the
percentage of open-ended-prompt response pairs whose pairwise cosine
similarity falls into each of ten bins covering [0, 1] in 0.1-wide
steps. The paper does not publish a single per-model mean similarity;
we estimate one by weighting each bin's percentage by its midpoint.

For bin percentages $p_b$ over bins $b \in \{[0.9,1.0], [0.8,0.9],
\dots, [0.0,0.1]\}$ with midpoints $m_b \in \{0.95, 0.85, \dots,
0.05\}$:

    intra_model_mean_similarity = sum_b (p_b * m_b) / 100

This is exact when the within-bin distribution is concentrated at the
bin midpoint and approximate (within ±0.05 worst-case) otherwise. The
paper itself does not publish a finer-grained estimate, so this is the
best reproducible mean we can derive from Table 6 alone.

We also compute the share of pairs with similarity at or above 0.8:

    pct_queries_similarity_above_0_8 = (p[0.9-1.0] + p[0.8-0.9]) / 100

Adds two keys per matched model to configs/comb_eval/benchmarks.json:
    hivemind_intra_sim:           mean similarity in [0, 1] (higher = more
                                  homogeneous output)
    hivemind_pct_sim_above_0_8:   fraction of response pairs in the top
                                  two similarity bins

Usage:
    uv run python src/comb_eval/scripts/add_hivemind_scores.py
"""

import json
from pathlib import Path

# Bin percentages from Table 6 of the Hivemind paper, transcribed from
# the paper PDF. Tuple order: (0.9-1.0, 0.8-0.9, 0.7-0.8, 0.6-0.7,
# 0.5-0.6, 0.4-0.5, 0.3-0.4, 0.2-0.3, 0.1-0.2, 0.0-0.1). Each row sums
# to ~100 (paper rounds to integer percentages, so a row may sum to
# 99 or 101).
HIVEMIND_TABLE6: dict[str, tuple[float, ...]] = {
    "gpt-4o-2024-11-20":                             (51.0, 36.0, 10.0,  1.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "gpt-4o-2024-08-06":                             (44.0, 37.0, 12.0,  4.0,  1.0,  0.0,  2.0,  0.0,  0.0,  0.0),
    "gpt-4o-2024-05-13":                             (40.0, 36.0, 16.0,  5.0,  1.0,  1.0,  1.0,  0.0,  0.0,  0.0),
    "gpt-4o-mini-2024-07-18":                        (53.0, 34.0,  9.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "gpt-4-turbo-2024-04-09":                        (38.0, 44.0, 11.0,  5.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0),
    "claude-3-5-sonnet-20241022":                    (61.0, 22.0,  9.0,  5.0,  2.0,  1.0,  0.0,  0.0,  0.0,  0.0),
    "claude-3-5-haiku-20241022":                     (56.0, 33.0,  7.0,  3.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "claude-3-sonnet-20240229":                      (48.0, 36.0, 10.0,  5.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "claude-3-haiku-20240307":                       (48.0, 33.0, 15.0,  2.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0),
    "claude-3-opus-20240229":                        (59.0, 27.0,  7.0,  3.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "deepseek-ai/DeepSeek-V3":                       (42.0, 39.0, 13.0,  3.0,  0.0,  3.0,  0.0,  0.0,  0.0,  0.0),
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": (43.0, 38.0, 11.0,  3.0,  4.0,  0.0,  0.0,  1.0,  0.0,  0.0),
    "meta-llama/Llama-3.1-8B-Instruct":              (19.0, 52.0, 14.0,  5.0,  6.0,  1.0,  2.0,  1.0,  0.0,  0.0),
    "meta-llama/Llama-3.1-70B-Instruct":             (23.0, 44.0, 23.0,  5.0,  2.0,  2.0,  0.0,  1.0,  0.0,  0.0),
    "meta-llama/Llama-3.2-1B-Instruct":              ( 5.0, 34.0, 38.0,  8.0,  5.0,  6.0,  1.0,  3.0,  0.0,  0.0),
    "meta-llama/Llama-3.2-3B-Instruct":              (20.0, 44.0, 20.0,  5.0,  5.0,  2.0,  2.0,  2.0,  0.0,  0.0),
    "meta-llama/Llama-3.3-70B-Instruct":             (51.0, 30.0, 12.0,  4.0,  1.0,  1.0,  1.0,  0.0,  0.0,  0.0),
    "google/gemma-2-2b-it":                          (19.0, 46.0, 19.0,  9.0,  3.0,  2.0,  2.0,  0.0,  0.0,  0.0),
    "google/gemma-2-9b-it":                          (30.0, 41.0, 19.0,  6.0,  0.0,  2.0,  2.0,  0.0,  0.0,  0.0),
    "google/gemma-2-27b-it":                         (33.0, 43.0, 14.0,  7.0,  0.0,  2.0,  1.0,  0.0,  0.0,  0.0),
    "google/gemma-1.1-2b-it":                        (17.0, 39.0, 30.0,  8.0,  3.0,  2.0,  1.0,  0.0,  0.0,  0.0),
    "google/gemma-1.1-7b-it":                        (22.0, 45.0, 18.0, 12.0,  2.0,  1.0,  0.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen2.5-0.5B-Instruct":                    ( 1.0, 23.0, 24.0, 27.0,  7.0, 13.0,  2.0,  3.0,  0.0,  0.0),
    "Qwen/Qwen2.5-1.5B-Instruct":                    ( 3.0, 39.0, 20.0, 16.0, 11.0,  8.0,  2.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen2.5-3B-Instruct":                      (14.0, 41.0, 26.0, 12.0,  3.0,  2.0,  1.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen2.5-7B-Instruct":                      (31.0, 35.0, 23.0,  5.0,  4.0,  1.0,  0.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen2.5-14B-Instruct":                     (34.0, 35.0, 22.0,  5.0,  2.0,  1.0,  0.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen2.5-32B-Instruct":                     (36.0, 44.0, 16.0,  2.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen2.5-72B-Instruct":                     (48.0, 26.0, 19.0,  4.0,  1.0,  1.0,  1.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen2.5-7B-Instruct-1M":                   (31.0, 36.0, 16.0, 12.0,  3.0,  0.0,  2.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen2.5-14B-Instruct-1M":                  (43.0, 27.0, 16.0,  9.0,  3.0,  1.0,  0.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen2-0.5B-Instruct":                      ( 0.0, 20.0, 21.0, 23.0, 20.0,  9.0,  4.0,  2.0,  1.0,  0.0),
    "Qwen/Qwen2-1.5B-Instruct":                      ( 2.0, 32.0, 24.0, 22.0, 11.0,  3.0,  4.0,  2.0,  0.0,  0.0),
    "Qwen/Qwen2-72B-Instruct":                       (35.0, 35.0, 24.0,  5.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen1.5-0.5B-Chat":                        ( 0.0, 23.0, 27.0, 24.0, 12.0,  6.0,  6.0,  2.0,  0.0,  0.0),
    "Qwen/Qwen1.5-1.8B-Chat":                        (12.0, 41.0, 27.0,  9.0,  4.0,  3.0,  3.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen1.5-4B-Chat":                          ( 6.0, 46.0, 24.0, 14.0,  4.0,  2.0,  3.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen1.5-7B-Chat":                          (26.0, 40.0, 16.0, 10.0,  4.0,  2.0,  1.0,  1.0,  0.0,  0.0),
    "Qwen/Qwen1.5-14B-Chat":                         (37.0, 33.0, 22.0,  3.0,  5.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen1.5-32B-Chat":                         (37.0, 37.0, 17.0,  7.0,  1.0,  1.0,  0.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen1.5-72B-Chat":                         (50.0, 35.0,  9.0,  3.0,  1.0,  1.0,  1.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen1.5-110B-Chat":                        (48.0, 36.0, 10.0,  2.0,  3.0,  0.0,  1.0,  0.0,  0.0,  0.0),
    "mistralai/Mistral-Small-24B-Instruct-2501":     (31.0, 39.0, 17.0,  4.0,  4.0,  2.0,  2.0,  1.0,  0.0,  0.0),
    "mistralai/Mistral-7B-Instruct-v0.1":            (24.0, 42.0, 23.0,  5.0,  1.0,  2.0,  2.0,  1.0,  0.0,  0.0),
    "mistralai/Mistral-7B-Instruct-v0.2":            (38.0, 40.0, 13.0,  4.0,  3.0,  1.0,  1.0,  0.0,  0.0,  0.0),
    "mistralai/Mistral-7B-Instruct-v0.3":            (30.0, 39.0, 18.0,  8.0,  3.0,  1.0,  1.0,  0.0,  0.0,  0.0),
    "mistralai/Ministral-8B-Instruct-2410":          (10.0, 38.0, 25.0, 14.0,  7.0,  2.0,  3.0,  1.0,  0.0,  0.0),
    "mistralai/Mistral-Nemo-Instruct-2407":          (19.0, 42.0, 22.0,  9.0,  3.0,  1.0,  4.0,  0.0,  0.0,  0.0),
    "mistralai/Mistral-Small-Instruct-2409":         (23.0, 42.0, 20.0,  6.0,  6.0,  3.0,  0.0,  0.0,  0.0,  0.0),
    "mistralai/Mistral-Large-Instruct-2411":         (43.0, 34.0, 15.0,  5.0,  2.0,  0.0,  1.0,  0.0,  0.0,  0.0),
    "mistralai/Mixtral-8x7B-Instruct-v0.1":          (45.0, 41.0,  9.0,  4.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "microsoft/phi-4":                               (38.0, 40.0, 11.0,  9.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "microsoft/Phi-3.5-mini-instruct":               (33.0, 42.0, 12.0,  8.0,  2.0,  2.0,  1.0,  0.0,  0.0,  0.0),
    "microsoft/Phi-3-mini-128k-instruct":            (15.0, 40.0, 22.0, 13.0,  5.0,  2.0,  2.0,  1.0,  0.0,  0.0),
    "o1-2024-12-17":                                 (27.0, 37.0, 20.0,  9.0,  4.0,  0.0,  3.0,  0.0,  0.0,  0.0),
    "o1-mini-2024-09-12":                            (38.0, 40.0, 17.0,  4.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "o1-preview-2024-09-12":                         (40.0, 33.0, 16.0,  8.0,  1.0,  2.0,  0.0,  0.0,  0.0,  0.0),
    "o3-mini-2025-01-31":                            (34.0, 35.0, 18.0, 10.0,  0.0,  2.0,  1.0,  0.0,  0.0,  0.0),
    "CohereForAI/aya-expanse-8b":                    (40.0, 44.0,  7.0,  7.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0),
    "CohereForAI/aya-expanse-32b":                   (50.0, 36.0,  8.0,  5.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "CohereForAI/c4ai-command-r-plus-08-2024":       (24.0, 32.0, 19.0, 10.0,  7.0,  3.0,  3.0,  1.0,  1.0,  0.0),
    "CohereForAI/c4ai-command-r-08-2024":            (27.0, 35.0, 14.0, 16.0,  2.0,  4.0,  1.0,  1.0,  0.0,  0.0),
    "allenai/OLMo-2-1124-13B-Instruct":              (29.0, 39.0, 12.0,  9.0,  7.0,  2.0,  1.0,  1.0,  0.0,  0.0),
    "allenai/OLMo-2-1124-7B-Instruct":               (30.0, 38.0, 15.0, 10.0,  3.0,  3.0,  1.0,  0.0,  0.0,  0.0),
    "allenai/Llama-3.1-Tulu-3-8B":                   (27.0, 36.0, 22.0,  8.0,  3.0,  1.0,  2.0,  0.0,  1.0,  0.0),
    "allenai/Llama-3.1-Tulu-3-70B":                  (28.0, 39.0, 18.0, 11.0,  1.0,  2.0,  0.0,  1.0,  0.0,  0.0),
    "qwen-max-2025-01-25":                           (55.0, 37.0,  4.0,  3.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "qwen-plus-2025-01-25":                          (56.0, 28.0, 11.0,  3.0,  2.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "qwen-turbo-2024-11-01":                         (37.0, 33.0, 20.0,  8.0,  1.0,  1.0,  0.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen3-0.6B":                               (13.0, 43.0, 27.0,  8.0,  6.0,  1.0,  2.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen3-1.7B":                               (44.0, 30.0, 14.0, 11.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen3-4B":                                 (51.0, 37.0,  4.0,  4.0,  2.0,  2.0,  0.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen3-8B":                                 (56.0, 26.0, 15.0,  1.0,  1.0,  0.0,  1.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen3-14B":                                (52.0, 31.0, 10.0,  4.0,  0.0,  1.0,  2.0,  0.0,  0.0,  0.0),
    "Qwen/Qwen3-32B":                                (40.0, 36.0, 14.0,  6.0,  1.0,  1.0,  2.0,  0.0,  0.0,  0.0),
    "gemini-1.5-flash":                              (62.0, 26.0,  5.0,  6.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "gemini-1.5-pro":                                (53.0, 32.0, 11.0,  3.0,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0),
    "gemini-2.0-flash":                              (40.0, 41.0, 11.0,  4.0,  2.0,  0.0,  2.0,  0.0,  0.0,  0.0),
    "gemini-2.0-flash-lite-preview-02-05":           (40.0, 41.0,  8.0,  6.0,  3.0,  1.0,  1.0,  0.0,  0.0,  0.0),
}

# Map our OpenRouter filesystem keys to Hivemind model names. Models in
# our pool that don't appear in Table 6 (e.g., Claude 4.x, GPT-5,
# QwQ-32b) are simply absent from this dict.
OPENROUTER_KEY_TO_HIVEMIND: dict[str, str] = {
    # Anthropic
    "anthropic_claude-3-haiku":                "claude-3-haiku-20240307",
    "anthropic_claude-3-5-haiku":              "claude-3-5-haiku-20241022",
    # OpenAI
    "openai_gpt-4o":                           "gpt-4o-2024-11-20",
    "openai_gpt-4o-mini":                      "gpt-4o-mini-2024-07-18",
    "openai_gpt-4-turbo":                      "gpt-4-turbo-2024-04-09",
    "openai_o3-mini":                          "o3-mini-2025-01-31",
    # DeepSeek (V3 corresponds to deepseek-chat-v3-0324 on OpenRouter)
    "deepseek_deepseek-chat-v3-0324":          "deepseek-ai/DeepSeek-V3",
    "deepseek_deepseek-chat":                  "deepseek-ai/DeepSeek-V3",
    # Google
    "google_gemma-2-9b-it":                    "google/gemma-2-9b-it",
    "google_gemma-2-27b-it":                   "google/gemma-2-27b-it",
    "google_gemini-2-0-flash-001":             "gemini-2.0-flash",
    # Meta
    "meta-llama_llama-3-1-70b-instruct":       "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama_llama-3-1-8b-instruct":        "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama_llama-3-2-3b-instruct":        "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama_llama-3-3-70b-instruct":       "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen
    "qwen_qwen-2-5-72b-instruct":              "Qwen/Qwen2.5-72B-Instruct",
    "qwen_qwen3-32b":                          "Qwen/Qwen3-32B",
    "qwen_qwen3-14b":                          "Qwen/Qwen3-14B",
    "qwen_qwen3-8b":                           "Qwen/Qwen3-8B",
    # Mistral
    "mistralai_mistral-7b-instruct-v0-1":      "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai_mistral-nemo":                  "mistralai/Mistral-Nemo-Instruct-2407",
    "mistralai_mistral-small-24b-instruct-2501":"mistralai/Mistral-Small-24B-Instruct-2501",
    "mistralai_mistral-large-2411":            "mistralai/Mistral-Large-Instruct-2411",
    # Microsoft
    "microsoft_phi-4":                         "microsoft/phi-4",
    # Cohere
    "cohere_command-r-plus-08-2024":           "CohereForAI/c4ai-command-r-plus-08-2024",
    "cohere_command-r-08-2024":                "CohereForAI/c4ai-command-r-08-2024",
}


# Bin midpoints for the ten bins, in the order Table 6 lists them.
BIN_MIDPOINTS: tuple[float, ...] = (
    0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05,
)


def midpoint_mean(percentages: tuple[float, ...]) -> float:
    """Bin-midpoint-weighted mean similarity (in [0, 1]).

    Given the ten bin percentages from Table 6, returns
        sum_b (p_b * m_b) / 100
    where m_b are the midpoints (0.95, 0.85, ..., 0.05).
    """
    assert len(percentages) == len(BIN_MIDPOINTS)
    return sum(p * m for p, m in zip(percentages, BIN_MIDPOINTS)) / 100.0


def pct_above_0_8(percentages: tuple[float, ...]) -> float:
    """Fraction of pairs in the [0.8, 1.0] range (top two bins)."""
    return (percentages[0] + percentages[1]) / 100.0


def main():
    bench_path = Path("configs/comb_eval/benchmarks.json")
    with open(bench_path) as f:
        benchmarks = json.load(f)

    print(f"Hivemind models in Table 6: {len(HIVEMIND_TABLE6)}")
    print(f"Models in our benchmarks.json: {len(benchmarks)}")
    print()

    matched = 0
    unmatched = []

    for or_key, hv_name in OPENROUTER_KEY_TO_HIVEMIND.items():
        if hv_name not in HIVEMIND_TABLE6:
            unmatched.append((or_key, f"Hivemind name {hv_name!r} not in Table 6"))
            continue
        pcts = HIVEMIND_TABLE6[hv_name]
        if or_key not in benchmarks:
            benchmarks[or_key] = {}
        benchmarks[or_key]["hivemind_intra_sim"] = round(midpoint_mean(pcts), 4)
        benchmarks[or_key]["hivemind_pct_sim_above_0_8"] = round(pct_above_0_8(pcts), 4)
        print(f"  {or_key} <- {hv_name}  sim={benchmarks[or_key]['hivemind_intra_sim']:.4f}")
        matched += 1

    with open(bench_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nMatched {matched}/{len(OPENROUTER_KEY_TO_HIVEMIND)} mappings.")
    if unmatched:
        print(f"Unmatched ({len(unmatched)}):")
        for k, why in unmatched:
            print(f"  - {k}: {why}")
    print(f"\nSaved to {bench_path}")


if __name__ == "__main__":
    main()
