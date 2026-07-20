"""cost_tracker.py — estimate API spend from local response files.

Walks data/dat_eval/run_v1 and counts API calls by (model, eval, temp),
then estimates USD spend using hardcoded OpenRouter pricing.

This is an *estimate* — actual charges depend on OpenRouter's per-request
accounting and any provider-side adjustments. For truth, check
https://openrouter.ai/activity

Usage:
    uv run python scripts/safety/cost_tracker.py
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# Rough OpenRouter pricing (input / output per 1M tokens, USD)
# Update periodically from https://openrouter.ai/models
PRICING = {
    # Anthropic
    "anthropic/claude-opus-4.6":        (15.00, 75.00),
    "anthropic/claude-opus-4.5":        (15.00, 75.00),
    "anthropic/claude-opus-4":          (15.00, 75.00),
    "anthropic/claude-sonnet-4.6":      (3.00, 15.00),
    "anthropic/claude-sonnet-4.5":      (3.00, 15.00),
    "anthropic/claude-sonnet-4":        (3.00, 15.00),
    "anthropic/claude-haiku-4.5":       (0.80, 4.00),
    "anthropic/claude-3.5-haiku":       (0.80, 4.00),
    "anthropic/claude-3-haiku":         (0.25, 1.25),
    # OpenAI
    "openai/gpt-5.4":                   (10.00, 30.00),
    "openai/gpt-5.4-mini":              (1.50, 6.00),
    "openai/gpt-5.4-nano":              (0.30, 1.20),
    "openai/gpt-5":                     (5.00, 15.00),
    "openai/gpt-5-mini":                (1.50, 6.00),
    "openai/gpt-5-nano":                (0.30, 1.20),
    "openai/gpt-4o":                    (2.50, 10.00),
    "openai/gpt-4o-mini":               (0.15, 0.60),
    "openai/gpt-oss-120b":              (0.09, 0.45),  # CREATE factuality judge
    "google/gemini-2.5-flash-lite":     (0.10, 0.40),
    "openai/gpt-4.1":                   (2.00, 8.00),
    "openai/gpt-4.1-mini":              (0.40, 1.60),
    "openai/gpt-4.1-nano":              (0.10, 0.40),
    "openai/gpt-4-turbo":               (10.00, 30.00),
    "openai/gpt-3.5-turbo":             (0.50, 1.50),
    "openai/o3":                        (10.00, 40.00),
    "openai/o3-mini":                   (1.10, 4.40),
    "openai/o4-mini":                   (1.10, 4.40),
    # Google
    "google/gemini-2.5-pro":            (1.25, 10.00),
    "google/gemini-2.5-flash":          (0.15, 0.60),
    "google/gemini-2.0-flash-001":      (0.10, 0.40),
    "google/gemma-3-27b-it":            (0.10, 0.20),
    "google/gemma-2-27b-it":            (0.10, 0.20),
    "google/gemma-2-9b-it":             (0.06, 0.06),
    # Meta
    "meta-llama/llama-4-maverick":      (0.20, 0.60),
    "meta-llama/llama-4-scout":         (0.15, 0.40),
    "meta-llama/llama-3.3-70b-instruct":(0.13, 0.40),
    "meta-llama/llama-3.1-70b-instruct":(0.13, 0.40),
    "meta-llama/llama-3.1-8b-instruct": (0.06, 0.06),
    "meta-llama/llama-3.2-1b-instruct": (0.02, 0.02),
    "meta-llama/llama-3.2-3b-instruct": (0.03, 0.03),
    # DeepSeek
    "deepseek/deepseek-r1":             (0.55, 2.19),
    "deepseek/deepseek-chat-v3-0324":   (0.27, 1.10),
    "deepseek/deepseek-chat":           (0.27, 1.10),
    # Qwen
    "qwen/qwen3-235b-a22b":             (0.20, 0.60),
    "qwen/qwen3-32b":                   (0.10, 0.20),
    "qwen/qwen3-14b":                   (0.06, 0.12),
    "qwen/qwen3-8b":                    (0.04, 0.10),
    "qwen/qwq-32b":                     (0.10, 0.20),
    "qwen/qwen-2.5-72b-instruct":       (0.13, 0.40),
    # Mistral
    "mistralai/mistral-large-2407":     (2.00, 6.00),
    "mistralai/mistral-large-2411":     (2.00, 6.00),
    "mistralai/mistral-nemo":           (0.07, 0.07),
    "mistralai/mistral-7b-instruct-v0.1": (0.03, 0.06),
    "mistralai/mistral-small-24b-instruct-2501": (0.10, 0.30),
    # Cohere
    "cohere/command-a":                 (2.50, 10.00),
    "cohere/command-r-plus-08-2024":    (2.50, 10.00),
    "cohere/command-r-08-2024":         (0.15, 0.60),
    "cohere/command-r7b-12-2024":       (0.04, 0.15),
    # NVIDIA
    "nvidia/llama-3.1-nemotron-70b-instruct": (0.13, 0.40),
    # Microsoft
    "microsoft/phi-4":                  (0.07, 0.14),
    # --- NoveltyBench-coverage additions (2026-04-22) ---
    "anthropic/claude-3.5-sonnet":      (3.00, 15.00),
    "google/gemini-pro-1.5":            (1.25, 5.00),
    "google/gemini-2.0-flash-lite-001": (0.075, 0.30),
    "google/gemini-2.0-pro-exp-02-05":  (0.0, 0.0),       # experimental / free
    "google/gemma-2-2b-it":             (0.03, 0.03),
    "meta-llama/llama-3.1-405b-instruct": (3.00, 3.00),
    # --- ARC-AGI-coverage additions (2026-04-23) ---
    "google/gemini-3.1-pro-preview":    (2.00, 12.00),
    "openai/gpt-5.2":                   (1.75, 14.00),
    "google/gemini-3-flash-preview":    (0.50, 3.00),
    "x-ai/grok-4":                      (3.00, 15.00),
    # --- LIB-pool gap-fill additions (2026-05-10) ---
    # Prices conservative (rounded up from OpenRouter listed rates).
    "anthropic/claude-3-opus":          (15.00, 75.00),
    "anthropic/claude-3.7-sonnet":      (3.00, 15.00),
    "anthropic/claude-3.7-sonnet:thinking": (3.00, 15.00),
    "deepseek/deepseek-r1-distill-llama-70b": (0.20, 0.60),
    "deepseek/deepseek-r1-distill-qwen-32b": (0.20, 0.60),
    "qwen/qwen-max":                    (1.60, 6.40),
    "qwen/qwen-2.5-coder-32b-instruct": (0.10, 0.30),
    "qwen/qwen-2.5-7b-instruct":        (0.05, 0.10),
    "amazon/nova-pro-v1":               (0.80, 3.20),
    "amazon/nova-lite-v1":              (0.06, 0.24),
}


def model_key_to_id(key: str) -> str:
    """Convert filesystem model key back to OpenRouter model ID."""
    # e.g. "anthropic_claude-opus-4-6" -> "anthropic/claude-opus-4.6"
    parts = key.split("_", 1)
    if len(parts) != 2:
        return key
    provider, rest = parts
    # Heuristic: the last "-X" before end that is purely digits is probably a version
    # like "4-6" -> "4.6". But it's risky. Just replace all "-N-M" -> ".N.M" for common patterns.
    # Simpler: just return both versions and check.
    return f"{provider}/{rest}"


def estimate_tokens(text: str) -> int:
    """Crude token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def analyze_responses(base_dir: Path) -> dict:
    """Walk the data dir and count tokens per model."""
    stats = defaultdict(lambda: {
        "dat_calls": 0, "dat_in_tokens": 0, "dat_out_tokens": 0,
        "cdat_calls": 0, "cdat_in_tokens": 0, "cdat_out_tokens": 0,
        "pace_calls": 0, "pace_in_tokens": 0, "pace_out_tokens": 0,
    })

    if not base_dir.exists():
        return stats

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model_key = model_dir.name

        # DAT responses (per-temp files)
        for f in model_dir.glob("dat_responses*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                for r in data:
                    if r.get("raw_response") is not None:
                        stats[model_key]["dat_calls"] += 1
                        # DAT prompt ~500 chars input, response varies
                        stats[model_key]["dat_in_tokens"] += 125  # ~500 chars / 4
                        stats[model_key]["dat_out_tokens"] += estimate_tokens(r["raw_response"] or "")
            except Exception:
                pass

        # CDAT responses
        for f in model_dir.glob("cdat_responses*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                for cue, r in data.items():
                    if r.get("raw_response") is not None:
                        stats[model_key]["cdat_calls"] += 1
                        stats[model_key]["cdat_in_tokens"] += 125
                        stats[model_key]["cdat_out_tokens"] += estimate_tokens(r["raw_response"] or "")
            except Exception:
                pass

        # PACE responses
        for f in model_dir.glob("pace_responses*.json"):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                for seed, d in data.items():
                    # Stage 1
                    if d.get("stage1_raw"):
                        stats[model_key]["pace_calls"] += 1
                        stats[model_key]["pace_in_tokens"] += 125
                        stats[model_key]["pace_out_tokens"] += estimate_tokens(d["stage1_raw"])
                    # Stage 2 (up to 3 chains)
                    for c in d.get("chains", []):
                        if c.get("raw_response"):
                            stats[model_key]["pace_calls"] += 1
                            stats[model_key]["pace_in_tokens"] += 200  # longer prompt
                            stats[model_key]["pace_out_tokens"] += estimate_tokens(c["raw_response"])
            except Exception:
                pass

    return stats


def match_pricing(model_key: str) -> tuple[float, float] | None:
    """Try to find pricing for a model key (with filesystem mangling)."""
    # Try direct reconstruction
    # Replace first underscore with /, then convert digit-dash-digit patterns back
    parts = model_key.split("_", 1)
    if len(parts) == 2:
        reconstructed = f"{parts[0]}/{parts[1]}"
        if reconstructed in PRICING:
            return PRICING[reconstructed]
        # Try replacing hyphens between digits with dots: "4-6" -> "4.6"
        import re
        dotted = re.sub(r"(\d)-(\d)", r"\1.\2", reconstructed)
        if dotted in PRICING:
            return PRICING[dotted]

    # Fallback: substring match
    key_lower = model_key.lower()
    for price_key, price in PRICING.items():
        # normalize price_key
        normalized = price_key.lower().replace("/", "_").replace(".", "-")
        if normalized == key_lower:
            return price
        # partial match
        if key_lower in normalized or normalized in key_lower:
            return price
    return None


def main():
    base = Path(__file__).parent.parent.parent / "data" / "dat_eval" / "run_v1"
    stats = analyze_responses(base)

    print("=" * 80)
    print("ESTIMATED API SPEND (from local response files)")
    print("=" * 80)
    print()
    print(f"{'Model':<45} {'Calls':>7} {'In tok':>10} {'Out tok':>10} {'Est $':>8}")
    print("-" * 80)

    total_cost = 0.0
    total_calls = 0
    unmatched = []

    for model_key in sorted(stats.keys()):
        s = stats[model_key]
        calls = s["dat_calls"] + s["cdat_calls"] + s["pace_calls"]
        in_tok = s["dat_in_tokens"] + s["cdat_in_tokens"] + s["pace_in_tokens"]
        out_tok = s["dat_out_tokens"] + s["cdat_out_tokens"] + s["pace_out_tokens"]

        pricing = match_pricing(model_key)
        if pricing is None:
            cost_str = "   ???"
            unmatched.append(model_key)
        else:
            inp_price, out_price = pricing
            cost = (in_tok / 1_000_000) * inp_price + (out_tok / 1_000_000) * out_price
            total_cost += cost
            cost_str = f"${cost:>6.3f}"

        total_calls += calls
        print(f"{model_key:<45} {calls:>7} {in_tok:>10,} {out_tok:>10,} {cost_str}")

    print("-" * 80)
    print(f"{'TOTAL':<45} {total_calls:>7}{'':>21} ${total_cost:>6.2f}")
    print()

    if unmatched:
        print("⚠ Unmatched models (cost not included):")
        for m in unmatched:
            print(f"    {m}")

    print()
    print("NOTE: This is an estimate. Token counts are approximated from text length")
    print("at ~4 chars/token. For actual charges check https://openrouter.ai/activity")


if __name__ == "__main__":
    main()
