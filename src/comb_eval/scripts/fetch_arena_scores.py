"""Fetch Chatbot Arena scores for benchmark correlation analysis.

Fetches Arena Overall and Arena Creative Writing scores via Playwright
browser automation from arena.ai. The CW category requires clicking a
button in the UI (URL params alone don't switch categories).

Outputs a benchmarks.json file compatible with the analyze.py pipeline.

Usage:
    uv run python src/comb_eval/scripts/fetch_arena_scores.py configs/comb_eval/fetch_arena_scores.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from playwright.sync_api import sync_playwright

from src.utils import load_config, init_directory, save_config


def _extract_table(page) -> dict[str, int]:
    """Extract model scores from the currently visible Arena leaderboard table.

    Returns:
        Dict mapping model display names to Elo scores.
    """
    rows = page.query_selector_all("table tbody tr")
    scores = {}
    for row in rows:
        cells = row.query_selector_all("td")
        if len(cells) < 4:
            continue
        texts = [c.inner_text().strip() for c in cells]
        # Column 2: model name (first line, before org/license on next line)
        model_name = texts[2].split("\n")[0].strip()
        # Column 3: score (first line, before ± CI on next line)
        score_text = texts[3].split("\n")[0].replace("±", "").strip()
        try:
            score = int(score_text)
            scores[model_name] = score
        except ValueError:
            continue
    return scores


def fetch_arena_scores() -> tuple[dict[str, int], dict[str, int]]:
    """Fetch Arena Overall and Creative Writing scores via Playwright.

    Returns:
        (overall_scores, cw_scores) — both dicts mapping model names to Elo.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Fetch Overall
        print("  Loading Overall leaderboard...")
        page.goto("https://arena.ai/leaderboard/text", timeout=60000)
        page.wait_for_timeout(10000)
        overall = _extract_table(page)
        print(f"  Overall: {len(overall)} models")

        # Click Creative Writing button to switch category
        print("  Switching to Creative Writing...")
        cw_button = page.query_selector('button:has-text("Creative Writing")')
        if not cw_button:
            raise RuntimeError("Creative Writing button not found on arena.ai")
        cw_button.click()
        page.wait_for_timeout(5000)
        cw = _extract_table(page)
        print(f"  Creative Writing: {len(cw)} models")

        browser.close()

    return overall, cw


# Mapping from OpenRouter model IDs to Arena display name substrings.
# We do case-insensitive substring matching against Arena names.
OPENROUTER_TO_ARENA = {
    # Anthropic
    "anthropic/claude-opus-4.6": "claude-opus-4-6",
    "anthropic/claude-opus-4.5": "claude-opus-4-5-20251101",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "anthropic/claude-sonnet-4": "claude-sonnet-4-20250514",
    "anthropic/claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "anthropic/claude-3.5-haiku": "claude-3-5-haiku-20241022",
    "anthropic/claude-3-haiku": "claude-3-haiku-20240307",
    # OpenAI
    "openai/gpt-5.4": "gpt-5.4",
    "openai/gpt-5.4-mini": "gpt-5.4-mini-high",
    "openai/gpt-5.4-nano": "gpt-5.4-nano-high",
    "openai/gpt-5": "gpt-5-high",
    "openai/gpt-5-mini": "gpt-5-mini-high",
    "openai/gpt-5-nano": "gpt-5-nano-high",
    "openai/gpt-4o": "chatgpt-4o-latest",
    "openai/gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "openai/gpt-4.1": "gpt-4.1-2025-04-14",
    "openai/gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "openai/gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
    "openai/gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "openai/gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "openai/o3": "o3-2025-04-16",
    "openai/o3-mini": "o3-mini",
    "openai/o4-mini": "o4-mini-2025-04-16",
    # Google
    "google/gemini-2.5-pro": "gemini-2.5-pro",
    "google/gemini-2.5-pro-preview": "gemini-2.5-pro",
    "google/gemini-2.5-flash": "gemini-2.5-flash",
    "google/gemini-2.5-flash-preview": "gemini-2.5-flash",
    "google/gemini-2.0-flash-001": "gemini-2.0-flash-001",
    "google/gemma-3-27b-it": "gemma-3-27b-it",
    "google/gemma-2-27b-it": "gemma-2-27b-it",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    # Meta
    "meta-llama/llama-4-maverick": "llama-4-maverick",
    "meta-llama/llama-4-scout": "llama-4-scout",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    # DeepSeek
    "deepseek/deepseek-r1": "deepseek-r1",
    "deepseek/deepseek-chat-v3-0324": "deepseek-v3-0324",
    "deepseek/deepseek-chat": "deepseek-v3",
    # Qwen
    "qwen/qwen3-235b-a22b": "qwen3-235b-a22b",
    "qwen/qwen-2.5-72b-instruct": "qwen2.5-72b-instruct",
    "qwen/qwen3-32b": "qwen3-32b",
    "qwen/qwq-32b": "qwq-32b",
    # Mistral
    "mistralai/mistral-large-2411": "mistral-large-2411",
    "mistralai/mistral-large-2407": "mistral-large-2407",
    "mistralai/mistral-nemo": "mistral-7b-instruct",
    # Cohere
    "cohere/command-a": "command-a-03-2025",
    "cohere/command-r-plus-08-2024": "command-r-plus-08-2024",
    # Other
    "microsoft/phi-4": "phi-4",
    "nvidia/llama-3.1-nemotron-70b-instruct": "llama-3.1-nemotron-70b-instruct",
}


def match_arena_name(
    target: str,
    arena_scores: dict[str, int],
) -> tuple[str | None, int | None]:
    """Match a target substring against Arena model names (case-insensitive).

    Returns:
        (matched_name, score) or (None, None).
    """
    target_lower = target.lower()
    arena_lower = {k.lower(): k for k in arena_scores}

    # Exact match
    if target_lower in arena_lower:
        real = arena_lower[target_lower]
        return real, arena_scores[real]

    # Substring match
    for arena_key_lower, arena_real in arena_lower.items():
        if target_lower in arena_key_lower:
            return arena_real, arena_scores[arena_real]

    return None, None


def build_benchmarks(
    model_ids: list[str],
    arena_overall: dict[str, int],
    arena_cw: dict[str, int],
) -> dict:
    """Build benchmarks.json by matching OpenRouter model IDs to Arena names."""
    benchmarks = {}

    for model_id in model_ids:
        model_key = model_id.replace("/", "_").replace(".", "-")
        arena_hint = OPENROUTER_TO_ARENA.get(model_id, model_id.split("/")[-1])

        entry = {}

        name, score = match_arena_name(arena_hint, arena_overall)
        if score is not None:
            entry["arena_overall"] = score
            print(f"  {model_id} -> overall: {name} = {score}")
        else:
            print(f"  {model_id} -> overall: NO MATCH (tried '{arena_hint}')")

        name, score = match_arena_name(arena_hint, arena_cw)
        if score is not None:
            entry["arena_cw"] = score
            print(f"  {model_id} -> cw: {name} = {score}")
        else:
            print(f"  {model_id} -> cw: NO MATCH")

        if entry:
            benchmarks[model_key] = entry

    return benchmarks


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    model_ids = config["models"]

    # Fetch scores
    print("Fetching Arena scores via Playwright...")
    arena_overall, arena_cw = fetch_arena_scores()

    # Save raw scores
    with open(output_dir / "arena_overall_raw.json", "w") as f:
        json.dump(arena_overall, f, indent=2)
    with open(output_dir / "arena_cw_raw.json", "w") as f:
        json.dump(arena_cw, f, indent=2)

    # Match and build benchmarks
    print("\nMatching models to Arena names...")
    benchmarks = build_benchmarks(model_ids, arena_overall, arena_cw)

    # Save
    out_path = output_dir / "benchmarks.json"
    with open(out_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    standard_path = Path(config.get("benchmarks_output", "configs/comb_eval/benchmarks.json"))
    with open(standard_path, "w") as f:
        json.dump(benchmarks, f, indent=2)

    print(f"\nSaved benchmarks to {out_path}")
    print(f"Also saved to {standard_path}")
    print(f"Matched {len(benchmarks)}/{len(model_ids)} models")

    missing = [m for m in model_ids if m.replace("/", "_").replace(".", "-") not in benchmarks]
    if missing:
        print(f"\nModels without any Arena scores ({len(missing)}):")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
