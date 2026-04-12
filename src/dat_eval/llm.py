"""LLM calling infrastructure for DAT/CDAT/PACE evaluation.

Thin wrapper around OpenRouter via the OpenAI SDK. Shared across all three
evaluation methods in this track.
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_client() -> OpenAI:
    """Create an OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("FATAL: OPENROUTER_API_KEY not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Call an LLM via OpenRouter. Returns raw response text."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def extract_words_from_response(raw: str, expected_count: int = 10) -> list[str]:
    """Extract a list of words from an LLM response.

    Handles various response formats:
    - JSON arrays: ["word1", "word2", ...]
    - Numbered lists: 1. word1\n2. word2\n...
    - Comma-separated: word1, word2, word3
    - One per line: word1\nword2\n...

    Returns:
        List of extracted words (lowercased, stripped).
    """
    raw = raw.strip()

    # Try JSON array first
    try:
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return [w.strip().lower() for w in data if w.strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try JSON object with a list field
    try:
        json_match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for v in data.values():
                if isinstance(v, list) and len(v) >= expected_count - 3:
                    return [str(w).strip().lower() for w in v if str(w).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try numbered list: "1. word" or "1) word"
    numbered = re.findall(r"^\s*\d+[\.\)]\s*(.+)$", raw, re.MULTILINE)
    if len(numbered) >= expected_count - 3:
        # Take just the first word from each line (in case there's explanation)
        words = []
        for line in numbered:
            # Remove anything after a dash, colon, or parenthetical
            word = re.split(r"[\-–—:(\[]", line)[0].strip()
            # Take first word only if multi-word
            word = word.split()[0] if word.split() else ""
            if word:
                words.append(word.strip(".,;\"'").lower())
        if words:
            return words

    # Try comma-separated
    if "," in raw:
        parts = raw.split(",")
        words = [p.strip().strip(".,;\"'[]").lower() for p in parts]
        words = [w.split()[0] if w.split() else "" for w in words]
        words = [w for w in words if w and w.isalpha()]
        if len(words) >= expected_count - 3:
            return words

    # Fallback: one word per line
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    words = []
    for line in lines:
        word = line.strip(".,;\"'0123456789.) ").lower()
        if word and " " not in word and word.isalpha():
            words.append(word)

    return words


def model_id_to_key(model_id: str) -> str:
    """Convert OpenRouter model ID to filesystem-safe key."""
    return model_id.replace("/", "_").replace(".", "-")
