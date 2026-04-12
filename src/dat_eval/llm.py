"""LLM calling infrastructure for DAT/CDAT/PACE evaluation.

Thin wrapper around OpenRouter via the OpenAI SDK. Shared across all three
evaluation methods in this track.
"""

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

load_dotenv()


def get_client() -> OpenAI:
    """Create a synchronous OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("FATAL: OPENROUTER_API_KEY not set")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def get_async_client() -> AsyncOpenAI:
    """Create an async OpenRouter client for concurrent requests."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("FATAL: OPENROUTER_API_KEY not set")
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    """Call an LLM via OpenRouter. Returns raw response text.

    Args:
        messages: Chat messages.
        model: OpenRouter model ID.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        seed: Optional seed for reproducibility-per-seed and variance-across-seeds.
            Passing different seeds breaks the model's prior on "first token" behavior
            even when temperature alone produces near-deterministic output.
        top_p: Nucleus sampling threshold. Set to 1.0 to disable (full distribution).
            Provider defaults are often 0.9, which suppresses tail tokens and limits
            diversity even at high temperature.
        top_k: Top-k sampling threshold. Set to 0 to disable (where supported).
            Not all providers/models accept this — passed via extra_body for OpenRouter.
    """
    client = get_client()
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        # top_k is non-standard in OpenAI chat completions; pass via extra_body
        # for OpenRouter to forward to providers that support it (Anthropic, most open models)
        kwargs["extra_body"] = {"top_k": top_k}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


async def call_llm_async(
    async_client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> str:
    """Async version of call_llm. Caller provides the AsyncOpenAI client so
    many concurrent calls can share one connection pool.
    """
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed
    if top_p is not None:
        kwargs["top_p"] = top_p
    if top_k is not None:
        kwargs["extra_body"] = {"top_k": top_k}
    response = await async_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def extract_words_from_response(raw: str | None, expected_count: int = 10) -> list[str]:
    """Extract a list of words from an LLM response.

    Handles various response formats:
    - JSON arrays: ["word1", "word2", ...]
    - Numbered lists: 1. word1\n2. word2\n...
    - Comma-separated: word1, word2, word3
    - One per line: word1\nword2\n...

    Returns:
        List of extracted words (lowercased, stripped). Empty list if raw is None/empty.
    """
    if raw is None:
        return []
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
