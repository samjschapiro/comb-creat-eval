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


def _endpoint() -> tuple[str, str]:
    """(base_url, api_key). Defaults to OpenRouter; override with
    LLM_BASE_URL + LLM_API_KEY to point harnesses at a local vLLM
    OpenAI-compatible server (e.g. http://localhost:8000/v1). Behavior
    is unchanged when those env vars are unset.
    """
    base_url = os.environ.get("LLM_BASE_URL")
    if base_url:
        # Local vLLM accepts any api_key string; default to a placeholder.
        return base_url, os.environ.get("LLM_API_KEY", "EMPTY")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("FATAL: OPENROUTER_API_KEY not set")
    return "https://openrouter.ai/api/v1", api_key


def get_client() -> OpenAI:
    """Synchronous client (OpenRouter, or local vLLM via LLM_BASE_URL)."""
    base_url, api_key = _endpoint()
    return OpenAI(base_url=base_url, api_key=api_key)


def get_async_client() -> AsyncOpenAI:
    """Async client (OpenRouter, or local server via LLM_BASE_URL)."""
    base_url, api_key = _endpoint()
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def get_async_client_openrouter() -> AsyncOpenAI:
    """Always OpenRouter, ignoring LLM_BASE_URL. For calls that must hit a
    hosted model (e.g. a NoveltyBench quality judge) even while a local
    server is serving the model under test."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("FATAL: OPENROUTER_API_KEY not set")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


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
    reasoning: dict | None = None,
    capture_reasoning: bool = False,
) -> str:
    """Async version of call_llm. Caller provides the AsyncOpenAI client so
    many concurrent calls can share one connection pool.

    Args:
        reasoning: Optional dict forwarded to OpenRouter's unified reasoning API.
            Keys: "effort" ("low"/"medium"/"high" for o-series), "max_tokens"
            (hard cap for Anthropic extended thinking), "exclude" (hide reasoning
            in response), "enabled" (false disables reasoning entirely where
            supported). For models like QwQ where reasoning is baked into the
            fine-tune, "effort: low" may still speed things up via provider-side.
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

    extra_body = {}
    if top_k is not None:
        extra_body["top_k"] = top_k
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        response = await async_client.chat.completions.create(**kwargs)
    except Exception as e:
        # Some providers reject the reasoning param (e.g., SiliconFlow for QwQ
        # rejects `enable_thinking`). Retry once without reasoning if that's
        # the likely cause.
        msg = str(e).lower()
        if reasoning is not None and (
            "reasoning" in msg or "enable_thinking" in msg or "thinking" in msg
        ):
            if "reasoning" in extra_body:
                del extra_body["reasoning"]
            if extra_body:
                kwargs["extra_body"] = extra_body
            else:
                kwargs.pop("extra_body", None)
            response = await async_client.chat.completions.create(**kwargs)
        else:
            raise
    # Some providers return an error-shaped 200 with choices=None; treat as
    # empty rather than crashing on the subscript.
    if not response.choices:
        return (None, None) if capture_reasoning else None
    msg = response.choices[0].message
    if capture_reasoning:
        # OpenRouter surfaces the trace as message.reasoning (string) and/or
        # message.reasoning_details (structured); both live in model_extra on the SDK object.
        extra = getattr(msg, "model_extra", None) or {}
        reasoning_trace = (getattr(msg, "reasoning", None) or extra.get("reasoning")
                           or extra.get("reasoning_details"))
        return msg.content, reasoning_trace
    return msg.content


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
