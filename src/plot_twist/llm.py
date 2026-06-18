"""LLM calling infrastructure for the plot_twist track.

Self-contained thin wrapper around OpenRouter via the OpenAI SDK (per
repo_usage.md: tracks do not import each other's code). Mirrors the shape of
src/dat_eval/llm.py so behaviour is consistent across tracks.
"""

import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


def get_async_client_openrouter() -> AsyncOpenAI:
    """Async client pinned to OpenRouter. Judges must hit a hosted model even
    when a local server is serving the model under test, so we do not honour
    LLM_BASE_URL here."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("FATAL: OPENROUTER_API_KEY not set")
    return AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


async def call_llm_async(
    async_client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int | None = None,
    reasoning: dict | None = None,
) -> str:
    """Single chat completion. Caller supplies the client so many concurrent
    calls share one connection pool.

    `reasoning` (optional) is forwarded verbatim to OpenRouter's unified reasoning
    API, e.g. {"effort": "high"} or {"enabled": true} to think, {"enabled": false}
    or {"max_tokens": 0} to disable thinking. Used for the thinking-vs-non-thinking
    ablation; None leaves the model at its provider default."""
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if seed is not None:
        kwargs["seed"] = seed
    if reasoning is not None:
        kwargs["extra_body"] = {"reasoning": reasoning}
    response = await async_client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


async def call_llm_async_full(
    async_client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    seed: int | None = None,
    reasoning: dict | None = None,
) -> dict:
    """Like call_llm_async but returns the content AND the reasoning trace + token
    counts, for the thinking-intervention experiment (Exp 1).

    OpenRouter returns the chain-of-thought on `message.reasoning` (a string) and a
    structured `message.reasoning_details` when reasoning is enabled and not excluded;
    reasoning-token usage appears under `usage.completion_tokens_details.reasoning_tokens`.
    Providers differ, so every field is read defensively and may be None.

    Returns {content, reasoning, reasoning_details, reasoning_tokens, completion_tokens}."""
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed
    if reasoning is not None:
        # exclude=False ensures the trace is RETURNED (not just billed) so we can keep it.
        kwargs["extra_body"] = {"reasoning": {"exclude": False, **reasoning}}
    response = await async_client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    usage = getattr(response, "usage", None)
    ctd = getattr(usage, "completion_tokens_details", None) if usage else None
    return {
        "content": msg.content,
        "reasoning": getattr(msg, "reasoning", None),
        "reasoning_details": getattr(msg, "reasoning_details", None),
        "reasoning_tokens": getattr(ctd, "reasoning_tokens", None) if ctd else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
    }


def model_id_to_key(model_id: str) -> str:
    """OpenRouter model ID -> filesystem-safe key."""
    return model_id.replace("/", "_").replace(".", "-")
