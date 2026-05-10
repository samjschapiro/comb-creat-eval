"""Generation and embedding clients shared across the new_tests benchmarks.

Generation is via OpenRouter (re-using the existing dat_eval OpenAI-SDK
client). Embeddings are via OpenAI direct (text-embedding-3-small), which
NoveltyBench and Hivemind both rely on for diversity scoring; OpenRouter
does not expose embeddings, so a separate OPENAI_API_KEY env var is
required for any benchmark that calls embed_texts().
"""

from __future__ import annotations

import asyncio
import os
from typing import Sequence

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

# Re-export the OpenRouter clients from dat_eval so callers don't have to
# pick a path. Generation pipelines can import these directly.
from src.dat_eval.llm import (  # noqa: F401
    get_client,
    get_async_client,
    call_llm,
    call_llm_async,
    model_id_to_key,
)

load_dotenv()


def get_openai_embeddings_client() -> OpenAI:
    """Direct OpenAI client for embeddings (OpenRouter does not proxy these)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "FATAL: OPENAI_API_KEY not set. Required for text-embedding-3-small "
            "used by NoveltyBench and Hivemind scoring."
        )
    return OpenAI(api_key=api_key)


def get_openai_embeddings_async_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "FATAL: OPENAI_API_KEY not set. Required for text-embedding-3-small "
            "used by NoveltyBench and Hivemind scoring."
        )
    return AsyncOpenAI(api_key=api_key)


_OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}


def embed_texts(
    texts: Sequence[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 256,
) -> list[list[float]]:
    """Embed a sequence of texts; returns a list of vectors aligned to texts.

    If `model` is one of the known OpenAI embedding models, calls the OpenAI
    API directly (requires OPENAI_API_KEY). Otherwise treats `model` as a
    sentence-transformers identifier (e.g. 'sentence-transformers/all-mpnet-
    base-v2') and runs locally — useful when the only API key available is
    OPENROUTER_API_KEY.

    Note for Hivemind / NoveltyBench: the canonical paper uses OpenAI's
    text-embedding-3-small. Substituting a sentence-transformers model
    changes the absolute similarity scale; relative orderings between
    models are usually preserved within a single embedding choice. The
    output config records the embedding model used.
    """
    if model in _OPENAI_EMBEDDING_MODELS:
        client = get_openai_embeddings_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            resp = client.embeddings.create(model=model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out
    # Local sentence-transformers fallback.
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(model)
    arr = st.encode(list(texts), batch_size=batch_size, show_progress_bar=False)
    return [list(v) for v in arr]


async def generate_many_async(
    model: str,
    prompts: Sequence[str],
    n_per_prompt: int,
    temperature: float,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    concurrency: int = 16,
    system: str | None = None,
) -> list[list[str]]:
    """For each prompt, generate n_per_prompt responses. Returns a list of
    response lists aligned to prompts.

    Uses asyncio.Semaphore to bound concurrency at the call level. Errors are
    raised as soon as they occur — no silent fallback. This matches the
    repo's fail-fast convention.
    """
    client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    async def _one(prompt: str) -> str:
        messages: list[dict] = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        async with sem:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=1,
            )
        return resp.choices[0].message.content or ""

    tasks: list = []
    for p in prompts:
        for _ in range(n_per_prompt):
            tasks.append(_one(p))
    flat = await asyncio.gather(*tasks)
    grouped: list[list[str]] = []
    for i, _ in enumerate(prompts):
        grouped.append(list(flat[i * n_per_prompt : (i + 1) * n_per_prompt]))
    return grouped
