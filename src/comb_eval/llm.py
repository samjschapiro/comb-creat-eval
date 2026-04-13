"""LLM prompting infrastructure for combinatorial creativity evaluation.

Handles formatting graph and query prompts, calling LLM APIs via OpenRouter
(OpenAI-compatible), and parsing responses. All models are accessed through
a single OpenRouter API key.

The eval asks each model to produce k distinct valid paths per prompt.
Intra-response diversity over the valid subset is the creativity signal.
"""

import json
import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI

from src.comb_eval.prompts import EvalPrompt

load_dotenv()


# --- OpenRouter client ---


def get_client() -> OpenAI:
    """Create a synchronous OpenRouter client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "FATAL: OPENROUTER_API_KEY not set. "
            "Set it in .env or as an environment variable."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def get_async_client() -> AsyncOpenAI:
    """Create an async OpenRouter client for concurrent requests."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "FATAL: OPENROUTER_API_KEY not set. "
            "Set it in .env or as an environment variable."
        )
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Call an LLM via OpenRouter synchronously. Returns raw response text."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def call_llm_async(
    async_client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    top_p: float | None = None,
    top_k: int | None = None,
    reasoning: dict | None = None,
) -> str:
    """Async LLM call via OpenRouter. Caller provides the AsyncOpenAI client
    so many concurrent calls share one connection pool.

    Args:
        reasoning: Optional dict forwarded to OpenRouter's unified reasoning API.
            Keys: "effort" ("low"/"medium"/"high"), "max_tokens", "exclude",
            "enabled". We want reasoning models to do minimal thinking on this
            task — the eval is about creative variation, not multi-step deduction.
    """
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if top_p is not None:
        kwargs["top_p"] = top_p

    extra_body: dict = {}
    if top_k is not None:
        extra_body["top_k"] = top_k
    if reasoning is not None:
        extra_body["reasoning"] = reasoning
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        response = await async_client.chat.completions.create(**kwargs)
    except Exception as e:
        # Some providers reject the reasoning param. Retry once without it.
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
    return response.choices[0].message.content


def model_id_to_key(model_id: str) -> str:
    """Convert an OpenRouter model ID to a filesystem-safe key.

    E.g., 'openai/gpt-4o' -> 'openai_gpt-4o'
    """
    return model_id.replace("/", "_").replace(".", "-")


# --- Prompt formatting ---


def format_graph_prompt(adjacency_text: str) -> str:
    """Format the graph memorization prompt."""
    return (
        "Below is a graph structure. Each line shows a node and its neighbors, "
        "with edge labels in parentheses. For example, 'ABC: DEF(a), GHI(b)' means "
        "node ABC connects to DEF via edge labeled 'a' and to GHI via edge labeled 'b'.\n"
        "\n"
        "Study this graph carefully.\n"
        "\n"
        f"{adjacency_text}"
    )


def format_query_prompt(prompt: EvalPrompt, k: int) -> str:
    """Format a k-distinct-paths query for the LLM.

    Asks the model to find k distinct valid paths satisfying the given
    constraints, returning structured output.
    """
    parts = [
        f"Find {k} DISTINCT paths of exactly {prompt.hop_count} hops "
        f"from {prompt.start} to {prompt.end} in the graph above."
    ]

    parts.append(
        f"The {k} paths must be different from each other — they should use "
        "different sequences of nodes (and ideally different edges)."
    )

    if prompt.include_labels:
        labels_str = ", ".join(f"'{l}'" for l in prompt.include_labels)
        parts.append(f"Every path MUST use at least one edge with each of these labels: {labels_str}.")

    if prompt.exclude_labels:
        labels_str = ", ".join(f"'{l}'" for l in prompt.exclude_labels)
        parts.append(f"No path may use any edge with these labels: {labels_str}.")

    parts.append(
        "\nRespond with ONLY a JSON object in this exact format:\n"
        '{"paths": ['
        '{"path": ["NODE1", "NODE2", ...], "edge_labels": ["a", "b", ...]}, '
        "... "
        f"]}}\n"
        f"The paths array must contain exactly {k} entries. For each entry, "
        "`path` is the sequence of node labels and `edge_labels` are the labels "
        "of the edges traversed (one fewer than nodes)."
    )

    return " ".join(parts)


# --- Response parsing ---


@dataclass
class LLMResponse:
    """Parsed k-paths response from an LLM for a single query.

    Attributes:
        paths: List of node-label sequences, one per returned path.
        edge_label_sequences: List of edge-label sequences, parallel to paths.
        n_paths_returned: Number of path entries the model returned (may differ from k).
        raw_response: Full raw response text.
        parse_success: Whether at least one well-formed path was parsed.
    """

    paths: list[list[str]] = field(default_factory=list)
    edge_label_sequences: list[list[str]] = field(default_factory=list)
    n_paths_returned: int = 0
    raw_response: str = ""
    parse_success: bool = False


def _extract_json_object(raw: str) -> dict | None:
    """Best-effort extraction of the outermost JSON object containing `paths`.

    Handles code-fence wrapping and preambles. Returns None on failure.
    """
    # Strip code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else raw

    # Find first balanced { ... } containing "paths"
    start = candidate.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(candidate)):
            c = candidate[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    blob = candidate[start:i + 1]
                    try:
                        data = json.loads(blob)
                        if isinstance(data, dict) and "paths" in data:
                            return data
                    except json.JSONDecodeError:
                        pass
                    break
        start = candidate.find("{", start + 1)

    return None


def parse_multi_path_response(raw: str) -> LLMResponse:
    """Parse a k-paths JSON response from the LLM.

    Expected shape:
        {"paths": [
            {"path": ["ABC", "DEF", ...], "edge_labels": ["a", "b", ...]},
            ...
        ]}

    Tolerates extra preamble/commentary and markdown code fences. Paths that
    fail to parse individually are skipped (not filled with empties) — the
    caller can compare `n_paths_returned` to the requested k.
    """
    if not raw:
        return LLMResponse(raw_response=raw or "", parse_success=False)

    data = _extract_json_object(raw)
    if data is None:
        return LLMResponse(raw_response=raw, parse_success=False)

    raw_paths = data.get("paths")
    if not isinstance(raw_paths, list):
        return LLMResponse(raw_response=raw, parse_success=False)

    paths: list[list[str]] = []
    edge_label_sequences: list[list[str]] = []
    for item in raw_paths:
        if not isinstance(item, dict):
            continue
        p = item.get("path", [])
        e = item.get("edge_labels", [])
        if not isinstance(p, list) or not isinstance(e, list):
            continue
        paths.append([str(n) for n in p])
        edge_label_sequences.append([str(l) for l in e])

    return LLMResponse(
        paths=paths,
        edge_label_sequences=edge_label_sequences,
        n_paths_returned=len(paths),
        raw_response=raw,
        parse_success=len(paths) > 0,
    )
