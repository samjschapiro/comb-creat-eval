"""LLM prompting infrastructure for combinatorial creativity evaluation.

Handles formatting graph and query prompts, calling LLM APIs via OpenRouter
(OpenAI-compatible), and parsing responses. All models are accessed through
a single OpenRouter API key.
"""

import json
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from src.comb_eval.prompts import EvalPrompt

load_dotenv()


# --- OpenRouter client ---


def get_client() -> OpenAI:
    """Create an OpenRouter client using the OpenAI SDK."""
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


def call_llm(
    messages: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> str:
    """Call an LLM via OpenRouter. Returns raw response text.

    Args:
        messages: List of message dicts with 'role' and 'content'.
        model: OpenRouter model ID (e.g., 'openai/gpt-4o', 'anthropic/claude-sonnet-4').
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.

    Returns:
        Raw response text.
    """
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


# --- Prompt formatting ---


def format_graph_prompt(adjacency_text: str) -> str:
    """Format the graph memorization prompt.

    This is the first stage: present the graph structure for the model to absorb.
    """
    return (
        "Below is a graph structure. Each line shows a node and its neighbors, "
        "with edge labels in parentheses. For example, 'ABC: DEF(a), GHI(b)' means "
        "node ABC connects to DEF via edge labeled 'a' and to GHI via edge labeled 'b'.\n"
        "\n"
        "Study this graph carefully.\n"
        "\n"
        f"{adjacency_text}"
    )


def format_query_prompt(prompt: EvalPrompt) -> str:
    """Format a path-finding query for the LLM.

    Asks the model to find a path satisfying the given constraints,
    returning structured output.
    """
    parts = [
        f"Find a path of exactly {prompt.hop_count} hops from {prompt.start} to {prompt.end} "
        f"in the graph above."
    ]

    if prompt.include_labels:
        labels_str = ", ".join(f"'{l}'" for l in prompt.include_labels)
        parts.append(f"The path MUST use edges with these labels: {labels_str}.")

    if prompt.exclude_labels:
        labels_str = ", ".join(f"'{l}'" for l in prompt.exclude_labels)
        parts.append(f"The path must NOT use edges with these labels: {labels_str}.")

    parts.append(
        "\nRespond with ONLY a JSON object in this format:\n"
        '{"path": ["NODE1", "NODE2", ...], "edge_labels": ["a", "b", ...]}\n'
        "where path is the sequence of nodes and edge_labels are the labels of "
        "edges traversed (one fewer than nodes)."
    )

    return " ".join(parts)


def format_batch_query_prompt(prompts: list[EvalPrompt]) -> str:
    """Format multiple path-finding queries in a single prompt.

    Each query is numbered for easy parsing.
    """
    parts = [
        "Answer each of the following path-finding queries about the graph above. "
        "For each query, respond with a JSON object.\n"
    ]

    for i, prompt in enumerate(prompts):
        query = [f"Query {i + 1}: "]
        query.append(
            f"Find a path of exactly {prompt.hop_count} hops "
            f"from {prompt.start} to {prompt.end}."
        )
        if prompt.include_labels:
            labels_str = ", ".join(f"'{l}'" for l in prompt.include_labels)
            query.append(f" MUST use edge labels: {labels_str}.")
        if prompt.exclude_labels:
            labels_str = ", ".join(f"'{l}'" for l in prompt.exclude_labels)
            query.append(f" Must NOT use edge labels: {labels_str}.")
        parts.append("".join(query))

    parts.append(
        '\nFor each query, provide: {"path": ["NODE1", "NODE2", ...], "edge_labels": ["a", "b", ...]}'
        '\nFormat your response as a JSON array of objects, one per query: [{"path": ..., "edge_labels": ...}, ...]'
    )

    return "\n".join(parts)


# --- Response parsing ---


@dataclass
class LLMResponse:
    """Parsed response from an LLM for a single path query."""

    path: list[str]  # Node labels
    edge_labels: list[str]  # Edge labels between consecutive nodes
    raw_response: str  # Full raw response text
    parse_success: bool  # Whether parsing succeeded


def parse_single_response(raw: str) -> LLMResponse:
    """Parse a single JSON path response from the LLM."""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if not json_match:
            return LLMResponse(path=[], edge_labels=[], raw_response=raw, parse_success=False)

        data = json.loads(json_match.group())
        path = data.get("path", [])
        edge_labels = data.get("edge_labels", [])

        if not isinstance(path, list) or not isinstance(edge_labels, list):
            return LLMResponse(path=[], edge_labels=[], raw_response=raw, parse_success=False)

        return LLMResponse(
            path=[str(n) for n in path],
            edge_labels=[str(l) for l in edge_labels],
            raw_response=raw,
            parse_success=True,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return LLMResponse(path=[], edge_labels=[], raw_response=raw, parse_success=False)


def parse_batch_response(raw: str, n_queries: int) -> list[LLMResponse]:
    """Parse a batch response containing multiple path results."""
    try:
        # Try to find a JSON array
        array_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if array_match:
            data = json.loads(array_match.group())
            if isinstance(data, list):
                results = []
                for item in data:
                    if isinstance(item, dict):
                        results.append(LLMResponse(
                            path=[str(n) for n in item.get("path", [])],
                            edge_labels=[str(l) for l in item.get("edge_labels", [])],
                            raw_response=raw,
                            parse_success=True,
                        ))
                    else:
                        results.append(LLMResponse(
                            path=[], edge_labels=[], raw_response=raw, parse_success=False
                        ))
                return results
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Fallback: try to find individual JSON objects
    json_objects = re.findall(r"\{[^{}]*\}", raw, re.DOTALL)
    results = []
    for obj_str in json_objects[:n_queries]:
        try:
            data = json.loads(obj_str)
            results.append(LLMResponse(
                path=[str(n) for n in data.get("path", [])],
                edge_labels=[str(l) for l in data.get("edge_labels", [])],
                raw_response=raw,
                parse_success=True,
            ))
        except (json.JSONDecodeError, KeyError, TypeError):
            results.append(LLMResponse(
                path=[], edge_labels=[], raw_response=raw, parse_success=False
            ))

    # Pad with failures if we didn't get enough
    while len(results) < n_queries:
        results.append(LLMResponse(
            path=[], edge_labels=[], raw_response=raw, parse_success=False
        ))

    return results
