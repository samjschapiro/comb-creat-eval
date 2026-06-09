"""Annotate each story with setup / reveal / score-rationale (plot_twist track).

For every story (LLM-generated and human gold), an annotator LLM reads the story
plus the rubric scores it already received, and produces three one-sentence labels:

  - setup:      the premise and what the reader believes before the twist
  - reveal:     the twist itself and what it makes the reader re-interpret
  - why_scored: why it earned that surprise/coherence (recasts the whole story =
                high surprise; well-prepared = high coherence; predictable or
                arbitrary = low)

This is a labelling/analysis pass, not a scoring pass: it explains the existing
judge scores and gives a short twist-concept summary (the `reveal`) that the UMAP
diversity plot embeds. Each annotation is written to its own file the instant it
completes (durable + resumable), so a kill never loses paid annotator calls.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from src.plot_twist.llm import call_llm_async, get_async_client_openrouter

ANNOTATE_PROMPT = """You are a literary analyst. Read the short story below and \
the scores a separate plot-twist rubric judge already gave it.

A "plot twist" is a reveal that makes the reader re-interpret earlier parts of the \
story. The rubric rated (1-5): SURPRISE (how much the ending forces a re-read) and \
COHERENCE (whether the earlier story still holds up, even fits better, after the \
reveal). twist_present indicates whether the judge thought there was a genuine \
twist at all.

Scores for this story: surprise={surprise}, coherence={coherence}, \
overall={overall}, twist_present={twist_present}.

Produce a JSON object with EXACTLY these keys, each a single sentence:
- "setup": the premise and what the reader is led to believe before the twist.
- "reveal": the twist itself -- what is revealed and what it makes the reader \
re-interpret. If there is no genuine twist, begin with "No genuine twist:" and say \
what happens instead.
- "why_scored": why this earned that surprise/coherence (e.g. the reveal recasts \
the whole story so surprise is high; it was quietly prepared so coherence is high; \
or it was predictable/arbitrary so a dimension is low).

Story:
\"\"\"{story}\"\"\"

Output only the JSON object, nothing else."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class AnnotateConfig:
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 400
    concurrency: int = 16


def _parse(raw: Optional[str]) -> Optional[dict]:
    if not raw:
        return None
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {k: data.get(k) for k in ("setup", "reveal", "why_scored")}


async def _annotate_one(
    async_client,
    sem: asyncio.Semaphore,
    item: dict,  # {"id", "source", "story", "scores"}
    cfg: AnnotateConfig,
    cache_dir: Path,
) -> dict:
    sid = str(item["id"])
    path = cache_dir / f"{sid}.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("reveal"):
            return existing  # resume: already annotated, no re-spend

    sc = item.get("scores", {})
    prompt = ANNOTATE_PROMPT.format(
        story=item["story"],
        surprise=sc.get("surprise"),
        coherence=sc.get("coherence"),
        overall=sc.get("overall"),
        twist_present=sc.get("twist_present"),
    )
    async with sem:
        try:
            raw: Optional[str] = await call_llm_async(
                async_client=async_client,
                messages=[{"role": "user", "content": prompt}],
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
            err = None
        except Exception as exc:
            raw, err = None, f"{type(exc).__name__}: {exc}"

    parsed = _parse(raw) or {}
    rec = {
        "id": sid,
        "source": item["source"],
        "scores": sc,
        "setup": parsed.get("setup"),
        "reveal": parsed.get("reveal"),
        "why_scored": parsed.get("why_scored"),
        "error": err,
    }
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


async def annotate_stories(
    cfg: AnnotateConfig, items: Sequence[dict], cache_dir: Path
) -> list[dict]:
    """Annotate every item. Each annotation is saved to cache_dir/<id>.json the
    moment it completes (durable + resumable)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(cfg.concurrency)
    return list(
        await asyncio.gather(*(_annotate_one(client, sem, it, cfg, cache_dir) for it in items))
    )
