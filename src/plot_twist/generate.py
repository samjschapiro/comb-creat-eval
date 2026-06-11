"""Generate LLM plot-twist stories for the plot_twist benchmark (paper Sec.3 / H5).

Open-ended generation: each model is asked, N times, to write a short story with a
plot twist -- it chooses its own subject (no premise, so the premise does not do
the creative work for the model; this measures the model's OWN transformational
creativity). Sampling is at temperature > 0 with a distinct seed per sample, so the
N stories from a model vary.

The quality SPREAD needed to test whether the rubric judge discriminates comes from
the model pool (deliberately weak -> strong) and sampling variance, not from
explicit predictable/random conditions.

Generators are kept DISJOINT from the judge ensemble (anti-circularity): a model
must not grade its own (or a sibling provider's) output. Enforced in the runner.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.plot_twist.llm import call_llm_async, get_async_client_openrouter, model_id_to_key

# Length band is matched to the human gold set (median ~2,469 words, IQR
# 1,873-4,561) so that story length does not confound the human-vs-LLM
# comparison (H5). The band is a config parameter (target_words), surfaced in the
# prompt below.
TWIST_PROMPT_TEMPLATE = (
    "Write a complete short story (roughly {lo:,}-{hi:,} words). The story must "
    "contain a genuine PLOT TWIST: a late reveal that makes the reader re-interpret "
    "earlier events in a new light, yet is consistent with everything that came "
    "before (in hindsight it should feel prepared, not arbitrary). Choose your own "
    "subject and characters. Do not announce or explain the twist; let it land.\n\n"
    "Write only the story, no title or commentary."
)


def twist_prompt(target_words: tuple[int, int]) -> str:
    lo, hi = target_words
    return TWIST_PROMPT_TEMPLATE.format(lo=lo, hi=hi)


@dataclass
class GenerateConfig:
    generator_models: list[str]
    n_samples: int = 10  # samples per (model x temperature)
    temperatures: list[float] = field(default_factory=lambda: [0.9, 1.0, 1.2])
    max_tokens: int = 4500  # fit ~3k words (~1.4 tokens/word) + margin
    concurrency: int = 16
    target_words: tuple[int, int] = (2000, 3000)  # matched to human-gold median band
    prompt: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.prompt = twist_prompt(self.target_words)


async def _generate_one(
    async_client,
    sem: asyncio.Semaphore,
    model: str,
    temperature: float,
    sample_idx: int,
    cfg: GenerateConfig,
    stories_dir: Path,
) -> dict:
    mkey = model_id_to_key(model)
    rec_id = f"{mkey}__t{int(round(temperature * 10)):02d}__s{sample_idx:02d}"
    path = stories_dir / mkey / f"{rec_id}.json"  # one subfolder per model
    path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: if a good story is already on disk, return it WITHOUT re-calling the
    # API (so a re-run never re-spends on stories we already have).
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("story"):
            return existing

    async with sem:
        try:
            story: Optional[str] = await call_llm_async(
                async_client=async_client,
                messages=[{"role": "user", "content": cfg.prompt}],
                model=model,
                temperature=temperature,
                max_tokens=cfg.max_tokens,
                seed=sample_idx + 1,  # distinct, reproducible (>=1; some providers reject seed 0)
            )
            err = None
        except Exception as exc:  # surfaced in the record, not silently dropped
            story, err = None, f"{type(exc).__name__}: {exc}"

    rec = {
        "id": rec_id,
        "model": model,
        "temperature": temperature,
        "sample": sample_idx,
        "story": story,
        "error": err,
    }
    # SAVE this story to its own file the instant it is generated, so a crash or
    # kill never loses completed work.
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


async def generate_stories(cfg: GenerateConfig, stories_dir: Path) -> list[dict]:
    """Generate n_samples twist stories per (model x temperature). Each story is
    written to its own file in stories_dir the moment it is produced (durable +
    resumable). One shared client and semaphore cap concurrency across all calls."""
    stories_dir = Path(stories_dir)
    stories_dir.mkdir(parents=True, exist_ok=True)
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(cfg.concurrency)
    tasks = [
        _generate_one(client, sem, model, temp, i, cfg, stories_dir)
        for model in cfg.generator_models
        for temp in cfg.temperatures
        for i in range(cfg.n_samples)
    ]
    return list(await asyncio.gather(*tasks))
