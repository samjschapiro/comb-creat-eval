"""Prompt-method intervention for plot-twist generation (Exp 2: prompting methods).

Tests whether a PROMPTING strategy -- as opposed to the API-side reasoning-effort dial of
the thinking experiment (Exp 1) -- buys transformational creativity. On a small
reasoning-capable subset we run, at fixed temperature, IN-CONTEXT REGENERATION and compare
it to the reused main-run baseline (independent samples). In-context regeneration is
NoveltyBench's / CREATE's one prompt-based lever that consistently moves the needle -- for
DIVERSITY (arXiv:2504.05228, arXiv:2603.09970); the analysis asks whether it also lifts
surprise/coherence/realism, or only diversity (by construction).

COST NOTE: keeping every full prior story in the conversation is prohibitively expensive
(a 3k-word story is ~4k tokens; by sample 8 the context is ~30k input tokens *per call*).
So we use SUMMARY-CONDITIONED regeneration: after each story a cheap model distils its
twist to one line, and the next prompt lists the twists already used and asks for a
different one. Per-call input stays tiny; the expensive generator never re-reads full prose.
This is a cost-bounded variant of the papers' in-context regeneration -- note it in writeups.

Generators stay DISJOINT from the judge ensemble (enforced in the runner). Each story (and
its one-line twist summary) is persisted the instant it is produced (durable + resumable);
a sequential cell rebuilds its prior-twist list from disk, so a re-run never re-spends.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from src.plot_twist.generate import twist_prompt
from src.plot_twist.llm import call_llm_async, get_async_client_openrouter, model_id_to_key

# "Be creative" instruction appended to the base twist prompt (NoveltyBench system-prompt
# instruction / CREATE "be creative"). Independent samples, just with this nudge.
BE_CREATIVE_SUFFIX = (
    "\n\nBe as creative and original as you possibly can: avoid clichéd, predictable, or "
    "commonly-used twists, and aim for a genuinely surprising, non-obvious reveal that a "
    "reader would not see coming."
)


def be_creative_prompt(target_words: tuple[int, int]) -> str:
    return twist_prompt(target_words) + BE_CREATIVE_SUFFIX


# Cheap one-line distillation of a story's twist, fed back to steer the next generation.
SUMMARIZE_PROMPT = (
    "In ONE sentence (at most 25 words), describe the core PLOT TWIST of the story below "
    "-- the late reveal and the assumption it overturns. Output only the sentence.\n\n"
    'Story:\n"""{story}"""'
)


def regen_prompt(base_prompt: str, prior_twists: list[str]) -> str:
    """The (single-turn) prompt for the k-th in-context sample: the base twist prompt plus,
    if any stories already exist in this cell, the list of twists to avoid repeating."""
    if not prior_twists:
        return base_prompt
    lines = "\n".join(f"- {t}" for t in prior_twists)
    return (
        base_prompt
        + "\n\nIMPORTANT: you have ALREADY written stories with the plot twists listed "
        "below. Your new story's twist must be GENUINELY DIFFERENT from every one of them "
        "-- a different underlying mechanism and a different subject, not a variation:\n"
        + lines
    )


def rec_id_for(model: str, temperature: float, method: str, sample_idx: int) -> str:
    """Per-story id, tagged with the prompt method so conditions never collide."""
    mkey = model_id_to_key(model)
    return f"{mkey}__t{int(round(temperature * 10)):02d}__p{method}__s{sample_idx:02d}"


@dataclass
class PromptMethodsConfig:
    generator_models: list[str]
    methods: list[str] = field(default_factory=lambda: ["incontext_regen"])
    n_samples: int = 8
    temperatures: list[float] = field(default_factory=lambda: [1.0])
    max_tokens: int = 4500
    concurrency: int = 12
    target_words: tuple[int, int] = (2000, 3000)
    summarizer_model: str = "openai/gpt-4o-mini"  # cheap twist distiller for regeneration
    prompt: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.prompt = twist_prompt(self.target_words)


def _record(model: str, temperature: float, method: str, sample_idx: int, story, err) -> dict:
    return {
        "id": rec_id_for(model, temperature, method, sample_idx),
        "model": model,
        "temperature": temperature,
        "sample": sample_idx,
        "prompt_method": method,
        # mirror prompt_method into reasoning_level so generic downstream tooling (which
        # groups thinking stories by reasoning_level) can also group these.
        "reasoning_level": method,
        "story": story,
        "error": err,
    }


async def _summarize_twist(async_client, sem, model: str, story: str) -> str | None:
    """One-line twist summary via the cheap summarizer (full story sent; the twist is at
    the END, so we never front-truncate). Returns None on failure."""
    async with sem:
        try:
            s = await call_llm_async(
                async_client=async_client,
                messages=[{"role": "user", "content": SUMMARIZE_PROMPT.format(story=story)}],
                model=model, temperature=0.0, max_tokens=80,
            )
            return (s or "").strip().splitlines()[0][:300] if s else None
        except Exception:
            return None


async def _gen_independent(
    async_client, sem, model, temperature, sample_idx, method, prompt_text, cfg, stories_dir,
) -> dict:
    """One independent sample under an independent-sampling method (`baseline` or
    `be_creative`): a single user prompt, no knowledge of prior stories. (`baseline` is
    normally reused from the main run rather than regenerated here.)"""
    mkey = model_id_to_key(model)
    path = stories_dir / mkey / f"{rec_id_for(model, temperature, method, sample_idx)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("story"):
            return existing
    async with sem:
        try:
            story = await call_llm_async(
                async_client=async_client,
                messages=[{"role": "user", "content": prompt_text}],
                model=model, temperature=temperature, max_tokens=cfg.max_tokens,
                seed=sample_idx + 1,
            )
            err = None
        except Exception as exc:
            story, err = None, f"{type(exc).__name__}: {exc}"
    rec = _record(model, temperature, method, sample_idx, story, err)
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return rec


async def _gen_incontext_cell(
    async_client, sem, model, temperature, cfg, stories_dir,
) -> list[dict]:
    """All n summary-conditioned in-context samples for one (model, temperature) cell.
    Each prompt is a single user turn = base prompt + the list of twists already used in
    this cell (so input stays small). Resumes by rebuilding that list from saved summaries;
    stops the chain if a generation fails."""
    mkey = model_id_to_key(model)
    dirp = stories_dir / mkey
    dirp.mkdir(parents=True, exist_ok=True)
    prior_twists: list[str] = []
    out: list[dict] = []
    for i in range(cfg.n_samples):
        path = dirp / f"{rec_id_for(model, temperature, 'incontext_regen', i)}.json"
        rec = None
        if path.exists():
            existing = json.loads(path.read_text())
            if existing.get("story"):
                rec = existing
        if rec is None:
            prompt = regen_prompt(cfg.prompt, prior_twists)
            async with sem:
                try:
                    story = await call_llm_async(
                        async_client=async_client,
                        messages=[{"role": "user", "content": prompt}],
                        model=model, temperature=temperature, max_tokens=cfg.max_tokens,
                        seed=i + 1,
                    )
                    err = None
                except Exception as exc:
                    story, err = None, f"{type(exc).__name__}: {exc}"
            summary = await _summarize_twist(async_client, sem, cfg.summarizer_model, story) if story else None
            rec = _record(model, temperature, "incontext_regen", i, story, err)
            rec["twist_summary"] = summary
            path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        out.append(rec)
        if not rec.get("story"):
            break  # chain broken; later samples can't condition on a missing story
        if rec.get("twist_summary"):
            prior_twists.append(rec["twist_summary"])
    return out


async def generate_prompt_methods(cfg: PromptMethodsConfig, stories_dir: Path) -> list[dict]:
    """Generate the configured methods for every (model, temperature). Each in-context cell
    runs its samples sequentially (different cells run concurrently) under one shared
    semaphore. Every story is written to its own file the moment it is produced."""
    stories_dir = Path(stories_dir)
    stories_dir.mkdir(parents=True, exist_ok=True)
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(cfg.concurrency)
    # prompt text per independent-sampling method
    indep_prompt = {"baseline": cfg.prompt, "be_creative": be_creative_prompt(cfg.target_words)}
    tasks = []
    for model in cfg.generator_models:
        for temp in cfg.temperatures:
            for method in cfg.methods:
                if method == "incontext_regen":
                    tasks.append(_gen_incontext_cell(client, sem, model, temp, cfg, stories_dir))
                elif method in indep_prompt:
                    tasks += [_gen_independent(client, sem, model, temp, i, method,
                                               indep_prompt[method], cfg, stories_dir)
                              for i in range(cfg.n_samples)]
                else:
                    raise ValueError(f"unknown prompt method: {method!r}")
    gathered = await asyncio.gather(*tasks)
    records: list[dict] = []
    for g in gathered:
        records.extend(g if isinstance(g, list) else [g])
    return records
