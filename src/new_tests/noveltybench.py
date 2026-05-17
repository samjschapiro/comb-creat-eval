"""NoveltyBench scoring implementation.

Reference: Zhang et al. 2025, "NoveltyBench: Evaluating Language Models for
Humanlike Diversity" (arXiv 2504.05228). This module reimplements the
utility_k metric from the paper, since the authors do not publish a
runnable scoring pipeline.

Canonical formula (paper §3):

    utility_k = (1 - p) / (1 - p^k)
                * sum_{i=1..k} p^{i-1} * 1[c_i != c_j for all j < i] * u_i

where:
- k = generations per prompt (paper uses k=10)
- p = patience parameter (paper uses p=0.8)
- c_i = the equivalence-class assignment of generation i. Paper uses a
  fine-tuned DeBERTa-v3-large classifier published as
  [yimingzhang/deberta-v3-large-generation-similarity](https://huggingface.co/yimingzhang/deberta-v3-large-generation-similarity).
  This module loads that classifier by default (see
  src/new_tests/distinctness.py) and falls back to an LLM-judge only
  when explicitly configured via distinctness_judge_model.
- u_i = quality of generation i in {1,...,10} (paper uses
  Skywork-Reward-Gemma-2-27B-v0.2 calibrated against MT-Bench / GPT-4
  judgments; this implementation substitutes an LLM-judge with a
  rubric prompt — see _quality_score)

Deviations from the canonical pipeline are flagged inline. Anyone
running this should (a) consider these deviations when comparing
absolute numbers to the paper's leaderboard, and (b) remember the
relative ordering between models is what matters here, not the
absolute number — which is why we explicitly mark the per-judge
configuration in the output JSON for reproducibility.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.new_tests.llm import call_llm_async, get_async_client
from src.dat_eval.llm import get_async_client_openrouter


async def _judge_call(
    async_client,
    sem: "asyncio.Semaphore",
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    async with sem:
        return await call_llm_async(
            async_client=async_client,
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
        )


# -----------------------------------------------------------------------------
# LLM-judge prompts
# -----------------------------------------------------------------------------

DISTINCTNESS_PROMPT = """Two responses to the same prompt are below. Decide \
whether a user who has already seen Response A would gain MEANINGFULLY \
NEW VALUE from also seeing Response B.

Apply a STRICT bar for distinctness. Two responses are EQUIVALENT (NOT \
distinct) if they share the SAME UNDERLYING CONTENT, even if worded \
differently. Specifically, if any of the following hold, they are \
EQUIVALENT:

- They tell essentially the same story / give the same answer / make the \
  same recommendation, even with renamed characters, paraphrasing, or \
  reordered sentences.
- They follow the same plot structure, the same overall argument, or the \
  same conceptual direction.
- The differences between them are only at the level of surface wording, \
  proper nouns, or tone, while the core content is the same.

Two responses are DISTINCT only when Response B offers genuinely \
different content: a different story arc, a different answer, a different \
recommendation, a different conceptual direction — something a careful \
reader would consider a substantively different contribution rather than \
a rewording of the same idea.

Prompt:
\"\"\"{prompt}\"\"\"

Response A:
\"\"\"{a}\"\"\"

Response B:
\"\"\"{b}\"\"\"

Answer with EXACTLY one word: either EQUIVALENT or DISTINCT."""


QUALITY_PROMPT = """Rate the quality of the following response to the given \
prompt on an integer scale from 1 (lowest quality) to 10 (highest \
quality), considering helpfulness, correctness, clarity, and how well it \
addresses the prompt. Output ONLY the integer score, nothing else.

Prompt:
\"\"\"{prompt}\"\"\"

Response:
\"\"\"{response}\"\"\"

Score (1-10):"""


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class NoveltyBenchConfig:
    """All knobs the implementation exposes. Defaults match the paper.

    distinctness_method:
        - "deberta" (default, canonical): use the fine-tuned
          yimingzhang/deberta-v3-large-generation-similarity classifier.
        - "llm": use an LLM-judge via OpenRouter — set
          distinctness_judge_model accordingly. Less faithful to the
          paper but useful when the local classifier can't be loaded.
    """

    k: int = 10
    patience: float = 0.8
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    distinctness_method: str = "deberta"
    distinctness_judge_model: str = "anthropic/claude-haiku-4-5"
    distinctness_threshold: float = 0.5
    distinctness_device: str = "cpu"
    quality_judge_model: str = "openai/gpt-4o-mini"
    # Quality is judge-dependent, so it is scored by an ensemble of
    # independent judges (different providers) and aggregated per
    # generation by the median — robust to a single outlier/broken judge.
    quality_judge_models: list[str] = field(
        default_factory=lambda: ["openai/gpt-4o-mini"]
    )
    judge_concurrency: int = 16
    generation_concurrency: int = 16


@dataclass
class PerPromptResult:
    prompt_id: str
    prompt: str
    generations: list[str]
    classes: list[int]  # generation i -> equivalence class id
    qualities: list[float]  # generation i -> u_i (median over judges)
    utility_k: float
    # judge model id -> per-generation raw scores (None if that judge
    # failed for that generation).
    quality_by_judge: dict[str, list[int | None]] = field(default_factory=dict)


@dataclass
class NoveltyBenchResult:
    test_model: str
    config: dict
    per_prompt: list[PerPromptResult]
    mean_utility_k: float
    fraction_distinct: float = field(default=0.0)
    mean_quality: float = field(default=0.0)
    # judge model id -> mean raw score over all cells it scored.
    mean_quality_by_judge: dict = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Utility math
# -----------------------------------------------------------------------------


def utility_k(classes: Sequence[int], qualities: Sequence[int], p: float) -> float:
    """Implements the paper's utility_k formula.

    classes[i] is the equivalence-class ID of generation i; the indicator
    1[c_i != c_j for all j < i] is true iff classes[i] is novel (i.e.
    classes[i] does not appear in classes[:i]). qualities[i] is u_i.
    """
    assert len(classes) == len(qualities)
    k = len(classes)
    if k == 0:
        return 0.0
    seen: set[int] = set()
    raw = 0.0
    for i in range(k):
        if classes[i] not in seen:
            seen.add(classes[i])
            raw += (p ** i) * qualities[i]
    norm = (1 - p) / (1 - p ** k) if abs(p - 1.0) > 1e-12 else 1.0 / k
    return norm * raw


# -----------------------------------------------------------------------------
# Equivalence-class clustering via LLM judge
# -----------------------------------------------------------------------------


async def _is_distinct_from(
    async_client,
    judge_model: str,
    sem: asyncio.Semaphore,
    prompt: str,
    a: str,
    b: str,
) -> bool:
    """LLM-judge stand-in for the paper's DeBERTa-v3-large equivalence
    classifier. Returns True iff the responses are judged DISTINCT
    (functionally different), False if EQUIVALENT.
    """
    raw = await _judge_call(
        async_client,
        sem,
        judge_model,
        DISTINCTNESS_PROMPT.format(prompt=prompt, a=a, b=b),
        max_tokens=8,
    )
    if raw is None:
        raise RuntimeError("Distinctness judge returned None")
    text = raw.strip().upper()
    if text.startswith("DISTINCT"):
        return True
    if text.startswith("EQUIVALENT"):
        return False
    print(f"[noveltybench] unexpected distinctness reply: {text!r}")
    return False


async def _assign_classes(
    async_client,
    cfg: NoveltyBenchConfig,
    prompt: str,
    generations: Sequence[str],
) -> list[int]:
    """Greedy class assignment matching the paper:
    'For each new generation, we compare it against a random generation
    from each existing class. If the classifier determines it is
    functionally equivalent to any existing class, we assign it to the
    first such class found; otherwise, we create a new class.'

    Two backends per cfg.distinctness_method: 'deberta' (canonical) and
    'llm' (OpenRouter judge fallback).
    """
    if not generations:
        return []

    classes: list[int] = [0]
    representatives: list[str] = [generations[0]]

    if cfg.distinctness_method == "deberta":
        # Local classifier — synchronous; batch all comparisons for one i.
        from src.new_tests.distinctness import batch_distinct

        for i in range(1, len(generations)):
            cur = generations[i]
            pairs = [(rep, cur) for rep in representatives]
            distinct_flags = batch_distinct(
                pairs,
                threshold=cfg.distinctness_threshold,
                device=cfg.distinctness_device,
            )
            assigned: int | None = None
            for j, distinct in enumerate(distinct_flags):
                if not distinct:
                    assigned = j
                    break
            if assigned is None:
                classes.append(len(representatives))
                representatives.append(cur)
            else:
                classes.append(assigned)
        return classes

    if cfg.distinctness_method == "llm":
        sem = asyncio.Semaphore(cfg.judge_concurrency)
        for i in range(1, len(generations)):
            cur = generations[i]

            async def _vs(j: int) -> tuple[int, bool]:
                return j, await _is_distinct_from(
                    async_client,
                    cfg.distinctness_judge_model,
                    sem,
                    prompt,
                    representatives[j],
                    cur,
                )

            results = await asyncio.gather(
                *(_vs(j) for j in range(len(representatives)))
            )
            assigned = None
            for j, distinct in results:
                if not distinct:
                    assigned = j
                    break
            if assigned is None:
                classes.append(len(representatives))
                representatives.append(cur)
            else:
                classes.append(assigned)
        return classes

    raise ValueError(f"Unknown distinctness_method {cfg.distinctness_method!r}")


# -----------------------------------------------------------------------------
# Quality scoring
# -----------------------------------------------------------------------------


_INT_RE = re.compile(r"\d+")


async def _quality_score(
    async_client,
    judge_model: str,
    sem: asyncio.Semaphore,
    prompt: str,
    response: str,
) -> int:
    raw = await _judge_call(
        async_client,
        sem,
        judge_model,
        QUALITY_PROMPT.format(prompt=prompt, response=response),
        max_tokens=4,
    )
    if raw is None:
        raise RuntimeError("Quality judge returned None")
    m = _INT_RE.search(raw)
    if m is None:
        raise ValueError(f"Quality judge produced no integer: {raw!r}")
    val = int(m.group(0))
    return max(1, min(10, val))


async def _quality_score_safe(
    async_client,
    judge_model: str,
    sem: asyncio.Semaphore,
    prompt: str,
    response: str,
) -> int | None:
    """_quality_score that returns None instead of raising, so one flaky
    judge/provider does not abort the ensemble for that generation."""
    try:
        return await _quality_score(async_client, judge_model, sem, prompt, response)
    except Exception:
        return None


async def _score_qualities(
    async_client,
    cfg: NoveltyBenchConfig,
    prompt: str,
    generations: Sequence[str],
) -> tuple[list[float], dict[str, list[int | None]]]:
    """Score each generation with every judge in cfg.quality_judge_models,
    then aggregate per generation by the median of the judges that
    returned a score. Robust to a single outlier or broken judge.

    Returns (aggregated, by_judge): aggregated[i] is the median quality
    for generation i; by_judge[m][i] is judge m's raw score (or None).
    """
    judges = cfg.quality_judge_models or [cfg.quality_judge_model]
    sem = asyncio.Semaphore(cfg.judge_concurrency)
    # by_judge[m] = [score per generation]; gather all (judge, gen) cells.
    by_judge: dict[str, list[int | None]] = {}
    for m in judges:
        by_judge[m] = list(
            await asyncio.gather(
                *(
                    _quality_score_safe(async_client, m, sem, prompt, g)
                    for g in generations
                )
            )
        )
    aggregated: list[float] = []
    for i in range(len(generations)):
        scores = [by_judge[m][i] for m in judges if by_judge[m][i] is not None]
        if not scores:
            raise RuntimeError(
                f"All {len(judges)} quality judges failed for a generation"
            )
        aggregated.append(float(statistics.median(scores)))
    return aggregated, by_judge


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


async def _generate_for_prompt(
    cfg: NoveltyBenchConfig,
    async_client,
    test_model: str,
    prompt: str,
    sem: asyncio.Semaphore,
) -> list[str]:
    async def _one() -> str:
        async with sem:
            r = await async_client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                n=1,
            )
        return r.choices[0].message.content or ""

    return list(await asyncio.gather(*(_one() for _ in range(cfg.k))))


# -----------------------------------------------------------------------------
# Top-level orchestration
# -----------------------------------------------------------------------------


async def score_model(
    cfg: NoveltyBenchConfig,
    test_model: str,
    prompts: list[dict],  # each dict has 'id' and 'prompt' keys
) -> NoveltyBenchResult:
    # Generation hits async_client (the local server when LLM_BASE_URL is
    # set). The quality/distinctness LLM judges must hit a hosted model,
    # so they use a separate always-OpenRouter client. (Distinctness via
    # 'deberta' ignores its client arg; only 'llm' uses it.)
    async_client = get_async_client()
    judge_client = get_async_client_openrouter()
    gen_sem = asyncio.Semaphore(cfg.generation_concurrency)
    per_prompt: list[PerPromptResult] = []
    for item in prompts:
        pid = str(item["id"])
        ptxt = item["prompt"]
        gens = await _generate_for_prompt(cfg, async_client, test_model, ptxt, gen_sem)
        classes = await _assign_classes(judge_client, cfg, ptxt, gens)
        qualities, quality_by_judge = await _score_qualities(
            judge_client, cfg, ptxt, gens
        )
        u = utility_k(classes, qualities, cfg.patience)
        per_prompt.append(
            PerPromptResult(
                prompt_id=pid,
                prompt=ptxt,
                generations=gens,
                classes=classes,
                qualities=qualities,
                utility_k=u,
                quality_by_judge=quality_by_judge,
            )
        )

    mean_u = sum(r.utility_k for r in per_prompt) / max(1, len(per_prompt))
    # Fraction of (generation index, prompt) cells that are novel within
    # their prompt's k-set:
    total_cells = sum(len(r.generations) for r in per_prompt)
    distinct_cells = sum(
        sum(1 for j in range(len(r.classes)) if r.classes[j] not in r.classes[:j])
        for r in per_prompt
    )
    frac_distinct = distinct_cells / max(1, total_cells)
    mean_q = sum(sum(r.qualities) for r in per_prompt) / max(1, total_cells)
    # Per-judge means (over the cells each judge actually scored) so an
    # outlier judge is visible in the summary.
    judge_ids = cfg.quality_judge_models or [cfg.quality_judge_model]
    mean_q_by_judge: dict = {}
    for m in judge_ids:
        vals = [
            s
            for r in per_prompt
            for s in r.quality_by_judge.get(m, [])
            if s is not None
        ]
        mean_q_by_judge[m] = (sum(vals) / len(vals)) if vals else None

    return NoveltyBenchResult(
        test_model=test_model,
        config={
            "k": cfg.k,
            "patience": cfg.patience,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "distinctness_method": cfg.distinctness_method,
            "distinctness_judge_model": (
                cfg.distinctness_judge_model
                if cfg.distinctness_method == "llm"
                else None
            ),
            "distinctness_threshold": cfg.distinctness_threshold,
            "quality_judge_models": (
                cfg.quality_judge_models or [cfg.quality_judge_model]
            ),
            "quality_aggregation": "median",
            "implementation_notes": (
                "Distinctness via canonical yimingzhang/deberta-v3-large-"
                "generation-similarity classifier (paper-faithful). "
                "Quality via ENSEMBLE of independent LLM judges (1-10), "
                "aggregated per generation by median (paper: single "
                "Skywork-Reward-Gemma-2-27B calibrated to MT-Bench/GPT-4)."
                if cfg.distinctness_method == "deberta"
                else "Distinctness via LLM judge fallback (NOT paper-faithful). "
                "Quality via LLM judge 1-10."
            ),
        },
        per_prompt=per_prompt,
        mean_utility_k=mean_u,
        fraction_distinct=frac_distinct,
        mean_quality=mean_q,
        mean_quality_by_judge=mean_q_by_judge,
    )


def save_result(result: NoveltyBenchResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    detail_path = output_dir / "per_prompt.json"
    summary = {
        "test_model": result.test_model,
        "config": result.config,
        "n_prompts": len(result.per_prompt),
        "mean_utility_k": result.mean_utility_k,
        "fraction_distinct": result.fraction_distinct,
        "mean_quality": result.mean_quality,
        "mean_quality_by_judge": result.mean_quality_by_judge,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    detail = [
        {
            "prompt_id": r.prompt_id,
            "prompt": r.prompt,
            "generations": r.generations,
            "classes": r.classes,
            "qualities": r.qualities,
            "quality_by_judge": r.quality_by_judge,
            "utility_k": r.utility_k,
        }
        for r in result.per_prompt
    ]
    detail_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False))
    return summary_path
