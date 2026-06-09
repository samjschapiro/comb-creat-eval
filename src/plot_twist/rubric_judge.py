"""Fixed-rubric LLM-as-judge for plot-twist quality (the T2C benchmark metric).

This is the primary, scalable score for the plot_twist benchmark (paper §3). An
LLM judge applies a FIXED rubric to a story and rates it on four dimensions:

  - surprise:       how much the ending forces a re-read of the earlier story
                    (the theory's T_mod / novelty term)
  - coherence:  whether the earlier story still holds together, even fits
                    better, after the reveal (the theory's preservation /
                    appropriateness term)
  - prose_quality:  general writing quality, INDEPENDENT of the twist. This is a
                    COVARIATE, not a creativity measure: H5 (human vs LLM) is only
                    meaningful if the twist dimensions, not prose, drive any gap.
  - overall:        overall plot-twist quality (high only when a twist is both
                    surprising AND coherent; predictable = low surprise, random
                    = low coherence)

It also returns `twist_present` (does the story contain a plot twist at all?),
used by H4 Test A (can a metric even detect a twist?).

The rubric is operationalized from the SBV graphical theory of transformational
creativity (Schapiro, Black, Varshney, ICCC 2025): a twist is an axiom
modification of the reader's story-DAG, so quality = surprise (re-read) x
coherence (preservation). The theory sets the rubric; the judge applies it.

IMPORTANT (reliability): the benchmark rests on this judge. Before trusting it,
calibrate against a human-rated subset and report judge-human agreement (paper
§3, the make-or-break number). Scores here are an ENSEMBLE of independent judge
models aggregated per dimension by the median, so one outlier/broken judge does
not corrupt a score. The exact rubric text is versioned (RUBRIC_VERSION) so runs
are comparable.
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from src.plot_twist.llm import call_llm_async, get_async_client_openrouter

RUBRIC_VERSION = "v1"

# -----------------------------------------------------------------------------
# The fixed rubric (versioned). Anchored 1-5 scales reduce judge variance.
# -----------------------------------------------------------------------------

RUBRIC_PROMPT = """You are a careful, consistent literary judge. You will read a \
short story and rate its PLOT TWIST using a fixed rubric. A "plot twist" is a \
reveal that makes the reader re-interpret earlier parts of the story. Judge only \
what is on the page; do not reward mere shock, gore, or randomness.

Rate the story on each dimension using the 1-5 anchors below. Use the FULL range.

SURPRISE -- How much does the ending force you to re-read the earlier story in a \
new light?
  1: No re-reading. The ending follows straightforwardly from the setup; expected.
  2: Minor. One small detail is recast; the overall reading is unchanged.
  3: Moderate. A subplot or one character is recast, but the core reading stays.
  4: Strong. A central assumption is overturned; much of the story reads differently.
  5: Total. A foundational assumption is overturned; nearly the whole story must be \
re-interpreted.

COHERENCE -- After the reveal, does the earlier story still hold together -- \
even fit better -- or is it contradicted?
  1: Contradicted. The reveal conflicts with earlier events; arbitrary or tacked-on.
  2: Weak. The reveal mostly does not fit; little in the story supports it.
  3: Mixed. Some earlier details support the reveal; others must be ignored.
  4: Strong. The earlier story is consistent with the reveal, and several details \
now make better sense.
  5: Seamless. The reveal was quietly prepared throughout; in hindsight it feels \
coherent and earlier details snap into place. Nothing has to be retracted.

PROSE_QUALITY -- Independent of the twist, how good is the writing (fluency, \
imagery, control)? Rate 1 (poor) to 5 (excellent). This is NOT about the twist.

OVERALL -- Overall, how good is this as a PLOT TWIST? A great twist is high on \
BOTH surprise and coherence. A predictable ending scores low on surprise; a \
random or incoherent ending scores low on coherence. Rate 1 (no/poor twist) \
to 5 (excellent twist).

Also decide TWIST_PRESENT: does the story contain a genuine plot twist at all \
(a reveal that re-interprets earlier events)? true or false.

Story:
\"\"\"{story}\"\"\"

First, in AT MOST three sentences, note what (if anything) the twist re-interprets \
and how well the earlier story supports it. Then output a single JSON object on \
its own final line, EXACTLY in this form and nothing after it:
{{"twist_present": <true|false>, "surprise": <1-5>, "coherence": <1-5>, \
"prose_quality": <1-5>, "overall": <1-5>}}"""


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------


@dataclass
class RubricConfig:
    """Knobs for the rubric judge.

    judge_models: ensemble of independent judges; each dimension is aggregated
        across judges by the median (robust to one broken/outlier judge). Use
        models from different providers for a meaningful reliability estimate.
    """

    judge_models: list[str] = field(
        default_factory=lambda: [
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "google/gemini-2.5-flash",
        ]
    )
    temperature: float = 0.0
    max_tokens: int = 400
    concurrency: int = 16
    rubric_version: str = RUBRIC_VERSION


DIMENSIONS = ("surprise", "coherence", "prose_quality", "overall")


@dataclass
class StoryScores:
    story_id: str
    # median over judges (None only if every judge failed for a dimension)
    surprise: float | None
    coherence: float | None
    prose_quality: float | None
    overall: float | None
    twist_present: bool | None  # majority vote over judges
    # judge model id -> its raw parsed dict (or None if it failed)
    by_judge: dict[str, dict | None] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_rubric_reply(raw: str | None) -> dict | None:
    """Extract the trailing JSON object and validate it. Returns a dict with
    keys twist_present(bool), surprise/coherence/prose_quality/overall
    (int 1-5), or None if the reply cannot be parsed/validated."""
    if not raw:
        return None
    matches = _JSON_RE.findall(raw)
    if not matches:
        return None
    # The rubric asks for the JSON on the final line; take the last object.
    for blob in reversed(matches):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        out: dict = {}
        ok = True
        for dim in DIMENSIONS:
            v = data.get(dim)
            if not isinstance(v, (int, float)):
                ok = False
                break
            out[dim] = max(1, min(5, int(round(v))))
        if not ok:
            continue
        tp = data.get("twist_present")
        if isinstance(tp, str):
            tp = tp.strip().lower() in ("true", "yes", "1")
        out["twist_present"] = bool(tp)
        return out
    return None


# -----------------------------------------------------------------------------
# Judging
# -----------------------------------------------------------------------------


async def _judge_one(
    async_client,
    sem: asyncio.Semaphore,
    model: str,
    story: str,
    cfg: RubricConfig,
) -> dict | None:
    async with sem:
        try:
            raw = await call_llm_async(
                async_client=async_client,
                messages=[{"role": "user", "content": RUBRIC_PROMPT.format(story=story)}],
                model=model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        except Exception:
            return None
    return _parse_rubric_reply(raw)


def _aggregate(by_judge: dict[str, dict | None]) -> dict:
    """Median per numeric dimension; majority vote for twist_present. Ignores
    judges that failed (None)."""
    parsed = [d for d in by_judge.values() if d is not None]
    agg: dict = {}
    for dim in DIMENSIONS:
        vals = [d[dim] for d in parsed if dim in d]
        agg[dim] = float(statistics.median(vals)) if vals else None
    tps = [d["twist_present"] for d in parsed if "twist_present" in d]
    agg["twist_present"] = (sum(tps) >= (len(tps) / 2)) if tps else None
    return agg


async def score_story(
    cfg: RubricConfig,
    story_id: str,
    story: str,
    async_client=None,
) -> StoryScores:
    """Score one story with the full judge ensemble."""
    client = async_client or get_async_client_openrouter()
    sem = asyncio.Semaphore(cfg.concurrency)
    by_judge: dict[str, dict | None] = {}
    results = await asyncio.gather(
        *(_judge_one(client, sem, m, story, cfg) for m in cfg.judge_models)
    )
    for m, r in zip(cfg.judge_models, results):
        by_judge[m] = r
    agg = _aggregate(by_judge)
    return StoryScores(
        story_id=story_id,
        surprise=agg["surprise"],
        coherence=agg["coherence"],
        prose_quality=agg["prose_quality"],
        overall=agg["overall"],
        twist_present=agg["twist_present"],
        by_judge=by_judge,
    )


def _score_to_dict(s: StoryScores) -> dict:
    return {
        "story_id": s.story_id,
        "surprise": s.surprise,
        "coherence": s.coherence,
        "prose_quality": s.prose_quality,
        "overall": s.overall,
        "twist_present": s.twist_present,
        "by_judge": s.by_judge,
    }


def _score_from_dict(d: dict) -> StoryScores:
    return StoryScores(
        story_id=d["story_id"],
        surprise=d["surprise"],
        coherence=d["coherence"],
        prose_quality=d["prose_quality"],
        overall=d["overall"],
        twist_present=d["twist_present"],
        by_judge=d.get("by_judge", {}),
    )


async def score_stories(
    cfg: RubricConfig,
    stories: Sequence[dict],  # each: {"id": str, "story": str}
    cache_dir: Path | None = None,
) -> list[StoryScores]:
    """Score many stories. One shared client/semaphore caps concurrency across
    all (story x judge) calls.

    If cache_dir is given, each story's score is written to cache_dir/<id>.json the
    instant it completes (durable), and a re-run reuses any score already on disk
    (resumable -- never re-spends on a story already judged). Mirrors the
    generation pipeline so a kill/crash never loses paid judge calls."""
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(cfg.concurrency)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    async def _one(item: dict) -> StoryScores:
        sid = str(item["id"])
        if cache_dir is not None:
            cached = cache_dir / f"{sid}.json"
            if cached.exists():
                return _score_from_dict(json.loads(cached.read_text()))
        by_judge: dict[str, dict | None] = {}
        results = await asyncio.gather(
            *(_judge_one(client, sem, m, item["story"], cfg) for m in cfg.judge_models)
        )
        for m, r in zip(cfg.judge_models, results):
            by_judge[m] = r
        agg = _aggregate(by_judge)
        scores = StoryScores(
            story_id=sid,
            surprise=agg["surprise"],
            coherence=agg["coherence"],
            prose_quality=agg["prose_quality"],
            overall=agg["overall"],
            twist_present=agg["twist_present"],
            by_judge=by_judge,
        )
        if cache_dir is not None:
            (cache_dir / f"{sid}.json").write_text(
                json.dumps(_score_to_dict(scores), indent=2, ensure_ascii=False)
            )
        return scores

    return list(await asyncio.gather(*(_one(it) for it in stories)))


# -----------------------------------------------------------------------------
# IO
# -----------------------------------------------------------------------------


def save_scores(scores: list[StoryScores], cfg: RubricConfig, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "rubric_scores.json"
    payload = {
        "rubric_version": cfg.rubric_version,
        "judge_models": cfg.judge_models,
        "aggregation": "median (numeric), majority (twist_present)",
        "scores": [
            {
                "story_id": s.story_id,
                "surprise": s.surprise,
                "coherence": s.coherence,
                "prose_quality": s.prose_quality,
                "overall": s.overall,
                "twist_present": s.twist_present,
                "by_judge": s.by_judge,
            }
            for s in scores
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return path
