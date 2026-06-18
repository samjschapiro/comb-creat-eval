"""Shared join + scoring helpers for the plot_twist experiments.

`annotations.json` is the unified per-story table (id, source, scores{surprise,
coherence, prose_quality, overall, twist_present}, reveal, ...), but it does NOT
carry the story text. Experiments that need the full story (DSI in Exp 2, the SBV
extractor in Exp 3) join each annotation back to its text:

  - LLM stories:  data/plot_twist/llm_twists/stories/<model_key>/<id>.json  (key "story")
  - human gold:   data/plot_twist/human_twists/texts/<id>.txt

This module also factors out the headline TC composite (`overall_eq`) so every
experiment scores models identically to make_tc_barplot.py: the equal-weight mean
of z-scored facets (surprise, coherence, diversity), NOT the tc=Div*mean(S*Coh)
product. See docs memory "plot-twist-headline-metric".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Default on-disk locations (override in callers/configs if a run uses other dirs).
ANNOTATIONS_JSON = "data/plot_twist/annotations/annotations.json"
LLM_STORIES_DIR = "data/plot_twist/llm_twists/stories"
HUMAN_TEXTS_DIR = "data/plot_twist/human_twists/texts"

EQ_FACETS = ("mean_surprise", "mean_coherence", "div")


def load_annotations(path: str | Path = ANNOTATIONS_JSON) -> list[dict]:
    """Load the unified per-story annotation table."""
    return json.loads(Path(path).read_text())


def _story_text(rec: dict, llm_stories_dir: Path, human_texts_dir: Path) -> str | None:
    """Resolve one annotation record to its full story text, or None if missing."""
    sid, source = rec["id"], rec["source"]
    if source == "human":
        p = human_texts_dir / f"{sid}.txt"
        return p.read_text(encoding="utf-8") if p.exists() else None
    # LLM: the story file lives under a per-model-key subfolder (id prefix before "__").
    mkey = sid.split("__")[0]
    p = llm_stories_dir / mkey / f"{sid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("story")


def attach_story_text(
    records: list[dict],
    llm_stories_dir: str | Path = LLM_STORIES_DIR,
    human_texts_dir: str | Path = HUMAN_TEXTS_DIR,
) -> list[dict]:
    """Return only the records whose story text could be resolved, each with a new
    `text` key added. Records with no text on disk are dropped (loud count by caller)."""
    llm_stories_dir = Path(llm_stories_dir)
    human_texts_dir = Path(human_texts_dir)
    out = []
    for r in records:
        txt = _story_text(r, llm_stories_dir, human_texts_dir)
        if txt:
            out.append({**r, "text": txt})
    return out


def score_num(rec: dict, key: str) -> float | None:
    """Pull a numeric rubric score out of an annotation record, or None if invalid."""
    v = rec.get("scores", {}).get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean_pairwise_distance(emb: np.ndarray) -> float:
    """Diversity = 1 - mean off-diagonal cosine, over L2-normalized embedding rows.
    Matches make_tc_barplot._div so diversity is computed identically everywhere."""
    if len(emb) < 2:
        return float("nan")
    sims = emb @ emb.T
    n = len(emb)
    return float(1.0 - (sims.sum() - np.trace(sims)) / (n * (n - 1)))


def add_overall_eq(rows: list[dict], facets: tuple[str, ...] = EQ_FACETS) -> list[dict]:
    """Add `overall_eq` (equal-weight mean of z-scored facets) to each row in place.

    Each row must already carry the raw facet values (default: mean_surprise,
    mean_coherence, div). z-scoring is across the rows passed in, so callers decide
    the reference set (e.g. all 74 sources, or the thinking-intervention cells)."""
    zs = {}
    for k in facets:
        v = np.array([d[k] for d in rows], dtype=float)
        zs[k] = (float(v.mean()), float(v.std()) or 1.0)
    for d in rows:
        d["overall_eq"] = float(np.mean([(d[k] - zs[k][0]) / zs[k][1] for k in facets]))
    return rows
