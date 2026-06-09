"""Shared set-membership helpers for the plot_twist track.

The human gold set is vetted into twist_type STRONG / BORDERLINE / NONE (recorded
in configs/plot_twist/pd_manifest.json). For the human ceiling we use only the
STRONG stories (genuine reinterpretation twists); LLM-generated stories are always
kept. These helpers let every analysis apply that rule consistently.
"""

from __future__ import annotations

import json
from pathlib import Path


def twist_types(manifest_path: str | Path) -> dict[str, str]:
    """slug -> twist_type for human gold stories that have been vetted."""
    man = json.loads(Path(manifest_path).read_text())
    return {s["slug"]: s["twist_type"] for s in man["stories"] if s.get("twist_type")}


def keep_story(story_id: str, types: dict[str, str], strong_only: bool) -> bool:
    """Keep predicate. LLM stories (ids not in the human manifest) are always kept.
    Human gold stories are kept iff STRONG when strong_only is set."""
    if not strong_only:
        return True
    if story_id in types:          # a vetted human gold story
        return types[story_id] == "STRONG"
    return True                    # LLM-generated (or unvetted) -> keep
