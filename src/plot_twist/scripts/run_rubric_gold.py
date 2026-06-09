"""Score the public-domain gold-set stories with the fixed-rubric LLM judge.

Reads the fetched gold-set story texts (the human ceiling for the human-vs-LLM
comparison, paper Sec.3), runs the ensemble judge (src.plot_twist.rubric_judge),
and writes per-story scores. This is the first run of the judge at scale; the
smoke runner only checks a twist/predictable/random contrast triple.

Headline per-story number is the geometric mean of SURPRISE and COHERENCE (a good
twist must be high on BOTH; geomean punishes being low on either). PROSE_QUALITY
is a covariate; OVERALL is a held-aside holistic check.

Usage:
    python src/plot_twist/scripts/run_rubric_gold.py configs/plot_twist/rubric_gold.yaml --overwrite [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from pathlib import Path

from src.utils import init_directory, load_config, save_config
from src.plot_twist.rubric_judge import RubricConfig, score_stories, save_scores


def _geomean(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(a * b)


def _load_stories(texts_dir: Path, manifest_path: Path, debug: bool) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    meta = {s["slug"]: s for s in manifest["stories"]}
    stories: list[dict] = []
    for txt in sorted(texts_dir.glob("*.txt")):
        slug = txt.stem
        stories.append(
            {
                "id": slug,
                "story": txt.read_text(encoding="utf-8"),
                "title": meta.get(slug, {}).get("title", slug),
            }
        )
    if not stories:
        raise FileNotFoundError(f"no .txt stories found in {texts_dir}")
    if debug:
        stories = stories[:3]
    return stories


def _fmt(x: float | None) -> str:
    return "  NA" if x is None else f"{x:4.1f}"


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg_dict = load_config(config_path)
    if "output_dir" not in cfg_dict:
        raise ValueError("FATAL: 'output_dir' required in config")
    for field_name in ("judge_models", "texts_dir", "manifest"):
        if field_name not in cfg_dict:
            raise ValueError(f"FATAL: '{field_name}' required in config")

    out = init_directory(cfg_dict["output_dir"], overwrite=overwrite)
    save_config(cfg_dict, out)

    texts_dir = Path(cfg_dict["texts_dir"])
    if not texts_dir.exists():
        raise FileNotFoundError(f"texts_dir not found: {texts_dir}")
    manifest_path = Path(cfg_dict["manifest"])
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    cfg = RubricConfig(
        judge_models=cfg_dict["judge_models"],
        temperature=cfg_dict.get("temperature", 0.0),
        max_tokens=cfg_dict.get("max_tokens", 400),
        concurrency=cfg_dict.get("concurrency", 16),
    )

    stories = _load_stories(texts_dir, manifest_path, debug)
    title = {s["id"]: s["title"] for s in stories}
    print(
        f"scoring {len(stories)} stories x {len(cfg.judge_models)} judges "
        f"(rubric {cfg.rubric_version})"
    )
    print(f"judges: {', '.join(cfg.judge_models)}\n")

    scores = asyncio.run(score_stories(cfg, stories))

    # Loud per-judge failure report: a judge that fails on everything (e.g. a
    # retired model ID) silently shrinks the ensemble, so surface it explicitly
    # BEFORE writing any output.
    n = len(scores)
    print("judge reliability (parsed replies / stories):")
    dead = []
    for m in cfg.judge_models:
        ok = sum(1 for s in scores if s.by_judge.get(m) is not None)
        flag = "  <-- ALL FAILED" if ok == 0 else ("  <-- some failures" if ok < n else "")
        print(f"  {m:<34} {ok}/{n}{flag}")
        if ok == 0:
            dead.append(m)
    print()
    if dead:
        raise RuntimeError(
            f"FATAL: judge(s) failed on every story: {dead}. "
            "Likely a bad/retired model ID or API/auth error -- fix before trusting scores."
        )

    save_scores(scores, cfg, out)

    rows = sorted(
        scores,
        key=lambda s: -(_geomean(s.surprise, s.coherence) or -1.0),
    )
    print(f"{'story':<24} {'pres':>5} {'surp':>5} {'coh':>5} {'prose':>6} {'ovr':>5} {'geo':>6}")
    for s in rows:
        g = _geomean(s.surprise, s.coherence)
        print(
            f"{s.story_id:<24} {str(s.twist_present):>5} "
            f"{_fmt(s.surprise)} {_fmt(s.coherence)} {_fmt(s.prose_quality)} "
            f"{_fmt(s.overall)} {('   NA' if g is None else f'{g:5.2f}')}"
        )

    csv_path = out / "scores.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "slug",
                "title",
                "twist_present",
                "surprise",
                "coherence",
                "prose_quality",
                "overall",
                "geomean_surp_coh",
            ]
        )
        for s in scores:
            w.writerow(
                [
                    s.story_id,
                    title.get(s.story_id, ""),
                    s.twist_present,
                    s.surprise,
                    s.coherence,
                    s.prose_quality,
                    s.overall,
                    _geomean(s.surprise, s.coherence),
                ]
            )

    print(f"\nsaved: {out / 'rubric_scores.json'}\n       {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
