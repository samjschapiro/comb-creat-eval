"""Locate the plot twist inside each LLM story, for the project page's in-text marker.

A cheap model is asked for the VERBATIM opening of the sentence that delivers the twist;
the runner then matches that string back into the story text. The matched substring is the
artifact — if the model paraphrased instead of quoting, the match fails and the story gets
no marker. A wrong marker is worse than none, so nothing is guessed.

Durable and resumable: results append to locations.jsonl and a re-run never re-spends on a
story already done.

    uv run python src/plot_twist/scripts/run_twist_points.py configs/plot_twist/twist_points.yaml
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, save_config
from src.plot_twist.join import attach_story_text, load_annotations
from src.plot_twist.llm import call_llm_async, get_async_client_openrouter
from src.plot_twist.predict import build_boundary_messages, locate_anchor


def llm_stories(annotations_path: str) -> list[dict]:
    """Every LLM story with resolvable prose. Human gold stories are excluded — their
    twists are hand-annotated in the paper's appendix table."""
    records = [r for r in load_annotations(annotations_path) if r["source"] != "human"]
    stories = attach_story_text(records)
    if not stories:
        raise ValueError("FATAL: no LLM stories with resolvable text")
    return stories


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


async def locate_one(client, sem, story, model, temperature, max_tokens) -> dict:
    """Ask for the verbatim twist opening, then match it back into the story."""
    async with sem:
        rec = {"story_id": story["id"], "source": story["source"]}
        try:
            locator = await call_llm_async(
                client, build_boundary_messages(story["text"]), model,
                temperature=temperature, max_tokens=max_tokens,
            )
            rec["locator"] = locator
            # The model must have QUOTED, not paraphrased: require >=3 words and an
            # actual match in the prose. Anything else yields no anchor.
            anchor = None
            if locator and len(locator.split()) >= 3:
                anchor = locate_anchor(story["text"], " ".join(locator.split()[:12]))
            rec["anchor"] = anchor
            rec["error"] = None
        except Exception as e:  # noqa: BLE001 — recorded per story, never aborts the run
            rec["locator"], rec["anchor"], rec["error"] = None, None, f"{type(e).__name__}: {e}"
        return rec


async def run(worklist, path, model, concurrency, temperature, max_tokens):
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(concurrency)
    tasks = [locate_one(client, sem, s, model, temperature, max_tokens) for s in worklist]
    with open(path, "a") as f:
        located = 0
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            rec = await coro
            located += bool(rec.get("anchor"))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} done, {located} located ({100*located/i:.0f}%)")


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for field in ("annotations_path", "model", "output_dir", "max_calls"):
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")

    output_dir = Path(config["output_dir"])
    if output_dir.exists() and overwrite:
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    jsonl_path = output_dir / "locations.jsonl"

    stories = llm_stories(config["annotations_path"])
    if debug:
        stories = stories[:20]
        print(f"DEBUG MODE: {len(stories)} stories -> {output_dir}")

    done = {r["story_id"] for r in load_jsonl(jsonl_path)}
    todo = [s for s in stories if s["id"] not in done]
    print(f"{len(stories)} LLM stories, {len(done)} already done, {len(todo)} to run.")

    cap = config["max_calls"]
    if len(todo) > cap:
        raise ValueError(f"FATAL: {len(todo)} calls exceeds max_calls={cap}.")

    if todo:
        words = sum(len(s["text"].split()) for s in todo)
        print(f"Planned: {len(todo)} calls, ~{words * 1.35 / 1e6:.2f}M input tokens.")
        asyncio.run(run(
            todo, jsonl_path, config["model"], config.get("concurrency", 16),
            config.get("temperature", 0.0), config.get("max_tokens", 60),
        ))

    # Collapse to the artifact the website consumes: story id -> verbatim anchor.
    records = load_jsonl(jsonl_path)
    anchors = {r["story_id"]: r["anchor"] for r in records if r.get("anchor")}
    (output_dir / "twist_points.json").write_text(json.dumps(anchors, indent=2))

    errored = sum(1 for r in records if r.get("error"))
    print(f"\nLocated {len(anchors)}/{len(records)} stories "
          f"({100 * len(anchors) / max(len(records), 1):.0f}%); {errored} errored.")
    print(f"Unmatched stories get no marker on the site -> {output_dir / 'twist_points.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
