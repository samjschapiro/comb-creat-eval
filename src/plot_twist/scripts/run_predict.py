"""Predict-the-twist runner: do models recover a story's withheld plot twist?

Two stages, both durable/resumable (no re-spend on re-run):

  1. Boundary: for each story, a cheap model locates where the twist begins; we split
     the full prose there to get the pre-twist text the reader would have. -> boundaries.jsonl
  2. Predict: each predictor model reads that pre-twist prose and predicts the twist;
     we score every guess by embedding similarity to the true reveal. -> predictions.jsonl

    uv run python src/plot_twist/scripts/run_predict.py configs/plot_twist/predict.yaml
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, save_config
from src.plot_twist.llm import (
    call_llm_async,
    get_async_client_openrouter,
    model_id_to_key,
)
from src.plot_twist.predict import (
    build_boundary_messages,
    build_predict_messages,
    cosine_similarity_scores,
    split_at_twist,
)


def load_stories(annotations_path, generations_path, n_stories, seed) -> list[dict]:
    """Stories with a setup+reveal annotation joined to their full prose, optionally sampled."""
    ann = json.load(open(annotations_path))
    ann = [a for a in ann if a.get("setup") and a.get("reveal")]
    story_text = {g["id"]: g["story"] for g in json.load(open(generations_path)) if g.get("story")}
    joined = [{**a, "story": story_text[a["id"]]} for a in ann if a["id"] in story_text]
    if not joined:
        raise ValueError("FATAL: no stories with both annotation and full prose")
    if n_stories and n_stories < len(joined):
        joined = random.Random(seed).sample(joined, n_stories)
    return joined


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


# --- Stage 1: boundaries ---

async def boundary_one(client, sem, story, model, max_tokens) -> dict:
    """Locate the twist and split off the pre-twist prose for one story."""
    async with sem:
        rec = {"story_id": story["id"]}
        try:
            locator = await call_llm_async(
                client, build_boundary_messages(story["story"]), model,
                temperature=0.0, max_tokens=max_tokens,
            )
            rec["locator"] = locator
            rec["pre_reveal"] = split_at_twist(story["story"], locator)
            rec["error"] = None
        except Exception as e:  # noqa: BLE001 — capture per story; report yield, don't crash run
            rec["pre_reveal"] = None
            rec["error"] = str(e)
        return rec


async def run_boundaries(stories, path, model, concurrency, max_tokens):
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(concurrency)
    tasks = [boundary_one(client, sem, s, model, max_tokens) for s in stories]
    with open(path, "a") as f:
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            f.write(json.dumps(await coro) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(tasks):
                print(f"  boundary {i}/{len(tasks)}")


# --- Stage 2: predictions ---

async def predict_one(client, sem, model, story, pre_reveal, temperature, max_tokens) -> dict:
    async with sem:
        rec = {
            "predictor": model,
            "predictor_key": model_id_to_key(model),
            "story_id": story["id"],
            "generator": story["source"],
            "reveal": story["reveal"],
        }
        try:
            rec["prediction"] = await call_llm_async(
                client, build_predict_messages(pre_reveal), model,
                temperature=temperature, max_tokens=max_tokens,
            )
            rec["error"] = None
        except Exception as e:  # noqa: BLE001
            rec["prediction"] = None
            rec["error"] = str(e)
        return rec


async def run_predictions(worklist, path, concurrency, temperature, max_tokens):
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(concurrency)
    tasks = [predict_one(client, sem, m, s, pre, temperature, max_tokens) for m, s, pre in worklist]
    with open(path, "a") as f:
        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            f.write(json.dumps(await coro) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(tasks):
                print(f"  predict {i}/{len(tasks)}")


def score_and_summarize(preds_path, embed_model, output_dir):
    records = load_jsonl(preds_path)
    ok = [r for r in records if r.get("prediction")]
    if not ok:
        raise ValueError("FATAL: no successful predictions to score")
    sims = cosine_similarity_scores(
        [r["prediction"] for r in ok], [r["reveal"] for r in ok], embed_model
    )
    for r, s in zip(ok, sims):
        r["similarity"] = s
    json.dump(ok, open(output_dir / "predictions_scored.json", "w"), indent=2)

    by_predictor: dict[str, list[float]] = {}
    for r in ok:
        by_predictor.setdefault(r["predictor"], []).append(r["similarity"])
    summary = {
        m: {"mean_similarity": sum(v) / len(v), "n": len(v)} for m, v in by_predictor.items()
    }
    summary = dict(sorted(summary.items(), key=lambda kv: -kv[1]["mean_similarity"]))
    json.dump(summary, open(output_dir / "summary.json", "w"), indent=2)

    print(f"\nScored {len(ok)} predictions ({len(records) - len(ok)} errored). "
          "Per-predictor mean similarity to true twist:")
    for m, s in summary.items():
        print(f"  {s['mean_similarity']:.3f}  (n={s['n']:>4})  {m}")


def guard(planned, cap, label):
    print(f"{label}: {planned} calls to run (cap {cap}).")
    if planned > cap:
        raise ValueError(f"FATAL: {planned} {label} calls exceeds max_calls={cap}.")


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for field in ("annotations_path", "generations_path", "predictor_models",
                  "boundary_model", "embed_model"):
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")

    output_dir = Path(config["output_dir"])
    if output_dir.exists() and overwrite:
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    bnd_path = output_dir / "boundaries.jsonl"
    preds_path = output_dir / "predictions.jsonl"

    stories = load_stories(
        config["annotations_path"], config["generations_path"],
        config.get("n_stories"), config.get("seed", 0),
    )
    if debug:
        stories = stories[:5]
        print(f"DEBUG MODE: {len(stories)} stories -> {output_dir}")

    max_calls = config.get("max_calls", 10 ** 9)

    # Stage 1: boundaries (skip stories already located).
    done_bnd = {r["story_id"] for r in load_jsonl(bnd_path)}
    todo = [s for s in stories if s["id"] not in done_bnd]
    guard(len(todo), max_calls, "boundary")
    if todo:
        asyncio.run(run_boundaries(
            todo, bnd_path, config["boundary_model"],
            config.get("concurrency", 16), config.get("boundary_max_tokens", 60),
        ))

    boundaries = {r["story_id"]: r for r in load_jsonl(bnd_path)}
    valid = [s for s in stories if boundaries.get(s["id"], {}).get("pre_reveal")]
    print(f"Boundary yield: {len(valid)}/{len(stories)} stories located.")

    # Stage 2: predictions on stories with a valid boundary (skip done pairs).
    done_pred = {(r["predictor_key"], r["story_id"]) for r in load_jsonl(preds_path)}
    worklist = [
        (m, s, boundaries[s["id"]]["pre_reveal"])
        for m in config["predictor_models"]
        for s in valid
        if (model_id_to_key(m), s["id"]) not in done_pred
    ]
    guard(len(worklist), max_calls, "predict")
    if worklist:
        asyncio.run(run_predictions(
            worklist, preds_path, config.get("concurrency", 16),
            config.get("temperature", 0.0), config.get("max_tokens", 200),
        ))

    score_and_summarize(preds_path, config["embed_model"], output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
