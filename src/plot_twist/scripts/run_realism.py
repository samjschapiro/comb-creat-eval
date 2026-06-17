"""Score every story on REALISM (grounded real-world vs fantastical/speculative) --
a 4th scoring dimension. Single cheap judge by default; durable + resumable.

Scores both LLM-generated stories (from generations.json) and the human gold set
(from the texts dir), so realism enters TC for every source.

Usage:
    python src/plot_twist/scripts/run_realism.py configs/plot_twist/realism.yaml --overwrite [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
from pathlib import Path

from src.utils import init_directory, load_config, save_config
from src.plot_twist.llm import call_llm_async, get_async_client_openrouter

REALISM_PROMPT = """You are a careful literary judge. Rate ONE property of the story: \
REALISM -- whether the story's actual world contains genuinely impossible (supernatural, \
science-fictional, or fantastical) elements, or whether everything that happens could \
occur in the ordinary real world.

DECISIVE TEST: does something genuinely impossible ACTUALLY happen in the story's world? \
Judge by what is real WITHIN the story, NOT by how surreal, vivid, or strange the telling \
is.

A story stays REALISTIC even if it contains any of the following -- do NOT lower the score \
for these:
- a dream, hallucination, vision, fever, or delirium experienced by a character (these are \
real psychological events, even when narrated as if real);
- a hoax, trick, illusion, lie, performance, or staged effect later revealed to be fake \
(e.g. "ghosts" that turn out to be ordinary people);
- vivid, surreal, exaggerated, or figurative prose, or a metaphorical/poetic title;
- an unreliable narrator, a coincidence, or an unusual but physically possible event.

A story is UNREAL only if the genuinely impossible actually occurs in its world: real \
ghosts or an afterlife, magic, time travel, artificial/synthetic beings or robots, aliens, \
a simulated/virtual reality that is literally real, monsters, or psychic/supernatural powers.

Rate 1-5 (use the FULL range):
  1: The central premise or twist depends on something genuinely impossible.
  2: Genuinely impossible elements are clearly present but secondary.
  3: Genuinely ambiguous -- the story leaves it truly unresolved whether something \
impossible occurred.
  4: Realistic, with at most a single minor, deniable brush with the impossible.
  5: Fully realistic -- everything that happens could happen in the real world (dreams, \
hallucinations, hoaxes, and surreal prose included).

Story:
\"\"\"{story}\"\"\"

First, in AT MOST one sentence, state whether anything genuinely impossible actually \
happens in the story's world (vs. only being dreamed, faked, or figurative). Then output a \
single JSON object on its own final line, exactly:
{{"realism": <1-5>}}"""

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse(raw):
    if not raw:
        return None
    for blob in reversed(_JSON_RE.findall(raw)):
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        v = d.get("realism")
        if isinstance(v, (int, float)):
            return max(1, min(5, int(round(v))))
    return None


async def _judge_one(client, sem, model, story, max_tokens):
    async with sem:
        try:
            raw = await call_llm_async(
                client, [{"role": "user", "content": REALISM_PROMPT.format(story=story)}],
                model, temperature=0.0, max_tokens=max_tokens)
        except Exception:
            return None
    return _parse(raw)


async def _score(cfg, items, cache_dir):
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(cfg.get("concurrency", 16))
    judges = cfg["judge_models"]
    mt = cfg.get("max_tokens", 120)

    async def one(it):
        sid = it["id"]
        p = cache_dir / f"{sid}.json"
        # Reuse any per-judge ratings already cached; only call judges still missing
        # (so adding judges to an existing run never re-pays for cached ones).
        cached = {}
        if p.exists():
            try:
                cached = json.loads(p.read_text()).get("by_judge") or {}
            except Exception:
                cached = {}
        todo = [m for m in judges if cached.get(m) is None]
        if todo:
            res = await asyncio.gather(*(_judge_one(client, sem, m, it["story"], mt) for m in todo))
            cached = {**cached, **dict(zip(todo, res))}
        vals = [cached[m] for m in judges if cached.get(m) is not None]
        rec = {"id": sid, "source": it["source"],
               "realism": float(statistics.median(vals)) if vals else None,
               "by_judge": cached}
        if todo:
            p.write_text(json.dumps(rec))
        return rec

    return list(await asyncio.gather(*(one(it) for it in items)))


def main(config_path, overwrite=False, debug=False):
    cfg = load_config(config_path)
    out = init_directory(cfg["output_dir"], overwrite=True) if overwrite else Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out)
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    items = []
    for r in json.loads(Path(cfg["llm_generations"]).read_text()):
        if r.get("story"):
            items.append({"id": r["id"], "source": r["model"], "story": r["story"]})
    for txt in sorted(Path(cfg["human_texts_dir"]).glob("*.txt")):
        items.append({"id": txt.stem, "source": "human", "story": txt.read_text(encoding="utf-8")})
    if debug:
        items = items[:5]
    print(f"scoring realism for {len(items)} stories with {cfg['judge_models']}")

    recs = asyncio.run(_score(cfg, items, cache))
    n_ok = sum(1 for r in recs if r.get("realism") is not None)
    print(f"realism scored {n_ok}/{len(recs)}")
    realism = {r["id"]: r["realism"] for r in recs if r.get("realism") is not None}
    (out / "realism_scores.json").write_text(json.dumps(realism, indent=2))
    print(f"saved: {out/'realism_scores.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    main(a.config_path, a.overwrite, a.debug)
