"""Re-judge a subset of already-scored paths in place, without redoing the whole corpus.

Judge spend dominates this pipeline, so a fix that touches one cell (a prompt change, a token
budget, a model swap) must not force a full re-score. This re-runs the judges only for records
matching ``--mode``/``--channel``, recomputes ``sat``, and rewrites path_scores.json + summary.json.

    .venv_mlx/bin/python src/kg_creat/scripts/rejudge.py data/kg_creat/scores_regimeA_all \
        --responses data/kg_creat/responses_regimeA_all --mode categorical --channel unjudged
"""

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.kg_creat.aggregate import aggregate  # noqa: E402
from src.kg_creat.embed import get_embedder  # noqa: E402
from src.kg_creat.scripts.score import run_judges, finalize_sat, finalize_regime_b  # noqa: E402


async def main(scores_dir, responses_dir, mode=None, channel=None, model="openai/gpt-oss-120b",
               concurrency=32):
    scores_dir, responses_dir = Path(scores_dir), Path(responses_dir)
    embed = get_embedder()
    for md in sorted(d for d in scores_dir.iterdir() if (d / "path_scores.json").exists()):
        recs = json.loads((md / "path_scores.json").read_text())
        responses = json.loads((responses_dir / md.name / "responses.json").read_text())
        by_prompt = {r["prompt_id"]: r for r in responses}
        targets = [r for r in recs
                   if (mode is None or r["mode"] == mode) and (channel is None or r.get("channel") == channel)]
        if not targets:
            print(f"  {md.name}: nothing to re-judge")
            continue
        # Clear the stale verdicts so run_judges' gates see a clean slate.
        for r in targets:
            for k in ("factual", "constraint_sat", "semantic_sat"):
                r.pop(k, None)
        print(f"  {md.name}: re-judging {len(targets)} paths ...")
        await run_judges(targets, by_prompt, model, concurrency)
        for r in targets:
            finalize_sat(r)
        finalize_regime_b(recs, embed)
        (md / "path_scores.json").write_text(json.dumps(
            recs, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))
        (md / "summary.json").write_text(json.dumps(aggregate(recs), indent=2))
        left = sum(1 for r in targets if r.get("channel") == "unjudged")
        print(f"    done — still unjudged: {left}/{len(targets)}")

    summ = {d.name: json.loads((d / "summary.json").read_text())
            for d in sorted(scores_dir.iterdir()) if (d / "summary.json").exists()}
    (scores_dir / "scores_summary.json").write_text(json.dumps(summ, indent=2))
    print(f"rewrote {scores_dir/'scores_summary.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scores_dir")
    p.add_argument("--responses", required=True)
    p.add_argument("--mode")
    p.add_argument("--channel")
    p.add_argument("--judge-model", default="openai/gpt-oss-120b")
    p.add_argument("--concurrency", type=int, default=32)
    a = p.parse_args()
    asyncio.run(main(a.scores_dir, a.responses, a.mode, a.channel, a.judge_model, a.concurrency))
