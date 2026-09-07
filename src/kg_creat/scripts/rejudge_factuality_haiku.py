"""Re-judge factuality for EVERY path with claude-haiku-4.5 on the Anthropic key.

This REPLACES the instrument rather than repairing it. The pool was factuality-judged by a single
cheap gpt-oss-120b, which is the weakest measurement in the pipeline: it batches 10 paths per call and
truncated on models emitting many long association paths, silently marking 2,039 paths (11.7%)
"unjudged" -- which the utility rule then counted as FAILURES. gpt-6-astra-flex lost 87% of its
association paths that way and scored 7.4 on a 100-point scale.

Two design points:

  NOT A POOL SUBJECT. claude-haiku-4.5 is already one of the three panel judges and is not itself in
  the 35-model pool, so it can grade every model without self-scoring bias. The other models reachable
  on these keys (gpt-5.6-sol, gpt-6-astra-flex, opus-4.7/4.8, sonnet-4.6, fable-5.1) are all pool
  entries and were rejected for that reason.

  BOTH VERDICTS ARE KEPT. The gpt-oss verdict is preserved on each record as ``factual_gptoss`` (and
  its channel as ``channel_gptoss``) before the new one is written, so the instrument change can be
  measured rather than assumed, and nothing is destroyed.

Batch size 3 -- calibration at that size parsed 20/20 with 780 in / 423 out tokens per call, against
the ~1,900 output tokens gpt-oss was spending at batch 10.

    .venv_mlx/bin/python -m src.kg_creat.scripts.rejudge_factuality_haiku
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.kg_creat import judge as J  # noqa: E402
from src.kg_creat.judge import FACTUALITY_BATCH_PROMPT, format_path, parse_factuality_batch  # noqa: E402
from src.kg_creat.scripts.score import finalize_sat, finalize_regime_b  # noqa: E402
from src.kg_creat.aggregate import aggregate  # noqa: E402

DEFAULT_SCORES = Path("data/kg_creat/kombine_test30/scores")
MODEL = "claude-haiku-4-5-20251001"
LEDGER_KEY = "anthropic/claude-haiku-4.5"
BATCH = 3
CONCURRENCY = 32
OUT_NAME = "factuality_judge_change.json"


async def main(scores_dir: Path, out_path: Path):
    global SCORES
    SCORES = scores_dir
    from anthropic import AsyncAnthropic
    cli = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(CONCURRENCY)
    tok = {"in": 0, "out": 0, "calls": 0, "failed": 0}

    async def one_batch(batch):
        block = "\n".join(f"Path {i + 1}: {format_path(r['triples'])}" for i, r in enumerate(batch))
        async with sem:
            try:
                r = await cli.messages.create(
                    model=MODEL, max_tokens=3000,
                    messages=[{"role": "user",
                               "content": FACTUALITY_BATCH_PROMPT.format(paths_block=block)}])
            except Exception:
                tok["failed"] += len(batch)
                return
        tok["calls"] += 1
        tok["in"] += r.usage.input_tokens
        tok["out"] += r.usage.output_tokens
        txt = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        res = parse_factuality_batch(txt, [len(x["triples"]) for x in batch])
        for rec, v in zip(batch, res or [None] * len(batch)):
            if v is not None:
                rec["factual"] = v
            else:
                tok["failed"] += 1

    mds = sorted(d for d in SCORES.iterdir() if (d / "path_scores.json").exists())
    print(f"re-judging factuality with {MODEL} on {len(mds)} models "
          f"(batch={BATCH}, concurrency={CONCURRENCY})\n")
    delta, summaries, t0 = {}, {}, time.time()
    for md in mds:
        recs = json.loads((md / "path_scores.json").read_text())
        todo = [r for r in recs
                if r.get("well_formed") and r["mode"] in ("baseline", "analogy") and r.get("triples")]
        for r in todo:                       # preserve the old instrument BEFORE overwriting
            r.setdefault("factual_gptoss", r.get("factual"))
            r.setdefault("channel_gptoss", r.get("channel"))
            r["factual"] = None
        before = {m: sum(1 for r in recs if r.get("mode") == m and r.get("channel_gptoss") == "ok")
                  for m in ("baseline", "analogy")}
        await asyncio.gather(*[one_batch(todo[i:i + BATCH]) for i in range(0, len(todo), BATCH)])
        for r in recs:
            finalize_sat(r)
        finalize_regime_b(recs, None)
        after = {m: sum(1 for r in recs if r.get("mode") == m and r.get("channel") == "ok")
                 for m in ("baseline", "analogy")}
        n = {m: sum(1 for r in recs if r.get("mode") == m and r.get("triples")) for m in before}
        (md / "path_scores.json").write_text(json.dumps(recs, indent=2))
        (md / "summary.json").write_text(json.dumps(aggregate(recs), indent=2))
        summaries[md.name] = json.loads((md / "summary.json").read_text())
        delta[md.name] = {m: {"n": n[m], "ok_before": before[m], "ok_after": after[m]} for m in before}
        b, a = before["baseline"], after["baseline"]
        nb = max(n["baseline"], 1)
        print(f"  {md.name:34s} assoc utility {100*b/nb:5.1f}% -> {100*a/nb:5.1f}%  ({a-b:+d} paths)")
    (SCORES / "scores_summary.json").write_text(json.dumps(summaries, indent=2))

    from src.kg_creat.cost_ledger import record
    e = record("rejudge", LEDGER_KEY, tok["calls"], tok["in"], tok["out"], config="rejudge_haiku")
    cost = f"${e['cost_usd']:.2f}" if e["cost_usd"] is not None else "unpriced"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": MODEL, "batch": BATCH, "usage": tok, "per_model": delta}, indent=1))
    print(f"\n{tok['calls']:,} calls, {tok['in']:,}+{tok['out']:,} tok -> {cost}   "
          f"unparsed paths: {tok['failed']}   wall {time.time()-t0:.0f}s")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default=str(DEFAULT_SCORES),
                    help="scores directory to re-judge (pool, or the effort study)")
    ap.add_argument("--out", default=None, help="where to write the before/after comparison")
    a = ap.parse_args()
    sd = Path(a.scores_dir)
    op = Path(a.out) if a.out else sd.parent / "analysis" / OUT_NAME
    asyncio.run(main(sd, op))
