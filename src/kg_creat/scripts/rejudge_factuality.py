"""Re-judge factuality for paths the batched judge left UNJUDGED (channel=='unjudged', factual None).

The main scoring batches 10 paths/call; on models that emit many long association paths, gpt-oss
truncates and returns None -> those paths are marked unjudged and (wrongly) counted as utility
failures. This re-runs factuality in SMALL batches so the judge does not truncate, updates each
model's path_scores.json + summary.json + the pooled scores_summary.json, and records ledger spend.

    .venv_mlx/bin/python -m src.kg_creat.scripts.rejudge_factuality
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.kg_creat import judge as J  # noqa: E402
from src.kg_creat.scripts.score import finalize_sat, finalize_regime_b  # noqa: E402
from src.kg_creat.aggregate import aggregate  # noqa: E402
from src.dat_eval.llm import get_async_client  # noqa: E402

SCORES = Path("data/kg_creat/kombine_test30/scores")
FACT_MODEL = "openai/gpt-oss-120b"
BATCH = 3          # small so the reasoning judge does not truncate on long paths
CONCURRENCY = 12


async def rejudge_model(client, sem, md):
    recs = json.loads((md / "path_scores.json").read_text())
    todo = [r for r in recs
            if r.get("channel") == "unjudged" and r.get("factual") is None
            and r.get("well_formed") and r["mode"] in ("baseline", "analogy")]
    if not todo:
        return None

    async def one_batch(batch):
        async with sem:
            res = await J.judge_factuality_batch(client, FACT_MODEL, [r["triples"] for r in batch])
        for r, v in zip(batch, res):
            if v is not None:
                r["factual"] = v

    await asyncio.gather(*[one_batch(todo[i:i + BATCH]) for i in range(0, len(todo), BATCH)])
    recovered = sum(1 for r in todo if r.get("factual") is not None)

    for r in recs:
        finalize_sat(r)
    finalize_regime_b(recs, None)
    (md / "path_scores.json").write_text(json.dumps(recs, indent=2))
    (md / "summary.json").write_text(json.dumps(aggregate(recs), indent=2))
    return {"model": md.name, "unjudged": len(todo), "recovered": recovered}


async def main():
    client = get_async_client()
    sem = asyncio.Semaphore(CONCURRENCY)
    J.reset_judge_usage()
    mds = sorted(d for d in SCORES.iterdir() if (d / "path_scores.json").exists())
    summaries = {}
    for md in mds:
        out = await rejudge_model(client, sem, md)
        if out:
            print(f"  {out['model']:34s} recovered {out['recovered']}/{out['unjudged']} unjudged")
        summaries[md.name] = json.loads((md / "summary.json").read_text())
    (SCORES / "scores_summary.json").write_text(json.dumps(summaries, indent=2))

    from src.kg_creat.cost_ledger import record
    for jm, u in J.get_judge_usage().items():
        e = record("rejudge", jm, u["calls"], u["in"], u["out"], config="rejudge_factuality")
        c = f"${e['cost_usd']:.4f}" if e["cost_usd"] is not None else "unpriced"
        print(f"  [ledger] {jm}: {u['calls']} calls, {u['in']:,}+{u['out']:,} tok -> {c}")
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
