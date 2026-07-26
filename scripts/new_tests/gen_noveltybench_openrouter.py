"""Generate NoveltyBench responses via OpenRouter with EXPLICIT sampling control.

Writes generations.jsonl in the exact format novelty-bench's partition.py/score.py expect:
    {"id":..., "prompt":..., "model":..., "generations":[...k...]}

Sampling is pinned to the paper's protocol: temperature=1.0, and nucleus/top-k
truncation explicitly DISABLED (top_p=1.0, top_k=0) -- the silent provider/HF
defaults are what previously suppressed diversity.

Usage:
  python3 gen_openrouter.py --model meta-llama/llama-3.1-8b-instruct \
      --data curated --out results/curated/llama-or --k 10 [--max-prompts N] [--concurrency 24]
"""
import argparse, asyncio, json, os, sys
from datasets import load_dataset
from openai import AsyncOpenAI


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="curated", choices=["curated", "wildchat"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-prompts", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--concurrency", type=int, default=24)
    a = ap.parse_args()

    ds = load_dataset("yimingzhang/novelty-bench", split=a.data)
    if a.max_prompts:
        ds = ds.shuffle(seed=0).select(range(min(a.max_prompts, len(ds))))
    print(f"{len(ds)} prompts from {a.data}", flush=True)

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"], timeout=180)
    sem = asyncio.Semaphore(a.concurrency)
    empties = 0

    async def one(prompt):
        nonlocal empties
        for attempt in range(10):
            try:
                async with sem:
                    r = await client.chat.completions.create(
                        model=a.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,          # paper protocol
                        top_p=1.0,                # disable nucleus truncation
                        max_tokens=a.max_tokens,
                        n=1,
                        extra_body={"top_k": 0},  # disable top-k truncation
                    )
                if r.choices and r.choices[0].message.content:
                    return r.choices[0].message.content
                wait = 1.0 + attempt           # empty/error-shaped 200
            except Exception as e:
                # 429 engine_overloaded and transient provider errors: back off and retry.
                # Exponential with cap; providers recover within seconds-minutes.
                wait = min(60, 3 * (2 ** attempt))
            await asyncio.sleep(wait)
        empties += 1
        return ""

    async def for_prompt(row):
        gens = await asyncio.gather(*(one(row["prompt"]) for _ in range(a.k)))
        return {"id": row["id"], "prompt": row["prompt"], "model": a.model,
                "generations": list(gens)}

    os.makedirs(a.out, exist_ok=True)
    results = []
    tasks = [for_prompt(r) for r in ds]
    for i, fut in enumerate(asyncio.as_completed(tasks), 1):
        results.append(await fut)
        if i % 25 == 0 or i == len(tasks):
            print(f"  {i}/{len(tasks)} prompts", flush=True)

    order = {pid: n for n, pid in enumerate(ds["id"])}
    results.sort(key=lambda r: order[r["id"]])
    with open(os.path.join(a.out, "generations.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(results)} prompts; empty generations after retries: {empties}")


if __name__ == "__main__":
    asyncio.run(main())
