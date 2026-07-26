"""Top-up regeneration for prompts with any empty/missing samples.
Regenerates a fresh k=10 for each incomplete prompt (same sampling protocol) and
writes back in place, preserving order. Safe to re-run (idempotent-ish)."""
import asyncio, json, os, sys
from openai import AsyncOpenAI

FILE = sys.argv[1]
MODEL = sys.argv[2]
K = 10
CONC = 8

def incomplete(gens):
    return len(gens) < K or any((not g) or (not g.strip()) for g in gens)

async def main():
    rows = [json.loads(l) for l in open(FILE)]
    todo = [i for i, r in enumerate(rows) if incomplete(r["generations"])]
    print(f"{len(rows)} prompts; {len(todo)} incomplete -> regenerating fresh k={K}", flush=True)
    if not todo:
        print("nothing to do"); return

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"], timeout=180)
    sem = asyncio.Semaphore(CONC)
    still_empty = 0

    async def one(prompt):
        nonlocal still_empty
        for attempt in range(12):
            try:
                async with sem:
                    r = await client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0, top_p=1.0, max_tokens=512, n=1,
                        extra_body={"top_k": 0},
                    )
                if r.choices and r.choices[0].message.content:
                    return r.choices[0].message.content
                wait = 1.0 + attempt
            except Exception:
                wait = min(60, 3 * (2 ** attempt))
            await asyncio.sleep(wait)
        still_empty += 1
        return ""

    async def redo(i):
        prompt = rows[i]["prompt"]
        rows[i]["generations"] = list(await asyncio.gather(*(one(prompt) for _ in range(K))))

    # process in chunks so we can report progress
    done = 0
    for start in range(0, len(todo), 25):
        chunk = todo[start:start+25]
        await asyncio.gather(*(redo(i) for i in chunk))
        done += len(chunk)
        print(f"  {done}/{len(todo)} regenerated", flush=True)

    with open(FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    remaining = sum(1 for r in rows if incomplete(r["generations"]))
    print(f"done. still-empty completions this pass: {still_empty}; prompts still incomplete: {remaining}")

if __name__ == "__main__":
    asyncio.run(main())
