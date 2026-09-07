"""Re-judge every blend by the SHARED-PROPERTY integration criterion: does at least one property of the
invented concept receive organizing structure from BOTH inputs (genuine double-scope fusion, e.g. the
Liquid Franchise where Democracy's 'allocates votes' and Banking's 'allocates credit' both map onto
'allocates vote-shares')? 3-judge panel (haiku-4.5, gpt-5.4, o3), majority vote.

    .venv/bin/python -m src.kg_creat.scripts.shared_property_judge
"""
import asyncio
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.kg_creat import judge as J
from src.dat_eval.llm import get_async_client

RESP = "data/kg_creat/kombine_test30/responses"
OUT = Path("data/kg_creat/kombine_test30/analysis/shared_property.json")
PANEL = ["anthropic/claude-haiku-4.5", "openai/gpt-5.4", "openai/o3"]

PROMPT = """You are judging whether a conceptual BLEND achieves genuine double-scope integration.

A genuine blend fuses two inputs so that BOTH inputs contribute organizing structure to the SAME property
of the invented concept -- one slot carries both frames. Canonical good example: blending Democracy +
Banking into 'Liquid Franchise', where Democracy's "allocates votes" AND Banking's "allocates credit"
both map onto one property, "allocates vote-shares".

A blend FAILS this criterion in one of two ways:
- CONCATENATION: the invention hangs some properties from input 1 and some from input 2 on a shared name,
  but NO single property is fed by both (e.g. "vote bank": (elects, leaders) from Democracy and (pays
  interest on, deposits) from Banking -- separate slots, never fused).
- CATEGORICAL ABSURDITY: the invention grafts one input's LITERAL properties onto the other until the
  artifact stops being what it is (e.g. a mattress that "eats krill" and "breathes through a blowhole").

Now judge this blend.
Input 1 (u): '{u}'
Input 2 (v): '{v}'
Generic space: {g}
Invented concept: {concept}
Blended structure (each triple tagged u / v / emergent):
{structure}

Decide: is there AT LEAST ONE property of the invention onto which BOTH inputs contribute organizing
structure (a genuine shared slot), rather than the inputs occupying separate slots? Ignore mere novelty
or factual nicety; judge only whether the two input structures are fused on a shared property.
Return ONLY JSON: {{"integrated": true or false, "shared_property": "name the property both feed, or none",
"failure": "none" or "concatenation" or "absurdity", "reason": "one sentence"}}"""


def _structure(it):
    tags = it.get("tags") or []
    return "\n".join(f"  ({t[0]}, {t[1]}, {t[2]}) [{tags[i] if i < len(tags) else '?'}]"
                     for i, t in enumerate(it["paths"][0]) if len(t) == 3)


async def judge_blend(client, sem, u, v, it):
    prompt = PROMPT.format(u=u, v=v, g=it.get("generic_space"), concept=it.get("concept"),
                           structure=_structure(it))
    async def one(model):
        async with sem:
            raw = await J._ask(client, model, prompt, max_tokens=1200)
        return J._extract_json(raw) if raw else None
    outs = await asyncio.gather(*[one(m) for m in PANEL])
    votes = [bool(o.get("integrated")) for o in outs if isinstance(o, dict) and "integrated" in o]
    verdict, agree = J._majority(votes)
    judges = [{"model": m, **{k: (o or {}).get(k) for k in ("integrated", "shared_property", "failure", "reason")}}
              for m, o in zip(PANEL, outs)]
    return verdict, agree, judges


async def main():
    client = get_async_client()
    sem = asyncio.Semaphore(10)
    J.reset_judge_usage()
    blends = []
    for f in sorted(glob.glob(f"{RESP}/*/responses.json")):
        m = f.split("/")[-2]
        for r in json.load(open(f)):
            if r.get("mode") == "blending" and r.get("items"):
                blends.append((m, r.get("u_label"), r.get("v_label"), r["prompt_id"], r["items"][0]))
    print(f"judging {len(blends)} blends with panel {PANEL} ...")
    res = await asyncio.gather(*[judge_blend(client, sem, u, v, it) for (m, u, v, pid, it) in blends])
    recs = []
    for (m, u, v, pid, it), (verdict, agree, judges) in zip(blends, res):
        recs.append({"model": m, "u": u, "v": v, "prompt_id": pid, "concept": it.get("concept"),
                     "integrated": verdict, "agreement": agree, "judges": judges})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(recs, indent=1))
    n = len(recs); yes = sum(1 for r in recs if r["integrated"] is True)
    print(f"\nGENUINE shared-property integration: {yes}/{n} ({100*yes/n:.0f}%)")
    unan = sum(1 for r in recs if r.get("agreement") == 1.0)
    print(f"panel unanimous: {unan}/{n} ({100*unan/n:.0f}%)")
    # per-model rate
    bym = defaultdict(lambda: [0, 0])
    for r in recs:
        bym[r["model"]][1] += 1; bym[r["model"]][0] += 1 if r["integrated"] else 0
    print("\nper-model integration rate (top/bottom):")
    order = sorted(bym.items(), key=lambda kv: -kv[1][0] / kv[1][1])
    for m, (a, b) in order[:5] + [("...", (0, 1))] + order[-5:]:
        if m == "...":
            print("  ...")
        else:
            print(f"  {m.split('_',1)[1]:22s} {100*a/b:3.0f}%  ({a}/{b})")
    print(f"\nsaved -> {OUT}")
    from src.kg_creat.cost_ledger import record
    for jm, u in J.get_judge_usage().items():
        e = record("shared_property", jm, u["calls"], u["in"], u["out"], config="blend_integration")
        c = f"${e['cost_usd']:.4f}" if e["cost_usd"] is not None else "unpriced"
        print(f"  [ledger] {jm}: {u['calls']} calls, {u['in']:,}+{u['out']:,} tok -> {c}")


if __name__ == "__main__":
    asyncio.run(main())
