"""Human-grounded coherence-taxonomy judge: classify each invented artifact (analogy h / blend c') as a
coherent concept or one of the failure modes a human rater flagged that the standard panel misses.

    .venv/bin/python -m src.kg_creat.scripts.coherence_taxonomy_judge validate --model openai/gpt-oss-120b
    .venv/bin/python -m src.kg_creat.scripts.coherence_taxonomy_judge full --model openai/o3
"""
import argparse
import asyncio
import glob
import json
from collections import Counter
from pathlib import Path

from src.kg_creat import judge as J
from src.dat_eval.llm import get_async_client

RESP = "data/kg_creat/kombine_test30/responses"
HR = Path("data/kg_creat/kombine_test30/human_review")
OUT = Path("data/kg_creat/kombine_test30/analysis/coherence_taxonomy.json")

PROMPT = """You are judging whether an INVENTED CONCEPT (created by an {mode} of two anchors) is a COHERENT, real idea.
A concept can be structurally valid (the mapping was applied, both inputs contribute) yet still FAIL as a real concept.
Anchors: '{u}' and '{v}'.

{body}

Classify the invention into EXACTLY ONE label:
- "coherent": it works as a real, sensible concept a person could use or picture.
- "categorical_absurdity": it takes one input's properties literally onto the other until the artifact stops being what it is (e.g. a mattress that dives and must surface to breathe; a film reel that exists only in memory and is never recorded).
- "mechanical_substitution": the mapping is applied slot-for-slot so the triples parse grammatically but are semantically empty (e.g. "Revolution death occurs when The Directory").
- "forced_swap": a chain of term-for-term substitutions with no real unifying idea (e.g. Filibuster->Liquidity Lock, Cloture vote->Margin call).
- "vague": too generic or underspecified to be a real concept.
Judge the concept's coherence, NOT its novelty or factual nicety. Return ONLY JSON: {{"label": "...", "reason": "..."}}"""

FAIL = {"categorical_absurdity", "mechanical_substitution", "forced_swap", "vague"}


def _body(mode, it):
    def tri(t):
        return f"({t[0]}, {t[1]}, {t[2]})"
    if mode == "analogy":
        paths = it.get("paths", [])
        pa = J.format_path(paths[0]) if paths else ""
        pb = J.format_path(paths[1]) if len(paths) > 1 else ""
        proj = "\n".join(f"  {tri(p['source'])} -> {tri(p['image'])}"
                         for p in it.get("projection", []) if isinstance(p, dict))
        return (f"Aligned paths:\n  A: {pa}\n  B: {pb}\n"
                f"Projected source concept: {it.get('projected')}\n"
                f"Invention: {it.get('invention')}\nProjection:\n{proj}")
    tags = it.get("tags") or []
    struct = "\n".join(f"  {tri(t)} [{tags[i] if i < len(tags) else '?'}]"
                       for i, t in enumerate(it["paths"][0]) if len(t) == 3)
    return (f"Generic space: {it.get('generic_space')}\nInvention: {it.get('concept')}\n"
            f"Blended structure:\n{struct}")


async def judge_one(client, model, sem, mode, u, v, it):
    prompt = PROMPT.format(mode=mode, u=u, v=v, body=_body(mode, it))
    async with sem:
        raw = await J._ask(client, model, prompt, max_tokens=1200)
    d = J._extract_json(raw) if raw else None
    lab = (d or {}).get("label", "unjudged")
    return lab, (d or {}).get("reason", "")


def load_inventions():
    out = []
    for f in sorted(glob.glob(f"{RESP}/*/responses.json")):
        m = f.split("/")[-2]
        for r in json.load(open(f)):
            if r.get("mode") in ("analogy", "blending") and r.get("items"):
                out.append((m, r["mode"], r.get("u_label"), r.get("v_label"), r["prompt_id"], r["items"][0]))
    return out


async def main_async(mode_arg, model, limit):
    client = get_async_client()
    sem = asyncio.Semaphore(12)
    J.reset_judge_usage()
    if mode_arg == "validate":
        items = {i["id"]: i for i in json.load(open(HR / "items.json"))}
        key = json.load(open(HR / "key.json"))
        rated = [json.loads(l) for l in (HR / "ratings.jsonl").read_text().splitlines() if l.strip()]
        # rebuild the full item (with paths/projection) from the response files, via key model+prompt
        resp = {}
        for f in glob.glob(f"{RESP}/*/responses.json"):
            mm = f.split("/")[-2]
            for r in json.load(open(f)):
                if r.get("items"):
                    resp[(mm, r["mode"], r["prompt_id"])] = (r.get("u_label"), r.get("v_label"), r["items"][0])
        tasks = []
        meta = []
        for rr in rated:
            rid = rr["id"]; k = key[rid]; task = rr["task"]
            ent = resp.get((k["model"], task, k["prompt_id"]))
            if not ent:
                continue
            u, v, it = ent
            tasks.append(judge_one(client, model, sem, task, u, v, it))
            meta.append((rid, task, rr["ratings"]))
        res = await asyncio.gather(*tasks)
        # agreement: judge-fail vs human coherent==0
        tp = fp = tn = fn = 0
        rows = []
        for (rid, task, hr), (lab, reason) in zip(meta, res):
            human_bad = hr.get("coherent") == 0
            judge_bad = lab in FAIL
            rows.append((task, human_bad, lab))
            if human_bad and judge_bad: tp += 1
            elif human_bad and not judge_bad: fn += 1
            elif not human_bad and judge_bad: fp += 1
            else: tn += 1
        print(f"validation ({model}) vs your coherent=0 on {len(rows)} items:")
        print(f"  human-bad caught (recall): {tp}/{tp+fn}   |   flagged-but-human-ok (FP): {fp}/{fp+tn}")
        print(f"  judge label distribution: {dict(Counter(l for _,_,l in rows))}")
    else:
        inv = load_inventions()
        if limit:
            inv = inv[:limit]
        res = await asyncio.gather(*[judge_one(client, model, sem, mode, u, v, it)
                                     for (m, mode, u, v, pid, it) in inv])
        recs = []
        for (m, mode, u, v, pid, it), (lab, reason) in zip(inv, res):
            name = it.get("invention") if mode == "analogy" else it.get("concept")
            recs.append({"model": m, "mode": mode, "u": u, "v": v, "prompt_id": pid,
                         "invention": name, "label": lab, "reason": reason})
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(recs, indent=1))
        print(f"judged {len(recs)} inventions -> {OUT}")
        print(f"label distribution: {dict(Counter(r['label'] for r in recs))}")
    from src.kg_creat.cost_ledger import record
    for jm, u in J.get_judge_usage().items():
        e = record("coherence_taxonomy", jm, u["calls"], u["in"], u["out"], config=mode_arg)
        c = f"${e['cost_usd']:.4f}" if e["cost_usd"] is not None else "unpriced"
        print(f"  [ledger] {jm}: {u['calls']} calls, {u['in']:,}+{u['out']:,} tok -> {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validate", "full"])
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(main_async(a.mode, a.model, a.limit))
