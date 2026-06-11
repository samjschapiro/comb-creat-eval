"""Classify each twist's MECHANISM into a literature-grounded taxonomy, to study the
semantic content of LLM vs human twists (deeper than opaque embedding clusters).

Taxonomy (one primary mechanism per reveal):
  IDENTITY      a character is secretly someone else / secretly related / the same
                person / an impostor / the true culprit (relational reinterpretation)
  ONTOLOGICAL   a character or the world is fundamentally not what it seemed --
                AI/robot, ghost/already-dead, dream/hallucination, simulation/afterlife
  ORCHESTRATION events were secretly engineered -- faked death, staged disappearance,
                manipulation, a con or a trap
  FACT_OBJECT   a concealed fact or the true nature of an object/situation is revealed
                (the worthless/priceless item, a hidden letter, the real cause)
  NARRATOR      unreliable narrator / the telling itself was deceptive / withheld view
  MORAL         moral or role inversion -- victim is perpetrator, helper is enemy,
                hero is villain (not via explicit orchestration)
  TEMPORAL      time loop, nonlinear time, events were past/future/repeating
  NONE          no genuine twist (predictable or absent reveal)

Durable + resumable (per-id cache); cheap classifier.

Usage: python src/plot_twist/scripts/classify_twists.py
"""

from __future__ import annotations

import asyncio
import collections
import json
from pathlib import Path

import numpy as np

from src.plot_twist.llm import call_llm_async, get_async_client_openrouter
from src.plot_twist.sets import twist_types

ANN = "data/plot_twist/annotations/annotations.json"
MANIFEST = "configs/plot_twist/pd_manifest.json"
CACHE = Path("data/plot_twist/twist_class/cache")
MODEL = "openai/gpt-4o-mini"
CODES = ["IDENTITY", "ONTOLOGICAL", "ORCHESTRATION", "FACT_OBJECT", "NARRATOR", "MORAL", "TEMPORAL", "NONE"]

PROMPT = """Classify the PRIMARY mechanism of this story's plot twist into exactly one code.

IDENTITY: a character is secretly someone else / secretly related / the same person / an impostor / the true culprit.
ONTOLOGICAL: a character or the world is fundamentally not what it seemed -- AI/robot, ghost/already-dead, dream/hallucination, simulation/afterlife.
ORCHESTRATION: events were secretly engineered -- faked death, staged disappearance, manipulation, a con or trap.
FACT_OBJECT: a concealed fact or the true nature of an object/situation is revealed (worthless/priceless item, hidden letter, real cause), not identity or ontology.
NARRATOR: unreliable narrator / the telling itself was deceptive / withheld perspective.
MORAL: moral/role inversion -- victim is perpetrator, helper is enemy, hero is villain (not via explicit orchestration).
TEMPORAL: time loop, nonlinear time, events were past/future/repeating.
NONE: no genuine twist (predictable or absent reveal).

REVEAL: {reveal}

Answer with ONLY the single code (one word)."""


async def _classify_one(client, sem, rec):
    sid = rec["id"]
    p = CACHE / f"{sid}.json"
    if p.exists():
        d = json.loads(p.read_text())
        if d.get("code"):
            return d
    async with sem:
        try:
            raw = await call_llm_async(client, [{"role": "user", "content": PROMPT.format(reveal=rec["reveal"])}],
                                       MODEL, temperature=0.0, max_tokens=8)
        except Exception as exc:
            raw = None
    code = "NONE"
    if raw:
        up = raw.strip().upper()
        code = next((c for c in CODES if c in up), "NONE")
    out = {"id": sid, "code": code}
    p.write_text(json.dumps(out))
    return out


async def main_async(items):
    CACHE.mkdir(parents=True, exist_ok=True)
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(16)
    return await asyncio.gather(*(_classify_one(client, sem, it) for it in items))


def main():
    recs = json.loads(Path(ANN).read_text())
    types = twist_types(MANIFEST)
    items = []
    for r in recs:
        if not r.get("reveal"):
            continue
        src = "human" if r["source"] == "human" else r["source"]
        if src == "human" and types.get(r["id"]) != "STRONG":
            continue
        items.append({"id": r["id"], "reveal": r["reveal"], "source": src,
                      "S": float(r["scores"].get("surprise") or 0), "Coh": float(r["scores"].get("coherence") or 0)})
    codes = {c["id"]: c["code"] for c in asyncio.run(main_async(items))}
    for it in items:
        it["code"] = codes[it["id"]]

    def dist(sel):
        c = collections.Counter(it["code"] for it in sel)
        n = len(sel)
        return {k: c.get(k, 0) / n for k in CODES}, n

    hum = [it for it in items if it["source"] == "human"]
    llm = [it for it in items if it["source"] != "human"]
    hd, hn = dist(hum)
    ld, ln = dist(llm)
    print(f"{'mechanism':<14}{'Human%':>8}{'LLM%':>8}{'LLM/Hum':>9}   mean S*Coh (LLM)")
    for c in sorted(CODES, key=lambda c: -ld[c]):
        sc = [it["S"] * it["Coh"] for it in llm if it["code"] == c]
        ratio = (ld[c] / hd[c]) if hd[c] > 0 else float("inf")
        rs = f"{ratio:.1f}x" if ratio != float("inf") else "  inf"
        print(f"{c:<14}{hd[c]*100:>7.0f}%{ld[c]*100:>7.0f}%{rs:>9}   {np.mean(sc):.1f}" if sc else
              f"{c:<14}{hd[c]*100:>7.0f}%{ld[c]*100:>7.0f}%{rs:>9}   --")
    print(f"(human n={hn}, llm n={ln})")

    # per provider family
    print("\n=== mechanism mix by model family (% ONTOLOGICAL+ORCHESTRATION = 'deus-ex' tropes) ===")
    fam = collections.defaultdict(list)
    for it in llm:
        fam[it["source"].split("/")[0]].append(it)
    fam["[HUMAN]"] = hum
    for f, its in sorted(fam.items(), key=lambda x: -sum(1 for it in x[1] if it["code"] in ("ONTOLOGICAL", "ORCHESTRATION")) / len(x[1])):
        d, _ = dist(its)
        deus = d["ONTOLOGICAL"] + d["ORCHESTRATION"]
        top = max(CODES, key=lambda c: d[c])
        print(f"  {f:<16} n={len(its):<4} deus-ex={deus*100:>3.0f}%  ident={d['IDENTITY']*100:>3.0f}%  fact={d['FACT_OBJECT']*100:>3.0f}%  none={d['NONE']*100:>3.0f}%  top={top}")

    json.dump([{**it, "e": None} for it in items], open("data/plot_twist/twist_class/classified.json", "w"),
              default=str, indent=0)
    print("\nsaved: data/plot_twist/twist_class/classified.json")


if __name__ == "__main__":
    main()
