"""LLM span-extraction of reasoning moves -- a semantic replacement for the regex locator.

Per trace (whole trace, no chunking) gpt-4o-mini judges, for each of the 10 reasoning steps,
whether it occurs and -- if so -- copies a SHORT VERBATIM snippet marking it. We then locate that
snippet in the trace ourselves (exact -> whitespace-flexible -> fuzzy) and take its character
offset / trace length as the step's position. So presence (semantic, by the LLM) and position
(by exact string match, NOT LLM character-counting) come from the same faithful source.

Durable + resumable (one cached JSON per trace). Prints the three validations: quote-match rate,
per-step presence + median[IQR] position, and the surprise-before-coherence ordering test.

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/run_move_spans.py [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import glob
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from src.plot_twist.llm import call_llm_async, get_async_client_openrouter
from src.plot_twist.scripts.make_move_positions import MOVES, STORIES

OUT = Path("data/plot_twist/thinking/downstream/move_spans")
STATS = OUT / "move_spans_stats.json"
MODEL = "openai/gpt-4o-mini"

LABELS = [lab for lab, _, _ in MOVES]
SIDE = {lab: side for lab, side, _ in MOVES}
DEFS = {
    "frame the task": "restating the prompt's constraints (length, no title, choosing the subject/characters)",
    "promise coherence": "asserting the twist must be consistent with / prepared by earlier events (fair play, not arbitrary)",
    "list potential twists": "brainstorming or enumerating candidate twists or known tropes",
    "plan setting": "deciding the story's subject, characters, or situation",
    "restate recontextualization goal": "restating that the ending must reframe / recontextualize earlier events",
    "propose, reject, & finalize twist": "weighing candidate twists, rejecting some (too cliche/obvious), settling on the final one",
    "plan clues to plant": "deciding what foreshadowing / clues / hints to plant earlier",
    "choose a reveal event": "choosing the concrete device or event that delivers the reveal (a letter, diary, photo, recording, ...)",
    "outline full plot": "laying out the structure, acts, beats, or scene order",
    "verify it coheres": "checking that, after the twist, the story still holds together / re-reads consistently",
}

PROMPT = """You are analysing the REASONING TRACE a model wrote while planning a short story with a \
plot twist. Below are {k} reasoning STEPS with definitions. For EACH step, decide whether it appears \
anywhere in the trace; if so, copy the SHORTEST verbatim snippet (at most 15 words, copied EXACTLY, \
character-for-character, from the trace) that marks where that step occurs.

STEPS:
{steps}

Return ONLY a JSON object whose keys are the step numbers ("1".."{k}") and whose values are \
{{"present": true|false, "quote": "<exact substring of the trace>" or null}}. The quote MUST be \
copyable verbatim from the trace. If a step is absent, use {{"present": false, "quote": null}}.

TRACE:
\"\"\"{trace}\"\"\""""


def build_prompt(trace: str) -> str:
    steps = "\n".join(f"{i + 1}. {lab} -- {DEFS[lab]}" for i, lab in enumerate(LABELS))
    return PROMPT.format(k=len(LABELS), steps=steps, trace=trace)


def locate(trace: str, quote: str):
    """Return (char_offset, match_kind) for `quote` in `trace`, or (None, 'miss')."""
    if not quote or not quote.strip():
        return None, None
    i = trace.find(quote)
    if i >= 0:
        return i, "exact"
    m = re.search(r"\s+".join(re.escape(w) for w in quote.split()), trace, re.I)  # whitespace-flex
    if m:
        return m.start(), "ws"
    L = max(8, len(quote)); qn = quote.lower(); tl = trace.lower()      # fuzzy sliding window
    best_r, best_i = 0.0, -1
    for s in range(0, max(1, len(trace) - L), max(1, L // 4)):
        r = difflib.SequenceMatcher(None, qn, tl[s:s + L]).ratio()
        if r > best_r:
            best_r, best_i = r, s
    return (best_i, "fuzzy") if best_r >= 0.7 else (None, "miss")


def parse_json(txt: str):
    m = re.search(r"\{.*\}", txt.strip(), re.S)
    return json.loads(m.group(0) if m else txt)


async def extract_one(client, sem, rec) -> dict:
    cache = OUT / f"{rec['id']}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    trace = rec["reasoning_trace"]; L = len(trace)
    async with sem:
        try:
            resp = await call_llm_async(client, [{"role": "user", "content": build_prompt(trace)}],
                                        model=MODEL, temperature=0.0, max_tokens=1000)
            obj, err = parse_json(resp), None
        except Exception as e:
            obj, err = None, f"{type(e).__name__}: {e}"
    steps = {}
    if obj:
        for i, lab in enumerate(LABELS):
            cell = obj.get(str(i + 1)) or {}
            present = bool(cell.get("present"))
            quote = cell.get("quote") if present else None
            pos, match = None, None
            if present and quote:
                idx, match = locate(trace, quote)
                if idx is not None:
                    pos = idx / L
            steps[lab] = {"present": present, "quote": quote, "pos": pos, "match": match, "side": SIDE[lab]}
    out = {"id": rec["id"], "model": rec.get("model"), "level": rec.get("reasoning_level"),
           "error": err, "steps": steps}
    cache.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return out


def validate(results, write_stats):
    mc, present, pos_by = Counter(), Counter(), {lab: [] for lab in LABELS}
    for o in results:
        for lab, c in o.get("steps", {}).items():
            if c.get("present"):
                present[lab] += 1
                if c.get("match"):
                    mc[c["match"]] += 1
                if c.get("pos") is not None:
                    pos_by[lab].append(c["pos"])
    tot = sum(mc.values()) or 1
    print("\n[1] quote-match rate:  " + "  ".join(f"{k}={v} ({100*v//tot}%)" for k, v in mc.most_common()))
    print(f"\n[2] per-step presence + position\n  {'step':<34}{'present':>8}{'medpos':>9}{'IQR':>16}")
    stats = []
    for lab in LABELS:
        ps = np.array(pos_by[lab])
        med, q1, q3 = (float(np.median(ps)), float(np.percentile(ps, 25)), float(np.percentile(ps, 75))) \
            if len(ps) else (float("nan"),) * 3
        print(f"  {lab:<34}{present[lab]:>8}{med:>9.2f}     [{q1:.2f},{q3:.2f}]")
        stats.append({"label": lab, "side": SIDE[lab], "present": present[lab],
                      "med": med, "q1": q1, "q3": q3, "n": len(ps)})
    s_med, c_med = [], []
    for o in results:
        sp = [c["pos"] for lab, c in o["steps"].items() if c.get("pos") is not None and SIDE[lab] == "S"]
        cp = [c["pos"] for lab, c in o["steps"].items() if c.get("pos") is not None and SIDE[lab] == "C"]
        if sp and cp:
            s_med.append(statistics.median(sp)); c_med.append(statistics.median(cp))
    if s_med:
        later = sum(c > s for s, c in zip(s_med, c_med))
        print(f"\n[3] ordering: surprise median {statistics.median(s_med):.2f} vs coherence "
              f"{statistics.median(c_med):.2f}; coherence later in {later}/{len(s_med)} traces")
        try:
            from scipy.stats import wilcoxon
            print(f"    Wilcoxon (coherence > surprise): p={wilcoxon(c_med, s_med, alternative='greater').pvalue:.1e}")
        except Exception:
            pass
    if write_stats:
        STATS.write_text(json.dumps(stats, indent=1))
        print(f"\nsaved -> {STATS}")


def main(debug=False):
    OUT.mkdir(parents=True, exist_ok=True)
    recs = [json.load(open(f)) for f in glob.glob(STORIES)]
    recs = [r for r in recs if (r.get("reasoning_trace") or "").strip()]
    if debug:
        recs = recs[:5]
    done = sum((OUT / f"{r['id']}.json").exists() for r in recs)
    print(f"extracting spans for {len(recs)} traces (model={MODEL}); {done} cached, {len(recs)-done} new")
    client_results = asyncio.run(_run(recs))
    validate(client_results, write_stats=not debug)


async def _run(recs, concurrency=12):
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(concurrency)
    return await asyncio.gather(*[extract_one(client, sem, r) for r in recs])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    main(ap.parse_args().debug)
