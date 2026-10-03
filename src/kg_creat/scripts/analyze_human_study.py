"""Human generation study (schapiro.ai/kombine): who counts, and how much usable data each item has.

Reads every responses file downloaded from the study's Drive folder (one JSON per session, duplicates from
retried uploads included) and writes:

  sessions.csv  one row per session: time, quick-check tries, AI signals computed here (own trap words,
                honeypot, paste attempts, text that appeared without typing, known AI phrasings), the
                manual screening verdict, and the final status (keep / flagged / excluded)
  coverage.csv  per benchmark item: answered responses, counting kept sessions only and kept + flagged
  summary.json  the headline counts

The manual verdict (data/kg_creat/human_study/screening_verdicts.csv) is authoritative for "no effort" and for the
judgment calls; the computed signals are reported next to it so every exclusion can be checked.

    .venv/bin/python -m src.kg_creat.scripts.analyze_human_study
"""
import csv
import glob
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

RESP = "data/kg_creat/responses"
VERDICTS = "data/kg_creat/human_study/screening_verdicts.csv"
OUT = Path("data/kg_creat/human_study")
EXCLUDE = {"ai_trap", "no_effort"}                                   # rejected on Prolific or due to be
FLAG = {"likely_ai", "suspected_ai", "borderline_no_effort"}        # kept, reported with and without
AI_PHRASES = [r"midday reset", r"chronomeal", r"noonrise", r"imperial lattice", r"scheduled\b.{0,60}\brefuel",
              r"breaks?\b.{0,30}\bovernight fast", r"interrupts?\b.{0,30}\bwork ?day", r"marks? (a|the) transition"]


def item_text(r):
    """All answer text of one item, as one lowercase string."""
    parts = []
    for t in r.get("triples") or []:
        parts += [t.get("head", ""), t.get("relation", ""), t.get("tail", "")]
    for key in ("path_a", "path_b"):
        for t in r.get(key) or []:
            parts += [t.get("head", ""), t.get("relation", ""), t.get("tail", "")]
    for p in r.get("projection") or []:
        parts += list(p.get("source") or []) + list(p.get("image") or [])
    parts += [r.get("blend_name") or "", r.get("generic_space") or ""]
    for s in r.get("structure") or []:
        for key in ("triple", "source", "source_u", "source_v"):
            parts += list(s.get(key) or [])
    return " ".join(str(p) for p in parts).lower()


def answered(r):
    """Did the participant give an answer to this item (not skipped, not empty)?"""
    if r.get("debug_skipped"):
        return False
    if r["task"] == "association":
        return any(t.get("relation") and t.get("tail") for t in r.get("triples") or [])
    if r["task"] == "analogy":
        return bool(r.get("path_a"))
    return not (r.get("skipped") or {}).get("blend") and bool(r.get("blend_name"))


def page_filled(r):
    """Characters the page writes into fields itself (start entity, concept names, blend name in the image column)."""
    u, v = len(r.get("u") or ""), len(r.get("v") or "")
    if r["task"] == "association":
        return u
    if r["task"] == "analogy":
        return u + v
    return u + v + len(r.get("blend_name") or "") * len(r.get("structure") or [])


def signals(d):
    hits, nontyped, phrases = [], 0, set()
    for r in d["responses"]:
        text = item_text(r)
        word = (r.get("trap_word") or "").lower()
        if word and word in text:
            hits.append(word)
        ty = r.get("typing") or {}
        typed_chars = (ty.get("chars") or 0) - page_filled(r)
        if answered(r) and typed_chars > 10 and (ty.get("keys") or 0) < 0.5 * typed_chars:
            nontyped += 1
        phrases |= {p for p in AI_PHRASES if re.search(p, text)}
    return {
        "trap_hits": len(hits), "trap_words": " ".join(hits),
        "honeypot": sum(bool((r.get("honeypot") or "").strip()) for r in d["responses"]),
        "paste": sum(r.get("paste_attempts") or 0 for r in d["responses"]),
        "nontyped_items": nontyped, "ai_phrases": len(phrases),
    }


def minutes(d):
    t0 = datetime.fromisoformat(d["consent"]["consent_timestamp"].replace("Z", "+00:00"))
    t1 = datetime.fromisoformat(d["submitted_at"].replace("Z", "+00:00"))
    return round((t1 - t0).total_seconds() / 60, 1)


def main():
    sessions = {}
    for f in sorted(glob.glob(f"{RESP}/*.json")):           # retried uploads are byte-identical copies
        d = json.load(open(f))
        sessions[d["session_id"]] = d
    verdict = {row["session_id"]: row["flag"] for row in csv.DictReader(open(VERDICTS))}
    missing = set(sessions) - set(verdict)
    assert not missing, f"sessions with no screening verdict: {sorted(missing)}"

    OUT.mkdir(parents=True, exist_ok=True)
    rows, cov = [], {"kept": Counter(), "kept_plus_flagged": Counter()}
    item_task = {}
    for sid, d in sessions.items():
        v = verdict[sid]
        status = "excluded" if v in EXCLUDE else "flagged" if v in FLAG else "kept"
        quiz = {q["task"]: q.get("attempts") for q in d.get("quiz") or []}
        tasks = Counter(r["task"] for r in d["responses"] if not r.get("is_control") and answered(r))
        rows.append({"session_id": sid, "slot": d.get("slot"), "minutes": minutes(d), "verdict": v or "fine",
                     "status": status, **signals(d),
                     "quiz_association": quiz.get("association"), "quiz_analogy": quiz.get("analogy"),
                     "quiz_blending": quiz.get("blending"),
                     "answered_association": tasks["association"], "answered_analogy": tasks["analogy"],
                     "answered_blending": tasks["blending"]})
        for r in d["responses"]:
            if r.get("is_control"):
                continue
            item_task[r["stimulus_id"]] = r["task"]
            if answered(r) and status == "kept":
                cov["kept"][r["stimulus_id"]] += 1
            if answered(r) and status in ("kept", "flagged"):
                cov["kept_plus_flagged"][r["stimulus_id"]] += 1

    with open(OUT / "sessions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(OUT / "coverage.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["stimulus_id", "task", "answered_kept", "answered_kept_plus_flagged"])
        for sid in sorted(item_task):
            w.writerow([sid, item_task[sid], cov["kept"][sid], cov["kept_plus_flagged"][sid]])

    by_task = defaultdict(list)
    for sid, t in item_task.items():
        by_task[t].append(cov["kept"][sid])
    summary = {
        "files": len(glob.glob(f"{RESP}/*.json")), "sessions": len(sessions),
        "status": Counter(r["status"] for r in rows), "verdict": Counter(r["verdict"] for r in rows),
        "median_minutes_kept": sorted(r["minutes"] for r in rows if r["status"] == "kept")[
            len([r for r in rows if r["status"] == "kept"]) // 2],
        "answered_per_item_kept": {t: {"n_items": len(v), "min": min(v), "median": sorted(v)[len(v) // 2],
                                       "max": max(v), "items_at_20": sum(x >= 20 for x in v)}
                                   for t, v in by_task.items()},
        "computed_vs_manual": {
            "trap_hits>=2_but_not_excluded": [r["session_id"] for r in rows if r["trap_hits"] >= 2 and r["status"] != "excluded"],
            "excluded_ai_with_<2_trap_hits": [r["session_id"] for r in rows if r["verdict"] == "ai_trap" and r["trap_hits"] < 2],
        },
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
