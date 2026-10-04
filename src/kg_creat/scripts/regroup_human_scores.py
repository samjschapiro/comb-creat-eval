"""Re-split already-scored human answers into human_kept / human_flagged after a screening verdict changes.

Judging is per answer (the panel never sees the group) and originality is pooled over BOTH groups, so moving a
session between groups needs no rescoring: its records move between the two files in score_input/, scores/ and
combined_scores/, and summaries / the composite are recomputed. Group membership comes from
data/kg_creat/human_study/sessions.csv (run analyze_human_study.py after editing screening_verdicts.csv).

    .venv/bin/python -m src.kg_creat.scripts.regroup_human_scores
"""
import csv
import json
from pathlib import Path

from src.kg_creat.aggregate import aggregate

ROOT = Path("data/kg_creat/human_study")
GROUP = {"kept": "human_kept", "flagged": "human_flagged"}


def main():
    status = {r["session_id"]: r["status"] for r in csv.DictReader(open(ROOT / "sessions.csv"))}
    # sample_idx -> session, from both groups' scorer input (sample_idx is the session's global index)
    session_of = {}
    for g in GROUP.values():
        for r in json.loads((ROOT / "score_input" / g / "responses.json").read_text()):
            session_of[r["sample_idx"]] = r["session_id"]
    target = {idx: GROUP.get(status[s]) for idx, s in session_of.items()}

    for sub, key, fname in (("score_input", "responses", "responses.json"),
                            ("scores", "path_scores", "path_scores.json"),
                            ("combined_scores", "path_scores", "path_scores.json")):
        allrecs = []
        for g in GROUP.values():
            allrecs += json.loads((ROOT / sub / g / fname).read_text())
        for g in GROUP.values():
            recs = [r for r in allrecs if target.get(r.get("sample_idx")) == g]
            (ROOT / sub / g / fname).write_text(json.dumps(recs, indent=1 if sub == "score_input" else 2))
            if key == "path_scores" and (ROOT / sub / g / "summary.json").exists():
                (ROOT / sub / g / "summary.json").write_text(json.dumps(aggregate(recs), indent=2))
            moved = len({r.get("sample_idx") for r in recs})
            print(f"  {sub:16s} {g:14s} {len(recs):5d} records, {moved} sessions")


if __name__ == "__main__":
    main()
