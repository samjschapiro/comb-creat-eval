"""Count "I can't think of one" (and any assigned item left without an answer) as a FAILED answer, i.e. 0.

score.py only scores answers that exist, so an item a participant skipped simply disappears and cannot lower
their score -- skipping the hard items flatters a person. This copies data/kg_creat/human_study/combined_scores to
combined_scores_skip0 and adds one failed record (utility False, every gated dimension 0, ``_missing``: true) for
every assigned test item with no usable answer:

  humans  per session: the shared items the session was ASSIGNED (from its raw responses file) but has no scored
          answer for (skipped, or emptied by the skip box). Association has no skip box, so nothing is added there.
  models  per model: the shared items with no usable answer (unparsed / empty draws), the same rule.

Then compute_composite.py runs on the copy. This is the main analysis; combined_scores (skips dropped) is the
sensitivity check. The "among passed answers" comparisons are unaffected: added records never pass.

    .venv/bin/python -m src.kg_creat.scripts.build_skip0_scores
"""
import csv
import glob
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/kg_creat/human_study")
SRC, DST = ROOT / "combined_scores", ROOT / "combined_scores_skip0"
RAW = "data/kg_creat/responses"
MODE = {"association": "baseline", "analogy": "analogy", "blending": "blending"}
GROUP = {"kept": "human_kept", "flagged": "human_flagged"}


def failed(mode, pid, sample_idx=None):
    r = {"mode": mode, "prompt_id": pid, "sample_idx": sample_idx, "temperature": None, "path_idx": 0,
         "_missing": True, "triples": [["(no answer)", "", ""]], "sat": False, "R": 0.0, "originality": 0.0,
         "em_originality": 0.0}
    if mode == "analogy":
        r.update(pair_sat=False, invention_utility=False, invention_integration=False)
    if mode == "blending":
        r.update(blend_utility=False, blend_integration=1)
    return r


def answered_keys(recs):
    return {(r.get("sample_idx"), r["mode"], r["prompt_id"]) for r in recs
            if (r["mode"] == "analogy" and "pair_sat" in r) or (r["mode"] != "analogy" and r.get("triples"))}


def main():
    if DST.exists():
        raise SystemExit(f"FATAL: {DST} exists; remove it first (this script never overwrites)")
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("composite.json"))
    shared = {r["prompt_id"] for r in json.loads((SRC / "human_kept" / "path_scores.json").read_text())}

    # humans: assigned test items per session, from the raw files
    status = {r["session_id"]: r["status"] for r in csv.DictReader(open(ROOT / "sessions.csv"))}
    idx_of = {}
    for g in GROUP.values():
        for r in json.loads((ROOT / "score_input" / g / "responses.json").read_text()):
            idx_of[r["session_id"]] = r["sample_idx"]
    assigned = defaultdict(list)                       # group -> [(sample_idx, mode, pid)]
    for f in glob.glob(f"{RAW}/*.json"):
        d = json.load(open(f))
        g = GROUP.get(status.get(d["session_id"]))
        if not g:
            continue
        if d["session_id"] not in idx_of:
            raise SystemExit(f"FATAL: kept/flagged session {d['session_id']} has no scorer input")
        for r in d["responses"]:
            if not r.get("is_control") and r["stimulus_id"] in shared:
                assigned[g].append((idx_of[d["session_id"]], MODE[r["task"]], r["stimulus_id"]))
    for g in GROUP.values():
        path = DST / g / "path_scores.json"
        recs = json.loads(path.read_text())
        have = answered_keys(recs)
        add = [failed(m, p, i) for i, m, p in set(assigned[g]) if (i, m, p) not in have]
        path.write_text(json.dumps(recs + add, indent=2))
        print(f"  {g:14s} +{len(add)} failed records for skipped items {dict(Counter(a['mode'] for a in add))}")

    # models: every shared item, one draw each
    for path in sorted(DST.glob("*/path_scores.json")):
        if path.parent.name in GROUP.values():
            continue
        recs = json.loads(path.read_text())
        have = {(m, p) for _i, m, p in answered_keys(recs)}
        need = {(next(r["mode"] for r in recs if r["prompt_id"] == p), p) for p in shared
                if any(r["prompt_id"] == p for r in recs)}
        need |= {("baseline" if p.startswith("A") else "analogy" if p.startswith("E") else "blending", p)
                 for p in shared if not any(r["prompt_id"] == p for r in recs)}
        add = [failed(m, p) for m, p in need if (m, p) not in have]
        if add:
            path.write_text(json.dumps(recs + add, indent=2))
            print(f"  {path.parent.name:34s} +{len(add)} failed records")
    print(f"wrote {DST}; next: compute_composite.py {DST}")


if __name__ == "__main__":
    main()
