"""Draft the human study's originality bonus (the study banner: "your bonus depends on how original and unusual
your responses are compared with other participants'").

Per kept participant, over every test item they were assigned (5 per task):
  answer quality gate  the answer passes utility (association: every triple factual; analogy: a valid pair and a
                       coherent invention, V(h) = 1; blending: a valid generic space and a coherent blend, V(c') = 1)
  originality          against OTHER PARTICIPANTS only: the mean, over the answer's non-anchor elements, of the mean
                       cosine distance to the k = 5 nearest elements among the other kept participants' answers to
                       the same item (score.py's rho, with a human-only pool; MiniLM, as in the paper)
  bonus score          the mean over assigned items of (originality if the answer passes the gate, else 0); skipped
                       and failed items count 0, so a participant cannot gain by skipping hard items
Writes data/kg_creat/human_study/compensation/originality_bonus_draft.csv (local; keyed by session_id, since the
Prolific IDs live in the Drive identity folder).

    .venv_mlx/bin/python -m src.kg_creat.scripts.draft_originality_bonus
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.score import _artifact_elements

HUMAN = "data/kg_creat/human_study/score_input/human_kept/responses.json"
SCORES = "data/kg_creat/human_study/paper_scores/human_kept/path_scores.json"
RAW = "data/kg_creat/responses"
OUT = Path("data/kg_creat/human_study/compensation/originality_bonus_draft.csv")
K = 5
MODE = {"association": "baseline", "analogy": "analogy", "blending": "blending"}


def main():
    hum = [r for r in json.load(open(HUMAN)) if not r.get("is_control")]
    # quality verdicts per (sample_idx, prompt_id, item index)
    ok = {}
    for r in json.load(open(SCORES)):
        if r.get("_missing"):
            continue
        key = (r.get("sample_idx"), r["prompt_id"])
        if r["mode"] == "analogy" and r["path_idx"] == 0 and "pair_sat" in r:
            ok[key + (0,)] = r.get("pair_sat") is True and bool(r.get("invention_utility"))
        elif r["mode"] == "blending" and r["path_idx"] == 0:
            ok[key + (0,)] = r.get("sat") is True and bool(r.get("blend_utility"))
        elif r["mode"] == "baseline":
            ok[key + (r["path_idx"],)] = r.get("sat") is True
    # answers: (session, sample_idx, prompt_id, item, element set)
    answers = []
    for r in hum:
        for i, it in enumerate(r.get("items") or []):
            els = {s for _k, s in _artifact_elements(it.get("paths") or [], r["u_label"], r["v_label"])}
            if els:
                answers.append((r["session_id"], r["sample_idx"], r["prompt_id"], r["mode"], i, els))
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    vec = {}
    for a in answers:
        for s in a[5]:
            if s not in vec:
                v = np.asarray(embed(s), float)
                vec[s] = v / (np.linalg.norm(v) or 1.0)
    by_item = defaultdict(list)
    for a in answers:
        by_item[a[2]].append(a)
    item_score = defaultdict(dict)   # session -> {(prompt_id, item): gated originality}
    for pid, rows in by_item.items():
        for a in rows:
            pool = sorted(set().union(*[b[5] for b in rows if b[0] != a[0]]) - a[5]) if len(rows) > 1 else []
            if len(pool) < K or not ok.get((a[1], pid, a[4]), False):
                rho = 0.0 if ok.get((a[1], pid, a[4]), False) is False else None
            else:
                P = np.stack([vec[s] for s in pool])
                rho = float(np.mean([np.sort(1 - P @ vec[s])[:K].mean() for s in a[5]]))
            item_score[a[0]][(pid, a[4])] = (rho, ok.get((a[1], pid, a[4]), False))
    # assigned test items per session (from the raw files), so skipped items count 0
    assigned = defaultdict(set)
    for f in glob.glob(f"{RAW}/*.json"):
        d = json.load(open(f))
        for r in d["responses"]:
            if not r.get("is_control"):
                assigned[d["session_id"]].add((MODE[r["task"]], r["stimulus_id"]))
    sessions = sorted({a[0] for a in answers} | {r["session_id"] for r in hum})
    out = []
    for s in sessions:
        per_task = {}
        for task, mode in MODE.items():
            items = [p for m, p in assigned[s] if m == mode]
            vals = []
            for p in items:
                got = [v for (pid, _i), v in item_score[s].items() if pid == p]
                if mode == "baseline":   # several chains per association item: the item's value is its best valid chain
                    vals.append(max([x for x, g in got if g and x is not None] or [0.0]))
                else:
                    vals.append(next((x for x, g in got if g and x is not None), 0.0))
            per_task[task] = (float(np.mean(vals)) if vals else 0.0, sum(v > 0 for v in vals), len(items))
        score = float(np.mean([v[0] for v in per_task.values()]))
        out.append({"session_id": s, "score": round(score, 4),
                    **{f"{t}_score": round(v[0], 4) for t, v in per_task.items()},
                    **{f"{t}_good_answers": f"{v[1]}/{v[2]}" for t, v in per_task.items()}})
    out.sort(key=lambda r: -r["score"])
    for i, r in enumerate(out, 1):
        r["rank"] = i
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "session_id", "score", "association_score", "analogy_score",
                                          "blending_score", "association_good_answers", "analogy_good_answers",
                                          "blending_good_answers"])
        w.writeheader()
        w.writerows(out)
    print(f"{len(out)} participants -> {OUT}")
    for r in out[:20]:
        print(f"  #{r['rank']:2d} {r['session_id'][:8]} score {r['score']:.3f}  good answers A {r['association_good_answers']} "
              f"An {r['analogy_good_answers']} B {r['blending_good_answers']}")
    print("  ...", "zero score:", sum(r["score"] == 0 for r in out))


if __name__ == "__main__":
    main()
