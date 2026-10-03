"""Human study answers -> the responses.json format score.py reads, so humans are judged exactly like the models.

Each kept (or flagged) session is one draw: its answer to an item becomes one response record with that item's
prompt_id, sample_idx = the session's index, and a single parsed item -- the same fields elicitation writes
for a model (paths, items[0] with inferences / projection / concept / generic_space / tags). Two pseudo-models:

  human_kept     sessions that passed screening
  human_flagged  likely-AI / borderline sessions, scored separately so every result can be shown with and without

Only answered items are written (a skipped item is no answer, not a failed one). The two warm-ups
(analogy_ctrl_1, blend_ctrl_1) are included: the comprehension screen needs the blend judge on Breakfast + Lunch.
Session status comes from data/kg_creat/human_study/sessions.csv (analyze_human_study.py).

    .venv/bin/python -m src.kg_creat.scripts.build_human_score_input
"""
import csv
import glob
import json
from pathlib import Path

from src.kg_creat.scripts.analyze_human_study import answered

RESP = "data/kg_creat/responses"
SESSIONS = "data/kg_creat/human_study/sessions.csv"
LLM_RESP = "data/kg_creat/kombine_test30/responses"
OUT = Path("data/kg_creat/human_study/score_input")
MODE = {"association": "baseline", "analogy": "analogy", "blending": "blending"}


def full(t):
    return [str(x).strip() for x in t] if t and len(t) == 3 and all(str(x).strip() for x in t) else None


def triple(d):
    return full([d.get("head", ""), d.get("relation", ""), d.get("tail", "")])


def to_item(r):
    """One human answer -> (paths, item) in the elicitation parse format."""
    if r["task"] == "association":
        path = [t for t in (triple(x) for x in r.get("triples") or []) if t]
        return [path], {"paths": [path], "inferences": []}
    if r["task"] == "analogy":
        pa = [t for t in (triple(x) for x in r.get("path_a") or []) if t]
        pb = [t for t in (triple(x) for x in r.get("path_b") or []) if t]
        proj = [{"source": full(p.get("source")), "image": full(p.get("image"))} for p in r.get("projection") or []]
        proj = [p for p in proj if p["source"] and p["image"]]
        item = {"paths": [pa, pb], "inferences": [],
                "projected": proj[0]["source"][0] if proj else "", "invention": proj[0]["image"][0] if proj else "",
                "projection": proj}
        return [pa, pb], item
    rows = [(s.get("from"), full(s.get("triple"))) for s in r.get("structure") or []]
    rows = [(f, t) for f, t in rows if t]
    structure = [t for _, t in rows]
    item = {"paths": [structure], "concept": r.get("blend_name") or "", "generic_space": r.get("generic_space") or "",
            "tags": [f for f, _ in rows], "inferences": [" ".join(t) for f, t in rows if f == "emergent"]}
    return [structure], item


def main():
    status = {row["session_id"]: row["status"] for row in csv.DictReader(open(SESSIONS))}
    sessions = {}
    for f in sorted(glob.glob(f"{RESP}/*.json")):
        d = json.load(open(f))
        sessions[d["session_id"]] = d
    # item metadata (labels, domains) from any model's elicitation, so the records match the models' exactly
    meta = {r["prompt_id"]: r for r in json.load(open(sorted(glob.glob(f"{LLM_RESP}/*/responses.json"))[0]))}

    groups = {"human_kept": "kept", "human_flagged": "flagged"}
    for name, want in groups.items():
        out = []
        for idx, (sid, d) in enumerate(sorted(sessions.items())):
            if status[sid] != want:
                continue
            for r in d["responses"]:
                if not answered(r):
                    continue
                pid = r["stimulus_id"]
                m = meta.get(pid, {})
                paths, item = to_item(r)
                if not any(paths):
                    continue
                mode = MODE[r["task"]]
                out.append({"prompt_id": pid, "bundle_id": pid, "regime": "A" if mode == "baseline" else "B",
                            "mode": mode, "u": r["u"], "v": r.get("v"), "u_label": r["u"], "v_label": r.get("v"),
                            "h": None, "k": 1, "constraint": None, "domain_u": m.get("domain_u"),
                            "domain_v": m.get("domain_v"), "cross_domain": m.get("cross_domain"),
                            "temperature": None, "sample_idx": idx, "session_id": sid, "is_control": bool(r.get("is_control")),
                            "items": [item], "paths": paths, "n_paths": len(paths), "n_items": 1,
                            "parse_success": True, "api_error": None})
        (OUT / name).mkdir(parents=True, exist_ok=True)
        (OUT / name / "responses.json").write_text(json.dumps(out, indent=1))
        by_mode = {mo: sum(1 for x in out if x["mode"] == mo) for mo in MODE.values()}
        print(f"{name}: {len(out)} answers  {by_mode}  (controls {sum(x['is_control'] for x in out)})")


if __name__ == "__main__":
    main()
