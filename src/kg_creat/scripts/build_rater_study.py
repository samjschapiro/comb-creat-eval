"""Build the co-author rater study: a self-contained web page in which several independent human
raters score the same Kombine artifacts on the dimensions the LLM judges score (utility) and on
originality, so that we can report (i) human-human inter-rater reliability, (ii) human-panel
agreement on every judged dimension, and (iii) how well pool-relative embedding originality (rho)
tracks human originality ratings.

Design
------
* Human answers come only from kept participants who passed the pre-registered comprehension screen
  (Breakfast + Lunch warm-up); every artifact also passed parsing and is well-formed (right endpoints,
  continuous, no revisited node) or, for analogy/blending, has both paths, a projection / generic space,
  a named invention and >= 2 blend links.
* The page reuses the human study's jsPsych build and stylesheet (rater_study_kombine.css), drawing each
  answer as a read-only copy of the widget participants filled in.
* Every rater sees the SAME artifacts (needed for inter-rater reliability); presentation order is
  shuffled per rater inside the page.
* Artifacts are grouped by entity pair (u, v): each group shows one human answer plus 2-3 model
  answers to the same pair, so originality is judged relative to the other answers to that item --
  the same comparison set rho uses.
* Within each task the model answers are stratified on the panel's verdict (pass / fail) so
  agreement is not dominated by one class.
* The page carries NO model names, NO human/model labels and NO judge verdicts. Those live only in
  the hidden key.

Outputs (data/kg_creat/rater_study/):
  items.json   blind content served to the page
  key.json     HIDDEN: artifact id -> source, prompt_id, panel votes, rho fields
  rate.html    the self-contained rating page (items embedded); share this file / host it

    .venv/bin/python -m src.kg_creat.scripts.build_rater_study --seed 0
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

SCORES = Path("data/kg_creat/human_study/paper_scores")
MODEL_RESP = Path("data/kg_creat/kombine_test30/responses")
HUMAN_RESP = Path("data/kg_creat/human_study/score_input/human_kept/responses.json")
TEMPLATE = Path(__file__).with_name("rater_study_template.html")
STUDY_CSS = Path(__file__).with_name("rater_study_kombine.css")   # vendored from the human study's index.html
WARMUP_SCORES = Path("data/kg_creat/human_study/scores/human_kept/path_scores.json")
OUT = Path("data/kg_creat/rater_study")

# groups per task x answers per group (1 human + rest models)
DESIGN = {"association": (4, 4), "analogy": (4, 3), "blending": (4, 3)}
WORKED_EXAMPLES = {"E0", "E26", "F0", "F26"}       # in the paper's tables; humans never saw them


def _rid(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]


def _maj(votes):
    vs = [bool(v) for v in votes if v is not None]
    return (sum(vs) * 2 > len(vs)) if vs else None


def _median(votes):
    vs = sorted(v for v in votes if v is not None)
    return vs[len(vs) // 2] if vs else None


def _clean(t):
    return [str(x) for x in t]


# ------------------------------------------------------------------ loading
def load_responses():
    """(source, prompt_id, sample_idx) -> response record."""
    out = {}
    for r in json.loads(HUMAN_RESP.read_text()):
        if not r.get("is_control"):
            out[("human", r["prompt_id"], r["sample_idx"])] = r
    for md in sorted(MODEL_RESP.iterdir()):
        rp = md / "responses.json"
        if rp.exists():
            for r in json.loads(rp.read_text()):
                out[(md.name, r["prompt_id"], r["sample_idx"])] = r
    return out


def screened_sample_idx():
    """sample_idx of kept participants who passed the pre-registered comprehension screen: the blend judge
    accepts their generic space on the Breakfast + Lunch warm-up (46 of 75; as in analyze_human_vs_models)."""
    resp = json.loads(HUMAN_RESP.read_text())
    sess = {r["sample_idx"]: r["session_id"] for r in resp}
    passed = {sess[r["sample_idx"]] for r in json.loads(WARMUP_SCORES.read_text())
              if r["prompt_id"] == "blend_ctrl_1" and r.get("generic_ok") is True}
    return {i for i, s in sess.items() if s in passed}, len(passed)


def load_scores(screen=True):
    """source -> list of path-score rows (human rows limited to screen-passing participants)."""
    keep, n_passed = screened_sample_idx() if screen else (None, None)
    if screen:
        print(f"human answers: comprehension screen on ({n_passed} participants pass)")
    out = {}
    for md in sorted(SCORES.iterdir()):
        ps = md / "path_scores.json"
        if not ps.exists() or md.name == "human_flagged":
            continue
        rows = json.loads(ps.read_text())
        if md.name == "human_kept" and keep is not None:
            rows = [r for r in rows if r["sample_idx"] in keep]
        out["human" if md.name == "human_kept" else md.name] = rows
    return out


# ------------------------------------------------------------------ candidate artifacts
def association_candidates(scores):
    """One candidate per scored path. Utility = every triple factual (panel: haiku factuality)."""
    cands = []
    for src, rows in scores.items():
        for r in rows:
            if r["mode"] != "baseline" or not r.get("well_formed") or not r.get("triples"):
                continue
            if len(r["triples"]) < 2:
                continue
            fact = r.get("factual") or []
            if any(f is None for f in fact):
                continue
            cands.append({
                "src": src, "pid": r["prompt_id"], "u": r["u_label"], "v": r["v_label"],
                "item": {"task": "association", "u": r["u_label"], "v": r["v_label"],
                         "path": [_clean(t) for t in r["triples"]]},
                "key": {"factual": fact, "sat": bool(r.get("sat")),
                        "originality": r.get("originality"), "surprise": r.get("R"),
                        "sample_idx": r["sample_idx"], "path_idx": r.get("path_idx")},
                "pass": bool(r.get("sat")),
            })
    return cands


def analogy_candidates(scores, resp):
    cands = []
    for src, rows in scores.items():
        by_pair = {}
        for r in rows:
            if r["mode"] == "analogy" and r.get("pair_idx") is not None:
                by_pair.setdefault((r["prompt_id"], r["sample_idx"], r["pair_idx"]), []).append(r)
        for (pid, sidx, pidx), rs in by_pair.items():
            r0 = rs[0]
            js = r0.get("invention_judges") or []
            if not js:
                continue
            rec = resp.get((src, pid, sidx))
            if not rec or not rec.get("items") or pidx >= len(rec["items"]):
                continue
            it = rec["items"][pidx]
            paths = it.get("paths") or []
            proj = [p for p in (it.get("projection") or [])
                    if isinstance(p, dict) and p.get("source") and p.get("image")]
            if len(paths) < 2 or not paths[0] or not paths[1] or not proj \
                    or not (it.get("invention") or "").strip() or not (it.get("projected") or "").strip():
                continue
            panel = {"valid": _maj([j.get("valid") for j in js]),
                     "coherent": _maj([j.get("coherent") for j in js])}
            origs = [x.get("originality") for x in rs if x.get("originality") is not None]
            cands.append({
                "src": src, "pid": pid, "u": r0["u_label"], "v": r0["v_label"],
                "item": {"task": "analogy", "u": r0["u_label"], "v": r0["v_label"],
                         "path_a": [_clean(t) for t in paths[0]],
                         "path_b": [_clean(t) for t in paths[1]],
                         "projected": it["projected"], "invention": it["invention"],
                         "projection": [{"source": _clean(p["source"]), "image": _clean(p["image"])}
                                        for p in proj]},
                "key": {"pair_sat": bool(r0.get("pair_sat")),
                        "pair_structural_ok": r0.get("pair_structural_ok"),
                        "factual": [x.get("factual") for x in rs],
                        "panel": panel, "judges": [{k: j.get(k) for k in ("model", "valid", "coherent")}
                                                   for j in js],
                        "originality": sum(origs) / len(origs) if origs else None,
                        "em_originality": r0.get("em_originality"), "surprise": r0.get("R"),
                        "sample_idx": sidx, "pair_idx": pidx},
                "pass": bool(r0.get("pair_sat")) and bool(panel["valid"]) and bool(panel["coherent"]),
            })
    return cands


def blending_candidates(scores, resp):
    cands = []
    for src, rows in scores.items():
        for r in rows:
            if r["mode"] != "blending" or not r.get("blend_judges"):
                continue
            rec = resp.get((src, r["prompt_id"], r["sample_idx"]))
            if not rec or not rec.get("items"):
                continue
            it = rec["items"][r.get("path_idx") or 0] if (r.get("path_idx") or 0) < len(rec["items"]) \
                else rec["items"][0]
            gs, concept = (it.get("generic_space") or "").strip(), (it.get("concept") or "").strip()
            triples = (it.get("paths") or [[]])[0]
            if not gs or not concept or len(triples) < 2:
                continue
            tags = it.get("tags") or []
            js = r["blend_judges"]
            panel = {"generic_ok": _maj([j.get("generic_ok") for j in js]),
                     "coherent": _maj([j.get("coherent") for j in js]),
                     "scope": _median([j.get("scope") for j in js])}
            cands.append({
                "src": src, "pid": r["prompt_id"], "u": r["u_label"], "v": r["v_label"],
                "item": {"task": "blending", "u": r["u_label"], "v": r["v_label"],
                         "generic_space": gs, "invention": concept,
                         "structure": [{"triple": _clean(t), "tag": tags[i] if i < len(tags) else "?"}
                                       for i, t in enumerate(triples)]},
                "key": {"panel": panel,
                        "judges": [{k: j.get(k) for k in ("model", "generic_ok", "coherent", "scope")}
                                   for j in js],
                        "generic_ok_field": r.get("generic_ok"),
                        "originality": r.get("originality"), "em_originality": r.get("em_originality"),
                        "surprise": r.get("R"), "sample_idx": r["sample_idx"]},
                "pass": bool(panel["generic_ok"]),
            })
    return cands


# ------------------------------------------------------------------ sampling
def build_groups(task, cands, n_groups, per_group, rng):
    by_pid = {}
    for c in cands:
        if c["pid"] in WORKED_EXAMPLES:
            continue
        by_pid.setdefault(c["pid"], {"human": [], "model": []})["human" if c["src"] == "human"
                                                                    else "model"].append(c)
    n_models = per_group - 1
    eligible = [pid for pid, d in sorted(by_pid.items())
                if d["human"] and len({c["src"] for c in d["model"] if c["pass"]}) >= n_models
                and len({c["src"] for c in d["model"] if not c["pass"]}) >= n_models
                and len({c["src"] for c in d["model"]}) >= n_models]
    pids = rng.sample(eligible, n_groups)
    groups = []
    for gi, pid in enumerate(pids):
        d = by_pid[pid]
        picks = [rng.choice(d["human"])]
        # aim each group (human included) at half panel-pass; odd group sizes alternate rounding by group
        target = per_group // 2 + (gi % 2 if per_group % 2 else 0)
        n_pass = min(n_models, max(0, target - int(picks[0]["pass"])))
        used = set()
        for want_pass, n in ((True, n_pass), (False, n_models - n_pass)):
            pool = [c for c in d["model"] if c["pass"] == want_pass]
            rng.shuffle(pool)
            for c in pool:
                if n == 0:
                    break
                if c["src"] in used:
                    continue
                picks.append(c)
                used.add(c["src"])
                n -= 1
        assert len(picks) == per_group, (task, pid, len(picks))
        rng.shuffle(picks)
        groups.append({"pid": pid, "u": picks[0]["u"], "v": picks[0]["v"], "answers": picks})
    return groups


def main(seed, screen=True):
    rng = random.Random(seed)
    resp, scores = load_responses(), load_scores(screen)
    cands = {"association": association_candidates(scores),
             "analogy": analogy_candidates(scores, resp),
             "blending": blending_candidates(scores, resp)}

    items, key = {"version": f"kombine-raters-s{seed}", "tasks": {}}, {}
    for task, (ng, pg) in DESIGN.items():
        groups = build_groups(task, cands[task], ng, pg, rng)
        out_groups = []
        for g in groups:
            answers = []
            for c in g["answers"]:
                aid = _rid(task, c["src"], c["pid"], json.dumps(c["key"].get("sample_idx")),
                           json.dumps(c["key"].get("path_idx", c["key"].get("pair_idx"))))
                answers.append({"id": aid, **c["item"]})
                key[aid] = {"task": task, "source": c["src"], "is_human": c["src"] == "human",
                            "prompt_id": c["pid"], "panel_pass": c["pass"], **c["key"]}
            out_groups.append({"gid": _rid(task, g["pid"]), "prompt_id_hidden": None,
                               "u": g["u"], "v": g["v"], "answers": answers})
        for og in out_groups:
            og.pop("prompt_id_hidden")
        items["tasks"][task] = out_groups
        n = sum(len(g["answers"]) for g in out_groups)
        npass = sum(key[a["id"]]["panel_pass"] for g in out_groups for a in g["answers"])
        print(f"{task:12s} {len(out_groups)} groups, {n} answers, panel-pass {npass}/{n} "
              f"(eligible candidates: {len(cands[task])})")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(items, indent=1))
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    html = (TEMPLATE.read_text().replace("/*__STUDY_CSS__*/", STUDY_CSS.read_text())
            .replace("/*__ITEMS__*/null", json.dumps(items)))
    (OUT / "rate.html").write_text(html)
    print(f"-> {OUT/'rate.html'} ({len(html)//1024} KB), hidden key -> {OUT/'key.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-screen", action="store_true",
                    help="include human answers from participants who failed the comprehension screen")
    a = ap.parse_args()
    main(a.seed, screen=not a.no_screen)
