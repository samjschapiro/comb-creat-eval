"""Sample analogy/blend artifacts for a BLIND human re-rating study.

The reviewer rates the SAME subjective dimensions the LLM judge panel scored -- blend
(generic_ok, coherent, scope) and analogy invention (valid, coherent) -- without seeing the panel's
verdicts or which model produced the artifact. Writes:
  human_review/items.json  -- blind content served to the UI (shuffled; no model name, no verdicts)
  human_review/key.json    -- HIDDEN: id -> model + per-judge panel votes, for the later
                              human-vs-panel agreement analysis.

    .venv/bin/python -m src.kg_creat.scripts.build_blind_review --n 60 --seed 0
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

SCORES = Path("data/kg_creat/kombine_test30/scores")
RESP = Path("data/kg_creat/kombine_test30/responses")
OUT = Path("data/kg_creat/kombine_test30/human_review")


def _maj_bool(votes):
    vs = [bool(v) for v in votes if v is not None]
    return (sum(vs) * 2 >= len(vs)) if vs else None      # >=half -> True


def _median_int(votes):
    vs = sorted(v for v in votes if v is not None)
    return vs[len(vs) // 2] if vs else None


def _rid(model, mode, pid):
    return hashlib.sha1(f"{model}|{mode}|{pid}".encode()).hexdigest()[:12]


def main(n: int, seed: int):
    rng = random.Random(seed)
    # content by (model, mode, prompt_id)
    content = {}
    for md in sorted(RESP.iterdir()):
        rp = md / "responses.json"
        if not rp.exists():
            continue
        for r in json.loads(rp.read_text()):
            if r["mode"] not in ("analogy", "blending") or not r.get("items"):
                continue
            content[(md.name, r["mode"], r["prompt_id"])] = r

    pool = {"analogy": [], "blending": []}
    for md in sorted(SCORES.iterdir()):
        ps = md / "path_scores.json"
        if not ps.exists():
            continue
        for r in json.loads(ps.read_text()):
            mode = r.get("mode")
            if mode == "blending" and r.get("blend_judges"):
                js = r["blend_judges"]
                panel = {"generic_ok": _maj_bool([j.get("generic_ok") for j in js]),
                         "coherent": _maj_bool([j.get("coherent") for j in js]),
                         "scope": _median_int([j.get("scope") for j in js])}
            elif mode == "analogy" and r.get("invention_judges") and "pair_sat" in r:
                js = r["invention_judges"]
                panel = {"valid": _maj_bool([j.get("valid") for j in js]),
                         "coherent": _maj_bool([j.get("coherent") for j in js])}
            else:
                continue
            key = (md.name, mode, r["prompt_id"])
            src = content.get(key)
            if not src:
                continue
            it = src["items"][0]
            u, v = src.get("u_label"), src.get("v_label")
            rid = _rid(*key)
            if mode == "blending":
                tags = it.get("tags") or []
                structure = [{"triple": t, "tag": (tags[i] if i < len(tags) else "?")}
                             for i, t in enumerate(it["paths"][0])]
                item = {"id": rid, "task": "blending", "u": u, "v": v,
                        "generic_space": it.get("generic_space"), "invention": it.get("concept"),
                        "structure": structure}
            else:
                item = {"id": rid, "task": "analogy", "u": u, "v": v,
                        "path_a": it["paths"][0], "path_b": it["paths"][1] if len(it["paths"]) > 1 else [],
                        "projected": it.get("projected"), "invention": it.get("invention"),
                        "projection": [{"source": p.get("source"), "image": p.get("image")}
                                       for p in (it.get("projection") or []) if isinstance(p, dict)]}
            pool[mode].append((item, {"id": rid, "model": md.name, "prompt_id": r["prompt_id"],
                                      "panel": panel, "judges": js}))

    per = n // 2
    picks = []
    for mode in ("analogy", "blending"):
        rng.shuffle(pool[mode])
        picks += pool[mode][:per]
    rng.shuffle(picks)
    items = [it for it, _ in picks]
    key = {k["id"]: k for _, k in picks}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(items, indent=1))
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    na = sum(1 for it in items if it["task"] == "analogy")
    print(f"Wrote {len(items)} blind items ({na} analogy, {len(items) - na} blend) -> {OUT/'items.json'}")
    print(f"Hidden key (model + panel votes) -> {OUT/'key.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="total items (split evenly analogy/blend)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.n, a.seed)
