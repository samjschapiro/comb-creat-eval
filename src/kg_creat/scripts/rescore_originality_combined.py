"""Originality for humans and models against ONE shared pool, so the composite can compare them.

Originality is pool-relative: an element's score is its mean distance to its 5 nearest neighbours among
everyone's answers to the same item. The models' stored scores used a models-only pool; the human scores used
humans + models. This recomputes every entity's originality against the same combined pool and writes copies
(the models' own score files are never touched):

  association           originality     = score.score_originality (all non-anchor elements)
  analogy / blending    originality     = base, em_originality = emergent invention -- the method of
                                          rescore_split_originality.py, but keyed per DRAW, because a human
                                          group holds many answers to one item where a model holds one.

Only items every entity answered are written (the human warm-ups and the two worked-example pairs drop out).
Then compute_composite.py runs on the copy. --pool models reproduces the stored model scores (a check).

    .venv_mlx/bin/python -m src.kg_creat.scripts.rescore_originality_combined --pool combined
"""
import argparse
import glob
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.rescore_split_originality import orig_elements
from src.kg_creat.scripts.score import _draw_key, build_item_element_pool, score_originality

LLM_SCORES = Path("data/kg_creat/kombine_test30/scores")
LLM_RESP = Path("data/kg_creat/kombine_test30/responses")
HUMAN_SCORES = Path("data/kg_creat/human_study/scores")
HUMAN_RESP = Path("data/kg_creat/human_study/score_input")
HUMAN_GROUPS = ["human_kept", "human_flagged"]
OUT = {"combined": Path("data/kg_creat/human_study/combined_scores"),
       "models": Path("data/kg_creat/human_study/combined_scores_check_models_only")}
_DUMP = dict(indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)


def entities(pool):
    scored = sorted(Path(p).parent.name for p in glob.glob(f"{LLM_SCORES}/*/summary.json"))
    ents = [(m, LLM_SCORES / m, LLM_RESP / m) for m in scored]
    if pool == "combined":
        ents += [(g, HUMAN_SCORES / g, HUMAN_RESP / g) for g in HUMAN_GROUPS]
    return ents


def main(pool):
    out = OUT[pool]
    if out.exists():
        raise SystemExit(f"FATAL: {out} exists; remove it first (this script never overwrites)")
    ents = entities(pool)
    embed = get_embedder()
    unit = lambda x: x / (np.linalg.norm(x) + 1e-9)

    # 1. base / association originality against the shared element pool (score.py's own method)
    item_pool = build_item_element_pool([resp for _, _, resp in ents], embed)
    recs_by, resp_by = {}, {}
    for name, sdir, rdir in ents:
        recs = json.loads((sdir / "path_scores.json").read_text())
        responses = json.loads((rdir / "responses.json").read_text())
        by_draw = {_draw_key(r): r for r in responses}
        score_originality(recs, by_draw, item_pool)
        recs_by[name], resp_by[name] = recs, by_draw

    # 2. analogy / blending: split into base and emergent, pooled per item over every entity's every draw
    per = defaultdict(dict)                     # prompt_id -> {(entity, draw): (base, emergent)}
    for name, by_draw in resp_by.items():
        for dk, r in by_draw.items():
            if r.get("mode") in ("analogy", "blending") and r.get("items"):
                per[r["prompt_id"]][(name, dk)] = orig_elements(r["items"][0], r["mode"], r.get("u_label"), r.get("v_label"))
    surfs = {s for d in per.values() for b, e in d.values() for s in b | e if s}
    E = {s: unit(np.asarray(embed(s), float)) for s in surfs}
    pools = {pid: ({s: E[s] for b, _ in d.values() for s in b if s in E},
                   {s: E[s] for _, e in d.values() for s in e if s in E}) for pid, d in per.items()}

    def rho(sset, pool, k=5):
        vals = []
        for s in sset:
            if s not in pool:
                continue
            d = sorted(1 - float(pool[s] @ o) for os, o in pool.items() if os != s)
            if d:
                vals.append(sum(d[:min(k, len(d))]) / min(k, len(d)))
        return (sum(vals) / len(vals)) if vals else None

    shared = set.intersection(*[{r["prompt_id"] for r in by_draw.values()} for by_draw in resp_by.values()]) \
        if pool == "combined" else {r["prompt_id"] for by_draw in resp_by.values() for r in by_draw.values()}
    print(f"{len(shared)} items answered by every entity")
    for name, recs in recs_by.items():
        n = 0
        for rec in recs:
            head = (rec.get("mode") == "analogy" and "pair_sat" in rec and rec.get("path_idx") == 0) or \
                   (rec.get("mode") == "blending" and rec.get("triples") and rec.get("path_idx") == 0)
            ent = per.get(rec["prompt_id"], {}).get((name, _draw_key(rec))) if head else None
            if ent is None:
                continue
            bp, ep = pools[rec["prompt_id"]]
            rec["originality"], rec["em_originality"] = rho(ent[0], bp), rho(ent[1], ep)
            n += 1
        # like-for-like items: drop the human warm-ups (no model answered them) and the 2 pairs only the
        # models answered (E0/F0, E26/F26, shown to humans as worked examples) -- AFTER pooling, so the pool is unchanged
        recs = [r for r in recs if r["prompt_id"] in shared]
        (out / name).mkdir(parents=True, exist_ok=True)
        (out / name / "path_scores.json").write_text(json.dumps(recs, **_DUMP))
        print(f"  {name:34s} rescored ({n} analogy/blend heads)")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["combined", "models"], required=True)
    main(ap.parse_args().pool)
