"""Split originality into BASE and EMERGENT for analogy and blending, and write both onto the head
records of every model's path_scores.json (originality = base, em_originality = emergent). Pool-relative,
element-level, k-NN -- the same method as score.score_originality, but on two element sets:

  analogy   base = anchor-path elements (the mapping)          emergent = invention h (projection images + name)
  blending  base = [u]/[v]/[uv] projected triples of the blend  emergent = [emergent] triples of the blend

Association is untouched (no emergent invention). Judge-free; local MLX embeddings, so no API cost.

    .venv_mlx/bin/python -m src.kg_creat.scripts.rescore_split_originality \\
        data/kg_creat/kombine_test30/scores data/kg_creat/kombine_test30/responses
"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.score import _artifact_elements, _norm

_DUMP = dict(indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)


def orig_elements(it, mode, u, v):
    """(base_surfaces, emergent_surfaces) as sets of surface strings."""
    if mode == "analogy":
        base = _artifact_elements(it.get("paths") or [], u, v)
        imgs = [pr["image"] for pr in (it.get("projection") or [])
                if isinstance(pr, dict) and len(pr.get("image", [])) == 3]
        emg = _artifact_elements([imgs], u, v)
        if it.get("invention"):
            emg.add(("c", _norm(it["invention"])))
    elif mode == "blending":
        tags = it.get("tags") or []
        tr = it["paths"][0] if it.get("paths") else []
        base = _artifact_elements([[t for i, t in enumerate(tr) if i < len(tags) and tags[i] in ("u", "v", "uv")]], u, v)
        emg = _artifact_elements([[t for i, t in enumerate(tr) if i < len(tags) and tags[i] == "emergent"]], u, v)
    else:
        return set(), set()
    return {s for _, s in base}, {s for _, s in emg}


def main(scores_dir, responses_dir, embed_model):
    scores_dir, responses_dir = Path(scores_dir), Path(responses_dir)
    embed = get_embedder(embed_model)
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)

    # per (prompt_id): {model: (base_set, emg_set)}
    per = defaultdict(dict)
    for f in glob.glob(f"{responses_dir}/*/responses.json"):
        m = f.split("/")[-2]
        for r in json.load(open(f)):
            mode = r.get("mode")
            if mode not in ("analogy", "blending") or not r.get("items"):
                continue
            b, e = orig_elements(r["items"][0], mode, r.get("u_label"), r.get("v_label"))
            per[r["prompt_id"]][m] = (mode, b, e)

    # embed all surfaces once (base + emergent pools per prompt)
    pools = {}   # prompt_id -> (base {surf:vec}, emg {surf:vec})
    allsurf = set()
    for pid, bymodel in per.items():
        for _mode, b, e in bymodel.values():
            allsurf |= b | e
    E = {s: un(np.asarray(embed(s), float)) for s in allsurf if s}
    for pid, bymodel in per.items():
        bs = set().union(*[b for _, b, _ in bymodel.values()]) if bymodel else set()
        es = set().union(*[e for _, _, e in bymodel.values()]) if bymodel else set()
        pools[pid] = ({s: E[s] for s in bs if s in E}, {s: E[s] for s in es if s in E})

    def rho(surfs, pool, k=5):
        vals = []
        for s in surfs:
            if s not in pool:
                continue
            d = sorted(1 - float(pool[s] @ o) for os, o in pool.items() if os != s)
            if d:
                vals.append(sum(d[:min(k, len(d))]) / min(k, len(d)))
        return (sum(vals) / len(vals)) if vals else None

    for md in sorted(scores_dir.iterdir()):
        ps = md / "path_scores.json"
        if not ps.exists():
            continue
        recs = json.loads(ps.read_text())
        # heads: analogy pair-head, blending structure record (both currently carry originality)
        n = 0
        by_head = {}
        for rec in recs:
            mode = rec.get("mode")
            if mode == "analogy" and "pair_sat" in rec:
                by_head[rec["prompt_id"]] = rec
            elif mode == "blending" and rec.get("triples") and rec.get("originality") is not None:
                by_head[rec["prompt_id"]] = rec
        for pid, rec in by_head.items():
            ent = per.get(pid, {}).get(md.name)
            if not ent:
                continue
            mode, b, e = ent
            bp, ep = pools[pid]
            rec["originality"] = rho(b, bp)          # base
            rec["em_originality"] = rho(e, ep)       # emergent invention
            n += 1
        ps.write_text(json.dumps(recs, **_DUMP))
        print(f"  {md.name:34s} split originality on {n} heads")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scores_dir")
    ap.add_argument("responses_dir")
    ap.add_argument("--embed_model", default="mlx-community/all-MiniLM-L6-v2-4bit")
    a = ap.parse_args()
    main(a.scores_dir, a.responses_dir, a.embed_model)
