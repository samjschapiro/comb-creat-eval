"""Embed every model's invention (blend c' / analogy h) and save vectors + metadata for the invention
landscape. Run in the MLX env; plot_invention_landscape.py (sklearn env) reads the .npz.

The embedded text is the invention's STRUCTURE ONLY -- each triple as "relation object", the coined
name dropped. Two models that call the same structure different things must land in the same place, and
two that share a label but no properties must not; the name is metadata here, never geometry.

    .venv_mlx/bin/python -m src.kg_creat.scripts.embed_inventions
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.kg_creat.embed import get_embedder  # noqa: E402
from src.kg_creat.scripts.analyze_inventive_multiples import slot_texts  # noqa: E402

RESP = Path("data/kg_creat/kombine_test30/responses")
SCORES = Path("data/kg_creat/kombine_test30/scores")
OUT = Path("data/kg_creat/kombine_test30/analysis/invention_vectors.npz")


def _struct(mode, it):
    """The invention's triples in the shape slot_texts expects."""
    if mode == "blending":
        tags = it.get("tags") or []
        return [list(t) + [tags[i] if i < len(tags) else ""]
                for i, t in enumerate((it.get("paths") or [[]])[0])]
    return [{"source": p.get("source"), "image": p.get("image")} for p in (it.get("projection") or [])]


def main():
    embed = get_embedder()
    vecs, models, tasks, items, names, us, vs, origs, utils, integs = ([] for _ in range(10))
    for md in sorted(SCORES.iterdir()):
        ps = md / "path_scores.json"
        rp = RESP / md.name / "responses.json"
        if not ps.exists() or not rp.exists():
            continue
        mid = md.name    # model key, e.g. "openai_gpt-5" (mapped to display/provider in the plot)
        # per-item emergent verdicts from the scored head records: originality, utility J^utl,
        # integration quality J^qua (analogy 0/1; blend scope 1-3 -> (scope-1)/2 in [0,1]).
        orig, util, integ = {}, {}, {}
        for r in json.loads(ps.read_text()):
            m_ = r["mode"]
            if m_ not in ("analogy", "blending"):
                continue
            if m_ == "analogy" and "pair_sat" not in r:
                continue                                   # only the pair-head carries the invention
            key = (m_, r["prompt_id"])
            if r.get("originality") is not None:
                orig[key] = r["originality"]
            if m_ == "blending":
                util[key] = 1.0 if r.get("blend_utility") else 0.0
                sc = r.get("blend_integration")
                integ[key] = ((sc - 1) / 2) if sc else np.nan
            else:
                util[key] = 1.0 if r.get("invention_utility") else 0.0
                integ[key] = 1.0 if r.get("invention_integration") else 0.0
        for r in json.loads(rp.read_text()):
            mode = r["mode"]
            if mode not in ("analogy", "blending") or not r.get("items"):
                continue
            it = r["items"][0]
            txt = " ; ".join(s for s, _ in slot_texts(mode, _struct(mode, it)))
            if len(txt) < 3:
                continue
            k = (mode, r["prompt_id"])
            vecs.append(np.asarray(embed(txt), dtype=float))
            models.append(mid); tasks.append(mode); items.append(r["prompt_id"])
            names.append(it.get("concept") if mode == "blending" else it.get("invention"))
            us.append(r.get("u_label")); vs.append(r.get("v_label"))
            origs.append(orig.get(k, np.nan)); utils.append(util.get(k, np.nan)); integs.append(integ.get(k, np.nan))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, vecs=np.array(vecs), models=np.array(models), tasks=np.array(tasks),
             items=np.array(items), names=np.array([n or "" for n in names]),
             u=np.array(us), v=np.array(vs), orig=np.array(origs, float),
             util=np.array(utils, float), integ=np.array(integs, float))
    print(f"saved {len(vecs)} invention vectors -> {OUT}")


if __name__ == "__main__":
    main()
