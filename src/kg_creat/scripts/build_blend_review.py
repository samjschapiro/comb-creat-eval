"""Sample the re-elicited (uv-tagged) blends for a BLIND human rating study, STRATIFIED by model:
the same fraction of each model's 30 blends, so the sample is representative across the pool.

The reviewer rates the same dimensions the LLM panel scored -- generic_ok, coherent, scope (1/2/3) --
plus a COGENCY judgment (is the invented concept a real idea you could reason with, or plausible-sounding
word association?), without seeing the model or the panel's verdicts. Writes:
  <out>/items.json  -- blind content served to the UI (shuffled; no model name, no verdicts)
  <out>/key.json    -- HIDDEN: id -> model + panel majority (generic_ok/coherent/scope) + shared_properties,
                       for the later human-vs-panel agreement analysis.

    .venv/bin/python -m src.kg_creat.scripts.build_blend_review --frac 0.2 --seed 0
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

SCORES = Path("data/kg_creat/kombine_test30/scores")
RESP = Path("data/kg_creat/kombine_test30/responses")
OUT_DEFAULT = Path("data/kg_creat/kombine_test30/human_review_blendv3")


def _maj_bool(votes):
    vs = [bool(v) for v in votes if v is not None]
    return (sum(vs) * 2 >= len(vs)) if vs else None


def _median_int(votes):
    vs = sorted(v for v in votes if v is not None)
    return vs[len(vs) // 2] if vs else None


def _rid(model, pid):
    return hashlib.sha1(f"{model}|blending|{pid}".encode()).hexdigest()[:12]


def main(frac: float, seed: int, out: Path):
    rng = random.Random(seed)
    # content by (model, prompt_id)
    content = {}
    for md in sorted(RESP.iterdir()):
        rp = md / "responses.json"
        if not rp.exists():
            continue
        for r in json.loads(rp.read_text()):
            if r.get("mode") == "blending" and r.get("items"):
                content[(md.name, r["prompt_id"])] = r

    # panel verdicts per (model, prompt_id)
    by_model = defaultdict(list)
    for md in sorted(SCORES.iterdir()):
        ps = md / "path_scores.json"
        if not ps.exists():
            continue
        for r in json.loads(ps.read_text()):
            if r.get("mode") != "blending" or not r.get("blend_judges"):
                continue
            js = r["blend_judges"]
            panel = {"generic_ok": _maj_bool([j.get("generic_ok") for j in js]),
                     "coherent": _maj_bool([j.get("coherent") for j in js]),
                     "scope": _median_int([j.get("scope") for j in js])}
            src = content.get((md.name, r["prompt_id"]))
            if not src:
                continue
            it = src["items"][0]
            tags = it.get("tags") or []
            structure = [{"triple": t, "tag": (tags[i] if i < len(tags) else "?")}
                         for i, t in enumerate(it["paths"][0])]
            rid = _rid(md.name, r["prompt_id"])
            item = {"id": rid, "task": "blending", "u": src.get("u_label"), "v": src.get("v_label"),
                    "generic_space": it.get("generic_space"), "invention": it.get("concept"),
                    "structure": structure}
            keyrec = {"id": rid, "model": md.name, "prompt_id": r["prompt_id"], "panel": panel,
                      "shared_properties": r.get("shared_properties"), "judges": js}
            by_model[md.name].append((item, keyrec))

    # stratified: the same fraction of EACH model's blends
    picks = []
    for model, blends in sorted(by_model.items()):
        rng.shuffle(blends)
        k = max(1, round(frac * len(blends)))
        picks += blends[:k]
        print(f"  {model:34s} {k}/{len(blends)} sampled")
    rng.shuffle(picks)   # present in random (blind) order
    items = [it for it, _ in picks]
    key = {kr["id"]: kr for _, kr in picks}

    out.mkdir(parents=True, exist_ok=True)
    (out / "items.json").write_text(json.dumps(items, indent=1))
    (out / "key.json").write_text(json.dumps(key, indent=1))
    print(f"\nWrote {len(items)} blind blend items ({frac:.0%} per model) -> {out/'items.json'}")
    print(f"Hidden key (model + panel votes) -> {out/'key.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2, help="fraction of each model's blends to sample (0.1-0.3)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args()
    main(a.frac, a.seed, a.out)
