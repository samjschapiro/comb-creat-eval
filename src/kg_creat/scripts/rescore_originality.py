"""Recompute pool-relative originality for every scored model against the CURRENT model pool.

Originality is the mean cosine distance of an artifact's non-anchor elements to their k nearest
neighbours among *all responses to the same item* -- so it is defined relative to the pool of models.
Add models to the pool and every earlier model's originality is measured against a different, smaller
pool than the new ones. ``score.py`` cannot fix this on a resume: it builds the element pool over all
models but then skips any model that already has a ``summary.json``, so the old scores keep their old
pool while the new ones get the full one. The two are not comparable, and every composite built from
the mix is wrong.

This recomputes the dimension for all models at once. It is judge-free -- no API calls, nothing paid
-- and it touches only the ``originality`` field and the aggregates derived from it. Judge verdicts,
utility flags, surprise and the emergent dimensions are read and written back untouched.

    .venv_mlx/bin/python -m src.kg_creat.scripts.rescore_originality data/kg_creat/kombine_test30
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from src.kg_creat.aggregate import aggregate
from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.score import _draw_key, build_item_element_pool, score_originality


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--embedding", default="mlx-community/all-MiniLM-L6-v2-4bit")
    ap.add_argument("--dry-run", action="store_true", help="report the shift, write nothing")
    args = ap.parse_args()

    resp_dir, score_dir = args.run_dir / "responses", args.run_dir / "scores"
    embed = get_embedder(args.embedding)
    model_dirs = sorted(d for d in resp_dir.iterdir() if (d / "responses.json").exists())
    scored = sorted(d for d in score_dir.iterdir() if (d / "path_scores.json").exists())
    print(f"{len(model_dirs)} models in the response pool, {len(scored)} scored")
    if len(model_dirs) != len(scored):
        print(f"  note: pool is built from all {len(model_dirs)} response dirs; "
              f"only the {len(scored)} scored ones are rewritten")

    item_pool = build_item_element_pool(model_dirs, embed)
    print(f"built embedded element pool over {len(item_pool)} items")

    all_summaries, shifts = {}, []
    for sd in scored:
        rd = resp_dir / sd.name
        recs = json.loads((sd / "path_scores.json").read_text())
        responses = json.loads((rd / "responses.json").read_text())
        by_prompt = {_draw_key(r): r for r in responses}
        before = {(r["prompt_id"], r.get("temperature"), r.get("sample_idx"), r["path_idx"]):
                  r.get("originality") for r in recs}
        score_originality(recs, by_prompt, item_pool)
        d = [r["originality"] - before[(r["prompt_id"], r.get("temperature"), r.get("sample_idx"),
                                        r["path_idx"])]
             for r in recs
             if r.get("originality") is not None
             and before.get((r["prompt_id"], r.get("temperature"), r.get("sample_idx"),
                             r["path_idx"])) is not None]
        mean_shift = float(np.mean(d)) if d else 0.0
        shifts.append((sd.name, mean_shift, len(d)))
        summary = aggregate(recs)
        all_summaries[sd.name] = summary
        if not args.dry_run:
            (sd / "path_scores.json").write_text(json.dumps(
                recs, indent=2,
                default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))
            (sd / "summary.json").write_text(json.dumps(summary, indent=2))
    if not args.dry_run:
        (score_dir / "scores_summary.json").write_text(json.dumps(all_summaries, indent=2))

    print(f"\nMEAN ORIGINALITY SHIFT PER MODEL ({'dry run, nothing written' if args.dry_run else 'written'})")
    for name, s, n in sorted(shifts, key=lambda t: t[1]):
        print(f"  {name:38s} {s:+.4f}   (n = {n} artifacts)")
    big = [s for _, s, _ in shifts]
    print(f"\nshift range {min(big):+.4f} to {max(big):+.4f}; "
          f"a pool that grew makes nearest neighbours closer, so shifts should be mostly negative")
    print("\nDownstream numbers built on originality must be regenerated: composite, leaderboard, "
          "radar/profile figures, and any report quoting an originality or composite value.")


if __name__ == "__main__":
    main()
