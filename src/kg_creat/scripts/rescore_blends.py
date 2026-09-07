"""Blend-only re-score: re-judge ONLY the blending records (the re-elicited v3 blends with the new "uv"
shared-slot tag) and MERGE them back into each model's existing path_scores.json, leaving the analogy /
baseline scores untouched (never re-spends the expensive o3 panel on association). Reuses score.py's
exact per-model flow (score_free -> pool-relative originality -> fusion + emergent panels -> finalize)
so the merged blend records are structurally identical to a full run. Resume-safe and per-model
incremental: each model is written the moment it finishes, so a mid-run key cap keeps completed models.

    .venv_mlx/bin/python -m src.kg_creat.scripts.rescore_blends configs/kg_creat/kombine_test30_panel_score.yaml
"""
import argparse
import asyncio
import json
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.utils import load_config  # noqa: E402
from src.kg_creat.embed import get_embedder  # noqa: E402
from src.kg_creat import judge as J  # noqa: E402
from src.kg_creat.aggregate import aggregate  # noqa: E402
from src.kg_creat.scripts.score import (  # noqa: E402
    build_item_element_pool, score_free, score_originality, run_judges, run_emergent_judge,
    finalize_sat, finalize_regime_b, _draw_key)

_DUMP = dict(indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)


async def main(config_path):
    config = load_config(config_path)
    responses_dir = Path(config["upstream_dir"])
    scores_dir = Path(config["output_dir"])
    jc = config.get("judge", {})
    judge_models = jc.get("models") or [jc.get("model", "openai/gpt-oss-120b")]
    factuality_model = jc.get("factuality_model", "openai/gpt-oss-120b")
    panel_open_ended = jc.get("panel_open_ended", True)
    concurrency = jc.get("concurrency", 8)
    embed = get_embedder(config.get("embedding", {}).get("model", "mlx-community/all-MiniLM-L6-v2-4bit"))

    model_dirs = [d for d in sorted(responses_dir.iterdir()) if (d / "responses.json").exists()]
    print(f"Blend re-score over {len(model_dirs)} models; panel: {', '.join(judge_models)}")
    # blend originality is pool-relative -> build the element pool over the NEW blends first.
    item_pool = build_item_element_pool(model_dirs, embed)
    print(f"Built element pool over {len(item_pool)} items.")
    J.reset_judge_usage()

    for md in model_dirs:
        sp = scores_dir / md.name / "path_scores.json"
        if not sp.exists():
            print(f"  {md.name}: no existing path_scores.json -- SKIP (run full score.py first)")
            continue
        responses = json.loads((md / "responses.json").read_text())
        by_prompt = {_draw_key(r): r for r in responses}
        blend_resp = [r for r in responses if r.get("mode") == "blending"]
        recs = []
        for r in blend_resp:
            recs.extend(score_free(r, embed))
        score_originality(recs, by_prompt, item_pool)          # base originality (all-triple; split step refines)
        n_paths = sum(1 for r in recs if r["triples"])
        print(f"  {md.name}: {len(blend_resp)} blends, {n_paths} to judge ...")
        await run_judges(recs, by_prompt, judge_models, concurrency, panel_open_ended, factuality_model)
        await run_emergent_judge(recs, by_prompt, judge_models, concurrency, panel_open_ended)
        for rec in recs:
            finalize_sat(rec)
        finalize_regime_b(recs, embed)

        # merge: keep non-blend scored recs, replace blend recs with the freshly scored ones.
        bak = sp.with_suffix(".json.bak_pre_blendv3")
        if not bak.exists():
            shutil.copy2(sp, bak)
        existing = json.loads(sp.read_text())
        kept = [r for r in existing if r.get("mode") != "blending"]
        merged = kept + recs
        sp.write_text(json.dumps(merged, **_DUMP))
        summary = aggregate(merged)
        (scores_dir / md.name / "summary.json").write_text(json.dumps(summary, indent=2))
        n_int = sum(1 for r in recs if (r.get("blend_integration") or 0) >= 2)
        print(f"    merged {len(recs)} blends (kept {len(kept)} non-blend); double-scope+ {n_int}/{len(recs)}")

    # refresh the top-level scores_summary.json from the per-model summaries
    all_sum = {}
    for md in model_dirs:
        f = scores_dir / md.name / "summary.json"
        if f.exists():
            all_sum[md.name] = json.loads(f.read_text())
    (scores_dir / "scores_summary.json").write_text(json.dumps(all_sum, indent=2))

    if not bool(os.environ.get("LLM_BASE_URL")):
        from src.kg_creat.cost_ledger import record
        for jm, u in J.get_judge_usage().items():
            e = record("score", jm, u["calls"], u["in"], u["out"], config="blendv3_rescore")
            c = f"${e['cost_usd']:.4f}" if e["cost_usd"] is not None else "unpriced"
            print(f"  [ledger] {jm}: {u['calls']} calls, {u['in']:,}+{u['out']:,} tok -> {c}")
    print("done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config_path")
    args = ap.parse_args()
    asyncio.run(main(args.config_path))
