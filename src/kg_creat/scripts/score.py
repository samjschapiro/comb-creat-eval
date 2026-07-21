"""Score elicited paths -> per-path records + the per-constraint ideation/execution roll-up.

Downstream of ``run_elicit.py``. Two phases:
  1. FREE  (always): parse -> well-formedness, exact constraints (exclusion/inclusion/
     ordering), novelty R (local MLX embeddings). No API.
  2. JUDGE (config ``judge.enabled``): factuality (CREATE K.2), categorical sat, and the
     Regime-B analogy/blending semantic sat -- via OpenRouter, budget-checked.

Then aggregates per mode (R_emit / R_valid / sat + failure channels) and writes the 2x2.
Run in the 3.12 env (.venv_mlx) so MLX embeddings are available:

    .venv_mlx/bin/python src/kg_creat/scripts/score.py configs/kg_creat/score.yaml --overwrite
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, init_directory, save_config  # noqa: E402
from src.kg_creat import scoring  # noqa: E402
from src.kg_creat.scoring import EmittedPath  # noqa: E402
from src.kg_creat.embed import get_embedder  # noqa: E402
from src.kg_creat import judge as J  # noqa: E402
from src.kg_creat import regime_b as RB  # noqa: E402
from src.kg_creat.aggregate import aggregate  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "safety"))
from cost_tracker import PRICING  # noqa: E402


def score_free(response: dict, embed) -> list[dict]:
    """Phase 1: per emitted path, the free signals (well-formedness + novelty). One record per path.

    Constraint satisfaction is judged (Phase 2) under open vocabulary, so nothing constraint-related
    is computed here.
    """
    recs = []
    spec_keys = ("prompt_id", "bundle_id", "regime", "mode", "u_label", "v_label", "h", "k",
                 "domain_u", "domain_v", "cross_domain")
    base = {k: response.get(k) for k in spec_keys}
    for pi, triples in enumerate(response["paths"]):
        rec = {**base, "path_idx": pi, "triples": triples, "n_hops": len(triples)}
        if not triples:
            rec.update({"well_formed": False, "wf_reason": "empty", "R": None})
            recs.append(rec)
            continue
        p = EmittedPath(triples)
        if response["regime"] == "A":
            wf, reason = scoring.well_formed(p, base["u_label"], base["v_label"], h=None)  # variable length (CREATE-style)
        else:  # Regime B: coherent chain, no revisited entities (reject circular / self-referential structures)
            if not scoring.is_continuous(p):
                wf, reason = False, "discontinuous"
            elif len(set(p.entities)) != len(p.entities):
                wf, reason = False, "revisits_node"
            else:
                wf, reason = True, "ok"
        rec["well_formed"] = wf
        rec["wf_reason"] = reason
        rec["R"] = scoring.novelty_R(p, embed, unit="triple")
        recs.append(rec)
    return recs


async def run_judges(recs: list[dict], responses_by_prompt: dict, model: str, concurrency: int):
    """Phase 2: fill factuality / categorical / semantic sat on the records (in place).

    Uses the LLM_BASE_URL-aware client, so the judge runs on the local MLX server when
    LLM_BASE_URL is set (fully-local, free) and on OpenRouter otherwise.
    """
    from src.dat_eval.llm import get_async_client
    client = get_async_client()
    sem = asyncio.Semaphore(concurrency)

    async def factual(rec):
        async with sem:
            rec["factual"] = await J.judge_factuality(client, model, rec["triples"]) if rec["triples"] else None

    async def categorical(rec):
        c = responses_by_prompt[rec["prompt_id"]]["constraint"]
        async with sem:
            out = await J.judge_categorical(client, model, c["type_label"], rec["triples"])
            rec["constraint_sat"] = bool(out.get("satisfied")) if out else None

    async def relation_con(rec):
        c = responses_by_prompt[rec["prompt_id"]]["constraint"]
        async with sem:
            out = await J.judge_relation_constraint(client, model, c, rec["triples"])
            rec["constraint_sat"] = bool(out.get("satisfied")) if out else None

    async def analogy(rec):
        paths = responses_by_prompt[rec["prompt_id"]]["paths"]
        if len(paths) < 2:
            rec["semantic_sat"] = False
            return
        async with sem:
            out = await J.judge_analogy(client, model, rec["u_label"], rec["v_label"], paths[0], paths[1])
            rec["semantic_sat"] = bool(out.get("valid")) if out else None

    async def blending(rec):
        paths = responses_by_prompt[rec["prompt_id"]]["paths"]
        if len(paths) < 2:
            rec["semantic_sat"] = False
            return
        async with sem:
            out = await J.judge_blending(client, model, rec["u_label"], paths[0], paths[1])
            rec["semantic_sat"] = bool(out.get("valid")) if out else None
            rec["blend_domains"] = out.get("domains") if out else None

    tasks, skipped = [], 0
    for rec in recs:
        if not rec["triples"]:
            continue
        # `sat` is first-failing-gate: a structurally malformed path fails on the 'structural'
        # channel regardless of factuality/constraint, so judging it is wasted spend.
        if not rec.get("well_formed"):
            skipped += 1
            continue
        tasks.append(factual(rec))
        if rec["mode"] == "categorical":
            tasks.append(categorical(rec))
        elif rec["mode"] in ("exclusion", "inclusion", "inclusion_rare", "ordering"):
            tasks.append(relation_con(rec))
        elif rec["mode"] == "analogy" and rec["path_idx"] == 0:  # judge the pair once
            tasks.append(analogy(rec))
        elif rec["mode"] == "blending" and rec["path_idx"] == 0:  # judge the branch pair once
            tasks.append(blending(rec))
    if skipped:
        print(f"    (skipping {skipped} structurally-failed paths — they fail on the "
              f"'structural' channel regardless, so no judge call is needed)")
    await asyncio.gather(*tasks)


def finalize_sat(rec: dict):
    """Combine gates into sat + the first failing channel."""
    if rec["regime"] == "A":
        if not rec["well_formed"]:
            rec["sat"], rec["channel"] = False, "structural"
            return
        factual = rec.get("factual")
        if factual is not None and not all(factual):
            rec["sat"], rec["channel"] = False, "factual"
            return
        con = True if rec["mode"] == "baseline" else rec.get("constraint_sat")
        if con is False:
            rec["sat"], rec["channel"] = False, "constraint"
            return
        if factual is None or con is None:  # judge not run / failed to parse
            rec["sat"], rec["channel"] = None, "unjudged"
            return
        rec["sat"], rec["channel"] = True, "ok"
    else:  # Regime B
        if not rec["well_formed"]:
            rec["sat"], rec["channel"] = False, "structural"
            return
        factual = rec.get("factual")
        sem = rec.get("semantic_sat")
        if factual is not None and not all(factual):
            rec["sat"], rec["channel"] = False, "factual"
        elif sem is False:
            rec["sat"], rec["channel"] = False, "semantic"
        elif factual is None or sem is None:
            rec["sat"], rec["channel"] = None, "unjudged"
        else:
            rec["sat"], rec["channel"] = True, "ok"


def finalize_regime_b(recs: list[dict], embed):
    """Pair-level verdict for the semantic tier, written onto each prompt's first path record.

    Analogy and blending are judged over a PAIR of structures, so per-path ``sat`` cannot express
    them: the unit of success is the mapping, not the path. This lives here (rather than in the
    plotters) so the figures and any downstream analysis inherit one definition of validity.

    Novelty for blending is the distance between the two branch TIPS -- how far apart the domains
    the model reached are -- which is the quantity the task actually asks it to maximise.
    """
    from collections import defaultdict
    by_prompt = defaultdict(dict)
    for r in recs:
        if r["mode"] in ("analogy", "blending"):
            by_prompt[r["prompt_id"]][r["path_idx"]] = r
    for paths in by_prompt.values():
        head = paths.get(0)
        if head is None:
            continue
        p0, p1 = head.get("triples"), (paths.get(1) or {}).get("triples")
        if head["mode"] == "analogy":
            ok, reason = RB.analogy_structural_ok(p0, p1)
        else:
            ok, reason = RB.blend_structural_ok(p0, p1, head["u_label"])
        fact = [f for r in paths.values() for f in (r.get("factual") or [])]
        sem = head.get("semantic_sat")
        head["pair_structural_ok"], head["pair_structural_reason"] = ok, reason
        if not ok:
            head["pair_sat"], head["pair_channel"] = False, "structural"
        elif not fact or not all(fact):
            head["pair_sat"], head["pair_channel"] = False, "factual"
        elif sem is False:
            head["pair_sat"], head["pair_channel"] = False, "semantic"
        elif sem is None:
            head["pair_sat"], head["pair_channel"] = None, "unjudged"
        else:
            head["pair_sat"], head["pair_channel"] = True, "ok"
        if head["mode"] == "blending" and p0 and p1:
            a, b = RB.branch_tips(p0, p1)
            head["tip_distance"] = float(scoring.cosine_distance(embed(a), embed(b)))


async def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"FATAL: upstream_dir not found: {upstream_dir}")
    # Resume-safe: --overwrite starts fresh; otherwise keep the dir and skip already-scored
    # models (judge spend is expensive, never redo a completed model).
    if overwrite:
        output_dir = init_directory(config["output_dir"], overwrite=True)
    else:
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    judge_cfg = config.get("judge", {})
    judge_enabled = judge_cfg.get("enabled", False)
    judge_model = judge_cfg.get("model", "openai/gpt-oss-120b")
    concurrency = judge_cfg.get("concurrency", 8)
    embed = get_embedder(config.get("embedding", {}).get("model", "mlx-community/all-MiniLM-L6-v2-4bit"))

    model_dirs = [d for d in upstream_dir.iterdir() if (d / "responses.json").exists()]
    print(f"Scoring {len(model_dirs)} model(s); judge_enabled={judge_enabled} ({judge_model})")

    all_summaries = {}
    for md in model_dirs:
        # Resume: skip a model already scored in this output dir (unless --overwrite).
        done_scores = output_dir / md.name / "summary.json"
        if done_scores.exists():
            print(f"  {md.name}: already scored, skipping (use --overwrite to redo)")
            all_summaries[md.name] = json.loads(done_scores.read_text())
            continue
        responses = json.loads((md / "responses.json").read_text())
        if debug:
            responses = responses[:6]
        by_prompt = {r["prompt_id"]: r for r in responses}
        recs = []
        for r in responses:
            recs.extend(score_free(r, embed))
        print(f"  {md.name}: {len(responses)} prompts, {len(recs)} paths (free scored)")

        if judge_enabled:
            n_paths = sum(1 for rec in recs if rec["triples"])
            local = bool(os.environ.get("LLM_BASE_URL"))
            if local:
                print(f"    running LOCAL judge ({judge_model}) on ~{n_paths} paths (free) ...")
            else:
                est = (n_paths * 700 * PRICING.get(judge_model, (9, 9))[0]
                       + n_paths * 300 * PRICING.get(judge_model, (9, 9))[1]) / 1e6
                print(f"    running judge on ~{n_paths} paths (est ~${est:.3f}) ...")
            await run_judges(recs, by_prompt, judge_model, concurrency)

        for rec in recs:
            finalize_sat(rec)
        finalize_regime_b(recs, embed)

        out_md = output_dir / md.name
        out_md.mkdir(parents=True, exist_ok=True)
        (out_md / "path_scores.json").write_text(json.dumps(recs, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))
        summary = aggregate(recs)
        (out_md / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries[md.name] = summary

    (output_dir / "scores_summary.json").write_text(json.dumps(all_summaries, indent=2))
    print(f"\nScores + per-constraint roll-up saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.config_path, args.overwrite, args.debug))
