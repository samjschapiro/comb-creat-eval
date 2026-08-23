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
from collections import Counter, defaultdict
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


def _norm(x) -> str:
    return str(x).strip().lower()


def _artifact_elements(triple_lists, u, v) -> set:
    """Non-anchor elements of an artifact: its concepts (minus anchors u,v) and its relations.

    Elements are tagged ('c', concept) / ('r', relation) so a concept and a relation that share a
    surface string are counted separately. Used for the item-specific originality frequency.
    """
    au, av = _norm(u), _norm(v)
    els = set()
    for triples in triple_lists:
        for t in triples:
            if len(t) != 3:
                continue
            h, r, tl = _norm(t[0]), _norm(t[1]), _norm(t[2])
            els.add(("r", r))
            if h not in (au, av):
                els.add(("c", h))
            if tl not in (au, av):
                els.add(("c", tl))
    return els


def build_item_frequency(model_dirs) -> dict:
    """Per-item element frequency p_a(e): fraction of RESPONSES to an item (a prompt/pair) whose
    artifacts contain element e, pooled across all models (the psychometric item-bank analogue,
    \\citealt Stevenson2022). Each (model, draw) response counts once; empty responses are skipped."""
    from collections import Counter, defaultdict
    total = Counter()
    elem = defaultdict(Counter)
    for md in model_dirs:
        try:
            responses = json.loads((md / "responses.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        for r in responses:
            paths = r.get("paths") or []
            if not any(paths):
                continue
            pid = r["prompt_id"]
            total[pid] += 1
            for e in _artifact_elements(paths, r.get("u_label"), r.get("v_label")):
                elem[pid][e] += 1
    return {pid: {e: c / total[pid] for e, c in ec.items()} for pid, ec in elem.items()}


def score_originality(recs, responses_by_prompt, item_freq):
    """Item-specific originality per artifact: mean inverse frequency p_a(e)^-1 over its non-anchor
    elements E_a (Table 3). Written onto the artifact's head record (per path for association, the
    even pair-head for analogy, the single structure record for blending)."""
    rec_index = {_draw_key(r) + (r["path_idx"],): r for r in recs}
    for r in responses_by_prompt.values():
        freq = item_freq.get(r["prompt_id"], {})
        u, v = r.get("u_label"), r.get("v_label")
        stride = 2 if r["mode"] == "analogy" else 1
        for idx, it in enumerate(r.get("items") or []):
            head = rec_index.get(_draw_key(r) + (idx * stride,))
            if head is None:
                continue
            ea = _artifact_elements(it["paths"], u, v)
            vals = [1.0 / freq[e] for e in ea if freq.get(e, 0) > 0]
            head["originality"] = (sum(vals) / len(vals)) if vals else None


def _draw_key(r: dict) -> tuple:
    """Unique key for one elicitation DRAW. All temperatures/samples of a prompt share ``prompt_id``,
    so judges and pair-finalizers must key on (prompt_id, temperature, sample_idx) -- keying on
    prompt_id alone collapses the draws and mixes one draw's structure with another's verdict."""
    return (r["prompt_id"], r.get("temperature"), r.get("sample_idx"))


def score_free(response: dict, embed) -> list[dict]:
    """Phase 1: per emitted path, the free signals (well-formedness + novelty). One record per path.

    Constraint satisfaction is judged (Phase 2) under open vocabulary, so nothing constraint-related
    is computed here.
    """
    recs = []
    # temperature + sample_idx are carried so judges/finalizers can key by DRAW: all temps of one
    # prompt share prompt_id, so keying on prompt_id alone collapses them and judges the wrong draw.
    spec_keys = ("prompt_id", "bundle_id", "regime", "mode", "u_label", "v_label", "h", "k",
                 "domain_u", "domain_v", "cross_domain", "temperature", "sample_idx")
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
        elif response["mode"] == "blending":
            # Fusion blend: "structure" is a STAR of triples about the blend, not a continuous chain,
            # so the chain/revisit checks don't apply -- well-formed = at least one valid triple.
            wf, reason = (True, "ok") if p.triples else (False, "empty")
        else:  # Regime B analogy: coherent chain, no revisited entities (reject circular structures)
            if not scoring.is_continuous(p):
                wf, reason = False, "discontinuous"
            elif len(set(p.entities)) != len(p.entities):
                wf, reason = False, "revisits_node"
            else:
                wf, reason = True, "ok"
        rec["well_formed"] = wf
        rec["wf_reason"] = reason
        if response["mode"] == "blending":
            # Blending surprise is the remoteness of the two FUSED inputs, d_cos(u,v) -- item-driven,
            # not the within-structure spread (docs/tracks/kg_creat/blending_fusion.md).
            a, b = base["u_label"], base["v_label"]
            rec["R"] = float(scoring.cosine_distance(embed(a), embed(b))) if a and b else None
        else:
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
    # Keyed by DRAW + path_idx: analogy emits a SET of pairs flattened as [a0, b0, a1, b1, ...]; each
    # even path_idx is a pair head, its partner path_idx+1. Keying includes temperature/sample_idx so a
    # pair head finds its OWN draw's partner, not another temperature's.
    rec_index = {_draw_key(r) + (r["path_idx"],): r for r in recs}

    FACT_BATCH = 10   # paths per batched factuality call (one call judges the whole batch)

    async def factual_batch(batch):
        async with sem:
            results = await J.judge_factuality_batch(client, model, [r["triples"] for r in batch])
        for r, res in zip(batch, results):
            r["factual"] = res

    async def categorical(rec):
        c = responses_by_prompt[_draw_key(rec)]["constraint"]
        async with sem:
            out = await J.judge_categorical(client, model, c["type_label"], rec["triples"])
            rec["constraint_sat"] = bool(out.get("satisfied")) if out else None

    async def relation_con(rec):
        c = responses_by_prompt[_draw_key(rec)]["constraint"]
        async with sem:
            out = await J.judge_relation_constraint(client, model, c, rec["triples"])
            rec["constraint_sat"] = bool(out.get("satisfied")) if out else None

    async def analogy(rec):
        paths = responses_by_prompt[_draw_key(rec)]["paths"]
        i = rec["path_idx"]
        partner = rec_index.get(_draw_key(rec) + (i + 1,))
        if i + 1 >= len(paths) or partner is None:
            rec["semantic_sat"] = False
            return
        async with sem:
            out = await J.judge_analogy(client, model, rec["u_label"], rec["v_label"], paths[i], paths[i + 1])
        v = bool(out.get("valid")) if out else None
        rec["semantic_sat"] = partner["semantic_sat"] = v  # the verdict is over the pair

    async def blending(rec):
        # Fusion blend: utility is a single judge-only verdict over the one structure (genuine coherent
        # fusion of u,v, hard gate on the generic space). One blend per prompt -> one item (idx 0).
        item = (responses_by_prompt[_draw_key(rec)].get("items") or [{}])[0]
        async with sem:
            out = await J.judge_blend_fusion(client, model, rec["u_label"], rec["v_label"],
                                             item.get("concept", ""), item.get("generic_space", ""),
                                             rec["triples"])
        rec["semantic_sat"] = bool(out.get("valid")) if out else None
        rec["generic_space_ok"] = bool(out.get("generic_space_ok")) if out else None
        rec["double_scope"] = bool(out.get("double_scope")) if out else None

    tasks, skipped, fact_recs = [], 0, []
    for rec in recs:
        if not rec["triples"]:
            continue
        # `sat` is first-failing-gate: a structurally malformed path fails on the 'structural'
        # channel regardless of factuality/constraint, so judging it is wasted spend.
        if not rec.get("well_formed"):
            skipped += 1
            continue
        # Blending utility is judge-only: a blend's structure is intentionally non-factual (novel
        # concept), so factuality is neither a gate nor meaningful here -- skip it (saves spend).
        if rec["mode"] != "blending":
            fact_recs.append(rec)   # factuality is judged in batches (below), not one call per path
        if rec["mode"] == "categorical":
            tasks.append(categorical(rec))
        elif rec["mode"] in ("exclusion", "inclusion", "inclusion_rare", "ordering"):
            tasks.append(relation_con(rec))
        elif rec["mode"] == "analogy" and rec["path_idx"] % 2 == 0:  # judge each pair (head = even idx)
            tasks.append(analogy(rec))
        elif rec["mode"] == "blending":  # one blend per prompt: judge its single structure
            tasks.append(blending(rec))
    for i in range(0, len(fact_recs), FACT_BATCH):   # batched factuality: ~10x fewer calls
        tasks.append(factual_batch(fact_recs[i:i + FACT_BATCH]))
    if skipped:
        print(f"    (skipping {skipped} structurally-failed paths — they fail on the "
              f"'structural' channel regardless, so no judge call is needed)")
    await asyncio.gather(*tasks)


async def run_emergent_judge(recs, responses_by_prompt, model, concurrency):
    """Emergent-creativity tier: per item, count candidate inferences the judge confirms are BOTH true
    and licensed by the whole but not any single part. Result written onto the item's head record
    (``emergent_count`` + the raw ``emergent_inferences``). This is the ``tab:scoring`` emergent metric.
    """
    from src.dat_eval.llm import get_async_client
    client = get_async_client()
    sem = asyncio.Semaphore(concurrency)
    rec_index = {_draw_key(r) + (r["path_idx"],): r for r in recs}

    async def judge_item(draw, head_idx, mode, u_label, v_label, item):
        head = rec_index.get(draw + (head_idx,))
        if head is None:
            return
        paths, inferences = item["paths"], item["inferences"]
        if not inferences:
            head["emergent_count"] = 0
            return
        if mode == "blending":
            # Fusion blend: emergent structure of the blend -- true of the blend, of NEITHER input alone.
            # Judged against the two INPUT concepts (u,v), not against sub-parts of the structure.
            async with sem:
                verdict = await J.judge_blend_emergent(client, model, u_label, v_label,
                                                       item.get("concept", ""), paths[0], inferences)
            head["emergent_count"] = J.count_blend_emergent(verdict, len(inferences))
            head["emergent_inferences"] = inferences
            return
        if mode == "analogy":
            src, tgt = paths[0], paths[1]
            artifact = f"{u_label}'s side: {J.format_path(src)}\n{v_label}'s side: {J.format_path(tgt)}"
            parts = [f"{u_label} side alone: {J.format_path(src)}",
                     f"{v_label} side alone: {J.format_path(tgt)}"]
        else:  # association: parts are the individual links
            path = paths[0]
            artifact = J.format_path(path)
            parts = [f"Link {i + 1}: {J.format_path([t])}" for i, t in enumerate(path)]
        async with sem:
            verdict = await J.judge_emergent(client, model, artifact, parts, inferences)
        head["emergent_count"] = J.count_emergent(verdict, len(inferences))
        head["emergent_inferences"] = inferences

    tasks = []
    for r in responses_by_prompt.values():
        items = r.get("items") or []
        stride = 2 if r["mode"] == "analogy" else 1  # blending is one path per item (like association)
        for item_idx, it in enumerate(items):
            tasks.append(judge_item(_draw_key(r), item_idx * stride, r["mode"],
                                    r.get("u_label"), r.get("v_label"), it))
    if tasks:
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
    elif rec["mode"] == "blending":
        # Fusion utility is JUDGE-ONLY (docs/tracks/kg_creat/blending_fusion.md): a blend is a novel
        # concept whose structure is intentionally false of the real world ("elements false or
        # impossible in both inputs", F&T), so factuality is NOT a gate here -- only a genuine,
        # coherent fusion (the fusion judge, with its hard generic-space gate).
        sem = rec.get("semantic_sat")
        if not rec["well_formed"]:
            rec["sat"], rec["channel"] = False, "structural"
        elif sem is False:
            rec["sat"], rec["channel"] = False, "semantic"
        elif sem is None:
            rec["sat"], rec["channel"] = None, "unjudged"
        else:
            rec["sat"], rec["channel"] = True, "ok"
    else:  # Regime B analogy
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
    """Pair-level verdict for the ANALOGY semantic tier, written onto each prompt's pair-head record.

    Analogy is judged over a PAIR of structures, so per-path ``sat`` cannot express it: the unit of
    success is the mapping, not the path. Blending is NOT pair-based anymore (one fused blend per
    prompt) -- its utility rides the per-path ``sat`` set in ``finalize_sat`` (well-formed -> factual
    -> semantic fusion verdict), so it is handled there, not here.
    """
    from collections import defaultdict
    by_prompt = defaultdict(dict)
    for r in recs:
        if r["mode"] == "analogy":
            by_prompt[_draw_key(r)][r["path_idx"]] = r   # per-draw: don't merge pairs across temps
    for paths in by_prompt.values():
        # each consecutive (even, even+1) is one emitted pair; verdict written onto the even head.
        for i in sorted(k for k in paths if k % 2 == 0):
            head, partner = paths.get(i), paths.get(i + 1)
            if head is None:
                continue
            p0 = head.get("triples")
            p1 = partner.get("triples") if partner else None
            ok, reason = RB.analogy_structural_ok(p0, p1)
            fact = list(head.get("factual") or []) + (list(partner.get("factual") or []) if partner else [])
            sem = head.get("semantic_sat")
            head["pair_idx"] = i // 2
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

    from src.dat_eval.llm import model_id_to_key
    exclude_keys = {model_id_to_key(m) for m in config.get("exclude_models", [])}
    model_dirs = [d for d in upstream_dir.iterdir()
                  if (d / "responses.json").exists() and d.name not in exclude_keys]
    if exclude_keys:
        print(f"Excluding {len(exclude_keys)} model(s): {sorted(exclude_keys)}")
    print(f"Scoring {len(model_dirs)} model(s); judge_enabled={judge_enabled} ({judge_model})")

    # Item-specific originality needs element frequencies pooled across ALL models, so build the
    # per-item table once, up front, before scoring any single model.
    item_freq = build_item_frequency(model_dirs)
    print(f"Built item-frequency table over {len(item_freq)} items (for originality).")

    all_summaries = {}
    genuine_dist = defaultdict(list)   # mode -> [verified-genuine count per (model, prompt)]
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
        by_prompt = {_draw_key(r): r for r in responses}   # per-draw, not per-prompt (temps share id)
        recs = []
        for r in responses:
            recs.extend(score_free(r, embed))
        score_originality(recs, by_prompt, item_freq)   # judge-free; item-specific inverse frequency
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
            await run_emergent_judge(recs, by_prompt, judge_model, concurrency)

        for rec in recs:
            finalize_sat(rec)
        finalize_regime_b(recs, embed)

        # per-prompt verified-genuine count (judge-passed combinations), for the scarcity check
        per_prompt = defaultdict(lambda: [0, None])
        for r in recs:
            per_prompt[r["prompt_id"]][1] = r["mode"]
            genuine = (r.get("pair_sat") is True) if r["mode"] == "analogy" \
                else (r.get("sat") is True)   # blending is one blend per prompt: rides per-path sat
            if genuine:
                per_prompt[r["prompt_id"]][0] += 1
        for cnt, mode in per_prompt.values():
            genuine_dist[mode].append(cnt)

        out_md = output_dir / md.name
        out_md.mkdir(parents=True, exist_ok=True)
        (out_md / "path_scores.json").write_text(json.dumps(recs, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))
        summary = aggregate(recs)
        (out_md / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries[md.name] = summary

    (output_dir / "scores_summary.json").write_text(json.dumps(all_summaries, indent=2))

    # Verified-genuine-count distribution per mode (combinatorial-scarcity check): how many
    # judge-passed combinations each (model, prompt) yielded. A pile-up at 0/1 => the items
    # themselves admit few genuine combinations, not just model homogenization.
    print("\nVerified-genuine combinations per (model, prompt):")
    for mode in ("baseline", "analogy", "blending"):
        vals = genuine_dist.get(mode) or []
        if not vals:
            continue
        hist = Counter(min(v, 3) for v in vals)
        bars = "  ".join(f"{k if k < 3 else '3+'}:{hist.get(k,0)}" for k in (0, 1, 2, 3))
        print(f"  {mode:9s} n={len(vals):3d}  mean={sum(vals)/len(vals):.2f}   [{bars}]")
    print(f"\nScores + per-constraint roll-up saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.config_path, args.overwrite, args.debug))
