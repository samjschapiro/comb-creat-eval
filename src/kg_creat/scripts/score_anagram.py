"""Score the anagram (exploratory-creativity) run -- fully judge-free.

Per candidate: deterministic letter-multiset validity, lexicon/Wikidata meaningfulness, and novelty
(embedding distance from the source, the same remoteness measure as the combinatorial tasks). Per
draw: yield (valid+meaningful count) and within-response diversity (mean pairwise distance among the
draw's valid candidates -- the set-level signal, since the model lists many in one response). Rolls up
per (model, temperature).

Run with the MLX venv (embedder):
    .venv_mlx/bin/python src/kg_creat/scripts/score_anagram.py \
        --responses data/kg_creat/responses_anagram --out data/kg_creat/anagram_scores \
        [--no-entity-check]  # disable the Wikidata proper-noun fallback (offline/faster)
"""

import argparse
import itertools
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.kg_creat import anagram  # noqa: E402
from src.kg_creat.embed import get_embedder  # noqa: E402
from src.kg_creat.scoring import cosine_distance  # noqa: E402


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(st.mean(xs), 4) if xs else None


def score_draw(rec, embed, entity_check):
    """Score one draw: annotate each candidate, and compute per-draw yield + within-response diversity."""
    src = rec["u_label"]
    cands = rec.get("candidates") or []
    scored = []
    for c in cands:
        s = anagram.score_candidate(src, c, embed=embed, entity_check=entity_check)
        scored.append(s)
    valid = [s for s in scored if s["valid_anagram"]]
    util = [s for s in scored if s["utility"]]              # valid AND meaningful
    # within-response diversity: mean pairwise embedding distance among the valid candidates' vectors
    div = None
    if len(valid) >= 2:
        vecs = [embed(s["candidate"]) for s in valid]
        div = _mean([cosine_distance(a, b) for a, b in itertools.combinations(vecs, 2)])
    return {
        "prompt_id": rec["prompt_id"], "u_label": src, "temperature": rec["temperature"],
        "domain_u": rec.get("domain_u"),
        "n_candidates": len(scored),
        "n_valid": len(valid), "n_meaningful": len(util),
        "hallucinated": len(scored) - len(valid),          # right-shaped letters but wrong multiset
        "yield": len(util),                                # usable anagrams this draw
        "abstained": len(scored) == 0,
        "within_div": div,
        "novelty_util": _mean([s.get("novelty") for s in util]),   # distance of usable anagrams
        "candidates": scored,
    }


def rollup(draws):
    """Aggregate scored draws into (model-level) rates. Rates over candidates use candidate totals."""
    n_draws = len(draws)
    tot_c = sum(d["n_candidates"] for d in draws)
    tot_v = sum(d["n_valid"] for d in draws)
    tot_m = sum(d["n_meaningful"] for d in draws)
    return {
        "n_draws": n_draws,
        "abstention_rate": round(sum(d["abstained"] for d in draws) / n_draws, 4) if n_draws else None,
        "mean_candidates": round(tot_c / n_draws, 3) if n_draws else None,
        "mean_yield": round(sum(d["yield"] for d in draws) / n_draws, 3) if n_draws else None,
        "valid_rate": round(tot_v / tot_c, 4) if tot_c else None,          # of emitted candidates
        "hallucination_rate": round((tot_c - tot_v) / tot_c, 4) if tot_c else None,
        "meaningful_rate": round(tot_m / tot_c, 4) if tot_c else None,
        "mean_novelty_util": _mean([d["novelty_util"] for d in draws]),
        "mean_within_div": _mean([d["within_div"] for d in draws]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--no-entity-check", action="store_true",
                    help="disable Wikidata proper-noun fallback (default: enabled)")
    args = ap.parse_args()

    entity_check = None
    if not args.no_entity_check:
        from src.kg_creat.wikidata import is_entity_label
        entity_check = is_entity_label

    embed = get_embedder()
    resp_root = Path(args.responses)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(d for d in resp_root.iterdir() if (d / "responses.json").exists())
    overall = {}
    for md in model_dirs:
        recs = json.loads((md / "responses.json").read_text())
        recs = [r for r in recs if r.get("api_error") is None]
        scored = [score_draw(r, embed, entity_check) for r in recs]
        (out / f"{md.name}.json").write_text(json.dumps(scored, indent=2, default=str))
        model_id = recs[0]["prompt_id"] if False else md.name
        by_temp = {}
        for t, grp in itertools.groupby(sorted(scored, key=lambda d: d["temperature"]),
                                        key=lambda d: d["temperature"]):
            by_temp[str(t)] = rollup(list(grp))
        overall[md.name] = {"all": rollup(scored), "by_temperature": by_temp}
        a = overall[md.name]["all"]
        print(f"{md.name:<40} draws={a['n_draws']:<4} abst={a['abstention_rate']} "
              f"yield={a['mean_yield']} valid={a['valid_rate']} halluc={a['hallucination_rate']} "
              f"mean_nov={a['mean_novelty_util']} div={a['mean_within_div']}")

    (out / "summary.json").write_text(json.dumps(overall, indent=2))
    print(f"\nWrote {len(model_dirs)} model score files + summary.json to {out}")


if __name__ == "__main__":
    main()
