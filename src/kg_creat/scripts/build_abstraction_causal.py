"""Build the abstraction-bottleneck CAUSAL test (rebuttal: reviewers eN17, cGKy).

Claim under test: on blending, finding the generic space g is the bottleneck. Correlationally, models find
a valid g only ~43% of the time, but given a valid g of their own they write a coherent blend ~97% of the
time. Here we intervene on g directly. Six models x the 30 blend items x three conditions:

  control  the original benchmark prompt, freshly re-sampled (measures regression to the mean on items a
           model failed in the paper run -- the baseline any improvement must beat)
  hint     the original prompt + a CONTENT-FREE procedural hint for finding g (map each input's roles,
           then abstract the corresponding ones); tests whether scaffolding the abstraction step helps
  oracle   step 1 replaced by a GIVEN, panel-validated g; tests whether, once the abstraction is supplied,
           models can build a coherent, double-scope blend on it

The oracle g for an item is the SAME for every test model: a generic space from a model OUTSIDE the test
set that all three panel judges accepted (generic_ok unanimous) in the paper run, picked at random with a
fixed seed among the eligible ones; one item with no unanimous candidate falls back to a 2/3-majority one
(recorded as panel_yes in oracle_g.json).

Writes data/kg_creat/abstraction_causal/{control,hint,oracle}/prompts/prompts.json and oracle_g.json.

    .venv/bin/python -m src.kg_creat.scripts.build_abstraction_causal --seed 0
"""
import argparse
import json
import random
from pathlib import Path

PROMPTS = Path("data/kg_creat/kombine_test30/prompts/prompts.json")
SCORES = Path("data/kg_creat/human_study/paper_scores")
RESP = Path("data/kg_creat/kombine_test30/responses")
OUT = Path("data/kg_creat/abstraction_causal")
TEST_MODELS = ["qwen/qwen-2.5-72b-instruct", "openai/gpt-4o-mini", "deepseek/deepseek-chat-v3-0324",
               "meta-llama/llama-3.3-70b-instruct", "openai/gpt-5.2", "anthropic/claude-opus-5"]


def model_dir(mid):
    return mid.replace("/", "_").replace(".", "-")


def main(seed):
    rng = random.Random(seed)
    blends = [s for s in json.loads(PROMPTS.read_text()) if s["mode"] == "blending"]
    test_dirs = {model_dir(m) for m in TEST_MODELS}
    cands = {}
    for md in sorted(SCORES.iterdir()):
        ps = md / "path_scores.json"
        if md.name.startswith("human") or md.name in test_dirs or not ps.exists():
            continue
        rp = RESP / md.name / "responses.json"
        if not rp.exists():
            continue
        resp = {(r["prompt_id"], r["sample_idx"]): r for r in json.loads(rp.read_text())}
        for r in json.loads(ps.read_text()):
            js = r.get("blend_judges") or []
            n_yes = sum(j.get("generic_ok") is True for j in js)
            if r["mode"] != "blending" or len(js) != 3 or n_yes < 2:
                continue
            rec = resp.get((r["prompt_id"], r["sample_idx"]))
            g = ((rec or {}).get("items") or [{}])[0].get("generic_space")
            if g:
                cands.setdefault(r["prompt_id"], []).append({"model": md.name, "generic_space": g.strip(),
                                                             "panel_yes": n_yes})
    oracle = {}
    for s in blends:
        allc = cands.get(s["prompt_id"], [])
        best = max((c["panel_yes"] for c in allc), default=0)     # unanimous (3/3) if any, else 2/3 majority
        pool = sorted((c for c in allc if c["panel_yes"] == best), key=lambda c: c["model"])
        if not pool:
            raise SystemExit(f"no unanimous generic space outside the test set for {s['prompt_id']}")
        oracle[s["prompt_id"]] = {**rng.choice(pool), "n_eligible": len(pool), "u": s["u_label"], "v": s["v_label"]}
    for cond in ("control", "hint", "oracle"):
        specs = []
        for s in blends:
            t = dict(s, blend_condition=cond)
            if cond == "oracle":
                t["given_generic_space"] = oracle[s["prompt_id"]]["generic_space"]
            specs.append(t)
        d = OUT / cond / "prompts"
        d.mkdir(parents=True, exist_ok=True)
        (d / "prompts.json").write_text(json.dumps(specs, indent=1))
    (OUT / "oracle_g.json").write_text(json.dumps(oracle, indent=1))
    print(f"{len(blends)} items x 3 conditions -> {OUT}; oracle g eligible per item: "
          f"min {min(o['n_eligible'] for o in oracle.values())}, "
          f"median {sorted(o['n_eligible'] for o in oracle.values())[len(oracle) // 2]}; "
          f"majority-only fallback: {[k for k, o in oracle.items() if o['panel_yes'] < 3]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args().seed)
