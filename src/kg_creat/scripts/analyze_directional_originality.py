"""Directional originality: how far one group's answers sit from its own group vs. from the other group.

For an artifact a from group X and a reference group Y, O(a | Y) is the benchmark's originality computed
against Y alone: the mean, over a's non-anchor elements (concepts and relations, score.py
_artifact_elements), of the mean cosine distance to the element's k=5 nearest neighbours among Y's
elements for the same item. Four quantities per task:

  O(H|H)  human answers vs. other humans      O(H|L)  human answers vs. models
  O(L|L)  model answers vs. other models      O(L|H)  model answers vs. humans

Same-producer answers are left out: a model answered each item up to 5 times, and those draws are not
"other models", so O(L|L) compares a model's answer only with OTHER models' answers.

Pool size is controlled. kNN distance falls as the reference pool grows, and the model pool per item is
roughly 15x the human one, so every reference pool is a random draw of the SAME number of element
surfaces per item (M = the human pool's size minus the query artifact), excluding the query artifact's
own elements, averaged over REPEATS draws. Within-group references exclude the query artifact itself.

Groups: humans = kept sessions (data/kg_creat/human_study/score_input/human_kept), models = the 35 scored
models' kombine_test30 answers (all draws). The two warm-up items have no model answers and are skipped.
Item-level means; 95% bootstrap CIs over items.

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_directional_originality
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.embed import get_embedder
from src.kg_creat.scripts.score import _artifact_elements

HUMAN = "data/kg_creat/human_study/score_input/human_kept/responses.json"
LLM = "data/kg_creat/kombine_test30/responses"
SCORED = "data/kg_creat/kombine_test30/scores"
OUT = Path("data/kg_creat/human_study/analysis/directional_originality.json")
K, REPEATS, BOOT, SEED = 5, 20, 2000, 0
TASK = {"baseline": "association", "analogy": "analogy", "blending": "blending"}


def artifacts(responses, source):
    """{prompt_id: [element-surface set per artifact]} (non-anchor concepts + relations), the producer of each
    artifact ({prompt_id: [source]}: the session for humans, the model for models), and the mode per item."""
    arts, src, mode = defaultdict(list), defaultdict(list), {}
    for r in responses:
        for it in r.get("items") or []:
            els = {s for _k, s in _artifact_elements(it.get("paths") or [], r.get("u_label"), r.get("v_label"))}
            if els:
                arts[r["prompt_id"]].append(els)
                src[r["prompt_id"]].append(source(r))
                mode[r["prompt_id"]] = r["mode"]
    return arts, src, mode


def score(query, ref_arts, keep, M, vec, rng):
    """O(query | ref): mean over query elements of the mean distance to K nearest in a size-M draw of the
    surfaces of the reference artifacts with keep[i] True (the query's own producer is left out)."""
    pool = sorted(set().union(*[a for i, a in enumerate(ref_arts) if keep[i]]) - query)
    if len(pool) < M or M < K:
        return None
    vals = []
    for _ in range(REPEATS):
        draw = rng.choice(len(pool), size=M, replace=False)
        P = np.stack([vec[pool[j]] for j in draw])
        Q = np.stack([vec[s] for s in sorted(query)])
        d = 1.0 - Q @ P.T
        vals.append(np.sort(d, axis=1)[:, :K].mean(axis=1).mean())
    return float(np.mean(vals))


def main():
    rng = np.random.default_rng(SEED)
    human, hsrc, mode = artifacts(json.load(open(HUMAN)), lambda r: r["session_id"])
    scored = {Path(p).parent.name for p in glob.glob(f"{SCORED}/*/summary.json")}
    llm_resp = [{**r, "_model": d.name} for d in sorted(Path(LLM).iterdir()) if d.name in scored
                for r in json.load(open(d / "responses.json"))]
    llm, lsrc, _ = artifacts(llm_resp, lambda r: r["_model"])
    items = sorted(pid for pid in human if pid in llm)
    print(f"{len(items)} items; {sum(len(human[p]) for p in items)} human and "
          f"{sum(len(llm[p]) for p in items)} model artifacts ({len(scored)} models)")

    embed = get_embedder()
    surfaces = {s for p in items for a in human[p] + llm[p] for s in a}
    vec = {}
    for s in surfaces:
        v = embed(s)
        vec[s] = v / (np.linalg.norm(v) or 1.0)
    print(f"embedded {len(surfaces)} element surfaces")

    per_item = {}
    for pid in items:
        H, L = human[pid], llm[pid]
        if len(H) < 2:
            continue
        # equal reference size: the human pool without one artifact, in distinct surfaces (smallest over queries)
        M = min(len(set().union(*[a for j, a in enumerate(H) if j != i]) - H[i]) for i in range(len(H)))
        out = {"HH": [], "HL": [], "LL": [], "LH": []}
        allH, allL = [True] * len(H), [True] * len(L)
        for i, a in enumerate(H):
            out["HH"].append(score(a, H, [s != hsrc[pid][i] for s in hsrc[pid]], M, vec, rng))
            out["HL"].append(score(a, L, allL, M, vec, rng))
        for i, a in enumerate(L):   # a model's other draws for the same item are not "other models"
            out["LL"].append(score(a, L, [s != lsrc[pid][i] for s in lsrc[pid]], M, vec, rng))
            out["LH"].append(score(a, H, allH, M, vec, rng))
        per_item[pid] = {"task": TASK[mode[pid]], "n_human": len(H), "n_llm": len(L), "M": M,
                         **{k: float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
                            for k, v in out.items()}}

    summary = {}
    for task in ("association", "analogy", "blending"):
        rows = [r for r in per_item.values() if r["task"] == task and all(r[k] is not None for k in ("HH", "HL", "LL", "LH"))]
        X = np.array([[r["HH"], r["HL"], r["LL"], r["LH"]] for r in rows])
        boot = X[rng.integers(0, len(X), size=(BOOT, len(X)))].mean(axis=1)
        diffs = {"HL-HH": (1, 0), "LH-LL": (3, 2), "HH-LL": (0, 2)}
        summary[task] = {
            "n_items": len(rows), "median_M": int(np.median([r["M"] for r in rows])),
            **{k: {"mean": round(float(X[:, j].mean()), 4),
                   "ci95": [round(float(x), 4) for x in np.percentile(boot[:, j], [2.5, 97.5])]}
               for j, k in enumerate(["O(H|H)", "O(H|L)", "O(L|L)", "O(L|H)"])},
            **{k: {"mean": round(float((X[:, a] - X[:, b]).mean()), 4),
                   "ci95": [round(float(x), 4) for x in np.percentile(boot[:, a] - boot[:, b], [2.5, 97.5])]}
               for k, (a, b) in diffs.items()},
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "per_item": per_item,
               "params": {"k": K, "repeats": REPEATS, "boot": BOOT, "seed": SEED}}, open(OUT, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
