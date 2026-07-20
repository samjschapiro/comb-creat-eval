"""Build a BLIND human-review set to measure LLM-judge reliability (CREATE-style).

Samples a stratified subset of the judge's verdicts, strips the verdict, and writes CSVs a
human fills in without seeing what the judge said. The judge verdicts are saved separately in
``_judge_key.json`` (do NOT open it while reviewing). ``score_review.py`` then computes
human-vs-judge agreement (Cohen's kappa, and precision/recall/balanced-acc for factuality).

Two review sets:
  - factuality: individual triples (with full-path context), balanced judge-true / judge-hallucinated;
  - analogy:    analogy pairs (both structures), balanced judge-valid / judge-invalid.

    .venv_mlx/bin/python src/kg_creat/scripts/sample_review.py data/kg_creat/scores_analogy_v2
"""

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def _chain(triples):
    return triples[0][0] + "".join(f"  --[{t[1]}]-->  {t[2]}" for t in triples)


def collect(scores_dir):
    fact_items, ana_items = [], []
    for md in Path(scores_dir).glob("*/path_scores.json"):
        model = md.parent.name
        recs = json.loads(md.read_text())
        byp = defaultdict(dict)
        for r in recs:
            if r["mode"] == "analogy":
                byp[r["prompt_id"]][r["path_idx"]] = r
        for r in recs:
            fv = r.get("factual")
            if not isinstance(fv, list) or not r.get("triples"):
                continue
            for i, (t, verdict) in enumerate(zip(r["triples"], fv)):
                fact_items.append({
                    "model": model, "prompt_id": r["prompt_id"], "triple_idx": i,
                    "path": _chain(r["triples"]),
                    "triple": f"({t[0]}, {t[1]}, {t[2]})",
                    "judge": "true" if verdict else "hallucinated",
                })
        for pid, paths in byp.items():
            p0, p1 = paths.get(0), paths.get(1)
            if not p0 or p1 is None or p0.get("semantic_sat") is None:
                continue
            ana_items.append({
                "model": model, "prompt_id": pid,
                "concept_A": p0["u_label"], "concept_B": p0["v_label"],
                "structure_A": _chain(p0["triples"]), "structure_B": _chain(p1["triples"]),
                "judge": "valid" if p0["semantic_sat"] else "invalid",
            })
    return fact_items, ana_items


def _stratified(items, n, key, seed):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for it in items:
        groups[it[key]].append(it)
    per = max(1, n // max(1, len(groups)))
    picked = []
    for g in groups.values():
        rng.shuffle(g)
        picked += g[:per]
    rng.shuffle(picked)
    return picked[:n]


def main(scores_dir, n_factuality=60, n_analogy=40, seed=0):
    out = Path(scores_dir) / "human_review"
    out.mkdir(parents=True, exist_ok=True)
    fact_items, ana_items = collect(scores_dir)
    fact = _stratified(fact_items, n_factuality, "judge", seed)
    ana = _stratified(ana_items, n_analogy, "judge", seed)

    key = {}
    with open(out / "review_factuality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "path (context)", "triple_to_judge", "YOUR_VERDICT (true / hallucinated)"])
        for i, it in enumerate(fact):
            iid = f"F{i}"
            key[iid] = it["judge"]
            w.writerow([iid, it["path"], it["triple"], ""])
    with open(out / "review_analogy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "concept_A", "concept_B", "structure_A", "structure_B",
                    "YOUR_VERDICT (valid / invalid)"])
        for i, it in enumerate(ana):
            iid = f"A{i}"
            key[iid] = it["judge"]
            w.writerow([iid, it["concept_A"], it["concept_B"], it["structure_A"], it["structure_B"], ""])

    (out / "_judge_key.json").write_text(json.dumps(key, indent=2))
    (out / "README.md").write_text(
        "# Blind judge-reliability review\n\n"
        "Fill the `YOUR_VERDICT` column in each CSV. **Do not open `_judge_key.json`** until done.\n\n"
        "- **review_factuality.csv** — for each triple (judged in the context of its path), is it a\n"
        "  true/plausible real-world fact? Write `true` or `hallucinated`.\n"
        "- **review_analogy.csv** — do the two structures form a genuine analogy (same relations, entities\n"
        "  playing corresponding roles in two distinct systems)? Write `valid` or `invalid`.\n\n"
        "Then run: `score_review.py <this scores dir>` for human-vs-judge agreement.\n")

    print(f"Wrote blind review set -> {out}")
    print(f"  review_factuality.csv : {len(fact)} triples "
          f"({sum(1 for k in key if k.startswith('F') and key[k]=='true')} judge-true / "
          f"{sum(1 for k in key if k.startswith('F') and key[k]=='hallucinated')} judge-hallucinated)")
    print(f"  review_analogy.csv    : {len(ana)} analogies "
          f"({sum(1 for k in key if k.startswith('A') and key[k]=='valid')} judge-valid / "
          f"{sum(1 for k in key if k.startswith('A') and key[k]=='invalid')} judge-invalid)")


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_analogy_v2"
    main(d)
