"""The paper's scores: the shared human + model originality pool, with every benchmark item kept for the models.

combined_scores_skip0 holds only the 86 items humans also saw. The models answered 88: the two worked-example pairs
(Democracy/Banking, The blue whale/The mattress; analogy E0, E26 and blending F0, F26) were shown to no human, so
their originality pool is the models alone either way -- their records in the original model-only scores
(data/kg_creat/kombine_test30/scores) are exactly what a shared pool would give. This copies combined_scores_skip0 to
paper_scores and adds, for every model, its original records for those items, plus one failed record (as in
build_skip0_scores.py) for any of them the model left without a usable answer. The human groups are copied unchanged.

    .venv/bin/python -m src.kg_creat.scripts.build_paper_scores
    .venv/bin/python src/kg_creat/scripts/compute_composite.py data/kg_creat/human_study/paper_scores
"""
import json
import shutil
from pathlib import Path

from src.kg_creat.scripts.build_skip0_scores import answered_keys, failed

SRC = Path("data/kg_creat/human_study/combined_scores_skip0")
ORIG = Path("data/kg_creat/kombine_test30/scores")
DST = Path("data/kg_creat/human_study/paper_scores")
EXTRA = ("E0", "E26", "F0", "F26")   # the worked-example items, shown to no human
MODE = lambda p: "baseline" if p.startswith("A") else "analogy" if p.startswith("E") else "blending"


def main():
    if DST.exists():
        raise SystemExit(f"FATAL: {DST} exists; remove it first (this script never overwrites)")
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("composite.json"))
    for path in sorted(DST.glob("*/path_scores.json")):
        m = path.parent.name
        if m.startswith("human"):
            continue
        recs = json.loads(path.read_text())
        have_items = {r["prompt_id"] for r in recs}
        orig = json.loads((ORIG / m / "path_scores.json").read_text())
        extra = [r for r in orig if r["prompt_id"] in EXTRA and r["prompt_id"] not in have_items]
        items = set(EXTRA) - have_items      # all four, including any the model left with no scored record at all
        answered = {(mo, p) for _i, mo, p in answered_keys(extra)}
        add = [failed(MODE(p), p) for p in items if (MODE(p), p) not in answered]
        path.write_text(json.dumps(recs + extra + add, indent=2))
        print(f"  {m:34s} +{len(extra)} records on {sorted(items)}, +{len(add)} failed")
    print(f"wrote {DST}; next: compute_composite.py {DST}")


if __name__ == "__main__":
    main()
