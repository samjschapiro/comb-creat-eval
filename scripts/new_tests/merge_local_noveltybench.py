"""Merge locally-generated NoveltyBench scores into benchmarks.json.

Reads data/new_tests/noveltybench/<pool_key>_served/summary.json produced by
run_noveltybench_served_batch.sh and writes each mean_utility_k into the
'noveltybench_utility' field for that model. Dry-run by default.

Usage:
    .venv-mlx/bin/python scripts/new_tests/merge_served_noveltybench.py           # dry run
    .venv-mlx/bin/python scripts/new_tests/merge_served_noveltybench.py --write   # apply
"""
import json, sys, glob, os

BENCH = "configs/comb_eval/benchmarks.json"
FIELD = "noveltybench_utility"

def main(write: bool):
    b = json.load(open(BENCH))
    rows = []
    for d in sorted(glob.glob("data/new_tests/noveltybench/*_served")):
        key = os.path.basename(d)[:-len("_served")]
        sp = os.path.join(d, "summary.json")
        if not os.path.exists(sp):
            continue
        val = json.load(open(sp)).get("mean_utility_k")
        if val is None:
            continue
        in_pool = key in b
        old = b.get(key, {}).get(FIELD)
        rows.append((key, round(val, 4), old, in_pool))

    print(f"{'model':44s} {'new':>8s} {'existing':>10s}  note")
    n_new = n_over = 0
    for key, val, old, in_pool in rows:
        note = "NOT in benchmarks.json!" if not in_pool else ("overwrite" if old not in (None,"","---") else "new field")
        if in_pool and old in (None,"","---"): n_new += 1
        if in_pool and old not in (None,"","---"): n_over += 1
        print(f"  {key:42s} {val:8.4f} {str(old):>10s}  {note}")
        if write and in_pool:
            b[key][FIELD] = val

    print(f"\n{len(rows)} local summaries | {n_new} new fields | {n_over} overwrites")
    if write:
        json.dump(b, open(BENCH, "w"), indent=2)
        print(f"WROTE {BENCH}")
    else:
        print("DRY RUN — pass --write to apply")

if __name__ == "__main__":
    main("--write" in sys.argv)
