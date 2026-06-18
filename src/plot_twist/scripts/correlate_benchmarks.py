"""Correlate the PT2CB scorecard columns with AGC-Bench + external benchmarks.

For each system in data/plot_twist/tc/tc.json we have the Overall z-composite and the
four facet means (surprise, coherence, diversity, realism). We correlate each against:
  - AGC-Bench mean_z   (sparc/data/external/creativity_benchmark.csv, `model` col)
  - Arena Creative Writing, EQ-Bench Creative Writing, Mazur CW, Arena Overall, MMLU-Pro
    (configs/comb_eval/benchmarks.json)

Pearson r is invariant to z-scoring, so r for a facet == r for its z-scored column;
the Overall column is the genuine 4-facet z-composite (overall_eq).

Usage:  python src/plot_twist/scripts/correlate_benchmarks.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

TC_JSON = Path("data/plot_twist/tc/tc.json")
AGC_CSV = Path("/Users/samuelschapiro/Desktop/Experiments/sparc/data/external/creativity_benchmark.csv")
BENCH_JSON = Path("configs/comb_eval/benchmarks.json")

# scorecard columns (a)-(e); key in tc.json -> display label
COLS = [
    ("overall_eq", "(a) Overall z"),
    ("mean_surprise", "(b) Surprise"),
    ("mean_coherence", "(c) Coherence"),
    ("div", "(d) Diversity"),
    ("mean_realism", "(e) Realistic"),
]

# external benchmark -> display label. Only benchmarks with adequate, representative
# coverage of our pool are kept; Mazur CW (n=3), NoveltyBench (n=5), and Hivemind (n=10)
# overlap only a small, weak-model-biased tail of our 71 systems, so their correlations
# are uninterpretable and are excluded.
MIN_N = 15
BENCHES = [
    ("agc_mean_z", "AGC-Bench mean_z"),
    ("arena_cw", "Arena Creative-Writing"),
    ("eq_bench_cw", "EQ-Bench Creative-Writing"),
    ("arena_overall", "Arena Overall (capability)"),
    ("mmlu_pro", "MMLU-Pro (capability)"),
]


def bench_key(source: str) -> str:
    """tc source 'anthropic/claude-opus-4.8' -> benchmarks.json key 'anthropic_claude-opus-4-8'."""
    return source.replace("/", "_").replace(".", "-")


def main() -> None:
    tc = [r for r in json.loads(TC_JSON.read_text()) if r["source"] != "human"]

    # AGC mean_z keyed by the same 'provider/model' string as tc.source
    agc = {}
    for r in csv.DictReader(AGC_CSV.open()):
        try:
            agc[r["model"]] = float(r["mean_z"])
        except (TypeError, ValueError):
            pass

    bench = json.loads(BENCH_JSON.read_text())

    def ext_value(source: str, metric: str):
        if metric == "agc_mean_z":
            return agc.get(source)
        row = bench.get(bench_key(source))
        if not row:
            return None
        v = row.get(metric)
        return float(v) if isinstance(v, (int, float)) else None

    # header
    width = 17
    print(f"\n{'PT2CB column ->':<28}" + "".join(f"{lab.split(') ')[-1][:width]:>{width+2}}" for _, lab in COLS))
    print(f"{'external benchmark':<28}" + "".join(f"{'(a)Overall' if i==0 else f'({chr(98+i-1)})':>{width+2}}" for i in range(len(COLS))))
    print("-" * (28 + (width + 2) * len(COLS)))

    for metric, blabel in BENCHES:
        cells = []
        for ckey, _ in COLS:
            xs, ys = [], []
            for r in tc:
                ev = ext_value(r["source"], metric)
                if ev is not None and r.get(ckey) is not None:
                    xs.append(r[ckey])
                    ys.append(ev)
            if len(xs) >= 3:
                r_, p_ = pearsonr(xs, ys)
                cells.append(f"{r_:+.2f} (n={len(xs)})")
            else:
                cells.append(f"-- (n={len(xs)})")
        print(f"{blabel:<28}" + "".join(f"{c:>{width+2}}" for c in cells))
    print()


if __name__ == "__main__":
    main()
