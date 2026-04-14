"""Decompose PACE into interpretable components to understand what's
actually driving its correlation with creative writing benchmarks.

PACE's formula (Qiu & Hu, EMNLP 2025) averages, over each position i=2..n,
the mean cosine distance from position i to every prior position. Equivalent
to a positional-weighted mean of all pairwise cosine distances with
adjacent (i, i-1) pairs included.

This script computes, for each PACE chain, several simpler decompositions:

    mean_adjacent_dist:       mean cosine distance of consecutive pairs only
    mean_nonadjacent_dist:    mean cosine distance of non-adjacent pairs only
    pace_early_pos1_9:        PACE over chain positions 1-9
    pace_late_pos10_19:       PACE over chain positions 10-19
    first_edge_dist:          cosine distance of the seed -> first chain word
    return_dist_last_to_first: distance of position 19 back to seed
    max_edge_dist:            max consecutive-pair cosine distance
    edge_dist_variance:       variance of consecutive-pair cosine distances

Per-model means of each component are correlated against creative writing
benchmarks to identify which sub-structure of PACE actually carries the
predictive signal.

Usage:
    uv run python src/dat_eval/scripts/analyze_pace_mechanisms.py \\
        configs/dat_eval/analyze_pace_mechanisms.yaml
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import load_config, init_directory, save_config
from src.dat_eval.pace import FastTextEmbeddings


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    return float(np.dot(u, v) / (nu * nv)) if nu > 0 and nv > 0 else 0.0


def encode_chain(chain: list[str], emb: FastTextEmbeddings) -> list[np.ndarray]:
    vecs = []
    for w in chain:
        v = emb.encode(w.lower())
        if np.linalg.norm(v) > 0:
            vecs.append(v)
    return vecs


# --- component metrics ---


def pace_full(vecs):
    n = len(vecs)
    if n < 2:
        return float("nan")
    return float(np.mean([
        np.mean([1 - _cosine(vecs[i], vecs[j]) for j in range(i)])
        for i in range(1, n)
    ]))


def mean_adjacent_dist(vecs):
    n = len(vecs)
    if n < 2:
        return float("nan")
    return float(np.mean([1 - _cosine(vecs[i], vecs[i + 1]) for i in range(n - 1)]))


def mean_nonadjacent_dist(vecs):
    n = len(vecs)
    if n < 3:
        return float("nan")
    ds = [
        1 - _cosine(vecs[i], vecs[j])
        for i, j in itertools.combinations(range(n), 2)
        if j - i >= 2
    ]
    return float(np.mean(ds)) if ds else float("nan")


def pace_early_pos1_9(vecs):
    n = len(vecs)
    if n < 4:
        return float("nan")
    end = min(10, n)
    return float(np.mean([
        np.mean([1 - _cosine(vecs[i], vecs[j]) for j in range(i)])
        for i in range(1, end)
    ]))


def pace_late_pos10_19(vecs):
    n = len(vecs)
    if n < 12:
        return float("nan")
    return float(np.mean([
        np.mean([1 - _cosine(vecs[i], vecs[j]) for j in range(i)])
        for i in range(10, n)
    ]))


def first_edge_dist(vecs):
    if len(vecs) < 2:
        return float("nan")
    return 1 - _cosine(vecs[0], vecs[1])


def return_dist_last_to_first(vecs):
    if len(vecs) < 2:
        return float("nan")
    return 1 - _cosine(vecs[0], vecs[-1])


def max_edge_dist(vecs):
    n = len(vecs)
    if n < 2:
        return float("nan")
    return max(1 - _cosine(vecs[i], vecs[i + 1]) for i in range(n - 1))


def edge_dist_variance(vecs):
    n = len(vecs)
    if n < 3:
        return float("nan")
    edges = [1 - _cosine(vecs[i], vecs[i + 1]) for i in range(n - 1)]
    return float(np.var(edges))


COMPONENT_METRICS = {
    "pace_full": pace_full,
    "mean_adjacent_dist": mean_adjacent_dist,
    "mean_nonadjacent_dist": mean_nonadjacent_dist,
    "pace_early_pos1_9": pace_early_pos1_9,
    "pace_late_pos10_19": pace_late_pos10_19,
    "first_edge_dist": first_edge_dist,
    "return_dist_last_to_first": return_dist_last_to_first,
    "max_edge_dist": max_edge_dist,
    "edge_dist_variance": edge_dist_variance,
}


# --- correlation + hierarchical regression helpers ---


def _spearman(x, y):
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 3:
        return float("nan"), float("nan"), len(df)
    r, p = stats.spearmanr(df["x"], df["y"])
    return float(r), float(p), len(df)


def _pearson(x, y):
    df = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(df) < 3:
        return float("nan"), float("nan"), len(df)
    r, p = stats.pearsonr(df["x"], df["y"])
    return float(r), float(p), len(df)


def _hierarchical(y, x1, x2):
    df = pd.concat(
        [y.rename("y"), x1.rename("x1"), x2.rename("x2")], axis=1
    ).dropna()
    n = len(df)
    if n < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), n
    yv = df["y"].values
    X1 = np.column_stack([np.ones(n), df["x1"].values])
    X2 = np.column_stack([np.ones(n), df["x1"].values, df["x2"].values])
    def r2(X, y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        yh = X @ b
        return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2), np.sum((y - yh) ** 2)
    r2_1, ss1 = r2(X1, yv)
    r2_2, ss2 = r2(X2, yv)
    F = ((ss1 - ss2) / 1) / (ss2 / (n - 3)) if ss2 > 0 else float("inf")
    p = 1 - stats.f.cdf(F, 1, n - 3) if np.isfinite(F) else 0.0
    return r2_1, r2_2, r2_2 - r2_1, p, n


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    upstream_dir = Path(config["upstream_dir"])
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    fasttext_path = config.get("fasttext_path", "resources/crawl-300d-2M.vec")
    pace_response_filename = config.get("pace_response_filename", "pace_responses_t0-0.json")
    min_chains = int(config.get("min_chains", 50))

    print(f"Loading FastText from {fasttext_path}...")
    emb = FastTextEmbeddings(fasttext_path)

    # Score every model's PACE chains into per-component means
    per_model: dict[str, dict] = {}
    for d in sorted(upstream_dir.iterdir()):
        if not d.is_dir():
            continue
        p = d / pace_response_filename
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        agg = {k: [] for k in COMPONENT_METRICS}
        for _, seed_data in data.items():
            for c in seed_data.get("chains", []):
                chain = c.get("chain", [])
                vecs = encode_chain(chain, emb)
                if len(vecs) < 2:
                    continue
                for mk, mf in COMPONENT_METRICS.items():
                    v = mf(vecs)
                    if not np.isnan(v):
                        agg[mk].append(v)
        if any(len(agg[k]) < min_chains for k in ("pace_full",)):
            continue
        per_model[d.name] = {
            k: float(np.mean(v)) if v else float("nan")
            for k, v in agg.items()
        }
        per_model[d.name]["n_chains"] = len(agg["pace_full"])

    print(f"Scored {len(per_model)} models")
    with open(output_dir / "results" / "per_model_decomposition.json", "w") as f:
        json.dump(per_model, f, indent=2)
    pd.DataFrame.from_dict(per_model, orient="index").to_csv(
        output_dir / "results" / "per_model_decomposition.csv"
    )

    # --- correlation analysis vs benchmarks
    benchmark_path = config["benchmark_file"]
    with open(benchmark_path) as f:
        benchmarks = json.load(f)
    bm_df = pd.DataFrame.from_dict(benchmarks, orient="index")
    benchmark_cols = config.get(
        "benchmark_columns",
        ["arena_cw", "arena_overall", "eq_bench_cw", "mazur_cw_v2"],
    )

    corr_table = {}
    for mk in COMPONENT_METRICS:
        series = pd.Series({k: v[mk] for k, v in per_model.items() if not np.isnan(v[mk])})
        entry = {}
        for b in benchmark_cols:
            rs, ps, n = _spearman(series, bm_df[b])
            rp, pp, _ = _pearson(series, bm_df[b])
            entry[b] = {
                "spearman": {"rho": rs, "p": ps, "n": n},
                "pearson":  {"r": rp, "p": pp, "n": n},
            }
        corr_table[mk] = entry

    # PACE is also in component_metrics (under pace_full). Compute hierarchical
    # gain of each *other* component over pace_full.
    pace_series = pd.Series({k: v["pace_full"] for k, v in per_model.items()})
    hier_table = {}
    for mk in COMPONENT_METRICS:
        if mk == "pace_full":
            continue
        series = pd.Series({k: v[mk] for k, v in per_model.items() if not np.isnan(v[mk])})
        entry = {}
        # (a) does component add to PACE?
        for b in benchmark_cols:
            r1, r2, d, p, n = _hierarchical(bm_df[b], pace_series, series)
            entry[f"{b}__pace_plus_component"] = {
                "R2_pace": r1, "R2_combined": r2, "delta_R2": d, "F_p": p, "n": n,
            }
        # (b) does PACE add to component? (the more interesting direction)
        for b in benchmark_cols:
            r1, r2, d, p, n = _hierarchical(bm_df[b], series, pace_series)
            entry[f"{b}__component_plus_pace"] = {
                "R2_component": r1, "R2_combined": r2, "delta_R2": d, "F_p": p, "n": n,
            }
        hier_table[mk] = entry

    with open(output_dir / "results" / "correlation_analysis.json", "w") as f:
        json.dump(
            {
                "correlations": corr_table,
                "hierarchical": hier_table,
                "n_models": len(per_model),
            },
            f, indent=2,
        )

    # Print headline table
    def sig(p):
        if np.isnan(p): return ""
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        if p < 0.1:   return "."
        return ""

    print(f"\n=== Component metric vs benchmarks ===")
    header = f"{'metric':<28s}  " + "  ".join(f"{b:<30s}" for b in benchmark_cols)
    print(header)
    for mk in COMPONENT_METRICS:
        cells = []
        for b in benchmark_cols:
            c = corr_table[mk][b]
            rs = c["spearman"]["rho"]; ps = c["spearman"]["p"]; n = c["spearman"]["n"]
            rp = c["pearson"]["r"];    pp = c["pearson"]["p"]
            cells.append(f"ρ={rs:+.3f}{sig(ps):<3s} r={rp:+.3f}{sig(pp):<3s} n={n}")
        print(f"{mk:<28s}  " + "  ".join(f"{c:<30s}" for c in cells))

    print(f"\n=== Does PACE add info beyond simpler component? (ΔR² from adding pace_full) ===")
    print(f"{'component':<28s}  " + "  ".join(f"{b:<22s}" for b in benchmark_cols))
    for mk in COMPONENT_METRICS:
        if mk == "pace_full":
            continue
        cells = []
        for b in benchmark_cols:
            e = hier_table[mk][f"{b}__component_plus_pace"]
            cells.append(f"ΔR²={e['delta_R2']:+.3f}{sig(e['F_p']):<3s} (n={e['n']})")
        print(f"{mk:<28s}  " + "  ".join(f"{c:<22s}" for c in cells))

    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
