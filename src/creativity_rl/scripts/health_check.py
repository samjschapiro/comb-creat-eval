"""Holistic health check over the full metric log of a GRPO training run.

Parses wandb/latest-run/files/output.log, extracts all metric dicts,
and reports trajectories for the 7 key health indicators of MCNS-RL.
"""

from __future__ import annotations

import ast
import statistics
import sys
from pathlib import Path


def main(log_path: Path) -> None:
    if not log_path.exists():
        print(f"FATAL: {log_path} not found")
        sys.exit(1)

    metrics: list[dict] = []
    for line in log_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{'") and "'loss'" in line:
            try:
                d = ast.literal_eval(line)
                if isinstance(d, dict):
                    metrics.append(d)
            except (SyntaxError, ValueError):
                pass

    n = len(metrics)
    if n == 0:
        print("no metric records")
        return

    def col(k):
        return [m.get(k) for m in metrics if m.get(k) is not None]

    def stat(vals, fmt=".4f"):
        if not vals:
            return "n/a"
        sv = sorted(vals)
        return (
            f"min={format(min(vals), fmt)} "
            f"p25={format(sv[len(sv)//4], fmt)} "
            f"med={format(statistics.median(vals), fmt)} "
            f"p75={format(sv[3*len(sv)//4], fmt)} "
            f"max={format(max(vals), fmt)} "
            f"last={format(vals[-1], fmt)}"
        )

    print(f"=== Health check ({n} metric records, log_every=5 → ~step {n*5}) ===\n")

    print("KEY 1 — Gradient signal (rstd > 0 means GRPO has signal):")
    rstd = col("reward/std")
    zero_rstd = sum(1 for v in rstd if v < 1e-3)
    print(f"  reward/std {stat(rstd)}")
    print(f"  zero-gradient steps (rstd<0.001): {zero_rstd}/{n} = {100*zero_rstd/n:.1f}%\n")

    print("KEY 2 — Within-batch novelty differentiation:")
    nstd = col("novelty/std")
    nmean = col("novelty/mean")
    ncv = [s / m for s, m in zip(nstd, nmean) if m > 1e-9]
    print(f"  ncv (nstd/nmean) {stat(ncv, '.4f')}")
    print(f"  novelty/std {stat(nstd)}\n")

    print("KEY 3 — Gate behavior:")
    pr = col("appropriateness/pass_rate")
    gate_splits = sum(1 for v in pr if v < 1.0)
    all_pass = sum(1 for v in pr if v >= 1.0)
    all_fail = sum(1 for v in pr if v <= 1e-3)
    print(f"  appropriateness/pass_rate {stat(pr, '.2f')}")
    print(f"  batches with at least one gate-fail (pr<1.0): {gate_splits}/{n} = {100*gate_splits/n:.1f}%")
    print(f"  batches all-pass (pr=1.0): {all_pass}/{n}")
    print(f"  batches all-fail (pr=0):   {all_fail}/{n}\n")

    print("KEY 4 — KL divergence (policy drift from ref):")
    kl = col("kl")
    print(f"  kl {stat(kl, '.5f')}\n")

    print("KEY 5 — Archive growth:")
    arch = col("archive/total_size")
    acmean = col("archive/mean_size")
    print(f"  total {stat(arch, '.0f')}")
    print(f"  per-cluster mean (k=5; healthy when >>5) {stat(acmean, '.2f')}")
    if len(arch) > 1:
        growth_rate = (arch[-1] - arch[0]) / (len(arch) - 1)
        print(f"  admissions per metric-log: ~{growth_rate:.2f}\n")
    else:
        print()

    print("KEY 6 — Gradient norm:")
    gn = col("grad_norm")
    strong = sum(1 for v in gn if v > 0.5)
    print(f"  grad_norm {stat(gn, '.3f')}")
    print(f"  strong-gradient steps (gn>0.5): {strong}/{n} = {100*strong/n:.1f}%\n")

    print("KEY 7 — Reward trajectory (NOT expected monotonic in MCNS — archive treadmill):")
    r = col("reward/mean")
    W = max(1, len(r) // 5)
    for i in range(0, len(r), W):
        w_r = r[i : i + W]
        w_rstd = rstd[i : i + W]
        if w_r:
            print(
                f"  window {i//W+1} (steps ~{i*5}-{(i+W)*5-1}): "
                f"r/mean = {statistics.mean(w_r):.3f}, "
                f"rstd_mean = {statistics.mean(w_rstd):.4f}"
            )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path(__file__).parent.parent.parent.parent / "wandb" / "latest-run" / "files" / "output.log"
    main(path)
