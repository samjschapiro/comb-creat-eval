"""Print a one-line summary of the latest TRL/GRPO metric dict.

Reads from <wandb_dir>/latest-run/files/output.log, finds the most
recent line that looks like a Python dict literal, and emits a single
line with selected fields. Used by the monitoring loop on the remote.

Usage:
    python metric_snapshot.py [path-to-output.log]
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path


def find_latest_metric(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None
    last = None
    with open(log_path, "r", errors="replace") as f:
        # Buffer last ~500 lines to find most-recent dict line
        lines = f.readlines()[-2000:]
    for line in reversed(lines):
        line = line.strip()
        # Each TRL metric log appears on its own line starting with '{'
        if line.startswith("{'") and "'loss'" in line:
            try:
                d = ast.literal_eval(line)
                if isinstance(d, dict):
                    return d
            except (SyntaxError, ValueError):
                continue
    return None


def main():
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Default: latest-run symlink under wandb/
        repo_root = Path(__file__).parent.parent.parent.parent
        log_path = repo_root / "wandb" / "latest-run" / "files" / "output.log"

    d = find_latest_metric(log_path)
    if d is None:
        print(f"step=? r=? rstd=? nz=? pr=? nstd=? arch=? kl=? gn=? loss=?")
        return

    def f(key, fmt=".4f"):
        v = d.get(key)
        if v is None:
            return "?"
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return str(v)

    def cv(std_key: str, mean_key: str, fmt: str = ".3f") -> str:
        """Coefficient of variation = std/mean. Archive-size-invariant.

        For MCNS-RL: as the archive grows, absolute novelty/reward values
        drop (less room to be novel). CV stays meaningful because both
        scale the same way. Higher CV sustained = model maintaining
        within-batch differentiation, which is what GRPO's group-relative
        advantage actually consumes.
        """
        std = d.get(std_key)
        mean = d.get(mean_key)
        if std is None or mean is None:
            return "?"
        try:
            std_f, mean_f = float(std), float(mean)
            if abs(mean_f) < 1e-9:
                return "inf"
            return format(std_f / mean_f, fmt)
        except (TypeError, ValueError):
            return "?"

    epoch = d.get("epoch", 0)
    parts = [
        f"epoch={epoch:.3f}",
        f"r={f('reward/mean')}",
        f"rstd={f('reward/std')}",
        f"rcv={cv('reward/std', 'reward/mean')}",
        f"nz={f('reward/nonzero_frac', '.2f')}",
        f"pr={f('appropriateness/pass_rate', '.2f')}",
        f"nstd={f('novelty/std')}",
        f"ncv={cv('novelty/std', 'novelty/mean')}",
        f"arch={f('archive/total_size', '.0f')}",
        f"acmean={f('archive/mean_size', '.2f')}",
        f"kl={f('kl', '.5f')}",
        f"gn={f('grad_norm', '.4f')}",
        f"loss={f('loss')}",
    ]
    print(" ".join(parts))


if __name__ == "__main__":
    main()
