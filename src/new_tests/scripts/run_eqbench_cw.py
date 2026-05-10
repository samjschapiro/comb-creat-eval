"""Run EQ-Bench Creative Writing v3 on one model.

Thin wrapper around the vendored EQ-Bench CLI in
[resources/repos/creative-writing-bench](../../../resources/repos/creative-writing-bench/).
The vendored runner does generation + rubric scoring + Glicko Elo
matchups against the existing leaderboard pool.

ONE-TIME SETUP for the vendored repo (per repo_usage.md, run from the
project root using uv):

    cd resources/repos/creative-writing-bench
    uv pip install -r requirements.txt
    uv run python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict')"
    cp .env.example .env
    # Edit .env with OPENROUTER_API_KEY and JUDGE_API_KEY (often the same).
    # The default OPENROUTER endpoint works for both test-model and judge-model
    # routing.
    unzip creative_bench_runs.zip   # canonical leaderboard runs
    unzip elo_results.zip           # canonical Elo scores

Then this script invokes the CLI as a subprocess, with cwd set to the
vendored repo. Outputs land in resources/repos/creative-writing-bench/
results/ — we copy the per-run summary to data/new_tests/eqbench_cw/
<model_key>/ for consistency with the other benchmarks.

Usage:
    uv run python src/new_tests/scripts/run_eqbench_cw.py \\
        configs/new_tests/eqbench_cw.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.dat_eval.llm import model_id_to_key  # noqa: E402
from src.utils import init_directory  # noqa: E402

EQBENCH_DIR = Path(__file__).parent.parent.parent.parent / "resources" / "repos" / "creative-writing-bench"


def _check_setup() -> None:
    if not EQBENCH_DIR.exists():
        raise FileNotFoundError(
            f"EQ-Bench dir not found at {EQBENCH_DIR}. "
            "Did you clone resources/repos/creative-writing-bench?"
        )
    if not (EQBENCH_DIR / ".env").exists():
        raise FileNotFoundError(
            f"{EQBENCH_DIR}/.env not found. Copy .env.example and configure "
            "OPENROUTER_API_KEY and JUDGE_API_KEY before running."
        )
    if not (EQBENCH_DIR / "data" / "creative_writing_prompts_v3.json").exists():
        raise FileNotFoundError(
            f"{EQBENCH_DIR}/data/creative_writing_prompts_v3.json not found."
        )


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if "test_model" not in config:
        raise ValueError("FATAL: 'test_model' is required in config")
    if "judge_model" not in config:
        raise ValueError("FATAL: 'judge_model' is required in config")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    (output_dir / "config.yaml").write_text(Path(config_path).read_text())

    _check_setup()

    test_model: str = config["test_model"]
    judge_model: str = config["judge_model"]
    iterations: int = config.get("iterations", 3 if not debug else 1)
    threads: int = config.get("threads", 16)
    run_id: str = config.get("run_id", model_id_to_key(test_model))

    # Use the canonical leaderboard runs file so Elo is comparable.
    runs_file = config.get("runs_file", "creative_bench_runs.json")
    prompts_file = config.get(
        "creative_prompts_file", "data/creative_writing_prompts_v3.json"
    )

    cmd = [
        sys.executable,
        "creative_writing_bench.py",
        "--test-model", test_model,
        "--judge-model", judge_model,
        "--runs-file", runs_file,
        "--creative-prompts-file", prompts_file,
        "--run-id", run_id,
        "--threads", str(threads),
        "--iterations", str(iterations),
        "--verbosity", "INFO",
    ]

    print(f"Invoking EQ-Bench CW v3 in {EQBENCH_DIR}")
    print(f"  test={test_model}; judge={judge_model}; iters={iterations}; run_id={run_id}")
    proc = subprocess.run(cmd, cwd=str(EQBENCH_DIR), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"EQ-Bench CLI failed with code {proc.returncode}")

    # Pull canonical outputs back into our data dir.
    src_runs = EQBENCH_DIR / runs_file
    if src_runs.exists():
        shutil.copy(src_runs, output_dir / runs_file)
        with open(src_runs) as f:
            runs = json.load(f)
        # The relevant key is run_id (or run_id + suffix); match prefix.
        matching = {k: v for k, v in runs.items() if k.startswith(run_id)}
        (output_dir / "summary.json").write_text(
            json.dumps(
                {
                    "test_model": test_model,
                    "judge_model": judge_model,
                    "iterations": iterations,
                    "matching_runs": list(matching.keys()),
                },
                indent=2,
            )
        )
        print(f"Saved {len(matching)} run record(s) to {output_dir}/summary.json")
    else:
        print(f"WARNING: expected runs file not found at {src_runs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, overwrite=args.overwrite, debug=args.debug)
