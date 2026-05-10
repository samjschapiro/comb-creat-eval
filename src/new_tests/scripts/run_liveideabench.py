"""Run LiveIdeaBench on one model.

Thin wrapper around the vendored LiveIdeaBench CLI in
[resources/repos/liveideabench](../../../resources/repos/liveideabench/).
The vendored runner generates ideas + scores them with the rotating
critic-panel of top-performing models per the paper.

ONE-TIME SETUP:

    cd resources/repos/liveideabench
    uv pip install -r requirements.txt
    uv run python -c "from utils.database import init_database; init_database()"
    echo "$OPENROUTER_API_KEY" > apikey
    # Optional Gemini and Step keys for richer judge panel — see README.

This script invokes run.py as a subprocess. Outputs land in the vendored
repo's results/ and SQLite db at data/. We copy the per-run summary to
our data dir.

Usage:
    uv run python src/new_tests/scripts/run_liveideabench.py \\
        configs/new_tests/liveideabench.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory  # noqa: E402

LIVEIDEA_DIR = (
    Path(__file__).parent.parent.parent.parent / "resources" / "repos" / "liveideabench"
)


def _check_setup() -> None:
    if not LIVEIDEA_DIR.exists():
        raise FileNotFoundError(
            f"LiveIdeaBench dir not found at {LIVEIDEA_DIR}. "
            "Did you clone resources/repos/liveideabench?"
        )
    apikey_path = LIVEIDEA_DIR / "apikey"
    if not apikey_path.exists():
        # Try to write it from env.
        if "OPENROUTER_API_KEY" in os.environ:
            apikey_path.write_text(os.environ["OPENROUTER_API_KEY"])
            print(f"Wrote OPENROUTER_API_KEY to {apikey_path}")
        else:
            raise FileNotFoundError(
                f"{apikey_path} not found, and OPENROUTER_API_KEY not in env."
            )


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if "idea_model" not in config:
        raise ValueError("FATAL: 'idea_model' is required in config")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    (output_dir / "config.yaml").write_text(Path(config_path).read_text())

    _check_setup()

    idea_model: str = config["idea_model"]
    provider: str = config.get("provider", "openrouter")
    keywords = config.get("keywords")  # None = all 1180 keywords
    if debug and keywords is None:
        keywords = ["relativity"]
        print("[DEBUG] limiting to one keyword: 'relativity'")

    cmd = [
        sys.executable,
        "run.py",
        "--idea_model", idea_model,
        "--provider", provider,
    ]
    if keywords is not None:
        cmd += ["--keyword"] + list(keywords)

    print(f"Invoking LiveIdeaBench in {LIVEIDEA_DIR}")
    print(f"  idea_model={idea_model}; provider={provider}; keywords={keywords or 'ALL'}")
    proc = subprocess.run(cmd, cwd=str(LIVEIDEA_DIR), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"LiveIdeaBench CLI failed with code {proc.returncode}")

    # Copy SQLite db + any CSV exports into our output dir.
    src_db_dir = LIVEIDEA_DIR / "data"
    if src_db_dir.exists():
        for f in src_db_dir.iterdir():
            if f.is_file() and f.suffix in {".db", ".sqlite", ".csv", ".json"}:
                shutil.copy(f, output_dir / f.name)
    print(f"Outputs copied to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, overwrite=args.overwrite, debug=args.debug)
