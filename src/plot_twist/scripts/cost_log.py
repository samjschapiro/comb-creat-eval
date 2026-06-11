"""Append the current OpenRouter key spend to a running cost ledger.

OpenRouter's /key endpoint reports cumulative USD `usage` on the API key -- the
ground-truth spend (not an estimate). We log a timestamped reading after each
paid run so there's a running total and per-run delta.

Usage:
    python src/plot_twist/scripts/cost_log.py "label for this checkpoint"
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

LEDGER = Path("docs/tracks/plot_twist/cost_log.md")


def key_usage() -> float:
    load_dotenv()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
    )
    return float(json.load(urllib.request.urlopen(req, timeout=30))["data"]["usage"])


def _last_usage() -> float | None:
    if not LEDGER.exists():
        return None
    rows = re.findall(r"\|\s*[^|]+\|\s*[^|]*\|\s*\$([0-9.]+)\s*\|", LEDGER.read_text())
    return float(rows[-1]) if rows else None


def main(label: str) -> None:
    usage = key_usage()
    prev = _last_usage()
    delta = usage - prev if prev is not None else usage
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if not LEDGER.exists():
        LEDGER.write_text(
            "# plot_twist OpenRouter cost ledger\n\n"
            "Ground-truth cumulative spend on the API key (OpenRouter `/key` usage), logged per run.\n\n"
            "| timestamp | run | cumulative ($) | Δ this run ($) |\n"
            "|---|---|---|---|\n"
        )
    with LEDGER.open("a") as fh:
        fh.write(f"| {ts} | {label} | ${usage:.4f} | ${delta:.4f} |\n")
    print(f"cumulative key spend: ${usage:.4f}   (Δ this run: ${delta:.4f})")
    print(f"logged: {LEDGER}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "checkpoint")
