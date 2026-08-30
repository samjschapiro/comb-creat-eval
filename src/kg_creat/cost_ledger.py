"""Persistent, append-only USD cost ledger for the Kombine LLM experiments.

Every paid step (elicitation, scoring, ...) appends one row per model with the ACTUAL
token usage reported by the API and the USD cost computed from ``cost_tracker.PRICING``.
The ledger is the single running tally of spend across the whole experimentation process.

    python -m src.kg_creat.cost_ledger          # print the running total + breakdown

Rows are never rewritten, so re-running a step that was resume-skipped adds nothing (the
elicitation runner only records on a fresh, non-skipped model run).
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = _ROOT / "data" / "kg_creat" / "cost_ledger.jsonl"

sys.path.insert(0, str(_ROOT / "scripts" / "safety"))
from cost_tracker import PRICING  # noqa: E402


def cost_usd(model_id: str, in_tokens: int, out_tokens: int) -> float | None:
    """USD cost from PRICING (per-1M in/out prices). None if the model is unpriced."""
    price = PRICING.get(model_id)
    if price is None:
        return None
    in_price, out_price = price
    return (in_tokens * in_price + out_tokens * out_price) / 1_000_000


def record(phase: str, model_id: str, n_calls: int, in_tokens: int, out_tokens: int,
           config: str = "", note: str = "", cost_override: float | None = None) -> dict:
    """Append one ledger row and return it. ``cost_override`` bypasses PRICING when the
    caller already knows the charge (e.g. a provider-reported cost)."""
    c = cost_override if cost_override is not None else cost_usd(model_id, in_tokens, out_tokens)
    entry = {
        "ts": _dt.datetime.now().isoformat(timespec="seconds"),
        "phase": phase,
        "model": model_id,
        "config": config,
        "n_calls": int(n_calls),
        "in_tokens": int(in_tokens),
        "out_tokens": int(out_tokens),
        "cost_usd": round(c, 6) if c is not None else None,
        "note": note,
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load() -> list[dict]:
    if not LEDGER_PATH.exists():
        return []
    return [json.loads(ln) for ln in LEDGER_PATH.read_text().splitlines() if ln.strip()]


def total() -> float:
    return sum((r["cost_usd"] or 0.0) for r in load())


def summarize() -> None:
    from collections import defaultdict
    rows = load()
    if not rows:
        print(f"Cost ledger empty ({LEDGER_PATH}).")
        return
    by_phase: dict[str, float] = defaultdict(float)
    by_model: dict[str, float] = defaultdict(float)
    tok_in = tok_out = 0
    unpriced: set[str] = set()
    for r in rows:
        by_phase[r["phase"]] += r["cost_usd"] or 0.0
        by_model[r["model"]] += r["cost_usd"] or 0.0
        tok_in += r.get("in_tokens", 0)
        tok_out += r.get("out_tokens", 0)
        if r["cost_usd"] is None:
            unpriced.add(r["model"])
    print(f"Kombine cost ledger  ({LEDGER_PATH})")
    print(f"  {len(rows)} rows | {tok_in:,} in + {tok_out:,} out tokens | TOTAL ${total():.4f}\n")
    print("  by phase:")
    for k, v in sorted(by_phase.items(), key=lambda x: -x[1]):
        print(f"    {k:14s} ${v:8.4f}")
    print("\n  by model:")
    for k, v in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"    {k:38s} ${v:8.4f}")
    if unpriced:
        print(f"\n  (!) unpriced models, cost NOT counted: {sorted(unpriced)}")


if __name__ == "__main__":
    summarize()
