"""Repair failed elicitation draws in place: re-elicit the failures and merge the recoveries back into
each model's responses.json. Never overwrites good draws.

Three failure classes, three fixes (see the run's taxonomy):
  * 429 rate-limit (Mistral shared pool) -> retry at LOW concurrency.
  * "null content" (reasoning model exhausts the token budget before emitting) -> retry with a much
    larger max_tokens (16x) so reasoning has room to finish AND emit.
  * PARSE failure (the call returned text that is not the requested structure) -> re-draw at the same
    temperature with --parse-too. This is a resample, not a fix: a model that cannot emit the format
    will fail again, and that is itself the result. Draws are replaced only when the retry parses, so
    a failed resample never destroys the record of the original failure.

    .venv_mlx/bin/python -m src.kg_creat.scripts.repair_elicit --dry-run   # estimate, no API calls
    .venv_mlx/bin/python -m src.kg_creat.scripts.repair_elicit             # api errors only
    .venv_mlx/bin/python -m src.kg_creat.scripts.repair_elicit --parse-too # api errors + parse failures
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.kg_creat.scripts.run_elicit import _run_one, REASONING_MODELS, model_id_to_key  # noqa: E402
from src.dat_eval.llm import get_async_client  # noqa: E402
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "safety"))
from cost_tracker import PRICING  # noqa: E402

RESP_DIR = Path("data/kg_creat/kombine_test30/responses")
PROMPTS = Path("data/kg_creat/kombine_test30/prompts/prompts.json")
BASE_MAX_TOKENS = 1600
REASONING = {"exclude": False}      # default effort, matching the original run

# Per-model overrides. mistral-large's failures are upstream 429s (not token exhaustion): low
# concurrency, normal tokens. Everything else here is reasoning null-content: big token budget.
LOW_CONCURRENCY = {"mistralai/mistral-large-2512"}


def specs_by_id():
    return {s["prompt_id"]: s for s in json.loads(PROMPTS.read_text())}


def summarize(records):
    n_ok = sum(1 for r in records if r.get("parse_success"))
    n_api = sum(1 for r in records if r.get("api_error"))
    return n_ok, n_api


async def repair_model(client, model_id, specs, dry_run, parse_too=False, provider=None,
                       max_tokens_override=None):
    key = model_id_to_key(model_id)
    rp = RESP_DIR / key / "responses.json"
    if not rp.exists():
        return None
    records = json.loads(rp.read_text())
    failed_idx = [i for i, r in enumerate(records)
                  if r.get("api_error") or (parse_too and not r.get("parse_success"))]
    if not failed_idx:
        return None
    reasoning_model = model_id in REASONING_MODELS
    mt = BASE_MAX_TOKENS * 16 if reasoning_model else BASE_MAX_TOKENS   # 16x headroom for reasoners
    if max_tokens_override:
        mt = max_tokens_override   # e.g. a null-content model that exhausted its original ceiling
    reasoning = REASONING if reasoning_model else None
    conc = 2 if model_id in LOW_CONCURRENCY else 8

    if dry_run:
        pin, pout = PRICING.get(model_id, (9, 9))
        est_out = 15000 if reasoning_model else 500     # reasoners emit a lot even to succeed
        est = len(failed_idx) * (750 * pin + est_out * pout) / 1e6
        print(f"  {model_id:34s} retry {len(failed_idx):3d}  mt={mt:5d} conc={conc}  est ~${est:6.2f}")
        return {"model": model_id, "n_retry": len(failed_idx), "est": est}

    sem = asyncio.Semaphore(conc)
    coros = []
    for i in failed_idx:
        r = records[i]
        spec = specs[r["prompt_id"]]
        coros.append(_run_one(client, sem, model_id, spec, mt, r.get("temperature", 0.9),
                              r.get("sample_idx", 0), reasoning, provider))
    results = await asyncio.gather(*coros)
    recovered, in_tok, out_tok = 0, 0, 0
    for i, new in zip(failed_idx, results):
        u = new.get("usage") or {}
        in_tok += u.get("in", 0); out_tok += u.get("out", 0)
        was_parse_only = not records[i].get("api_error")
        if new.get("parse_success"):
            recovered += 1
            records[i] = new
        elif not was_parse_only:
            records[i] = new             # an api_error record can only improve; keep the retry outcome
        # a parse failure that fails again keeps the ORIGINAL draw: the failure is the datum
    rp.write_text(json.dumps(records, indent=2, default=str))
    n_ok, n_api = summarize(records)
    # ledger: only the retried calls' actual usage
    from src.kg_creat.cost_ledger import record
    e = record("elicit-repair", model_id, len(failed_idx), in_tok, out_tok,
               config="repair", note=f"recovered={recovered}/{len(failed_idx)} now parsed={n_ok}/{len(records)}")
    cost = f"${e['cost_usd']:.4f}" if e["cost_usd"] is not None else "unpriced"
    print(f"  {model_id:34s} recovered {recovered:3d}/{len(failed_idx):<3d} -> parsed {n_ok}/{len(records)} "
          f"api_fail {n_api}  ({in_tok:,}+{out_tok:,} tok, {cost})")
    return {"model": model_id, "recovered": recovered, "n_retry": len(failed_idx),
            "n_parsed": n_ok, "n_api_fail": n_api, "cost_usd": e["cost_usd"]}


async def main(dry_run, parse_too=False, only=None, config=None, max_tokens=None):
    """`only` restricts the repair to named models. Re-drawing a model whose artifacts are already
    JUDGED would leave its path_scores describing responses that no longer exist, so a repair after
    scoring must always be scoped."""
    specs = specs_by_id()
    models = []
    for d in sorted(RESP_DIR.iterdir()):
        if (d / "responses.json").exists():
            recs = json.loads((d / "responses.json").read_text())
            if any(r.get("api_error") or (parse_too and not r.get("parse_success")) for r in recs):
                # recover the model_id from the summary (key has dots replaced)
                s = json.loads((d / "summary.json").read_text())
                if only is None or s["model_id"] in only:
                    models.append(s["model_id"])
    print(f"Models with failed draws: {len(models)}")
    # A model elicited off OpenRouter must be REPAIRED on the same route, or the retry would come
    # from a different provider than the draws it is replacing. --config reuses that run's provider.
    provider = None
    if config:
        import yaml
        cfg = yaml.safe_load(open(config))
        if cfg.get("provider"):
            from src.kg_creat.providers import build as build_provider
            provider = build_provider(cfg["provider"])
            print(f"provider: {cfg['provider'].get('kind')} (from {config})")
        max_tokens = max_tokens or (cfg.get("eval") or {}).get("max_tokens")
    client = None if dry_run else get_async_client()
    total_est = 0.0
    for m in models:
        out = await repair_model(client, m, specs, dry_run, parse_too, provider, max_tokens)
        if dry_run and out:
            total_est += out["est"]
    if dry_run:
        print(f"\nTOTAL estimated repair cost: ~${total_est:.2f}  (rough; reasoner output is variable)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--parse-too", action="store_true",
                    help="also re-draw records that returned text which did not parse")
    ap.add_argument("--config", default=None,
                    help="elicit config whose provider/max_tokens the retry should reuse")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the retry ceiling (null-content failures need more than the original)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these model ids (required once a model has been scored)")
    args = ap.parse_args()
    asyncio.run(main(args.dry_run, args.parse_too, args.models, args.config, args.max_tokens))
