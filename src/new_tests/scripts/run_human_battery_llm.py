"""Run the human DRAT study's battery (DRAT, RAT, SCTT) on LLMs via OpenRouter.

Items are the human study's own: every DRAT anchor set the analysed main-run participants saw,
the 30 human RAT items, and the 12 SCTT scenarios. Prompts use the participant-facing wording
(src/new_tests/human_battery.py). The DAT is not re-run: the dat_eval pool already has it.

Resume-safe: responses are appended to <output_dir>/responses/<model>.jsonl as they arrive and
completed (test, item, sample) keys are skipped on restart. Spend is computed from each call's
billed usage at OpenRouter's current prices; the run stops before starting new calls once the
cumulative spend (including earlier, resumed calls) reaches budget_usd.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.dat_eval.llm import call_llm_async, get_async_client_openrouter
from src.new_tests.human_battery import SCTT_ITEMS, drat_prompt, rat_prompt, sctt_prompt
from src.utils import save_config

REQUIRED = ("upstream_dir", "output_dir", "models", "reasoning_models", "reasoning", "tests", "budget_usd", "model_concurrency",
            "concurrency", "max_retries", "seed")


def build_items(upstream, tests):
    """(test, item_id, sample, prompt, meta) for every call, from the human study's own items."""
    s = pd.read_csv(Path(upstream) / "drat_sets.csv")
    r = pd.read_csv(Path(upstream) / "responses.csv", low_memory=False)
    s = s[(s["sample"] == "main") & (~s["exclude"].astype(bool))]
    items = []
    if "drat" in tests:
        sets = s.groupby("anchors").agg(relation=("relation", "first"), k=("k", "first")).reset_index()
        for _, row in sets.iterrows():
            anchors = row["anchors"].split("|")
            for i in range(tests["drat"]["samples"]):
                items.append(("drat", row["anchors"], i, drat_prompt(anchors),
                              {"anchors": anchors, "relation": row["relation"], "k": int(row["k"])}))
    if "rat" in tests:
        rat = r[(r["test"] == "rat") & (r["sample"] == "main")].drop_duplicates("item_id")
        for _, row in rat.iterrows():
            cues = row["prompt"].split("|")
            for i in range(tests["rat"]["samples"]):
                items.append(("rat", row["item_id"], i, rat_prompt(cues), {"cues": cues, "solution": row["solution"]}))
    if "sctt" in tests:
        for item_id in SCTT_ITEMS:
            for i in range(tests["sctt"]["samples"]):
                items.append(("sctt", item_id, i, sctt_prompt(item_id), {}))
    return items


def fetch_prices(models):
    data = httpx.get("https://openrouter.ai/api/v1/models", timeout=60).json()["data"]
    prices = {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in data}
    missing = [m for m in models if m not in prices]
    if missing:
        raise ValueError(f"FATAL: not on OpenRouter now: {missing}")
    return {m: prices[m] for m in models}


def load_done(path):
    done, spent = set(), 0.0
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                spent += rec["cost_usd"]
                if rec["content"] is not None:
                    done.add((rec["test"], rec["item_id"], rec["sample"]))
    return done, spent


class Budget:
    def __init__(self, cap, spent):
        self.cap, self.spent, self.lock = cap, spent, asyncio.Lock()

    async def add(self, usd):
        async with self.lock:
            self.spent += usd

    def exhausted(self):
        return self.spent >= self.cap


async def run_model(model, items, config, prices, budget, out_path):
    done, _ = load_done(out_path)
    todo = [it for it in items if (it[0], it[1], it[2]) not in done]
    is_reasoning = model in config["reasoning_models"]
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(config["concurrency"])
    lock = asyncio.Lock()
    stats = {"ok": 0, "empty": 0, "error": 0, "skipped_budget": 0}

    async def one(test, item_id, sample, prompt, meta):
        tcfg = config["tests"][test]
        async with sem:
            if budget.exhausted():
                stats["skipped_budget"] += 1
                return
            content, usage, err = None, {"in": 0, "out": 0}, None
            for attempt in range(config["max_retries"]):
                try:
                    content, usage = await call_llm_async(
                        client, [{"role": "user", "content": prompt}], model,
                        temperature=tcfg["temperature"],
                        max_tokens=tcfg["max_tokens_reasoning"] if is_reasoning else tcfg["max_tokens"],
                        seed=config["seed"] + sample,
                        reasoning=config["reasoning"] if is_reasoning else None, capture_usage=True)
                    err = None
                    if content:
                        break
                except Exception as e:  # recorded per call, never silently dropped
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    await asyncio.sleep(2 ** attempt)
            cost = usage["in"] * prices[model][0] + usage["out"] * prices[model][1]
            await budget.add(cost)
            stats["ok" if content else ("error" if err else "empty")] += 1
            rec = {"model": model, "test": test, "item_id": item_id, "sample": sample, "content": content or None,
                   "error": err, "tokens_in": usage["in"], "tokens_out": usage["out"], "cost_usd": cost,
                   "reasoning": is_reasoning, **meta}
            async with lock:
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")

    await asyncio.gather(*(one(*it) for it in todo))
    return len(todo), stats


async def main_async(config, out):
    items = build_items(config["upstream_dir"], config["tests"])
    prices = fetch_prices(config["models"])
    (out / "responses").mkdir(exist_ok=True)
    spent = sum(load_done(out / "responses" / f"{m.replace('/', '__')}.jsonl")[1] for m in config["models"])
    budget = Budget(config["budget_usd"], spent)
    print(f"{len(items)} calls per model x {len(config['models'])} models; already spent ${spent:.4f} "
          f"of ${config['budget_usd']:.2f}")
    # Several models run at once (slow reasoning models otherwise serialise the run); models still
    # start in config order, and none starts once the shared budget is reached.
    summary, model_sem = {}, asyncio.Semaphore(config["model_concurrency"])

    async def one_model(m):
        async with model_sem:
            if budget.exhausted():
                print(f"BUDGET REACHED (${budget.spent:.2f}); not starting {m}", flush=True)
                return
            n, stats = await run_model(m, items, config, prices, budget, out / "responses" / f"{m.replace('/', '__')}.jsonl")
            summary[m] = stats
            print(f"{m:45s} todo={n:4d} {stats}  cumulative ${budget.spent:.4f}", flush=True)

    await asyncio.gather(*(one_model(m) for m in config["models"]))
    (out / "run_summary.json").write_text(json.dumps({"spent_usd": budget.spent, "models": summary}, indent=2))
    print(f"TOTAL SPENT ${budget.spent:.4f}")


def main(config_path, overwrite=False, debug=False):
    import yaml
    config = yaml.safe_load(open(config_path))
    for key in REQUIRED:
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    if overwrite:
        raise ValueError("FATAL: this runner is resume-safe and never overwrites; use a new output_dir")
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.yaml").exists() and yaml.safe_load(open(out / "config.yaml")) != config:
        raise ValueError(f"FATAL: {out}/config.yaml differs from {config_path}; resume only with the same config")
    save_config(config, out)
    asyncio.run(main_async(config, out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
