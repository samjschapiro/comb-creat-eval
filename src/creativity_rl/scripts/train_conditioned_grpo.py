"""Stage 2 (RL arm): online GRPO on conditioned (x, S) contexts.

Reads the build output (records.jsonl). Each row gives a prompt x and a
fixed set S of prior answers, and a formatted conditioned prompt. GRPO
samples K responses per context; the reward is the distance of a
response to S, gated by appropriateness A(x, y) > tau. S is fixed per
prompt, so the reward is a fixed function of (x, S, y): no archive, no
drift. This is the difference from full_run_v1.

Usage:
    uv run python src/creativity_rl/scripts/train_conditioned_grpo.py \\
        configs/creativity_rl/conditioned_v1.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory


@dataclass
class ConditionedDivergenceReward:
    scorer: object
    embedder: object
    reward: object  # MCNSReward
    _last_appr: np.ndarray | None = field(default=None)
    _last_nov: np.ndarray | None = field(default=None)
    _last_reward: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        self.__name__ = "conditioned_divergence"

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        from src.creativity_rl.reward_callable import _flatten_completion

        if "x" not in kwargs or "S" not in kwargs:
            raise ValueError(
                "FATAL: reward needs 'x' and 'S' columns; set "
                "remove_unused_columns=False and keep them in the dataset"
            )
        xs = list(kwargs["x"])
        Ss = list(kwargs["S"])
        comp = [_flatten_completion(c) for c in completions]
        if not (len(xs) == len(Ss) == len(comp)):
            raise ValueError(
                f"FATAL: length mismatch x={len(xs)} S={len(Ss)} comp={len(comp)}"
            )

        appr = self.scorer.score(xs, comp)
        emb = self.embedder.encode(comp)
        nov = np.zeros(len(comp), dtype=np.float32)
        for i, s_list in enumerate(Ss):
            if not s_list:
                raise ValueError("FATAL: empty S in a record; build must supply S")
            s_emb = self.embedder.encode(list(s_list))
            nov[i] = float(np.mean(1.0 - emb[i] @ s_emb.T))

        r = self.reward(appr, nov)
        self._last_appr, self._last_nov, self._last_reward = appr, nov, r
        return [float(v) for v in r]

    def telemetry(self) -> dict:
        if self._last_reward is None:
            return {}
        return {
            "reward/mean": float(self._last_reward.mean()),
            "reward/std": float(self._last_reward.std()),
            "reward/nonzero_frac": float((self._last_reward > 0).mean()),
            "appropriateness/mean": float(self._last_appr.mean()),
            "appropriateness/pass_rate": float((self._last_reward > 0).mean()),
            "novelty/mean": float(self._last_nov.mean()),
            "novelty/std": float(self._last_nov.std()),
        }


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["train_rl"]
    for key in ("output_dir", "upstream_dir"):
        if key not in cfg:
            raise ValueError(f"FATAL: 'train_rl.{key}' required in config")

    records_path = Path(cfg["upstream_dir"]) / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(
            f"FATAL: {records_path} not found. Run build_conditioned_data.py first."
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from src.creativity_rl.reward import MCNSReward
    from src.creativity_rl.scoring import AppropriatenessScorer, SBERTEmbedder

    output_dir = init_directory(cfg["output_dir"], overwrite=overwrite)
    shutil.copy(config_path, output_dir / "config.yaml")
    for sub in ("checkpoints", "logs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    rows = []
    with open(records_path) as f:
        for line in f:
            rec = json.loads(line)
            rows.append(
                {
                    "prompt": [{"role": "user", "content": rec["prompt"]}],
                    "x": rec["x"],
                    "S": rec["S"],
                }
            )
    if not rows:
        raise ValueError("FATAL: upstream produced no records")
    print(f"Loaded {len(rows)} conditioned contexts", flush=True)
    train_ds = Dataset.from_list(rows)

    scorer = AppropriatenessScorer(
        model_name=cfg["appropriateness"]["rm_model"],
        device="cuda",
        load_in_4bit=cfg["appropriateness"].get("rm_load_in_4bit", False),
        max_length=cfg["appropriateness"].get("rm_max_length", 512),
    )
    embedder = SBERTEmbedder(cfg["novelty"]["embedding_model"], device="cuda")
    mcns = MCNSReward(
        tau=float(cfg["appropriateness"]["threshold_tau"]),
        mode="hard" if cfg["reward"]["type"] == "mcns_hard" else "soft",
        soft_temperature=cfg["reward"].get("soft_temperature"),
    )
    reward_fn = ConditionedDivergenceReward(scorer=scorer, embedder=embedder, reward=mcns)

    peft_config = LoraConfig(
        r=cfg["policy"]["lora"]["rank"],
        lora_alpha=cfg["policy"]["lora"]["alpha"],
        lora_dropout=cfg["policy"]["lora"]["dropout"],
        target_modules=cfg["policy"]["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    wb = cfg["logging"]["wandb"]
    if wb["enabled"]:
        os.environ.setdefault("WANDB_PROJECT", wb["project"])

    grpo_args = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=float(cfg["rl"]["learning_rate"]),
        per_device_train_batch_size=cfg["rl"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["rl"]["gradient_accumulation_steps"],
        num_generations=cfg["rl"]["group_size_K"],
        max_completion_length=cfg["rl"]["generation"]["max_new_tokens"],
        temperature=cfg["rl"]["generation"]["temperature"],
        top_p=cfg["rl"]["generation"]["top_p"],
        beta=cfg["rl"]["kl_coefficient_beta"],
        max_steps=cfg["rl"]["total_steps"],
        save_steps=cfg["rl"]["save_every_steps"],
        logging_steps=cfg["logging"]["log_every_steps"],
        report_to=["wandb"] if wb["enabled"] else [],
        run_name=wb.get("run_name"),
        bf16=True,
        seed=cfg["seed"],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=cfg["policy"]["base_model"],
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    _orig_log = trainer.log

    def _log(logs, *a, **kw):
        logs.update(reward_fn.telemetry())
        return _orig_log(logs, *a, **kw)

    trainer.log = _log

    if debug:
        print(f"DEBUG: {len(train_ds)} contexts, output {output_dir}", flush=True)
    trainer.train()
    trainer.save_model(str(output_dir / "checkpoints" / "final"))
    print("Done.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_path")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    main(a.config_path, a.overwrite, a.debug)
