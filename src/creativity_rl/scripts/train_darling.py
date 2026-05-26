"""Faithful DARLING reproduction (and the GRPO quality-only baseline).

Mirrors facebookresearch/darling verl/wildchat_scripts/darling.sh in our
TRL/GRPO stack: Llama-3.1-8B-Instruct, n=8 rollouts/prompt, lr 1e-6,
low-variance KL in the loss (beta 0.001), advantage NOT std-normalized,
multiplicative quality x partition-diversity reward, WildChat prompts.
reward.type=quality_only drops the diversity factor (their GRPO baseline).

Usage:
    uv run python src/creativity_rl/scripts/train_darling.py \\
        configs/creativity_rl/darling_repro.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
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
class QualityOnlyReward:
    """GRPO quality-only baseline (their grpo_baseline, no diversity)."""

    quality_scorer: object
    _last: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.__name__ = "quality_only"

    def __call__(self, prompts, completions, **kwargs) -> list[float]:
        from src.creativity_rl.reward_callable import (
            _flatten_completion,
            _flatten_prompt,
        )

        fp = [_flatten_prompt(p) for p in prompts]
        fc = [_flatten_completion(c) for c in completions]
        q = np.asarray(self.quality_scorer.score(fp, fc), dtype=np.float32)
        self._last = {"reward/mean": float(q.mean()), "reward/std": float(q.std())}
        return [float(v) for v in q]

    def telemetry(self) -> dict:
        return dict(self._last)


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "output_dir" not in cfg:
        raise ValueError("FATAL: 'output_dir' required in config")
    rtype = cfg["reward"]["type"]
    if rtype not in ("darling", "quality_only"):
        raise ValueError(f"FATAL: reward.type must be darling|quality_only, got {rtype}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from src.creativity_rl.data import load_wildchat_prompts
    from src.creativity_rl.scoring import AppropriatenessScorer

    output_dir = init_directory(cfg["output_dir"], overwrite=overwrite)
    shutil.copy(config_path, output_dir / "config.yaml")
    for sub in ("checkpoints", "logs"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg["data"]["source"] != "wildchat":
        raise ValueError(f"FATAL: unsupported data.source {cfg['data']['source']}")
    prompts = load_wildchat_prompts(n=cfg["data"]["n_prompts"], seed=cfg["seed"])
    print(f"Loaded {len(prompts)} WildChat prompts", flush=True)
    train_ds = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": p}]} for p in prompts]
    )

    quality = AppropriatenessScorer(
        model_name=cfg["reward"]["quality_rm"],
        device="cuda",
        load_in_4bit=cfg["reward"].get("quality_rm_load_in_4bit", True),
        max_length=cfg["reward"].get("quality_rm_max_length", 1024),
    )

    if rtype == "darling":
        from src.creativity_rl.darling import DarlingReward, SimilarityClassifier

        clf = SimilarityClassifier(device=cfg["reward"].get("sim_classifier_device", "cuda"))
        reward_fn = DarlingReward(quality_scorer=quality, clf=clf)
    else:
        reward_fn = QualityOnlyReward(quality_scorer=quality)

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
        scale_rewards=cfg["rl"]["scale_rewards"],
        max_steps=cfg["rl"]["total_steps"],
        save_steps=cfg["rl"]["save_every_steps"],
        logging_steps=cfg["logging"]["log_every_steps"],
        report_to=["wandb"] if wb["enabled"] else [],
        run_name=wb.get("run_name"),
        bf16=True,
        seed=cfg["seed"],
        remove_unused_columns=False,
        # Memory accommodation for a 40GB GPU. Pure compute/memory
        # trade-offs: identical gradients, no change to N, lr, KL,
        # reward, or data, so the DARLING methodology is unchanged.
        gradient_checkpointing=cfg["rl"].get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Optional: log the actual K rollouts per prompt (wandb table +
        # stdout) so we can verify in-training generations are diverse.
        log_completions=cfg["logging"].get("log_completions", False),
        num_completions_to_print=cfg["logging"].get("num_completions_to_print", 0),
    )

    trainer = GRPOTrainer(
        model=cfg["policy"]["base_model"],
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    _orig = trainer.log

    def _log(logs, *a, **kw):
        logs.update(reward_fn.telemetry())
        return _orig(logs, *a, **kw)

    trainer.log = _log

    if debug:
        print(f"DEBUG: reward={rtype} prompts={len(train_ds)} out={output_dir}", flush=True)
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
