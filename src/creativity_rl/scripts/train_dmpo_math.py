"""DMPO math replication training (arXiv:2605.19461).

Trains Qwen2.5-Math-7B on a verifiable-reward math dataset (GSM8K or
NuminaMath) with the paper's hyperparameters. Two arms via cfg.method:
    dmpo  -> DMPOTrainer (GRPO + group-level distribution-matching MSE)
    grpo  -> vanilla GRPOTrainer (baseline)

Reward is rule-based (format + correctness), so no reward model is loaded
-- VRAM is just policy + GRPO rollouts + optional LoRA.

Usage:
    uv run python src/creativity_rl/scripts/train_dmpo_math.py \\
        configs/creativity_rl/dmpo_math.yaml [--overwrite]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory


# Paper §4.3 format spec: "Chain-of-Thought reasoning followed by
# Answer: [Solution]". We mirror that wording in the prompt so the
# format reward (which strictly checks for the `Answer:` keyword) is
# achievable. Note this differs from Qwen2.5-Math's natural \\boxed{}
# habit -- the model has to learn the requested format.
_FORMAT_INSTR = (
    "Think step by step about the problem, then write your final answer "
    "on a new line in the form\nAnswer: <your answer>"
)


def _problem_user(q: str) -> str:
    return f"{_FORMAT_INSTR}\n\nProblem: {q}"


def load_math_dataset(source: str, n: int | None, seed: int):
    """Build a chat-format dataset with 'prompt' (chat messages list) and
    'ground_truth' columns. TRL forwards ground_truth as a kwarg to the
    reward function when remove_unused_columns=False.

    Sources:
        openr1_math  -> open-r1/OpenR1-Math-220k (closest public artifact
                        to the paper's Openr1-Math-46K; subsample to ~46k
                        by setting n_prompts=46000 in config).
        numina_math  -> AI-MO/NuminaMath-CoT (the underlying source for
                        Open-R1's math data; broad math problem coverage).
        gsm8k        -> openai/gsm8k (smaller, fast for debug runs).
    """
    from datasets import load_dataset

    from src.creativity_rl.math_reward import extract_answer

    if source == "openr1_math":
        ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
        prob_col = "problem" if "problem" in ds.column_names else "question"
        # open-r1 ships a separate `answer` column with the canonical
        # ground-truth answer (not a chain-of-thought solution), so we
        # use it directly without re-parsing.
        if "answer" in ds.column_names:
            ans_col, raw_is_answer = "answer", True
        elif "solution" in ds.column_names:
            ans_col, raw_is_answer = "solution", False
        else:
            raise ValueError(f"no answer column in {ds.column_names}")

        def _fmt(ex):
            raw = ex[ans_col]
            if raw is None:
                gt = ""
            elif raw_is_answer:
                gt = str(raw).strip()
            else:
                gt = extract_answer(raw) or ""
            return {
                "prompt": [{"role": "user", "content": _problem_user(ex[prob_col])}],
                "ground_truth": gt,
            }

        ds = ds.map(_fmt, remove_columns=ds.column_names)
        ds = ds.filter(lambda ex: ex["ground_truth"] != "")
    elif source == "numina_math":
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")

        def _fmt(ex):
            return {
                "prompt": [{"role": "user", "content": _problem_user(ex["problem"])}],
                "ground_truth": extract_answer(ex["solution"]) or "",
            }

        ds = ds.map(_fmt, remove_columns=ds.column_names)
        ds = ds.filter(lambda ex: ex["ground_truth"] != "")
    elif source == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")

        def _fmt(ex):
            ans = ex["answer"].split("####")[-1].strip().replace(",", "")
            return {
                "prompt": [{"role": "user", "content": _problem_user(ex["question"])}],
                "ground_truth": ans,
            }

        ds = ds.map(_fmt, remove_columns=ds.column_names)
    else:
        raise ValueError(f"unsupported math source: {source!r}")

    if n is not None and len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if "output_dir" not in cfg:
        raise ValueError("FATAL: 'output_dir' required in config")
    method = cfg.get("method", "dmpo")
    if method not in ("dmpo", "grpo"):
        raise ValueError(f"FATAL: method must be 'dmpo' or 'grpo', got {method!r}")

    import torch
    from trl import GRPOConfig, GRPOTrainer

    from src.creativity_rl.math_reward import MathRewardFn
    from src.creativity_rl.dmpo import DMPOTrainer

    output_dir = init_directory(cfg["output_dir"], overwrite=overwrite)
    shutil.copy(config_path, output_dir / "config.yaml")

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    n = 32 if debug else cfg["data"]["n_prompts"]
    train_ds = load_math_dataset(cfg["data"]["source"], n=n, seed=cfg["seed"])
    print(f"Loaded {len(train_ds)} math prompts from {cfg['data']['source']}", flush=True)

    reward_fn = MathRewardFn(
        format_weight=cfg["reward"].get("format_weight", 0.1),
        correct_weight=cfg["reward"].get("correct_weight", 1.0),
    )

    # LoRA is optional. Drop the `policy.lora` block in the config to run
    # full fine-tuning (paper-faithful when sharded across multiple GPUs
    # via FSDP). On a single 40/80GB GPU LoRA is required.
    peft_config = None
    if "lora" in cfg["policy"]:
        from peft import LoraConfig
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
        remove_unused_columns=False,           # so ground_truth reaches reward_fn
        gradient_checkpointing=cfg["rl"].get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type=cfg["rl"].get("lr_scheduler_type", "cosine"),
        log_completions=cfg["logging"].get("log_completions", False),
        num_completions_to_print=cfg["logging"].get("num_completions_to_print", 0),
    )

    common = dict(
        model=cfg["policy"]["base_model"],
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    if method == "dmpo":
        trainer = DMPOTrainer(
            **common,
            dm_lambda=float(cfg["dmpo"]["lambda"]),
            dm_alpha=float(cfg["dmpo"]["alpha"]),
        )
        print(
            f"DMPOTrainer: lambda={cfg['dmpo']['lambda']} alpha={cfg['dmpo']['alpha']}",
            flush=True,
        )
    else:
        trainer = GRPOTrainer(**common)
        print("GRPOTrainer (baseline)", flush=True)

    if debug:
        print(f"DEBUG run: prompts={len(train_ds)} out={output_dir}", flush=True)
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
