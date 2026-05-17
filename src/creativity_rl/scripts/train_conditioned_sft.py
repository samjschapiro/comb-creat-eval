"""Stage 2: supervised training on the conditioned dataset.

Reads the JSONL produced by build_conditioned_data.py. Each training
example is the chat formatted conditioned prompt followed by the
supervised target (the appropriate candidate farthest from S). LoRA
adapter on a small base model via TRL's SFTTrainer.

Usage:
    uv run python src/creativity_rl/scripts/train_conditioned_sft.py \\
        configs/creativity_rl/conditioned_v1.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        full = yaml.safe_load(f)
    cfg = full["train"]
    if "output_dir" not in cfg:
        raise ValueError("FATAL: 'train.output_dir' required in config")
    if "upstream_dir" not in cfg:
        raise ValueError("FATAL: 'train.upstream_dir' required in config")

    upstream = Path(cfg["upstream_dir"])
    records_path = upstream / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(
            f"FATAL: {records_path} not found. Run build_conditioned_data.py first."
        )

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    output_dir = init_directory(cfg["output_dir"], overwrite=overwrite)
    shutil.copy(config_path, output_dir / "config.yaml")

    tokenizer = AutoTokenizer.from_pretrained(cfg["policy"]["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = []
    with open(records_path) as f:
        for line in f:
            r = json.loads(line)
            text = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["sft_target"]},
                ],
                tokenize=False,
            )
            rows.append({"text": text})
    if not rows:
        raise ValueError("FATAL: no training rows; upstream produced an empty dataset")
    print(f"Loaded {len(rows)} training rows", flush=True)
    ds = Dataset.from_list(rows)

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

    args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=float(cfg["optim"]["learning_rate"]),
        per_device_train_batch_size=cfg["optim"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["optim"]["gradient_accumulation_steps"],
        max_steps=cfg["optim"]["total_steps"],
        max_seq_length=cfg["optim"]["max_seq_length"],
        save_steps=cfg["optim"]["save_every_steps"],
        logging_steps=cfg["logging"]["log_every_steps"],
        report_to=["wandb"] if wb["enabled"] else [],
        run_name=wb.get("run_name"),
        bf16=True,
        seed=cfg["seed"],
    )

    trainer = SFTTrainer(
        model=cfg["policy"]["base_model"],
        args=args,
        train_dataset=ds,
        peft_config=peft_config,
    )
    if debug:
        print(f"DEBUG: {len(ds)} rows, output {output_dir}", flush=True)
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
