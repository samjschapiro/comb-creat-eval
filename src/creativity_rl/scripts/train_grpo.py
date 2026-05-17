"""MCNS-RL training entry point.

GRPO with MCNS reward (novelty conditional on appropriateness gate).
LoRA adapter on a small base model. wandb logging.

Usage:
    uv run python src/creativity_rl/scripts/train_grpo.py \\
        configs/creativity_rl/smoke_test.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory


REQUIRED_TOP_LEVEL = [
    "output_dir", "seed", "dataset", "policy", "appropriateness",
    "novelty", "reward", "rl", "logging", "budget",
]


def validate_config(cfg: dict) -> None:
    for key in REQUIRED_TOP_LEVEL:
        if key not in cfg:
            raise ValueError(f"FATAL: '{key}' is required in config")
    if cfg["reward"]["type"] == "mcns_soft" and cfg["reward"].get("soft_temperature") is None:
        raise ValueError("FATAL: reward.soft_temperature required when reward.type='mcns_soft'")
    # tau may be null in config and filled in by reading a calibration
    # file from the output_dir (see resolve_tau below). Hard-error only
    # if both routes are missing.


def resolve_tau(cfg: dict) -> float:
    """Resolve appropriateness threshold tau.

    Priority:
    1. config.appropriateness.threshold_tau (if not null).
    2. calibration JSON written by calibrate_tau.py at
       {output_dir}/calibration/calibration_tau.json.
    3. Otherwise fail (unless calibration.skip is set, in which case
       require tau to be in the config).
    """
    import json as _json

    explicit = cfg["appropriateness"].get("threshold_tau")
    if explicit is not None:
        return float(explicit)

    skip = cfg["appropriateness"].get("calibration", {}).get("skip", False)
    if skip:
        raise ValueError(
            "FATAL: threshold_tau is null and calibration.skip=true. "
            "Set tau explicitly in config."
        )

    cal_file = Path(cfg["output_dir"]) / "calibration" / "calibration_tau.json"
    if not cal_file.exists():
        raise ValueError(
            f"FATAL: threshold_tau is null and calibration file not found at "
            f"{cal_file}. Run src/creativity_rl/scripts/calibrate_tau.py first."
        )
    with open(cal_file) as f:
        cal = _json.load(f)
    print(f"  Loaded calibrated tau = {cal['tau']:.4f} from {cal_file}", flush=True)
    return float(cal["tau"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    seed_everything(cfg["seed"])

    # Resolve tau BEFORE init_directory potentially wipes the calibration
    # file that calibrate_tau.py wrote into output_dir/calibration/.
    tau = resolve_tau(cfg)

    output_dir = init_directory(cfg["output_dir"], overwrite=overwrite)
    shutil.copy(config_path, output_dir / "config.yaml")
    for sub in ("checkpoints", "logs", "archive", "calibration"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)
    # Re-save resolved tau into the new (post-init) calibration dir for
    # traceability (init_directory wiped the original).
    import json as _json
    with open(output_dir / "calibration" / "resolved_tau.json", "w") as f:
        _json.dump({"tau": tau}, f, indent=2)

    if debug:
        print(f"DEBUG: output_dir={output_dir}", flush=True)

    # Heavy imports deferred so config errors fail fast.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from src.creativity_rl.archive import ClusterArchiveSet
    from src.creativity_rl.data import cluster_prompts, load_prompts
    from src.creativity_rl.reward import MCNSReward
    from src.creativity_rl.reward_callable import MCNSRewardFunction
    from src.creativity_rl.scoring import AppropriatenessScorer, SBERTEmbedder

    # ---- 1. Load prompts ----
    print("[1/7] Loading prompts...", flush=True)
    prompts = load_prompts(
        hf_name=cfg["dataset"]["train"]["hf_name"],
        split=cfg["dataset"]["train"]["split"],
        text_field=cfg["dataset"]["train"]["text_field"],
        n=cfg["dataset"]["train"].get("max_prompts"),
        seed=cfg["seed"],
    )
    print(f"  Loaded {len(prompts)} prompts", flush=True)

    # ---- 2. Embed prompts + cluster ----
    print("[2/7] Embedding prompts and clustering...", flush=True)
    embedder = SBERTEmbedder(cfg["novelty"]["embedding_model"], device="cuda")
    prompt_embeddings = embedder.encode(prompts)
    cluster_ids = cluster_prompts(
        prompt_embeddings,
        n_clusters=cfg["novelty"]["archive"]["n_clusters"],
        seed=cfg["seed"],
    )
    print(f"  {cfg['novelty']['archive']['n_clusters']} clusters, sizes: "
          f"{np.bincount(cluster_ids).tolist()[:10]}...", flush=True)

    # ---- 3. Load RM ----
    print("[3/7] Loading appropriateness RM...", flush=True)
    scorer = AppropriatenessScorer(
        model_name=cfg["appropriateness"]["rm_model"],
        device="cuda",
        load_in_4bit=cfg["appropriateness"].get("rm_load_in_4bit", False),
        max_length=cfg["appropriateness"].get("rm_max_length", 1024),
    )
    print(f"  RM loaded; chat_template={scorer._use_chat_template}", flush=True)

    # ---- 4. Init archive + reward ----
    print("[4/7] Initializing archive and reward...", flush=True)
    archive_set = ClusterArchiveSet(
        n_clusters=cfg["novelty"]["archive"]["n_clusters"],
        dim=embedder.dim,
        hnsw_m=cfg["novelty"]["archive"]["hnsw_m"],
        hnsw_ef_construction=cfg["novelty"]["archive"]["hnsw_ef_construction"],
        max_size=cfg["novelty"]["archive"]["max_size_per_cluster"],
        warmup_admissions=cfg["novelty"]["archive"].get("warmup_admissions", 8),
    )
    # tau was already resolved before init_directory; reuse it.
    mcns_reward = MCNSReward(
        tau=tau,
        mode="hard" if cfg["reward"]["type"] == "mcns_hard" else "soft",
        soft_temperature=cfg["reward"].get("soft_temperature"),
    )
    reward_fn = MCNSRewardFunction(
        scorer=scorer,
        embedder=embedder,
        archive_set=archive_set,
        reward=mcns_reward,
        k_nearest=cfg["novelty"]["k_nearest"],
        admission_mode=cfg["novelty"]["archive"]["admission_novelty_threshold"],
    )

    # ---- 5. Build training dataset (conversational + cluster_id) ----
    print("[5/7] Building training dataset...", flush=True)
    train_ds = Dataset.from_list([
        {
            "prompt": [{"role": "user", "content": p}],
            "cluster_id": int(c),
        }
        for p, c in zip(prompts, cluster_ids)
    ])

    # ---- 6. Configure GRPO + LoRA ----
    print("[6/7] Configuring GRPO trainer...", flush=True)
    peft_config = LoraConfig(
        r=cfg["policy"]["lora"]["rank"],
        lora_alpha=cfg["policy"]["lora"]["alpha"],
        lora_dropout=cfg["policy"]["lora"]["dropout"],
        target_modules=cfg["policy"]["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    wandb_cfg = cfg["logging"]["wandb"]
    if wandb_cfg["enabled"]:
        os.environ.setdefault("WANDB_PROJECT", wandb_cfg["project"])

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
        report_to=["wandb"] if wandb_cfg["enabled"] else [],
        run_name=wandb_cfg.get("run_name"),
        bf16=True,
        seed=cfg["seed"],
        remove_unused_columns=False,    # keep cluster_id for the reward fn
    )

    trainer = GRPOTrainer(
        model=cfg["policy"]["base_model"],
        reward_funcs=reward_fn,
        args=grpo_args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    # Hook into trainer's logging to inject reward telemetry from our callable.
    _original_log = trainer.log
    def _log_with_telemetry(logs, *args, **kwargs):
        logs.update(reward_fn.telemetry())
        return _original_log(logs, *args, **kwargs)
    trainer.log = _log_with_telemetry

    # ---- 7. Train ----
    print("[7/7] Starting training...", flush=True)
    trainer.train()
    trainer.save_model(str(output_dir / "checkpoints" / "final"))
    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
