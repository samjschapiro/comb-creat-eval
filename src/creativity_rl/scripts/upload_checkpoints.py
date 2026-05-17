"""Upload LoRA checkpoints to wandb as artifacts on an existing run.

Resumes the wandb run associated with the training, then logs each
checkpoint directory as a separate Artifact of type "model".

Usage:
    uv run python src/creativity_rl/scripts/upload_checkpoints.py \\
        --run_id p9twhk40 \\
        --checkpoints_dir data/creativity_rl/runs/full_run_v1/checkpoints
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(run_id: str, checkpoints_dir: str, project: str, entity: str | None) -> None:
    import wandb

    cps = sorted(Path(checkpoints_dir).iterdir())
    cps = [p for p in cps if p.is_dir() and p.name.startswith("checkpoint-")]
    if not cps:
        raise SystemExit(f"FATAL: no checkpoint-* dirs in {checkpoints_dir}")

    print(f"Found {len(cps)} checkpoints: {[p.name for p in cps]}", flush=True)
    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must",
        job_type="upload-checkpoints",
    )
    print(f"Resumed wandb run {run.id} ({run.url})", flush=True)

    for cp in cps:
        step_str = cp.name.split("-")[1]
        artifact = wandb.Artifact(
            name=f"qwen-1.5b-mcns-{step_str}",
            type="model",
            description=f"LoRA adapter at training step {step_str} from full_run_v1",
            metadata={
                "step": int(step_str),
                "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
                "lora_rank": 16,
                "training_config": "configs/creativity_rl/full_run_v1.yaml",
            },
        )
        # Skip optimizer.pt (~150MB) — large and not needed for inference.
        for f in cp.iterdir():
            if f.name == "optimizer.pt":
                print(f"  [{cp.name}] skipping {f.name} (not needed for inference)", flush=True)
                continue
            artifact.add_file(str(f))
        print(f"  [{cp.name}] logging artifact...", flush=True)
        run.log_artifact(artifact)

    print("\nFlushing... (artifact uploads continue in background until finish)", flush=True)
    run.finish()
    print("Done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "comb-creat-eval"))
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY") or None)
    args = parser.parse_args()
    main(args.run_id, args.checkpoints_dir, args.project, args.entity)
