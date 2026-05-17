"""Calibrate the appropriateness threshold tau for the conditioned build.

Samples base-policy responses on the build's prompt source, scores them
with the appropriateness RM, and sets tau at the (1 - pass_rate_target)
quantile so that pass_rate_target of base responses pass the gate.

tau is base-model specific: a value calibrated on one model is invalid
for another. Run this whenever build.policy.base_model changes.

Reads the `build` section of the config. Writes
build.output_dir/../calibration_tau.json and prints the value to put in
build.appropriateness.threshold_tau.

Usage:
    uv run python src/creativity_rl/scripts/calibrate_conditioned_tau.py \\
        configs/creativity_rl/conditioned_v1.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["build"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.creativity_rl.data import load_prompts
    from src.creativity_rl.scoring import AppropriatenessScorer

    cal = cfg["appropriateness"]["calibration"]
    n = cal["n_samples"]
    pass_rate = cal["pass_rate_target"]
    gen = cfg["generation"]

    print(f"[1/4] Loading {n} prompts...", flush=True)
    prompts = load_prompts(
        hf_name=cfg["dataset"]["hf_name"],
        split=cfg["dataset"]["split"],
        text_field=cfg["dataset"]["text_field"],
        n=n,
        seed=cfg["seed"],
    )

    print(f"[2/4] Loading base policy {cfg['policy']['base_model']}...", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["policy"]["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # correct for batched decoder generation
    model = AutoModelForCausalLM.from_pretrained(
        cfg["policy"]["base_model"], torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    print("[3/4] Sampling base responses...", flush=True)
    responses: list[str] = []
    bs = 8
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        chats = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]
        toks = tok(chats, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=gen["max_new_tokens"],
                temperature=gen["temperature"],
                top_p=gen["top_p"],
                do_sample=True,
                pad_token_id=tok.pad_token_id,
            )
        for seq in out:
            responses.append(tok.decode(seq[toks["input_ids"].shape[1]:], skip_special_tokens=True))
        if (i // bs) % 5 == 0:
            print(f"  {len(responses)}/{len(prompts)}", flush=True)
    del model
    torch.cuda.empty_cache()

    print("[4/4] Scoring and computing tau...", flush=True)
    scorer = AppropriatenessScorer(
        model_name=cfg["appropriateness"]["rm_model"],
        device="cuda",
        load_in_4bit=cfg["appropriateness"].get("rm_load_in_4bit", False),
        max_length=cfg["appropriateness"].get("rm_max_length", 512),
    )
    scores = scorer.score(prompts[: len(responses)], responses)
    tau = float(np.quantile(scores, 1.0 - pass_rate))

    out_dir = Path(cfg["output_dir"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "tau": tau,
        "pass_rate_target": pass_rate,
        "n": len(scores),
        "base_model": cfg["policy"]["base_model"],
        "rm_model": cfg["appropriateness"]["rm_model"],
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "quantiles": {
            q: float(np.quantile(scores, v))
            for q, v in {"p10": 0.10, "p20": 0.20, "p50": 0.50, "p80": 0.80}.items()
        },
    }
    (out_dir / "calibration_tau.json").write_text(json.dumps(result, indent=2))
    print(f"\nCalibrated tau = {tau:.4f}", flush=True)
    print(f"  base={cfg['policy']['base_model']} mean={result['mean']:.3f} std={result['std']:.3f}", flush=True)
    print(f"  wrote {out_dir / 'calibration_tau.json'}", flush=True)
    print(f"\nSet build.appropriateness.threshold_tau: {tau:.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_path")
    a = p.parse_args()
    main(a.config_path)
