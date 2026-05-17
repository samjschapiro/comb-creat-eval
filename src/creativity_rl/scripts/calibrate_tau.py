"""Calibrate the appropriateness threshold tau on a base policy.

Samples N responses from the base policy (no LoRA, no training) on a
random subset of training prompts, scores them with the appropriateness
RM, and sets tau as the (1 - pass_rate_target)-quantile so that
pass_rate_target fraction of base-policy responses pass the gate.

Outputs a JSON file at <output_dir>/calibration_tau.json with the
calibrated value plus diagnostics (mean / std / quantiles).

Usage:
    uv run python src/creativity_rl/scripts/calibrate_tau.py \\
        configs/creativity_rl/full_run_v1.yaml [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def main(config_path: str, overwrite: bool = False) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_root = Path(cfg["output_dir"])
    cal_dir = out_root / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    out_file = cal_dir / "calibration_tau.json"
    if out_file.exists() and not overwrite:
        raise ValueError(f"FATAL: {out_file} exists. Use --overwrite.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.creativity_rl.data import load_prompts
    from src.creativity_rl.scoring import AppropriatenessScorer

    cal_cfg = cfg["appropriateness"]["calibration"]
    n_samples = cal_cfg["n_samples_for_tau"]
    pass_rate = cal_cfg["pass_rate_target"]

    print(f"[1/4] Loading {n_samples} prompts...", flush=True)
    prompts = load_prompts(
        hf_name=cfg["dataset"]["train"]["hf_name"],
        split=cfg["dataset"]["train"]["split"],
        text_field=cfg["dataset"]["train"]["text_field"],
        n=n_samples,
        seed=cfg["seed"],
    )

    print(f"[2/4] Loading base policy ({cfg['policy']['base_model']})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["policy"]["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["policy"]["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    print("[3/4] Sampling base-policy responses...", flush=True)
    responses: list[str] = []
    batch_size = 8
    gen_cfg = cfg["rl"]["generation"]
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        chat_inputs = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch
        ]
        toks = tokenizer(chat_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j, seq in enumerate(out):
            gen = seq[toks["input_ids"].shape[1] :]
            responses.append(tokenizer.decode(gen, skip_special_tokens=True))
        if (i // batch_size) % 4 == 0:
            print(f"  generated {len(responses)}/{len(prompts)}", flush=True)
    del model
    torch.cuda.empty_cache()

    print("[4/4] Scoring with RM and computing tau...", flush=True)
    scorer = AppropriatenessScorer(
        model_name=cfg["appropriateness"]["rm_model"],
        device="cuda",
        load_in_4bit=cfg["appropriateness"].get("rm_load_in_4bit", False),
        max_length=cfg["appropriateness"].get("rm_max_length", 1024),
    )
    scores = scorer.score(prompts, responses)
    tau = float(np.quantile(scores, 1.0 - pass_rate))

    result = {
        "tau": tau,
        "pass_rate_target": pass_rate,
        "n_samples": len(scores),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "quantiles": {
            "p05": float(np.quantile(scores, 0.05)),
            "p10": float(np.quantile(scores, 0.10)),
            "p20": float(np.quantile(scores, 0.20)),
            "p50": float(np.quantile(scores, 0.50)),
            "p80": float(np.quantile(scores, 0.80)),
            "p95": float(np.quantile(scores, 0.95)),
        },
        "config_path": str(config_path),
        "rm_model": cfg["appropriateness"]["rm_model"],
        "base_model": cfg["policy"]["base_model"],
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nCalibrated tau = {tau:.4f}", flush=True)
    print(f"  RM score mean={result['mean']:.3f} std={result['std']:.3f}", flush=True)
    print(f"  Quantiles: p20={result['quantiles']['p20']:.3f}  p50={result['quantiles']['p50']:.3f}", flush=True)
    print(f"  Wrote {out_file}", flush=True)
    print(f"\nUpdate configs/creativity_rl/full_run_v1.yaml:")
    print(f"  appropriateness.threshold_tau: {tau:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite)
