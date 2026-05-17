"""One-shot generation sampler for qualitative inspection.

Loads the base policy and RM, generates K samples per prompt for a
small set of prompts, scores with RM. Prints prompt + completions +
appropriateness + pass/fail relative to current tau.

Used during training to inspect what's flowing through the system.
With KL << 1, base-policy outputs are representative of trained-policy
outputs in early training.

Usage:
    uv run python src/creativity_rl/scripts/sample_examples.py \\
        configs/creativity_rl/full_run_v1.yaml [N_PROMPTS=3] [K=4]
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def main(config_path: str, n_prompts: int = 3, k: int = 4, adapter_path: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.creativity_rl.data import load_prompts
    from src.creativity_rl.scoring import AppropriatenessScorer

    tau = float(cfg["appropriateness"]["threshold_tau"])

    prompts = load_prompts(
        hf_name=cfg["dataset"]["test"]["hf_name"],
        split=cfg["dataset"]["test"]["split"],
        text_field=cfg["dataset"]["test"]["text_field"],
        n=n_prompts,
        seed=cfg["seed"] + 1,  # different prompts than calibration
    )

    print(f"Loading base policy {cfg['policy']['base_model']}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["policy"]["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # correct for batched decoder generation
    model = AutoModelForCausalLM.from_pretrained(
        cfg["policy"]["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path).eval()

    print(f"Loading RM {cfg['appropriateness']['rm_model']}...", flush=True)
    scorer = AppropriatenessScorer(
        model_name=cfg["appropriateness"]["rm_model"],
        device="cuda",
        load_in_4bit=cfg["appropriateness"].get("rm_load_in_4bit", False),
        max_length=cfg["appropriateness"].get("rm_max_length", 512),
    )

    gen_cfg = cfg["rl"]["generation"]
    print(f"\nGenerating K={k} responses per prompt, max_tokens={gen_cfg['max_new_tokens']}, "
          f"temp={gen_cfg['temperature']}, top_p={gen_cfg['top_p']}\n", flush=True)
    print(f"τ = {tau} (responses with RM score > τ pass the appropriateness gate)\n", flush=True)
    print("=" * 80, flush=True)

    for i, prompt in enumerate(prompts):
        print(f"\nPROMPT {i+1}: {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n", flush=True)
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        # Generate K samples in parallel
        toks = tokenizer([chat] * k, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = []
        for seq in out:
            gen = seq[toks["input_ids"].shape[1] :]
            completions.append(tokenizer.decode(gen, skip_special_tokens=True))

        scores = scorer.score([prompt] * k, completions)
        for j, (c, s) in enumerate(zip(completions, scores)):
            gate = "PASS" if s > tau else "FAIL"
            c_disp = c.replace("\n", " ")[:300]
            print(f"  [{j+1}] RM={s:+.3f} {gate}: {c_disp}{'...' if len(c) > 300 else ''}", flush=True)
        print("-" * 80, flush=True)


if __name__ == "__main__":
    cfg_path = sys.argv[1]
    n_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    adapter = sys.argv[4] if len(sys.argv) > 4 else None
    main(cfg_path, n_prompts, k, adapter)
