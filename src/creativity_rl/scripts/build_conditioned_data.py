"""Stage 1: build the conditioned divergence dataset.

For each prompt x: sample a set S of prior answers from the base model,
format the conditioned prompt (x + S + the divergence instruction),
sample candidates conditioned on that, score appropriateness, measure
each candidate's distance to S, then record the supervised target (the
appropriate candidate farthest from S) and, for the preference fallback,
a (chosen, rejected) pair.

Writes a JSONL dataset to build.output_dir/records.jsonl.

Usage:
    uv run python src/creativity_rl/scripts/build_conditioned_data.py \\
        configs/creativity_rl/conditioned_v1.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)["build"]
    if "output_dir" not in cfg:
        raise ValueError("FATAL: 'build.output_dir' required in config")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.creativity_rl.conditioned import (
        format_conditioned_prompt,
        select_pair,
        select_sft_target,
    )
    from src.creativity_rl.data import load_prompts
    from src.creativity_rl.scoring import AppropriatenessScorer, SBERTEmbedder

    output_dir = init_directory(cfg["output_dir"], overwrite=overwrite)
    shutil.copy(config_path, output_dir / "config.yaml")

    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    m = cfg["conditioning"]["s_size"]
    s_pool = cfg["conditioning"]["s_pool"]
    k = cfg["conditioning"]["n_candidates"]
    if s_pool < m:
        raise ValueError(f"FATAL: s_pool ({s_pool}) must be >= s_size ({m})")
    if cfg["appropriateness"]["threshold_tau"] is None:
        raise ValueError(
            "FATAL: build.appropriateness.threshold_tau is null. Run "
            "calibrate_conditioned_tau.py and set the value before the build "
            "(tau is base-model specific)."
        )
    tau = float(cfg["appropriateness"]["threshold_tau"])
    gen = cfg["generation"]

    print(f"[1/4] Loading prompts...", flush=True)
    prompts = load_prompts(
        hf_name=cfg["dataset"]["hf_name"],
        split=cfg["dataset"]["split"],
        text_field=cfg["dataset"]["text_field"],
        n=cfg["dataset"]["max_prompts"],
        seed=cfg["seed"],
    )

    print(f"[2/4] Loading base policy + scorers...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["policy"]["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # correct for batched decoder generation
    model = AutoModelForCausalLM.from_pretrained(
        cfg["policy"]["base_model"], torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    scorer = AppropriatenessScorer(
        model_name=cfg["appropriateness"]["rm_model"],
        device="cuda",
        load_in_4bit=cfg["appropriateness"].get("rm_load_in_4bit", False),
        max_length=cfg["appropriateness"].get("rm_max_length", 512),
    )
    embedder = SBERTEmbedder(cfg["novelty"]["embedding_model"], device="cuda")

    def sample(user_content: str, n: int) -> list[str]:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        toks = tokenizer([chat] * n, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=gen["max_new_tokens"],
                temperature=gen["temperature"],
                top_p=gen["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        return [
            tokenizer.decode(seq[toks["input_ids"].shape[1]:], skip_special_tokens=True)
            for seq in out
        ]

    print(f"[3/4] Building records ({len(prompts)} prompts)...", flush=True)
    records = []
    skipped_s = 0  # prompts dropped because too few base samples passed the gate
    for i, x in enumerate(prompts):
        # S = the m most appropriate base samples, filtered from a larger pool.
        pool = sample(x, s_pool)
        pool_appr = scorer.score([x] * s_pool, pool)
        passing_idx = [j for j in range(s_pool) if pool_appr[j] > tau]
        if len(passing_idx) < m:
            skipped_s += 1
            continue
        passing_idx.sort(key=lambda j: -float(pool_appr[j]))
        S = [pool[j] for j in passing_idx[:m]]

        cond_prompt = format_conditioned_prompt(x, S)
        cands = sample(cond_prompt, k)

        appr = scorer.score([x] * k, cands)
        emb = embedder.encode(cands)
        s_emb = embedder.encode(S)
        # novelty of a candidate = mean cosine distance to S (normalized embeddings)
        nov = np.array([float(np.mean(1.0 - e @ s_emb.T)) for e in emb])

        sft_idx = select_sft_target(appr, nov, tau)
        pair = select_pair(appr, nov, tau)
        if sft_idx is None:
            continue
        rec = {
            "x": x,
            "S": S,
            "prompt": cond_prompt,
            "sft_target": cands[sft_idx],
            "chosen": cands[pair.chosen_idx] if pair else None,
            "rejected": (
                cands[pair.rejected_idx]
                if pair and pair.rejected_idx is not None
                else None
            ),
            "appropriateness": appr.tolist(),
            "novelty_vs_S": nov.tolist(),
        }
        records.append(rec)
        if debug and i < 2:
            print(json.dumps(rec, indent=2)[:1500], flush=True)
        if i % 20 == 0:
            print(f"  {i}/{len(prompts)} prompts, {len(records)} records", flush=True)

    print(f"[4/4] Writing {len(records)} records...", flush=True)
    with open(output_dir / "records.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(
        f"Done. {len(records)}/{len(prompts)} prompts yielded a target; "
        f"{skipped_s} skipped (fewer than {m} base samples passed the gate).",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("config_path")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true")
    a = p.parse_args()
    main(a.config_path, a.overwrite, a.debug)
