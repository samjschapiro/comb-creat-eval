"""Held-out diversity evaluation — the Stanley headline test.

Compares base policy and MCNS-trained checkpoints on prompts the model
has never seen during training. For each model:
  - generate K samples per prompt
  - measure within-prompt diversity (mean pairwise SBERT cosine distance)
  - measure appropriateness (RM score)
  - measure gate pass rate

The diversity metric is computed under SBERT (training-time embedding)
and ideally also a second embedder for sanity (TODO).

Usage:
    uv run python src/creativity_rl/scripts/eval_diversity.py \\
        configs/creativity_rl/full_run_v1.yaml \\
        --checkpoints base,checkpoint-500,checkpoint-1000 \\
        --n_prompts 100 --k 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def generate_batch(model, tokenizer, prompts, k, gen_cfg, device="cuda"):
    """Generate K samples per prompt. Returns list of lists (len=len(prompts), each K)."""
    import torch

    all_completions: list[list[str]] = []
    # Generate per prompt with K replications to allow padding.
    for prompt in prompts:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        toks = tokenizer([chat] * k, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **toks,
                max_new_tokens=gen_cfg["max_new_tokens"],
                temperature=gen_cfg["temperature"],
                top_p=gen_cfg["top_p"],
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        comps = []
        for seq in out:
            gen = seq[toks["input_ids"].shape[1]:]
            comps.append(tokenizer.decode(gen, skip_special_tokens=True))
        all_completions.append(comps)
    return all_completions


def within_prompt_diversity(embeddings_per_prompt: list[np.ndarray]) -> np.ndarray:
    """For each prompt, mean cosine distance across its K embeddings.

    embeddings_per_prompt[i]: (K, D), L2-normalized.
    Returns: (N,) per-prompt mean pairwise cosine distance.
    """
    out = []
    for emb in embeddings_per_prompt:
        K = emb.shape[0]
        # Pairwise cosine distances; embeddings normalized → 1 - <a, b>.
        sim = emb @ emb.T
        dist = 1.0 - sim
        # Mask out diagonal (self-distance = 0)
        iu = np.triu_indices(K, k=1)
        out.append(float(dist[iu].mean()))
    return np.array(out)


def evaluate_one(
    label: str,
    base_model_name: str,
    adapter_path: str | None,
    prompts: list[str],
    k: int,
    gen_cfg: dict,
    rm_scorer,
    embedder,
    tau: float,
):
    """Generate, score, and embed for one model variant. Cleans up GPU after."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n=== {label} ===", flush=True)
    print(f"Loading base policy {base_model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # correct for batched decoder generation
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from {adapter_path}...", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path).eval()

    print(f"Generating {k} samples × {len(prompts)} prompts...", flush=True)
    all_completions = generate_batch(model, tokenizer, prompts, k, gen_cfg)

    # Free model memory before RM/SBERT batch.
    del model
    torch.cuda.empty_cache()

    flat_prompts = []
    flat_completions = []
    for p, comps in zip(prompts, all_completions):
        for c in comps:
            flat_prompts.append(p)
            flat_completions.append(c)

    print("Scoring with RM...", flush=True)
    rm_scores = rm_scorer.score(flat_prompts, flat_completions)
    pass_rate = float((rm_scores > tau).mean())

    print("Embedding with SBERT...", flush=True)
    flat_embeddings = embedder.encode(flat_completions)
    embeddings_per_prompt = [
        flat_embeddings[i * k : (i + 1) * k] for i in range(len(prompts))
    ]
    div = within_prompt_diversity(embeddings_per_prompt)

    return {
        "label": label,
        "rm_mean": float(rm_scores.mean()),
        "rm_std": float(rm_scores.std()),
        "pass_rate": pass_rate,
        "diversity_mean": float(div.mean()),
        "diversity_std": float(div.std()),
        "diversity_per_prompt": div.tolist(),
        "rm_scores": rm_scores.tolist(),
        "n_prompts": len(prompts),
        "k": k,
    }


def main(config_path: str, checkpoints: list[str], n_prompts: int, k: int, output_path: str | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from src.creativity_rl.data import load_prompts
    from src.creativity_rl.scoring import AppropriatenessScorer, SBERTEmbedder

    tau = float(cfg["appropriateness"]["threshold_tau"])
    base_model = cfg["policy"]["base_model"]
    gen_cfg = cfg["rl"]["generation"]

    print(f"Loading {n_prompts} held-out prompts from {cfg['dataset']['test']['hf_name']}...", flush=True)
    prompts = load_prompts(
        hf_name=cfg["dataset"]["test"]["hf_name"],
        split=cfg["dataset"]["test"]["split"],
        text_field=cfg["dataset"]["test"]["text_field"],
        n=n_prompts,
        seed=cfg["seed"] + 100,  # different from any seed used in training
    )

    print(f"Loading RM ({cfg['appropriateness']['rm_model']})...", flush=True)
    rm_scorer = AppropriatenessScorer(
        model_name=cfg["appropriateness"]["rm_model"],
        device="cuda",
        load_in_4bit=cfg["appropriateness"].get("rm_load_in_4bit", False),
        max_length=cfg["appropriateness"].get("rm_max_length", 512),
    )
    print(f"Loading SBERT ({cfg['novelty']['embedding_model']})...", flush=True)
    embedder = SBERTEmbedder(cfg["novelty"]["embedding_model"], device="cuda")

    results = []
    run_root = Path(cfg["output_dir"])
    for ckpt in checkpoints:
        if ckpt == "base":
            adapter = None
            label = "base"
        else:
            adapter = str(run_root / "checkpoints" / ckpt)
            label = ckpt
        r = evaluate_one(label, base_model, adapter, prompts, k, gen_cfg, rm_scorer, embedder, tau)
        results.append(r)

    print("\n" + "=" * 80, flush=True)
    print(f"HELD-OUT DIVERSITY EVAL — {n_prompts} prompts × K={k} samples", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Model':<25} {'Diversity':<14} {'RM mean':<14} {'Pass-rate':<10}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        div = f"{r['diversity_mean']:.4f} ± {r['diversity_std']:.4f}"
        rm = f"{r['rm_mean']:+.3f} ± {r['rm_std']:.3f}"
        pr = f"{r['pass_rate']:.2%}"
        print(f"{r['label']:<25} {div:<14} {rm:<14} {pr:<10}", flush=True)
    print("-" * 80, flush=True)

    # Compute deltas vs base
    if any(r["label"] == "base" for r in results):
        base_r = next(r for r in results if r["label"] == "base")
        print("\nDeltas vs base:", flush=True)
        for r in results:
            if r["label"] == "base":
                continue
            d_div = r["diversity_mean"] - base_r["diversity_mean"]
            d_rm = r["rm_mean"] - base_r["rm_mean"]
            d_pr = r["pass_rate"] - base_r["pass_rate"]
            print(f"  {r['label']}: diversity {d_div:+.4f}  RM {d_rm:+.3f}  pass-rate {d_pr:+.2%}", flush=True)

    if output_path:
        Path(output_path).write_text(json.dumps(results, indent=2))
        print(f"\nWrote detailed results to {output_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--checkpoints", type=str, default="base,checkpoint-500,checkpoint-1000",
                        help="Comma-separated. 'base' = no adapter; others map to checkpoints/<name>.")
    parser.add_argument("--n_prompts", type=int, default=100)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args.config_path, args.checkpoints.split(","), args.n_prompts, args.k, args.output)
