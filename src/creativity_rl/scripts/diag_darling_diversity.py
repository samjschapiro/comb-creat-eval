"""Generation-only DARLING diversity diagnostic.

Loads base Llama-3.1-8B-Instruct, samples K completions for a handful of
real WildChat prompts at the real DARLING sampling params, and runs the
exact darling.py partition diversity on each group. Answers one question
before committing to a 600-step run: do K samples to real prompts produce
nonzero diversity (a learning signal), or is the reward dead at cold start?

No training, no RM, no optimizer -- just generation + the CPU classifier.

    uv run python src/creativity_rl/scripts/diag_darling_diversity.py \
        [n_prompts] [K] [max_new_tokens]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def main() -> None:
    n_prompts = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_new = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.creativity_rl.darling import SimilarityClassifier, _partition_distinctness
    from src.creativity_rl.data import load_wildchat_prompts

    prompts = load_wildchat_prompts(n=n_prompts, seed=17)
    print(f"Loaded {len(prompts)} WildChat prompts", flush=True)

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16
    ).to("cuda").eval()
    clf = SimilarityClassifier(device="cpu")

    any_div = 0
    all_div = []
    for pi, p in enumerate(prompts):
        text = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                max_new_tokens=max_new,
                num_return_sequences=K,
                pad_token_id=tok.eos_token_id,
            )
        gens = [
            tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for o in out
        ]
        div = _partition_distinctness(gens, clf)
        all_div.append(float(div.mean()))
        if (div > 0).any():
            any_div += 1
        lens = [len(g.split()) for g in gens]
        print(
            f"\n[{pi}] prompt: {p[:90]!r}\n"
            f"    word-lens={lens}\n"
            f"    diversity={np.round(div, 3).tolist()} mean={div.mean():.3f}",
            flush=True,
        )
        print(f"    sample gen[0]: {gens[0][:160]!r}", flush=True)
        print(f"    sample gen[1]: {gens[1][:160]!r}", flush=True)

    print(
        f"\n=== SUMMARY: {any_div}/{len(prompts)} prompts have >0 diversity in "
        f"some response; mean per-prompt diversity = {np.mean(all_div):.3f} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()
