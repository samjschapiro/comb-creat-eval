"""Isolate why TRL GRPO rollouts collapse: for the exact prompts TRL
used (from the gencheck wandb completions tables), compare
  - TRL-logged 8 completions' partition diversity, vs
  - 8 fresh samples from plain model.generate(do_sample, temp=1, top_p=1)
on the SAME prompt. If plain >> TRL, the TRL generation path is the bug.

    uv run python src/creativity_rl/scripts/diag_trl_vs_plain.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def main() -> None:
    import wandb
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.creativity_rl.darling import SimilarityClassifier, _partition_distinctness

    api = wandb.Api()
    rs = sorted(
        api.runs("schapirolab/comb-creat-eval",
                 filters={"display_name": "darling_gencheck"}),
        key=lambda r: r.created_at,
    )
    r = rs[-1]
    arts = [a for a in r.logged_artifacts() if "completions" in a.name]
    pairs = []  # (prompt_text, [trl completions])
    for a in arts:
        d = a.download()
        tj = glob.glob(os.path.join(d, "**/*.json"), recursive=True)
        t = json.load(open(tj[0]))
        ci = {c: i for i, c in enumerate(t["columns"])}
        rows = t["data"]
        pr = rows[0][ci["prompt"]]
        comps = [row[ci["completion"]] for row in rows]
        pairs.append((pr, comps))
    print(f"{len(pairs)} TRL prompt-groups recovered", flush=True)

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model = (
        AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16
        ).to("cuda").eval()
    )
    clf = SimilarityClassifier(device="cuda")

    # Strip the chat-template wrapper TRL logged; re-extract the user turn.
    def user_text(p: str) -> str:
        if "<|start_header_id|>user<|end_header_id|>" in p:
            seg = p.split("<|start_header_id|>user<|end_header_id|>")[1]
            return seg.split("<|eot_id|>")[0].strip()
        return p

    for k, (praw, trl_comps) in enumerate(pairs):
        u = user_text(praw)
        trl_div = float(_partition_distinctness(trl_comps, clf).mean())

        text = tok.apply_chat_template(
            [{"role": "user", "content": u}],
            tokenize=False, add_generation_prompt=True,
        )
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc, do_sample=True, temperature=1.0, top_p=1.0,
                max_new_tokens=512, num_return_sequences=8,
                pad_token_id=tok.eos_token_id,
            )
        plain = [
            tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for o in out
        ]
        plain_div = float(_partition_distinctness(plain, clf).mean())
        print(
            f"\n[{k}] {u[:80]!r}\n"
            f"    TRL-logged diversity   = {trl_div:.3f}\n"
            f"    plain-generate diversity = {plain_div:.3f}",
            flush=True,
        )
        print(f"    TRL gen0 : {trl_comps[0][:140]!r}", flush=True)
        print(f"    TRL gen1 : {trl_comps[1][:140]!r}", flush=True)
        print(f"    plain g0 : {plain[0][:140]!r}", flush=True)
        print(f"    plain g1 : {plain[1][:140]!r}", flush=True)


if __name__ == "__main__":
    main()
