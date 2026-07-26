# 2026-07-24 · NoveltyBench self-run pipeline validated + pool expanded (n=10→17)

**Summary.** Built and validated a self-run NoveltyBench pipeline (OpenRouter generation +
authors' DeBERTa partition + bf16 Skywork-Reward-27B scoring) that reproduces the paper's
Table-1 utility within 1.8%, then used it to expand the NoveltyBench correlation pool from
n=10 (transcribed-only) to n=17. **Result: CDAT's NoveltyBench specificity regresses from
+0.60 (n=8) to +0.22 (n=13, n.s.)** — confirming reviewer 6Cbj's small-sample concern.

## Pipeline validation (the hard part)

Reproducing the paper's NoveltyBench utility required fixing three separate issues, each of
which alone put us far off (llama-3.1-8b, paper Utility=3.76):
1. **skip_special_tokens=False** in the repo's multi-generation decode path → `<|eot_id|>`
   pollution → utility 2.12. Fixed at source (`inference.py`).
2. **8-bit quantization** pushed Skywork's raw reward outside its narrow calibration band
   (thresholds −7.72…−2.05), collapsing 85% of quality scores to buckets 1/10 → utility 3.05.
   bf16 (authors' dtype) restores a healthy quality spread → 4.31.
3. **Prompt set + sampling.** Table 1 is the **union of 1,100 prompts** (100 curated + 1,000
   WildChat), not curated-only; and generation is pure temperature-1 (HF/provider defaults
   silently applied top_p=0.9/top_k=50, suppressing diversity). Fixing both → union utility
   **3.69 vs paper 3.76 (−1.8%)**. Validated → self-run values are mergeable with transcribed.

## Expansion

- Generated 9 non-frontier open models via OpenRouter (temp=1, top_p=1, top_k=0), locally
  (GPU-free) at ~$8. Integrity check caught **2 dead models** (mistral-7b-v0.1, nemotron-70b:
  100% empty — no OpenRouter endpoint) and **qwen-2.5-72b at 81%** (topped up incomplete
  prompts). **7 clean models** survived.
- Scored on a 2×H100 (bf16 Skywork fits fully, fast). Union utilities merged into
  `benchmarks.json` (`noveltybench_utility`, source-tagged): mistral-nemo 4.00, llama-3.1-70b
  3.72, phi-4 3.59, mistral-small-24b 3.29, llama-4-scout 3.14, qwen-2.5-72b 3.09,
  gemma-3-27b 2.26.

## Finding (n=10 → n=17)

| test | n=10 val/spec | n=17 val/spec |
|---|---|---|
| DAT | −0.21 / −0.20 | −0.31 / −0.28 |
| CDAT | +0.63 / +0.60 (n=8) | +0.41 / +0.22 (n=13) |
| CDAT-N | +0.56 / +0.45 | +0.25 / +0.04 |
| CDAT-A | −0.63 / −0.40 | −0.35 / −0.02 |
| PACE | −0.20 / −0.00 | −0.33 / −0.07 |

At n=17 **no test significantly predicts NoveltyBench utility** on either axis. CDAT's +0.60
specificity was a small-sample artifact — directly answers 6Cbj, but weakens the paper's
"CDAT best predicts divergent thinking" claim (revise accordingly).

**Caveat:** correlation mixes 10 transcribed (paper pipeline) + 7 self-run (our pipeline);
validated on one overlap model (llama-3.1-8b). More overlap validation would harden it.

## Reproducibility scripts (scripts/new_tests/)
`gen_noveltybench_openrouter.py` (pinned-sampling generation), `gen_nb_batch_local.sh` (local
batch), `nb_topup_empties.py` (regen incomplete prompts), `nb_provision_scoring.sh` (Lambda
provision, bf16 config), `nb_score_all.sh` (partition+score+union). Scores archived at
`data/new_tests/noveltybench_skywork/` (gitignored). All Lambda instances terminated.

## Next
Expand further with the mid-tier non-frontier models (llama-4-maverick, qwen3-235b,
deepseek-chat×2, claude-3-haiku; ~$19 OpenRouter → n≈22). Reasoning models (qwq, deepseek-r1)
excluded — inline `<think>` traces corrupt diversity/quality scoring.
