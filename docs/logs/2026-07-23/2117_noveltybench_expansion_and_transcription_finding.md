# 2026-07-23 · rebuttal — benchmark n-expansion attempt + the "transcribed values" finding

**Summary.** Worked on the NeurIPS rebuttal (papers/iccc-2026, submission 13188), targeting the
reviewers' biggest concern — small benchmark sample sizes (F63x "major weakness"; 6Cbj on
NoveltyBench n≈8–10). Tried to expand NoveltyBench n by generating benchmark scores ourselves.
**Key finding: every benchmark value in `benchmarks.json` is transcribed from an external
paper/leaderboard, so self-running any benchmark with a different pipeline yields incomparable
scores.** Three self-run attempts failed for this reason (or bugs); the valid path is to fill
*our* side (test + capability coverage) for models the external source already scored.

## Context: the rebuttal

Reviews (see the OpenReview PDF): rEB8 5/Accept (conf 2), 6Cbj 4/Borderline-accept (conf 3),
F63x 3/Borderline-reject (**conf 5** — the swing reviewer). Biggest cross-cutting issues:
1. **Small n / statistical robustness** (F63x + 6Cbj) — the #1 threat; undercuts the
   scientific-ideation null and the NoveltyBench CDAT +0.60.
2. Motivation "why tests not benchmarks?" (F63x). 3. creativity vs creative-achievement
   definitions (all three). 4. narrow construct coverage / RAT literature (6Cbj). 5. technical
   fixes (CDAT computation, green-cell rule, CDAT-A "non-positive throughout" overstatement,
   inter-test/benchmark corr, temperature effect, common-method bias). 6. base models.
   **DRAT is out of scope — user: "it's a completely different paper." Do not mention DRAT/RAT.**

## THE finding (most important)

`src/comb_eval/scripts/add_*_scores.py` shows all benchmark columns are **transcribed**, not
self-generated:
- NoveltyBench — "Direct transcription from NoveltyBench paper Table 1" (Skywork-Reward quality model).
- Hivemind — derived from **Table 6** of the Hivemind paper (per-model 10-bin similarity
  histogram → midpoint-weighted mean). 79 models in Table 6; only ~26 mapped to our pool.
- Mazur, EQ-Bench, ARC-AGI, MMLU-Pro — all scraped from their respective leaderboards.

**Implication:** you cannot raise a benchmark's n by re-running it with a different
generation/scoring pipeline — the scale won't match. The **valid, cheap** expansion is to take
the models the external source *already scored* and fill our side: run DAT/CDAT/PACE (cheap,
sampling works over OpenRouter) and pull public Arena Overall + MMLU-Pro.
- NoveltyBench: 20 paper models; 10 fully usable now. Of the missing 10: **7 need only
  Arena+MMLU** (we already have their test scores) → n 10→17; **3 need test runs too**
  (claude-3-opus, gemini-1.5-pro, gemini-2.0-pro) → n→20.
- Hivemind: ~28 unmapped open Table-6 models (Qwen2.5/Qwen3, OLMo-2, Tulu-3, aya, Mistral
  v0.2/v0.3, Ministral, Mixtral, Phi-3.5, gemma-1.1…) → map + our-side backfill.

## Attempts (all three self-run routes failed — recorded so they aren't repeated)

1. **Local MLX generation (M5).** Set up `.venv-mlx` + `mlx_lm.server`; ran 4 models (~5 hrs).
   **Invalid:** `mlx_lm.server` 0.31.3 decodes **greedily, ignoring temperature** (verified: all
   methods give identical samples; Python API `make_sampler(temp=1.0)` samples fine). Zeroed out
   diversity on a diversity benchmark. All local runs discarded.
2. **OpenRouter served generation + gpt-4o-mini judge.** Generation samples correctly; ran 6
   models (~$1.65). **Invalid scale:** existing pool is the paper's Skywork numbers, ours use a
   gpt-4o-mini judge → our gpt-4o-mini = **5.25** vs paper **3.11** for the same model. Not mergeable.
3. **Lambda A100 + real Skywork (authors' official code).** Cloned `novelty-bench/novelty-bench`
   (has `inference.py`/`partition.py`(DeBERTa)/`score.py`(Skywork, calibration = 9 hardcoded
   reward-bin thresholds → 1–10)). Downloaded Skywork-Reward-Gemma-2-27B-v0.2; generated
   llama-3.1-8b (their transformers mode) → partition → score (8-bit; bf16 OOMs on 40 GB).
   **Validation failed: our utility = 2.12 vs paper 3.76.** Diagnosed cause: the repo's
   transformers-mode generation left `<|eot_id|>` special tokens trailing every response, which
   tanks Skywork quality and makes DeBERTa collapse all responses to one class (partitions all
   `[0,…]`). A strip-tokens-and-rescore test was launched but the **user killed the VM** before it
   finished (cost ≈ $5–6). Unresolved whether stripping reproduces 3.76.

## Assets built this session

- `.venv-mlx/` — isolated Mac env (py3.12, mlx-lm, torch+MPS, transformers, datasets); **not
  for the CUDA-pinned repo env**. Used for local scoring/recompute.
- `scripts/new_tests/recompute_noveltybench_corr.py` — **VALIDATED**: reproduces the paper's
  NoveltyBench Overall column exactly (DAT −.21/−.20, CDAT +.63/+.60, …) from cached test scores
  + benchmarks.json, via the paper's own compositing + semi-partial specificity. Confirmed the
  headline point: **CDAT's +0.60 NoveltyBench specificity is n.s. (p≈0.21) at n=8–10** — exactly
  6Cbj's concern. Use this for the rebuttal's power/robustness numbers.
- `scripts/new_tests/merge_local_noveltybench.py` — merge self-run NB summaries into
  benchmarks.json (dry-run default). NOTE: only valid once a self-run pipeline is proven to match
  the paper scale (currently it is NOT — see attempt 3).
- `scripts/new_tests/run_noveltybench_{local,served}_batch.sh`, `..._served_parallel.sh` — batch
  runners (superseded; kept for record). Bug fixed: bash 3.2 lacks `wait -n`; and `pkill -f
  inference.py` self-matches its own SSH shell — use the `[i]nference` bracket trick.
- Code fixes: `src/new_tests/noveltybench.py` (`_generate_for_prompt` guards None `choices` from
  OpenRouter error-shaped 200s, retries then ""), `src/dat_eval/llm.py` (`call_llm_async` returns
  None instead of crashing on empty `choices`). Both legitimate robustness fixes — keep.

## Files modified / created

- Created: this log; `scripts/new_tests/{recompute_noveltybench_corr,merge_local_noveltybench}.py`;
  `scripts/new_tests/run_noveltybench_{local_batch,served_batch,served_parallel}.sh`;
  `configs/new_tests/{nb_local,nb_served}/`, `noveltybench_local_smoke.yaml`; `.venv-mlx/` (gitignored-ish).
- Modified: `src/new_tests/noveltybench.py`, `src/dat_eval/llm.py` (None-choices guards).
- `configs/comb_eval/benchmarks.json` — **reverted to original** (a merge was applied then reverted);
  `benchmarks.json.bak_pre_nblocal` is a backup == original and can be deleted.
- `.claude/settings.json` — auto-modified by the harness (session SSH allowlist); left unstaged.

## Cost

OpenRouter ≈ $1.65 (served NB gens, unused) + Lambda A100 ≈ $5–6 (~2.5–3 hrs @ $1.99/hr,
terminated). Net negative result on NoveltyBench self-run, but the transcription finding + the
validated recompute tool + the our-side-backfill plan are the durable outputs.

## Next steps

1. **Rebuttal-first, cheap, valid:** raise n by our-side backfill — Arena+MMLU lookups for the 7
   NoveltyBench paper-models we're missing (→ n 17, $0), then DAT/CDAT/PACE on the 3 remaining
   (→ n 20); same method for Hivemind's ~28 unmapped Table-6 models.
2. Use `recompute_noveltybench_corr.py` (and analogous per-benchmark recompute) to report
   validity/specificity + power at the new n.
3. If self-run NB is still wanted: resolve attempt-3 (strip special tokens or regenerate with
   proper EOS, re-score with Skywork, check reproduction of 3.76) **before** trusting/merging any
   self-run values. Needs a ≥48 GB GPU for bf16, or accept 8-bit after a reproduction check.
4. Non-experiment rebuttal points (motivation, definitions, temperature reanalysis, inter-test
   corr, CDAT-A wording) are writing tasks — see the reviewer synthesis above.
