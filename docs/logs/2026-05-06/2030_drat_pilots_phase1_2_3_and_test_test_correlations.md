# 2026-05-06 — DRAT pilots Phase 1+2+3, prompt ablation, multi-embedding, test-test correlations

## Summary

Long evening session continuing the DRAT design from earlier today.
Built the pilot pipeline incrementally: Phase 1 (6 cheap LIB-17
models, K=30 pre-registered hand-curated bank), Phase 2 (4 mid-tier),
Phase 3 (6 expensive/reasoning). Covered **16 of 17 LIB-17 models**
(qwq-32b skipped). Re-scored everything under all three embeddings
(SBERT, GloVe, FastText). Ablated prompt framing (default
"connects A and B" vs analogical "metaphorically applied to both").
Computed test-test correlations between DRAT and the four existing
dat_eval tests (DAT, CDAT, CDAT-N, CDAT-A, PACE).

Headline empirical result: **DRAT is significantly anti-correlated
with DAT** (r = −0.65, p = 0.006 under GloVe; r ≈ −0.5 at p < 0.05
under all three embeddings). This is the strongest p-value in the
analysis and the most defensible construct-validity signal we have.
It says DRAT is not "DAT with anchors" — the two measure structurally
different things. LIB specificity sits at +0.24 to +0.28 across
embeddings (matching the paper's +0.21 / +0.24 baselines on the same
benchmark) but does not reach conventional significance at n=16.

Total API spend across the three pilot phases plus the analogical
re-pilot: under \$5 at OpenRouter list prices.

## Tasks completed

### Phase 1 — six cheap LIB-17 models on default prompt
- Hand-curated $K=30$ anchor bank, organized as 3 pairs each across the
  10 inter-division pairings of {life, physical, social, formal,
  engineering}. Saved as `configs/new_tests/drat_pilot_v1.yaml`.
- Two degenerate pairs swapped from the initial draft after a review:
  (mutation, mutation operator) → (heartbeat, oscillator),
  (graph, network) → (algorithm, factory).
- 180 calls completed, 0 errors after adding retry-with-backoff to
  the runner (initial run died on a 429 from DeepInfra for
  mistral-small).
- Findings: pipeline robust; per-model means tightly clustered for
  5/6 models in 49–58 range, mistral-small clearly broken (mean
  9.26, refused some prompts, produced random scatter on others).
  At n=5 with mistral excluded, r(DRAT, LIB) = −0.76 — driven
  almost entirely by llama-3.3-70b being highest-DRAT lowest-LIB.
  Too small to draw conclusions.
- Bank flaw surfaced: 9/30 pairs have τ > 0.35 (high), causing
  gate failures concentrated on broad/abstract anchors like
  (heart, engine), (virus, crystal), (neuron, graph). Documented
  but not fixed for this run — pre-registration discipline.

### Phase 2 — four mid-tier LIB-17 models on default prompt
- Added gpt-4o-mini, claude-3.5-haiku, deepseek-chat, qwen-2.5-72b.
- Combined Phase 1+2 → n=10 LIB-17 (n=9 working).
- One non-fatal error: qwen returned None on (phase transition,
  revolution).
- Findings at n=9 working models: r(DRAT, LIB) = −0.36, r|g = −0.52
  (still in the wrong direction). r(DRAT, Arena) = +0.23.
  llama-3.3-70b still anchoring the negative pattern; pattern
  moderated but didn't reverse.

### Multi-embedding rescoring (no API)
- Built `src/new_tests/scripts/rescore_drat.py`. Reuses
  `GloVeEmbeddings` from `src/dat_eval/dat.py` and
  `FastTextEmbeddings` from `src/dat_eval/pace.py`. Multi-word
  anchors handled by averaging available token vectors; OOV tokens
  silently skipped.
- All 300 Phase 1+2 responses rescored under SBERT, GloVe, FastText.
- Findings: embedding choice meaningfully changes correlations.
  Under SBERT, r|g(LIB) ≈ 0; under FastText, r|g(LIB) = +0.13–+0.22
  (gated $n_{\min}=3$ or $n_{\min}=5$). Running only SBERT (the
  smoke-test default) had been actively misleading.
- Hivemind diversity strongly negative across all embeddings × all
  variants (r ≈ −0.6 to −0.7) at n=9; moderates to ~−0.10 by n=14
  later.

### Prompt-framing ablation
- Added `prompt_style` kwarg to `drat_prompt()` and threaded through
  the runner. New "analogical" framing:
  *"each of which could be metaphorically applied to both A and B"*
  vs default *"each of which connects A and B"*.
- Re-ran all 10 LIB-17 cheap+mid models with analogical prompt;
  saved to `data/new_tests/drat/pilot_analogical_v1/`.
- Findings: mean DRAT collapses (50 → 25 range) — most analogical
  attempts fail the gate. But model *rankings* under analogical
  correlate better with LIB: r|g(LIB, FastText, $n_{\min}=3$) = +0.48
  at n=10 (vs +0.22 default).
- Initially read the score collapse as a misalignment between
  prompt and metric. Closer look at responses showed the gate is
  filtering in favor of grounded analogies (firewall, network,
  mechanism — words that map onto specific structural roles in
  both domains) and against loose metaphors (river, labyrinth).
  Retracted the misalignment claim — the analogical prompt + current
  score is *more* faithful to the cognitive target, not less.

### Phase 3 — six expensive/reasoning LIB-17 models on analogical prompt
- Added llama-3.1-70b, mistral-large-2411, gpt-4o, gpt-4-turbo,
  deepseek-r1, o3-mini. qwq-32b deliberately skipped.
- Reasoning controls (`effort: low`, `exclude: true`,
  4× max_tokens multiplier) added to `run_drat_smoke.py` and
  forwarded to OpenRouter via `call_llm_async`.
- 180 calls completed, 0 errors.
- Combined Phase 1+2+3 analogical → **n = 16 LIB-17 coverage**.

### Headline correlation table at n=16, analogical, gated $n_{\min}=3$

| benchmark | n | best embedding | r | p | r\|g | p\|g |
|---|---|---|---|---|---|---|
| LIB | 16 | sbert | +0.34 | 0.19 | +0.28 | 0.31 |
| LIB | 16 | glove | +0.37 | 0.16 | +0.14 | 0.62 |
| LIB | 16 | fasttext | +0.38 | 0.14 | +0.24 | 0.39 |
| Arena CW | 16 | glove | +0.55 | **0.027** \* | +0.16 | 0.58 |
| Arena CW | 16 | fasttext | +0.48 | 0.060 . | +0.25 | 0.38 |
| EQ-Bench CW | 7 | (small n) | | | | |
| Mazur CW | 8 | (small n) | | | | |
| Hivemind div | 14 | fasttext | −0.29 | 0.32 | −0.12 | 0.71 |
| NovelB-U | 7 | sbert | +0.72 | 0.069 . | +0.72 | 0.11 |

- LIB specificity matches paper baselines but doesn't reach
  conventional significance.
- Arena CW raw correlation crosses p < 0.05 under GloVe; specificity
  drops near zero, so this is the capability-mediated path, not
  test-specific.
- NoveltyBench Utility specificity is the strongest construct claim
  on a benchmark, but n=7 means we can't put weight on the p-value.
- Hivemind weirdness from earlier n=9 analysis moderates to weak
  negative at n=14. Probably noise.

### Test-test correlations (n=16)

| test | r(SBERT) | r(GloVe) | r(FastText) | best p |
|---|---|---|---|---|
| DAT | −0.54 | **−0.65** | −0.51 | **p = 0.006** \*\* |
| CDAT | +0.24 | +0.31 | +0.44 | 0.089 . |
| CDAT-N | −0.21 | −0.31 | −0.41 | 0.114 |
| CDAT-A | +0.33 | +0.42 | +0.49 | 0.056 . |
| PACE | −0.05 | +0.07 | +0.05 | 0.86 |

- **DRAT × DAT: significantly negative**, p < 0.01 under GloVe and
  p < 0.05 under SBERT and FastText. The strongest signal in the
  analysis.
- DRAT × CDAT-A borderline positive (+0.49, p = 0.056) — both
  measure anchoring to source anchors.
- DRAT × CDAT-N weakly negative — same divergence-vs-bridging
  tension that drives the DAT correlation, weaker form.
- DRAT × PACE indistinguishable from zero — independent measurement.

### Sample sizes worth flagging

- LIB: n=16 (full LIB-17 coverage minus qwq-32b)
- Arena CW: n=16
- Hivemind: n=14 (some LIB-17 models lack hivemind_diversity)
- NovelB-U: n=7 (NoveltyBench has limited LIB-17 overlap)
- Mazur CW: n=8
- EQ-Bench CW: n=7

Small-n benchmarks (NovelB-U, Mazur, EQ-Bench) have OLS instability
when computing r|g with 2 capability covariates (df = n − 3 = 4–5).
The very large negative r|g values on EQ-Bench CW and Mazur CW
(−0.55 to −0.99 in places) are almost certainly OLS-fit artifacts,
not real signal.

## Files

### New
- `configs/new_tests/drat_pilot_v1.yaml` — Phase 1 (6 cheap models,
  default prompt). Contains the locked $K=30$ anchor bank.
- `configs/new_tests/drat_pilot_v2.yaml` — Phase 2 (4 mid-tier models,
  default prompt).
- `configs/new_tests/drat_pilot_analogical_v1.yaml` — analogical
  re-pilot of all 10 cheap+mid models.
- `configs/new_tests/drat_pilot_phase3.yaml` — Phase 3 (6 expensive/
  reasoning models, analogical prompt). Adds `reasoning_models`
  list and `reasoning` block.
- `src/new_tests/scripts/rescore_drat.py` — multi-embedding rescoring
  + multi-benchmark correlation analysis (no API).

### Modified
- `src/new_tests/drat.py` — `drat_prompt()` now takes `style` kwarg
  ("default" or "analogical").
- `src/new_tests/scripts/run_drat_smoke.py` —
  - `asyncio.gather` with semaphore for concurrency
  - retry-on-429/503/timeout with exponential backoff (1.5, 2.5,
    4.5, 8.5s) for up to `max_retries` attempts
  - `prompt_style` config option, threaded through to `drat_prompt`
  - `reasoning_models` + `reasoning` config options for OpenRouter
    reasoning API forwarding (effort, exclude, max_tokens multiplier)
  - returns error stub instead of raising on call failure, so one
    bad model doesn't kill the whole run

## Findings worth flagging

### 1. The DAT anti-correlation is the strongest construct-validity signal

Across all embeddings and all variants tested, DRAT and DAT correlate
negatively. p < 0.01 under GloVe at n=16. The mechanism follows from
the score: DAT rewards pure scatter, DRAT rewards anchored
divergence. A model that aces DAT scatters too widely to anchor in
the (A, B) intersection that DRAT requires.

This is the strongest empirical answer to "is DRAT just DAT with
anchors": no, statistically significantly no.

### 2. Analogical prompt is the cognitively faithful version

Reading the actual responses showed that the analogical framing
elicits Gentner-style structure mapping (firewall, network,
labyrinth as analogies for immune system / supply chain) where the
default prompt elicits literal field vocabulary (vaccine, logistics,
warehouse). The score's per-word utility gate filters in favor of
grounded analogies. This is the right behavior for the construct
DRAT was designed to measure.

The score-magnitude collapse under analogical (means 50 → 25) is a
real fact about how hard the task is, not evidence the metric is
broken.

### 3. The bank pre-registration matters more than I'd appreciated

9/30 pairs in the v1 bank had τ > 0.35, driving gate failures
concentrated on abstract or broad-vocabulary anchors. (heart,
engine) had τ = 0.41 and 5/6 models gate-failed. Pre-registered
τ-prescreening — sample candidate pairs, calibrate τ on each, keep
only those in $[0.20, 0.30]$ — is the next thing to build before any
paper-defensible run.

### 4. Statistical conclusions

- LIB validity / specificity at n=16 does not reach p < 0.05 under
  any embedding × variant combination.
- This matches the dat_eval paper's published finding for DAT and
  CDAT on LIB at n=17. We're at the same statistical wall.
- Test-test correlations *do* reach p < 0.01 (DRAT × DAT under
  GloVe). Test-vs-benchmark correlations do not.
- The honest read for any paper: DRAT measures something distinct
  from existing tests; whether what it measures predicts LIB
  specifically is an open empirical question that the current pool
  doesn't have power to answer.

## Process and tooling notes

- The runner refactors (concurrency, retry, prompt_style, reasoning)
  are general-purpose — `run_drat_smoke.py` is no longer
  smoke-specific. Worth renaming to `run_drat.py` next session.
- Multi-embedding rescoring is the right pattern: keep raw responses
  in `data/.../raw_results.json`, rescore offline under different
  embeddings/variants. No re-querying needed for design ablations.
- Cost discipline held: ~\$5 total across 4 pilot runs (Phase 1, 2,
  analogical-v1, Phase 3) at OpenRouter list prices.

## Next steps

1. **τ-prescreening for the bank.** Construct a larger candidate
   pool (50–80 pairs), calibrate τ for each across all three
   embeddings, keep the 30 with τ in a controlled range.
   Pre-register the new bank as `drat_pilot_bank_v2.json`. No API
   calls needed.
2. **Re-run with the new bank.** All 16 LIB-17 models, analogical
   prompt, both prompt styles for ablation, all three embeddings.
   ~\$5.
3. **Multi-seed sampling.** Currently $S=1$ seed per pair. The DAT
   pipeline uses 3 temperatures × 40 trials. For DRAT a smaller
   $S=3$ seeds at temp 1.0 would already give within-model variance
   estimates and reduce per-model SEM substantially. ~\$15.
4. **Decide whether to scale to the full dat_eval pool.** With the
   τ-prescreened bank and S=3 seeds, take DRAT to all 54 models in
   the dat_eval pool. This doesn't help LIB (still capped at 16–17)
   but tightens the other benchmarks. ~\$30.
5. **Possible design pivot if τ-prescreening doesn't help.** The
   per-word utility gate is the plausible bottleneck for analogical
   responses. A score that explicitly rewards intersection-region
   density (as the original min-utility design intended) might be
   worth revisiting under analogical prompt. Different paper
   structure if so.
