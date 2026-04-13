# 2026-04-13 — ICCC paper: data expansion, partial correlations, paper polish

## Summary

Extended the dat_eval evaluation across 53 LLMs, added EQ-Bench and
Hivemind benchmarks alongside Arena CW, computed partial correlations
controlling for general capability, and brought the ICCC short paper
to a near-camera-ready state with color-coded tables, scatter grids,
and an inter-metric heatmap.

## Tasks completed

### Eval expansion
- Added Hivemind matching script (`src/comb_eval/scripts/add_hivemind_scores.py`)
  pulling intra-model similarity from the Artificial Hivemind paper
  (arXiv:2510.22954). Initial coverage 20 models; later extended to 26.
- Added EQ-Bench Creative Writing scoring via `src/comb_eval/scripts/add_eqbench_scores.py`,
  matching 35 of 49 OpenRouter model IDs to the EQ-Bench v3 leaderboard.
- Added 6 new small open models to extend Hivemind coverage:
  Llama-3.2-1B/3B, Mistral-7B-v0.1, Mistral-Small-24B-2501, Qwen3-14B/8B.
  Updated pricing table and OpenRouter→Arena mappings accordingly.
- Final n: 52 (DAT/PACE), 51 (CDAT), 34 (EQ-Bench), 24 (Hivemind).

### Methodology improvements
- Added `partial_spearman` and `partial_pearson` to the scoring pipeline,
  controlling for Arena Overall to isolate creativity-specific signal
  from general capability.
- Added Pearson `r` alongside Spearman `ρ` for every correlation in the
  output and the paper.
- Added inter-metric correlation reporting (4×4 across DAT, CDAT
  Novelty, CDAT Appropriateness, PACE).

### Run-evals infrastructure
- Reasoning-model handling: added `reasoning.effort=low, exclude=true`
  config and a retry-without-reasoning fallback for providers that
  reject the param (e.g. SiliconFlow on QwQ).
- Model-aware `max_tokens`: 4× multiplier for known reasoning models so
  reasoning tokens don't crowd out the answer.
- Skipped QwQ entirely after discovering its baked-in reasoning was
  silently consuming thousands of tokens per call. Stubbed remaining
  files so the rest of the pipeline treats it as done.
- Added top_p=1.0/top_k=0 controls to bypass provider nucleus filtering
  for DAT/CDAT.

### Paper drafting and polish
- Wrote and pushed the ICCC 2026 short paper to Overleaf
  (`papers/iccc-2026/main.tex`).
- Added single shared "correlation summary" floating block containing:
  (a) Color-coded simple+partial table for Spearman ρ and Pearson r
      against Arena CW, EQ-Bench CW, Hivemind, with green for
      expected-direction significants and maroon for wrong-direction.
  (b) Triangular Batlow heatmap of inter-metric correlations.
  Both share one master caption via subcaption package.
- Added 2×3 (4-rows × 3-cols) scatter grid with per-cell ρ, n, and
  per-model labels.
- Added per-temperature CDAT bar chart.
- Added example-responses figure (DAT, CDAT, PACE outputs from
  Claude Sonnet 4.5) below Table 1.
- Switched figure styling to Helvetica + Batlow/vik (Crameri colormaps).
- Wrote partial correlation formula into the Method section.
- Converted all `\paragraph{}` headings to flowing prose at user
  request.

### Headline findings
- DAT does not predict LLM creative writing after partialling
  (ρ=0.03, p=0.85 vs Arena CW).
- CDAT Appropriateness's apparent CW signal collapses or reverses sign
  after partialling out Arena Overall — it tracks general capability,
  not creativity.
- PACE retains creativity-specific signal vs Arena CW (ρ=0.31, p=0.03)
  and replicates the original Qiu & Hu 0.74 across a different seed
  set and broader model mix.
- After partialling out capability, PACE also predicts Hivemind output
  diversity in the expected direction (ρ=−0.39).

## Files modified / created

- `papers/iccc-2026/` — full short paper (main.tex, iccc.bib, iccc.sty,
  iccc.bst, figures/)
- `src/dat_eval/scripts/score_evals.py` — partial Spearman/Pearson,
  EQ-Bench/Hivemind correlation, inter-metric matrix
- `src/dat_eval/scripts/run_evals.py` — async concurrency, reasoning
  config, model-aware max_tokens, budget cap
- `src/dat_eval/scripts/make_figures.py` — Batlow + Helvetica styling,
  4×3 scatter grid, triangular inter-metric heatmap, scatter labels
- `src/dat_eval/llm.py` — async client, reasoning param, top_p/top_k
- `src/comb_eval/scripts/add_hivemind_scores.py` — new
- `src/comb_eval/scripts/add_eqbench_scores.py` — new
- `src/comb_eval/scripts/fetch_arena_scores.py` — added new model
  mappings
- `scripts/safety/` — status.sh, kill_all.sh, cost_tracker.py (added
  earlier this session, kept up to date with new pricing)
- `configs/dat_eval/run_evals.yaml` — temps, top-p/k, max_tokens,
  reasoning, budget cap, full 55-model list
- `configs/comb_eval/benchmarks.json` — augmented with eq_bench_cw and
  hivemind_intra_sim per model
- `docs/reports/2026-04-12_preliminary_correlations/` — written and
  iteratively updated through the session
- `docs/AI_OPERATIONS_PROTOCOL.md` — created earlier; followed
  throughout (background tasks registered, status.sh used to verify
  no orphan processes)

## Key decisions and insights

- **Ground-truth source for spend is OpenRouter dashboard**, not local
  estimates. Local cost tracker is a proxy.
- **Reasoning models can't always be tamed** — QwQ in particular has
  reasoning baked into the fine-tune; provider-side `effort=low` is
  rejected by some providers; the only safe option is to skip.
- **Partial correlation flips the headline** — under simple
  correlation PACE looked like it failed to predict diversity; partial
  correlation revealed that capability was masking the underlying
  signal.
- **Two creativity dimensions exist**: output diversity (Hivemind) and
  creative-writing quality (Arena CW / EQ-Bench). Most metrics measure
  one, neither, or the wrong combination. Only PACE captures both
  after capability is controlled for.

## Open questions and next steps

- The Hivemind partial correlation for PACE went from −0.71 (n=19) to
  −0.39 (n=24) when 6 small models were added. Worth understanding
  whether the small-model addition diluted or revealed the true signal.
- Pearson and Spearman partials disagree on the Hivemind result
  (Spearman: PACE strongly negative after partialling; Pearson: small
  positive). Caption flags this; for camera-ready we may want a more
  decisive tie-breaker (e.g., bootstrap CIs on partials).
- Missing models for full Hivemind comparability include Llama-3.1-405B,
  o1 family, and several Qwen variants — Hivemind has them but they're
  not in our current eval set.
- The Overleaf draft is at the polished-figures stage; next is a
  full editorial pass on prose and bibliography verification (the bib
  entries are still flagged as AI-generated and need human verification).
