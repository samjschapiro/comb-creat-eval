# DAT/CDAT/PACE Correlation Study — Track Progress

## Overview

**ICCC 2026 Short Paper:** Does the Divergent Association Test Actually Measure Creativity in LLMs?

Evaluate a set of LLMs on three creativity metrics (DAT, CDAT, PACE), then correlate each with Chatbot Arena Creative Writing rankings. Test whether these psycholinguistic creativity measures actually predict creative ability in LLMs, and which metric does so most reliably.

## Claims

1. **DAT scores may not meaningfully predict LLM creative writing ability** — the CDAT paper (Nakajima et al., 2026) showed DAT is invalid for LLMs because random words outscore all models. But does it still correlate with Arena CW?
2. **CDAT (with appropriateness gating) is a more valid creativity metric for LLMs than DAT** — if CDAT correlates more strongly with Arena CW, the appropriateness constraint matters.
3. **Comparing DAT, CDAT, and PACE provides a multi-faceted view** — do they measure the same thing? If they correlate with each other AND with Arena CW, they tap into a shared creativity construct. If they diverge, they capture different facets.

## Metrics

- **DAT** (Olson et al., 2021): Generate 10 maximally different nouns. Score = mean pairwise cosine distance (GloVe 840B) of first 7 valid words × 100.
- **CDAT** (Nakajima et al., 2026): Generate 10 diverse words associated with a cue. Novelty (pairwise distance, SBERT) + appropriateness gate (similarity to cue). Score = novelty conditional on passing gate.
- **PACE** (Qiu & Hu, 2025): Generate 3 parallel 20-word association chains per seed. Score = average semantic distance (FastText) across positions and chains.

## Pipeline

```
Step 1: run_evals.py    → raw LLM responses for all three tasks
Step 2: score_evals.py  → scores + correlations with Arena benchmarks
```

## Embedding resources needed for scoring

- `resources/glove.840B.300d.txt` — for DAT scoring
- `resources/crawl-300d-2M.vec` — for PACE scoring
- `all-mpnet-base-v2` (SBERT, auto-downloaded) — for CDAT scoring

## Progress

### 2026-04-11 — Initial implementation
- [x] Track structure created
- [x] DAT module: GloVe-based scoring, validation, prompting (`src/dat_eval/dat.py`)
- [x] CDAT module: SBERT scoring, appropriateness gate, cue words (`src/dat_eval/cdat.py`)
- [x] PACE module: FastText scoring, 2-stage prompting, chain parsing (`src/dat_eval/pace.py`)
- [x] Unified LLM caller via OpenRouter (`src/dat_eval/llm.py`)
- [x] Orchestration: run_evals.py (Step 1) and score_evals.py (Step 2)
- [x] Smoke-tested with GPT-4o: all three tasks produce clean output
- [x] Reuses Arena benchmarks from comb_eval track (`configs/comb_eval/benchmarks.json`)

### Next steps (initial plan, mostly completed below)
- [x] Download GloVe and FastText embeddings to `resources/`
- [x] Run full eval (52 models × all three tasks)
- [x] Run scoring and correlation analysis
- [x] Check if DAT is really "invalid" — confirmed
- [x] Compare CDAT vs PACE vs DAT correlation strengths
- [x] Write up short paper draft

### 2026-04-12 / 2026-04-13 — Full eval, partial correlations, paper draft

Pipeline / infrastructure
- Async OpenRouter client with 20-way concurrency in `src/dat_eval/llm.py`
- Reasoning-model handling: `reasoning.effort=low,exclude=true` plus a
  retry-without-reasoning fallback for providers that reject the param
- Model-aware `max_tokens` (4× multiplier for known reasoning models)
- top_p=1.0 / top_k=0 controls in `run_evals.py` to bypass nucleus filtering
- Budget cap (`budget_usd`) in run config, with `cost_tracker.py` PRICING table
- Idempotent skip-if-exists per (model, eval, temperature) file

Sampling and parameters
- DAT: 40 trials per temp at $T \in \{1.0, 1.5, 2.0\}$, unique seeds
- CDAT: 50 cue words at the same three temperatures
- PACE: 50 seed words at $T = 0$ (per Qiu & Hu)

Benchmarks added
- Arena Overall (already had)
- EQ-Bench Creative Writing v3 — added via `add_eqbench_scores.py`
- Hivemind intra-model similarity (arXiv:2510.22954) — added via
  `add_hivemind_scores.py`

Correlation analysis
- Spearman ρ, Pearson r, 500-iter bootstrap CIs
- Partial correlations against Arena CW, EQ-Bench CW, Hivemind, all
  controlling for Arena Overall (formula in `partial_spearman` /
  `partial_pearson`)
- 4×4 inter-metric correlation matrix

Headline findings (final, n=52 / 51 / 24)
- DAT vs Arena CW: simple ρ=0.36** but partial collapses to ρ=0.03 (NS)
  — entire signal is general-capability driven
- CDAT Appropriateness vs Arena CW: simple ρ=0.45*** but partial flips
  to ρ=−0.16 (NS) — appropriateness indexes capability, not creativity
- PACE vs Arena CW: simple ρ=0.78***, partial ρ=0.31* — only metric
  with creativity-specific signal that survives partialling
- PACE vs Hivemind (partial): ρ=−0.39 — strongest diversity-predictive
  signal among the four

Paper artifacts (Overleaf-synced via `papers/iccc-2026/`)
- 4-page short paper draft pushed to Overleaf
- Camera-ready figures (Helvetica + Batlow/vik):
  - 4×3 scatter grid (metrics × benchmarks)
  - Triangular inter-metric heatmap (Batlow)
  - Per-temperature CDAT bar chart
  - Color-coded correlation table (green = expected-direction
    significant; maroon = wrong-direction significant)
  - Example-responses figure (DAT/CDAT/PACE outputs from Sonnet 4.5)
- Math definition of partial correlation in Method section
- Bibliography entries flagged as AI-generated (need human verification)

### Next steps
- [ ] Verify and clean up bibliography entries (currently flagged
      `% --- AI-GENERATED REFERENCE (VERIFY) ---`)
- [ ] Editorial pass on prose for short-paper concision
- [ ] Decide on Spearman vs Pearson tie-break for the Hivemind
      partial finding (they disagree; n=24 is small)
- [ ] Optionally extend Hivemind sample by adding the Hivemind-only
      Llama-3.1-405B / o1 family / smaller Qwen variants
