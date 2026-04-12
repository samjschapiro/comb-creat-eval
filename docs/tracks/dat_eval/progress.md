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

### Next steps
- [ ] Download GloVe and FastText embeddings to `resources/`
- [ ] Run full eval (17 models × all three tasks)
- [ ] Run scoring and correlation analysis
- [ ] Check if DAT is really "invalid" (low/no correlation with Arena CW)
- [ ] Compare CDAT vs PACE vs DAT correlation strengths
- [ ] Write up short paper draft
