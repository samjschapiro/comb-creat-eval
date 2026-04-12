# Combinatorial Creativity Eval — Track Progress

## Overview

Evaluate off-the-shelf LLMs on combinatorial creativity using a synthetic graph task, then correlate scores with Chatbot Arena Creative Writing rankings.

**Based on:**
- Combinatorial creativity framework: arXiv:2509.21043 (Sections 3+)
- Evaluation methodology (correlate with Arena CW): PACE paper, EMNLP 2025

**Core claims:**
1. Combinatorial creativity — producing novel combinations of familiar concepts under constraints — can be evaluated in off-the-shelf LLMs via a synthetic graph task presented in-context.
2. LLM performance correlates with Arena CW rankings (and more strongly than with Arena Overall or MMLU-Pro).
3. (Exploratory) The novelty-utility tradeoff persists in frontier LLMs.

**Key confound to address:** Larger models may simply be better at both graph search and creative writing. The partial correlation (Arena CW controlling for Arena Overall) is the critical test.

**Comparison targets:** PACE scores, DAT scores — if our eval + PACE + DAT all independently predict Arena CW but measure different things, that's the strongest finding.

## Design Decisions

- **Graph size**: 150 nodes, ~440 edges (avg degree 6). Adjacency text is ~7.6K chars — fits comfortably in context. Large enough that brute-force enumeration is infeasible; small enough for single-prompt presentation.
- **Graph presentation**: Full adjacency list in system prompt, queries in user messages. Not truly "in-memory" — we're testing constrained combinatorial search over known structure, not memorization.
- **Eval prompts**: 200 total. 4 hop counts (1-4) × 10 base paths × 5 difficulty levels. Constraints are inclusion/exclusion on edge labels, verified solvable via constrained BFS.
- **Scoring**: Creativity = Utility × Novelty. Utility is binary-gated by constraint satisfaction. Novelty = α_h × hops + α_r × avg_surprise. Fully automatic, no LLM-as-judge.

## Pipeline

```
Step 1: generate_eval.py  →  graph.json, prompts.json
Step 2: run_eval.py       →  per-model responses.json
Step 3: analyze.py        →  scores, correlations, comparisons
```

## Progress

### 2026-04-11 — Initial implementation
- [x] Project scaffolding (pyproject.toml, utils, track structure)
- [x] Graph construction module (`src/comb_eval/graph.py`)
- [x] Eval prompt generation with constrained BFS (`src/comb_eval/prompts.py`)
- [x] LLM prompting infrastructure via OpenRouter (`src/comb_eval/llm.py`) — single API for all models
- [x] Scoring module: Creativity = Utility × Novelty (`src/comb_eval/scoring.py`)
- [x] Correlation analysis: Spearman, bootstrap, partial correlations, PACE/DAT comparison (`src/comb_eval/analysis.py`)
- [x] Orchestration scripts for all 3 pipeline stages + Arena score fetcher
- [x] Tested Step 1: 150-node graph, 200 prompts generated successfully
- [x] Arena score fetcher via Playwright — scrapes arena.ai for both Overall and CW categories
- [x] Fetched real Arena scores: 17/17 models matched (Overall + CW)

### Next steps
- [ ] Run Step 2 against target LLMs (need OPENROUTER_API_KEY in .env)
- [ ] Run Step 3 analysis and evaluate correlation results
- [ ] Tune graph size if needed (sweep 100–300 nodes)
- [ ] Collect/compute PACE and DAT scores for comparison
- [ ] Investigate whether difficulty-stratified scores improve the correlation
