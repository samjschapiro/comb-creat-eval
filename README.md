# comb-creat-eval

Evaluating automatic creativity metrics for large language models.
The current focus is the `dat_eval` track: comparing DAT, CDAT, and
PACE across 52 LLMs and asking whether their correlations with
external creativity benchmarks survive partialling out general
capability. Headline results, figures, and the paper draft are in
`papers/iccc-2026/`.

## Setup

```bash
uv sync
cp .env.example .env  # then add OPENROUTER_API_KEY
```

For scoring you also need the GloVe and FastText embeddings:

```bash
# resources/glove.840B.300d.txt   (DAT)
# resources/crawl-300d-2M.vec     (PACE)
# all-mpnet-base-v2 SBERT auto-downloads on first use (CDAT)
```

## Main scripts

```bash
# 1. Run DAT/CDAT/PACE evals against a model list (OpenRouter)
bash scripts/dat_eval/run_evals.sh

# 2. Score responses; compute raw + partial Pearson r and bootstrap
#    CIs against Arena CW, EQ-Bench CW, Mazur CW v2, Hivemind,
#    NoveltyBench Utility, LiveIdeaBench, and ARC-AGI v2
bash scripts/dat_eval/score_evals.sh

# 3. Generate camera-ready figures
uv run python src/dat_eval/scripts/make_figures.py

# 4. Refresh external benchmark scores
uv run python src/comb_eval/scripts/fetch_arena_scores.py configs/comb_eval/fetch_arena_scores.yaml
uv run python src/comb_eval/scripts/add_eqbench_scores.py
uv run python src/comb_eval/scripts/add_hivemind_scores.py
uv run python src/comb_eval/scripts/add_mazur_scores.py
uv run python src/comb_eval/scripts/add_mmlu_pro_scores.py
uv run python src/comb_eval/scripts/add_noveltybench_scores.py
uv run python src/comb_eval/scripts/add_arc_agi_scores.py
```

## Safety

Before launching anything that hits the OpenRouter API:

```bash
bash scripts/safety/status.sh                    # see what's running
uv run python scripts/safety/cost_tracker.py     # estimate spend so far
bash scripts/safety/kill_all.sh                  # stop everything
```

See `docs/AI_OPERATIONS_PROTOCOL.md` for the full operational rules.
