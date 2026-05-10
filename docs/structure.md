# Project Structure

```
src/
├── utils.py                    # cross-track utilities (init_directory, load_config)
├── comb_eval/                  # earlier exploratory track (mostly dormant)
│   ├── *.py                    # graph / prompt / scoring / analysis modules
│   └── scripts/
│       ├── fetch_arena_scores.py     # scrape Chatbot Arena via Playwright
│       ├── add_eqbench_scores.py     # match OpenRouter ↔ EQ-Bench v3
│       ├── add_hivemind_scores.py    # match OpenRouter ↔ Hivemind paper
│       ├── add_mazur_scores.py       # match OpenRouter ↔ Mazur CW (lechmazur/writing@80b7f17)
│       ├── add_mmlu_pro_scores.py    # MMLU-Pro from TIGER-Lab leaderboard CSV
│       ├── add_noveltybench_scores.py # match OpenRouter ↔ NoveltyBench Utility
│       ├── add_arc_agi_scores.py     # ARC-AGI v2 via llm-stats leaderboard
│       └── build_per_benchmark_table.py  # regen appendix table from benchmarks.json
├── dat_eval/                   # primary track (ICML + NeurIPS 2026 paper)
│   ├── llm.py                  # OpenRouter sync + async client
│   ├── dat.py                  # DAT scoring (GloVe 840B)
│   ├── cdat.py                 # CDAT scoring (SBERT all-mpnet-base-v2)
│   ├── pace.py                 # PACE scoring (FastText crawl-300d-2M)
│   └── scripts/
│       ├── run_evals.py        # async DAT/CDAT/PACE runner with budget cap
│       ├── score_evals.py      # Pearson + semi-partial r(X, Y - Y_hat_g)
│       ├── multi_embed_appendix.py  # per-embedding pipeline (GloVe + FastText + SBERT)
│       └── make_figures.py     # camera-ready figures (Times + RdBu_r / Okabe-Ito)
└── new_tests/                  # active track (DRAT + new benchmark wrappers)
    ├── drat.py                 # DRAT scoring (k-anchor utility gate, max/min/avg)
    ├── distinctness.py         # pairwise-distance utilities shared across tests
    ├── llm.py                  # OpenRouter wrapper specific to this track
    ├── rat.py                  # zero-shot RAT scoring (30 classic items)
    ├── hivemind.py             # Hivemind divergence wrapper
    ├── noveltybench.py         # NoveltyBench utility wrapper
    └── scripts/
        ├── run_drat_smoke.py        # DRAT smoke runner used for pilots/ablations
        ├── run_eqbench_cw.py        # EQ-Bench CW wrapper
        ├── run_hivemind.py          # Hivemind wrapper runner
        ├── run_liveideabench.py     # LiveIdeaBench wrapper runner
        ├── run_noveltybench.py      # NoveltyBench wrapper runner
        └── run_rat.py               # RAT wrapper runner

configs/
├── comb_eval/
│   ├── benchmarks.json         # per-model: arena_overall, mmlu_pro, arena_cw,
│   │                           # eq_bench_cw, mazur_cw_v2, hivemind_diversity,
│   │                           # noveltybench_*, liveideabench (+ per-facet rows)
│   └── fetch_arena_scores.yaml
├── dat_eval/
│   ├── run_evals.yaml          # 53 models, temps, sampling, budget
│   └── score_evals.yaml        # embedding paths, bootstrap iters
└── new_tests/                  # DRAT pilot + ablation grids; benchmark configs
    ├── drat_pilot_*.yaml             # DRAT pilot phases (3anchor_v1, phase4{,_3anchor}, phase5_expansion)
    ├── drat_ablation_k{2,3,4}_{expert,conceptnet}{,_ext}.yaml
    │                                 # full (k, vocab) grid; _ext = phase-5 extension pool
    ├── eqbench_cw.yaml, hivemind.yaml, liveideabench.yaml, noveltybench.yaml
    └── rat_pilot.yaml, rat_expansion.yaml, rat_expansion_rerun.yaml

scripts/
├── comb_eval/                  # bash wrappers for comb_eval pipeline
├── dat_eval/                   # bash wrappers for dat_eval pipeline
├── new_tests/                  # bash wrappers for new_tests runners
│   ├── run_eqbench_cw.sh, run_hivemind.sh, run_liveideabench.sh, run_noveltybench.sh
└── safety/
    ├── status.sh               # see what's running, file activity, API conns
    ├── kill_all.sh             # SIGTERM + SIGKILL all eval processes
    └── cost_tracker.py         # estimated USD spend from local response files

data/
├── comb_eval/
├── dat_eval/run_v1/
│   ├── <model_key>/{dat,cdat,pace}_responses_t<temp>.json   # raw outputs
│   └── downstream/scores_v1/results/                          # scored
└── new_tests/                   # DRAT pilots + ablations + RAT runs
    ├── drat/{pilot_*, ablation_k*_{expert,conceptnet}{,_ext}}/raw_results.json
    └── rat/{pilot_v1, expansion_v1}/summary.json

docs/
├── AI_OPERATIONS_PROTOCOL.md    # safety rules I follow
├── HYPOTHESES.md                # working hypotheses across tracks
├── closing_tasks.md
├── repo_usage.md                # core repo conventions
├── research_context.md          # bird's-eye research summary
├── start.md
├── structure.md                 # this file
├── writing_advice.md            # writing-style guidance
├── literature.md
├── call_for_papers_icml.md
├── logs/<YYYY-MM-DD>/           # session logs
├── memos/                       # tacit-knowledge notes
├── reports/<date>_<name>/       # written-up findings + figures
└── tracks/
    └── new_tests/               # active track docs
        ├── progress.md          # working state of the track
        ├── drat_design.md       # DRAT design notes
        ├── proposals.md         # candidate-test proposals
        └── survey.md            # related-work survey

papers/                          # gitignored in outer repo (Overleaf-synced sub-repos)
├── iccc-2026/                   # main paper (validity / specificity framework + DRAT)
│   ├── main.tex, main_neurips.tex      # ICML 2026 / NeurIPS 2026 builds (sections/, tables/, sections_neurips/, tables_neurips/)
│   ├── main_jmlr.tex, main_preprint.tex  # JMLR / arXiv preprint builds (sections_jmlr/, tables_jmlr/)
│   ├── styles/                          # icml2026, neurips_2026, jmlr2e, preprint
│   ├── bib/main.bib
│   └── figures/
└── drat-icml-2026/              # standalone DRAT short paper (stub)
    └── main.tex

resources/                       # gitignored: GloVe, FastText, Numberbatch embeddings
```
