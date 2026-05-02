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
└── dat_eval/                   # primary track (ICML + NeurIPS 2026 paper)
    ├── llm.py                  # OpenRouter sync + async client
    ├── dat.py                  # DAT scoring (GloVe 840B)
    ├── cdat.py                 # CDAT scoring (SBERT all-mpnet-base-v2)
    ├── pace.py                 # PACE scoring (FastText crawl-300d-2M)
    └── scripts/
        ├── run_evals.py        # async DAT/CDAT/PACE runner with budget cap
        ├── score_evals.py      # Pearson + semi-partial r(X, Y - Y_hat_g)
        ├── multi_embed_appendix.py  # per-embedding pipeline (GloVe + FastText + SBERT)
        └── make_figures.py     # camera-ready figures (Times + RdBu_r / Okabe-Ito)

configs/
├── comb_eval/
│   ├── benchmarks.json         # per-model: arena_overall, arena_cw,
│   │                           # eq_bench_cw, hivemind_intra_sim
│   └── fetch_arena_scores.yaml
└── dat_eval/
    ├── run_evals.yaml          # 53 models, temps, sampling, budget
    └── score_evals.yaml        # embedding paths, bootstrap iters

scripts/
├── comb_eval/                  # bash wrappers for comb_eval pipeline
├── dat_eval/                   # bash wrappers for dat_eval pipeline
└── safety/
    ├── status.sh               # see what's running, file activity, API conns
    ├── kill_all.sh             # SIGTERM + SIGKILL all eval processes
    └── cost_tracker.py         # estimated USD spend from local response files

data/
├── comb_eval/
└── dat_eval/run_v1/
    ├── <model_key>/{dat,cdat,pace}_responses_t<temp>.json   # raw outputs
    └── downstream/scores_v1/results/                          # scored

docs/
├── AI_OPERATIONS_PROTOCOL.md    # safety rules I follow
├── closing_tasks.md
├── repo_usage.md                # core repo conventions
├── research_context.md          # bird's-eye research summary
├── start.md
├── structure.md                 # this file
├── writing_advice.md            # writing-style guidance
├── literature.md
├── logs/<YYYY-MM-DD>/           # session logs
├── memos/                       # tacit-knowledge notes (e.g. ai_operations_log.md)
├── reports/<date>_<name>/       # written-up findings + figures
└── tracks/<track>/progress.md   # per-track status

papers/
└── iccc-2026/                   # Overleaf-synced (separate git repo)
    ├── main.tex                       # ICML 2026 build
    ├── main_neurips.tex               # NeurIPS 2026 build
    ├── icml2026.{sty,bst}
    ├── neurips_2026.sty
    ├── main.bib
    ├── sections/, tables/             # ICML content
    ├── sections_neurips/, tables_neurips/  # NeurIPS-only content
    └── figures/

resources/                       # gitignored: GloVe, FastText embeddings
```
