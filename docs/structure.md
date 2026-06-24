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
│       ├── run_drat_smoke.py        # DRAT smoke runner used for pilots/ablations
│       ├── run_eqbench_cw.py        # EQ-Bench CW wrapper
│       ├── run_hivemind.py          # Hivemind wrapper runner
│       ├── run_liveideabench.py     # LiveIdeaBench wrapper runner
│       ├── run_noveltybench.py      # NoveltyBench wrapper runner
│       └── run_rat.py               # RAT wrapper runner
├── kg_creat/                   # active track (test-time Comb-Creat on a real KG)
│   │                           # scaffolded 2026-06-01; eval engine TBD (Phase 1)
│   └── scripts/                # graph.py (Wikidata loader) + scoring port to come
└── plot_twist/                 # active track (TwistBench: transformational creativity via plot twists)
    ├── llm.py                  # OpenRouter wrapper (+ optional `reasoning` param)
    ├── generate.py             # durable per-story twist generation (multi-temp, resumable)
    ├── annotate.py             # setup/reveal/why annotation
    ├── rubric_judge.py         # 3-judge rubric (surprise, coherence; judges ≠ generators)
    ├── dsi.py                  # DSI baseline metric
    ├── sets.py                 # STRONG/BORDERLINE/NONE human gold-set selection
    └── scripts/
        ├── fetch_pd_stories.py        # fetch the human gold plot-twist stories
        ├── run_generate.py            # open-ended "write a story with a plot twist" (3 temps × 10)
        ├── run_annotate.py            # setup/reveal/why annotation runner
        ├── run_rubric_{gold,smoke,stimuli}.py  # rubric scoring (gold / smoke / stimuli)
        ├── run_dsi.py                 # DSI baseline runner
        ├── run_realism.py             # realism (grounded vs fantastical) — 4th equal-weighted facet
        ├── classify_twists.py, analyze_collapse.py  # twist taxonomy + mode-collapse analysis
        ├── correlate_dsi.py           # DSI vs external creativity benchmarks
        ├── judge_reliability.py, grm_irt.py, bayes_grm_jrt.py  # inter-judge reliability
        ├── cost_log.py                # OpenRouter spend → docs/tracks/plot_twist/cost_log.md
        └── make_tc_barplot.py         # TC scorecard (Overall + 2×2 facet grid) + breakdown figures

configs/
├── comb_eval/
│   ├── benchmarks.json         # per-model: arena_overall, mmlu_pro, arena_cw,
│   │                           # eq_bench_cw, mazur_cw_v2, hivemind_diversity,
│   │                           # noveltybench_*, liveideabench (+ per-facet rows)
│   └── fetch_arena_scores.yaml
├── dat_eval/
│   ├── run_evals.yaml          # 53 models, temps, sampling, budget
│   └── score_evals.yaml        # embedding paths, bootstrap iters
├── new_tests/                  # DRAT pilot + ablation grids; benchmark configs
│   ├── drat_pilot_*.yaml             # DRAT pilot phases (3anchor_v1, phase4{,_3anchor}, phase5_expansion)
│   ├── drat_ablation_k{2,3,4}_{expert,conceptnet}{,_ext}.yaml
│   │                                 # full (k, vocab) grid; _ext = phase-5 extension pool
│   ├── eqbench_cw.yaml, hivemind.yaml, liveideabench.yaml, noveltybench.yaml
│   └── rat_pilot.yaml, rat_expansion.yaml, rat_expansion_rerun.yaml
└── plot_twist/                 # TwistBench configs
    ├── generate_llm_twists.yaml     # generator + excluded-judge model lists
    ├── rubric{,_gold,_llm_twists}.yaml  # rubric judge configs
    ├── realism.yaml                 # realism judge (single cheap judge, durable)
    ├── tc.yaml                      # scorecard / TC composite (4-facet, human STRONG-only)
    ├── annotate.yaml, dsi_quality.yaml
    ├── judge_reliability.yaml, grm_irt.yaml, bayes_grm_jrt.yaml
    └── pd_manifest.json             # human gold stories + twist_type (STRONG/BORDERLINE/NONE)

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
├── new_tests/                   # DRAT pilots + ablations + RAT runs
│   ├── drat/{pilot_*, ablation_k*_{expert,conceptnet}{,_ext}}/raw_results.json
│   └── rat/{pilot_v1, expansion_v1}/summary.json
└── plot_twist/                   # TwistBench outputs
    ├── human_twists/, llm_twists/   # gold + generated stories (per-model subfolders)
    ├── annotations/, rubric_gold/   # setup/reveal annotations; rubric scores
    ├── realism/realism_scores.json  # per-story realism (id → 1–5)
    ├── dsi/, dsi_quality/, twist_class/, collapse.json  # baselines + analyses
    ├── judge_reliability/, grm_irt/, bayes_grm_jrt/      # reliability outputs
    └── tc/                           # TC composite + scorecard figures (tc_scorecard.png)

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
    ├── new_tests/               # active track docs
    │   ├── progress.md          # working state of the track
    │   ├── drat_design.md       # DRAT design notes
    │   ├── proposals.md         # candidate-test proposals
    │   └── survey.md            # related-work survey
    ├── kg_creat/                # active track docs (real-KG Comb-Creat)
    │   ├── progress.md          # goal, status, phased roadmap
    │   ├── design.md            # task/scoring spec + reuse map + risks
    │   └── novelty_vs_create.md # methodological novelty table vs CREATE
    └── plot_twist/              # active track docs (TwistBench benchmark)
        ├── progress.md          # goal, status, phased roadmap
        ├── design.md            # SBV→story-DAG mapping, CSAM spec, baselines
        ├── paper_outline.md, experiments.md, rubric_design.md
        ├── pd_gold_set.md       # human gold plot-twist set notes
        └── cost_log.md          # cumulative OpenRouter spend

papers/                          # gitignored in outer repo (Overleaf-synced sub-repos)
├── iccc-2026/                   # main paper (validity / specificity framework + DRAT)
│   ├── main.tex, main_neurips.tex      # ICML 2026 / NeurIPS 2026 builds (sections/, tables/, sections_neurips/, tables_neurips/)
│   ├── main_jmlr.tex, main_preprint.tex  # JMLR / arXiv preprint builds (sections_jmlr/, tables_jmlr/)
│   ├── styles/                          # icml2026, neurips_2026, jmlr2e, preprint
│   ├── bib/main.bib
│   └── figures/
├── drat-icml-2026/              # standalone DRAT short paper (stub)
│   └── main.tex
└── pt2cb-iclr-2027/             # plot_twist paper (TwistBench; folder name predates ARR/EACL switch)
    ├── main.tex, sections/      # §3 benchmark, §4 results (scorecard), …
    └── figures/                 # tc_scorecard.png + fig_scorecard.tex

resources/                       # gitignored: GloVe, FastText, Numberbatch embeddings
└── fonts/inter/                 # Inter TTFs registered by make_tc_barplot.py
```
