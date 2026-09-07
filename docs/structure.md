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
├── kg_creat/                   # active track (Kombine: association / analogy / blending over a curated entity pool); 35-model run 2026-09-07
│   ├── graph.py                # LEGACY (seed-BFS era): KG-agnostic KnowledgeGraph (typed path enumeration)
│   ├── wikidata.py             # Wikidata REST client: entity grounding for the curated pool (+ legacy BFS builder)
│   ├── sample.py               # LEGACY: matched-bundle (Regime A) + random analogy/blend (Regime B) samplers
│   ├── prompts.py              # Kombine prompt renderer (association / analogy / blending; `uv` shared-slot tag)
│   ├── judge.py                # factuality gate (claude-haiku-4.5 since 09-07) + panel judges: blend generic_ok/coherent/scope + shared_properties, analogy invention
│   ├── providers.py            # non-OpenRouter elicitation routes (LiteLLM gateway + Anthropic SDK); bypasses LLM_BASE_URL so OpenRouter budget guards stay intact
│   ├── model_names.py          # single source for LOGO_SLUG / BRAND / DISPLAY (35 entries) / _provider; no matplotlib dependency
│   ├── relation_classes.py     # LEGACY: embedding-derived relation CLASSES + per-bundle baseline-derived targets
│   ├── regime_b.py             # shared structure-mapping predicates (analogy + single-anchor blending)
│   ├── diversity.py            # set-level diversity D over M-resamples (per temperature; all + valid)
│   ├── embed.py                # local MLX MiniLM embeddings (novelty R)
│   ├── scoring.py, parse.py ({triple, from} blend schema with u/v/uv/emergent tags), aggregate.py
│   ├── cost_ledger.py          # persistent per-phase/model USD ledger (data/kg_creat/cost_ledger.jsonl)
│   ├── anagram.py              # anagram task (exploratory-creativity side probe, judge-free scoring)
│   ├── vendor/create/          # vendored CREATE scorer (author-cleared)
│   └── scripts/
│       ├── sample_pool.py, resolve_pool.py, sample_flat.py   # curated entity pool + per-task item sampling (no graph)
│       ├── run_elicit.py       # elicitation; resume-safe, saves reasoning traces, actual-cost stop vs budget_usd
│       ├── score.py            # scoring pass: factuality gate + 3-judge panel (blend/invention), persists per-judge explanations
│       ├── repair_elicit.py, rejudge_factuality.py           # backfill failed draws; re-judge paths the batched judge truncated
│       ├── rescore_originality.py, rescore_split_originality.py, rescore_blends.py  # pool-relative rescore; base/emergent split; blend-v3 rescore
│       ├── build_blind_review.py, blind_review_server.py     # blind human re-rating of panel dimensions (+ hidden key)
│       ├── build_blend_review.py, blend_review_server.py     # blend-specific review UI (uv shared-slot format)
│       ├── leaderboard_unanimous.py, leaderboard_single_judge.py  # judge-robustness checks on the leaderboard
│       ├── shared_property_judge.py, coherence_taxonomy_judge.py  # blend shared-slot + failure-mode probes
│       ├── analyze_invention_homogeneity.py, analyze_inventive_multiples.py, embed_inventions.py  # cross-model convergence analyses
│       ├── analyze_failure_modes.py, catalogue_generic_space_failures.py, analyze_blend_integration.py  # what goes wrong, per channel
│       ├── analyze_facet_correlations.py, analyze_item_effects.py, embedding_robustness.py  # what the dimensions measure; encoder robustness
│       ├── analyze_task_dissociation.py  # analogy vs blending on the SAME model x pair cells (2x2 + disattenuation)
│       ├── analyze_blend_difficulty.py   # what makes an anchor pair hard to blend (exploratory, post-hoc coding)
│       ├── compute_composite.py, make_composite_table.py, make_appendix_tables.py, make_pool_appendix.py, datasheet.py
│       ├── plot_hivemind.py, plot_invention_landscape.py, plot_creativity_gallery.py, plot_profiles.py, plot_radar.py
│       ├── plot_multiples_matrix.py, plot_abstraction_failure.py, plot_bars.py, make_multiples_showcase.py
│       ├── make_paper_multiples_figure.py  # stacks the matrix over the landscape into the paper's single figure
│       ├── sample_anagram.py, run_anagram.py, score_anagram.py    # anagram side probe
│       └── build_gc.py, sample_bundles.py, plot_regime_a.py, compute_diversity.py, make_pass2.py  # legacy seed-BFS / Regime-A pipeline
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
        ├── build_website_data.py      # JSON payload for the project page (website/twistbench/)
        ├── build_human_eval_stimuli.py  # length-matched human-vs-top-LLM pairs -> jsPsych stimuli
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
├── kg_creat/                   # Kombine run + scoring configs (pilot / Regime-A configs kept for history)
│   ├── kombine_test30_run.yaml       # the canonical 30-item/task run (original 6 models)
│   ├── kombine_test30_frontier.yaml  # +13 frontier flagships, resume-safe, actual-cost stop (budget_usd 75)
│   ├── kombine_test30_anthropic3.yaml  # +3 legacy-priced Anthropic models -> 21-model pool (now 30 with spread9)
│   ├── kombine_test30_blendv3{,_gemini}.yaml  # blending-only re-elicitation with the `uv` shared-slot tag
│   ├── kombine_test30_panel_score.yaml  # 3-judge non-subject panel (subjective) + cheap single factuality judge
│   ├── kombine_{pilot,v1,v2,blend_v2}_*.yaml  # earlier pilot / polysemy / fusion-blend runs
│   └── run_anagram.yaml, build_gc.yaml, sample_bundles.yaml, score_regimeA.yaml  # side probe + legacy pipeline
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
    ├── human_eval_stimuli.yaml      # human-vs-top-LLM preference study: pair selection + length matching
    ├── comprehension_items.json     # authored reading-check options (4 per pool story)
    ├── judge_reliability.yaml, grm_irt.yaml, bayes_grm_jrt.yaml
    └── pd_manifest.json             # human gold stories + twist_type (STRONG/BORDERLINE/NONE)

scripts/
├── comb_eval/                  # bash wrappers for comb_eval pipeline
├── dat_eval/                   # bash wrappers for dat_eval pipeline
├── new_tests/                  # bash wrappers for new_tests runners
│   ├── run_eqbench_cw.sh, run_hivemind.sh, run_liveideabench.sh, run_noveltybench.sh
├── kg_creat/                   # deploy_study.sh (jsPsych human-generation study)
└── safety/
    ├── status.sh               # see what's running, file activity, API conns
    ├── kill_all.sh             # SIGTERM + SIGKILL all eval processes
    └── cost_tracker.py         # estimated USD spend from local response files

data/
├── comb_eval/
├── dat_eval/run_v1/
│   ├── <model_key>/{dat,cdat,pace}_responses_t<temp>.json   # raw outputs
│   └── downstream/scores_v1/results/                          # scored
├── kg_creat/                    # Kombine outputs (gitignored)
│   ├── entities_curated.json, pool_{candidates,reference,wikidata}.json  # the 283-anchor curated pool
│   ├── cost_ledger.jsonl        # persistent per-phase/model actual USD spend
│   ├── effort_study/            # thinking-effort study (2 models x low/medium/high)
│   │   ├── {low,medium,high}/, all/  # per-effort responses; `all/` is the pooled symlink tree used as scoring upstream
│   │   ├── scores/              # scored in ONE pooled pass so pool-relative originality is comparable across effort levels
│   │   └── figures/             # fig_effort_{composite,dimensions,delta}.{pdf,png} + effort_composite.json
│   └── kombine_test30/          # the 35-model run
│       ├── prompts/, responses/ # items; per-model responses.json (+ reasoning traces)
│       ├── blends_v3/           # `uv`-tag blend re-elicitation, merged back into responses/
│       ├── scores/<model>/path_scores.json  # per-artifact scores + per-judge explanations; composite.json
│       ├── human_review{,_blendv3}/  # blind review items + hidden key + ratings.jsonl
│       └── analysis/            # invention_homogeneity.json, invention_vectors.npz
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
    ├── kg_creat/                # active track docs (Kombine)
    │   ├── progress.md          # goal, status, phased roadmap
    │   ├── design.md            # task/scoring spec + reuse map + risks
    │   ├── blending_fusion.md   # two-concept fusion reframe of blending
    │   ├── methods.md, assessment.md, constraints.md  # scoring methods; legacy constraint taxonomy
    │   ├── literature_map_forms.md, primary_sources_motivations.md, related_work.md
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
├── kg_creat-iclr/               # Kombine paper (kg_creat track; ICLR 2027)
│   ├── main.tex, content/       # benchmark + results; appendices: entity pool, judge reliability + human corroboration
│   ├── media/                   # leaderboard, per-task tables, invention landscape, profiles
│   └── prompts/, setup/, build/
└── pt2cb-iclr-2027/             # plot_twist paper (TwistBench; folder name predates ARR/EACL switch)
    ├── main.tex, sections/      # §3 benchmark, §4 results (scorecard), …
    └── figures/                 # tc_scorecard.png + fig_scorecard.tex

website/                         # static project pages (no build step; plain HTML/CSS/JS)
└── twistbench/                  # TwistBench page: leaderboard + full-text story explorer
    ├── index.html, static/      # Nerfies-style layout, CSS/JS, downscaled paper figures
    ├── data/                    # generated by src/plot_twist/scripts/build_website_data.py
    └── README.md                # rebuild / serve / deploy notes

resources/                       # gitignored: GloVe, FastText, Numberbatch embeddings
└── fonts/inter/                 # Inter TTFs registered by make_tc_barplot.py
```
