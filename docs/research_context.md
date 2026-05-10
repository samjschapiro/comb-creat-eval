# Research Context

## Overall research

Investigation of automatic creativity metrics for large language models.
The recurring question across tracks: when a metric "predicts" creative
ability, is it picking up something creativity-specific, or just
tracking general model capability?

## Active tracks

### dat_eval (primary, ICML 2026 GenAI-Creativity workshop + NeurIPS 2026)

Evaluates three psycholinguistic semantic-distance creativity tests
— DAT, CDAT (and its novelty / appropriateness components), PACE —
on 54 LLMs across three embedding models (GloVe, FastText, SBERT)
and six external benchmarks spanning three target constructs:
creative writing (Arena CW, EQ-Bench CW, Mazur CW), divergent
thinking (Hivemind, NoveltyBench Utility), and scientific ideation
(LiveIdeaBench). Each test is measured on two criteria: *validity*
(raw Pearson r with the benchmark) and *specificity* (semi-partial
r(X, Y − Ŷ_g), Y residualised on a 2-proxy capability stack of
Arena Overall + MMLU-Pro). A covariance-PSD bound (proven in the
appendix) gives a per-benchmark theoretical ceiling on attainable
specificity.

**Status**: full eval run complete; analysis pipeline complete;
draft past second-pass rewrite, near submission. Two parallel
submission variants live in `papers/iccc-2026/`: `main.tex` (ICML
2026 GenAI-Creativity workshop) and `main_neurips.tex` (NeurIPS
2026 main track). Section files (`sections/`) and tables
(`tables/`) feed the ICML build; NeurIPS-only variants live in
`sections_neurips/` and `tables_neurips/`.

**Headline findings**:
- Specificity, not validity, is what separates a creativity test
  from a capability proxy.
- Test effectiveness varies sharply by construct: DAT is the best
  predictor of creative writing; CDAT is the best predictor of
  divergent thinking; PACE has high raw validity on creative
  writing but its specificity collapses under capability control,
  so it is mostly a capability proxy.
- None of the three tests is a valid-and-specific predictor of
  scientific ideation (LiveIdeaBench, n=17): all observed
  specificities are exploratory.
- Across all panels, observed tests sit well below the theoretical
  ceiling — leaving meaningful room for new test designs.

**Sources of truth** (key values updated 2026-05-02):
- MMLU-Pro: TIGER-Lab leaderboard CSV
  (`TIGER-Lab/mmlu_pro_leaderboard_submission`), not AA.
- Mazur CW: `lechmazur/writing` GitHub at commit `80b7f17`.
- EQ-Bench CW: `eqbench.com/creative_writing.js`
  (`leaderboardDataCreativeWritingV3`).
- Specificity computation: true semi-partial `r(X, Y − Ŷ_g)`
  (was full partial pre-2026-05-02; values shifted accordingly).

### new_tests (active, JMLR/ICML preprint in `papers/iccc-2026`)

Designs and evaluates *new* creativity tests on top of the dat_eval
infrastructure, motivated by the dat_eval finding that no existing
semantic-distance test is a valid-and-specific predictor of
LiveIdeaBench (scientific ideation). Headline test:
**DRAT (Divergent Remote Association Test)** — a hybrid of RAT and
DAT. The model produces $\geq 10$ words bridging a $k$-anchor
quadruple ($k = 4$ in the headline configuration); each word is
gated by an embedding-based "near any anchor" utility ($\max$
aggregator over per-anchor cosine), and survivors are scored by
mean pairwise embedding distance — DAT scoring on a RAT-style
constrained vocabulary.

Other benchmark wrappers added under `src/new_tests/` to support
the broader pool: `eqbench_cw`, `hivemind`, `liveideabench`,
`noveltybench`, `rat` (zero-shot strict accuracy on classic 30-item
RAT). Configs in `configs/new_tests/`, runners in
`scripts/new_tests/` and `src/new_tests/scripts/`. Detailed
ongoing notes in `docs/tracks/new_tests/progress.md`.

**Status (2026-05-09)**: DRAT pool n = 20 on LIB after the LIB
facet refresh. Headline $k=4$ scientific-terms cell is significant
on both axes ($v = +0.57^{**}$, $s = +0.50^{*}$). Anchor banks +
RAT prompt now documented in the appendix. Two ablations live —
anchor count $k \in \{2,3,4\}$ and vocabulary corpus
(scientific-terms vs. ConceptNet relation-distant) — both reported
in `04_drat.tex`. Utility-gate aggregator ablation
($\max$ / $\min$ / $\mathrm{avg}$) in `06_appendix.tex`.

### comb_eval (background / exploratory)

Earlier track exploring combinatorial-creativity-style evaluation.
Currently dormant; reused only for the Arena-score fetcher and
`benchmarks.json` schema that dat_eval consumes.

## Cross-track conventions

- All API calls go through OpenRouter via the `openai` Python SDK.
- Per-model scores live in `data/<track>/run_v1/<model_key>/`.
- `configs/comb_eval/benchmarks.json` is the shared per-model
  benchmark store, augmented in place by `add_eqbench_scores.py`,
  `add_hivemind_scores.py`, `add_mazur_scores.py`,
  `add_mmlu_pro_scores.py`, `add_noveltybench_scores.py`, and
  `add_arc_agi_scores.py`.
- Long-running scripts respect `budget_usd` in their config and abort
  before exceeding the cap.
- Safety scripts in `scripts/safety/` (`status.sh`, `kill_all.sh`,
  `cost_tracker.py`) and `docs/AI_OPERATIONS_PROTOCOL.md` are
  consulted before launching any expensive operation.
