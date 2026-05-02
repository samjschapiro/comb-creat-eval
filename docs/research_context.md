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
