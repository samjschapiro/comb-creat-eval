# Research Context

## Overall research

Investigation of automatic creativity metrics for large language models.
The recurring question across tracks: when a metric "predicts" creative
ability, is it picking up something creativity-specific, or just
tracking general model capability?

## Active tracks

### dat_eval (primary, ICML 2026 GenAI-Creativity workshop)

Evaluates three psycholinguistic semantic-distance creativity tests
— DAT, CDAT (and its novelty / appropriateness components), PACE —
on 52 LLMs across three embedding models (GloVe, FastText, SBERT)
and six external benchmarks spanning three target constructs:
creative writing (Arena CW, EQ-Bench CW, Mazur CW v2), divergent
thinking (Hivemind, NoveltyBench Utility), and scientific ideation
(LiveIdeaBench). Each test is measured on two criteria: *validity*
(raw Pearson r with the benchmark) and *specificity* (semi-partial r
after residualising the benchmark on a 2-proxy capability stack of
Arena Overall + MMLU-Pro). A covariance-PSD bound (proven in the
appendix) gives a per-benchmark theoretical ceiling on attainable
specificity.

**Status**: full eval run complete; analysis pipeline complete; ICML
draft on Overleaf (`papers/iccc-2026/`, name retained from the ICCC
short-paper precursor) past first-draft, mid-rewrite for ICML
submission. Outstanding writing holes: empty Results section, empty
Conclusion, abstract sentence 3 unfinished, contributions item 3
unfinished, background §2.4 placeholder. Bibliography contains
AI-generated entries flagged for human verification.

**Headline findings**:
- Semantic-distance test effectiveness varies sharply by construct.
  DAT is the most valid and specific predictor of creative writing;
  CDAT is the most valid and specific predictor of divergent thinking;
  PACE is *not* a valid-and-specific predictor on either construct
  once capability is partialled out.
- None of the three tests is a valid-and-specific predictor of
  scientific ideation (LiveIdeaBench, n=17): all observed
  specificities are exploratory.
- Across all panels, observed tests sit well below the theoretical
  ceiling — leaving meaningful room for new test designs.

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
