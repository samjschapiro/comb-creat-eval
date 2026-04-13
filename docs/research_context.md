# Research Context

## Overall research

Investigation of automatic creativity metrics for large language models.
The recurring question across tracks: when a metric "predicts" creative
ability, is it picking up something creativity-specific, or just
tracking general model capability?

## Active tracks

### dat_eval (primary, ICCC 2026 short paper)

Compares three psycholinguistic creativity metrics — DAT, CDAT, PACE —
on 53 LLMs and correlates each against three external benchmarks
(Chatbot Arena Creative Writing, EQ-Bench Creative Writing, Hivemind
intra-model similarity), with partial correlations controlling for
Arena Overall to isolate creativity-specific signal from general
capability.

**Status**: full eval run complete; correlations and figures done;
short paper draft pushed to Overleaf (`papers/iccc-2026/`); awaiting
editorial pass and bibliography verification.

**Headline findings**: DAT does not predict creative writing once
capability is partialled out; CDAT Appropriateness's apparent signal
is a general-capability artifact; PACE is the only metric whose
correlation with creative writing survives partialling, and (after
partialling) it also predicts output diversity in the expected
direction.

### comb_eval (background / exploratory)

Earlier track exploring combinatorial-creativity-style evaluation.
Currently dormant; reused only for the Arena-score fetcher and
`benchmarks.json` schema that dat_eval consumes.

## Cross-track conventions

- All API calls go through OpenRouter via the `openai` Python SDK.
- Per-model scores live in `data/<track>/run_v1/<model_key>/`.
- `configs/comb_eval/benchmarks.json` is the shared per-model
  benchmark store, augmented in place by `add_eqbench_scores.py` and
  `add_hivemind_scores.py`.
- Long-running scripts respect `budget_usd` in their config and abort
  before exceeding the cap.
- Safety scripts in `scripts/safety/` (`status.sh`, `kill_all.sh`,
  `cost_tracker.py`) and `docs/AI_OPERATIONS_PROTOCOL.md` are
  consulted before launching any expensive operation.
