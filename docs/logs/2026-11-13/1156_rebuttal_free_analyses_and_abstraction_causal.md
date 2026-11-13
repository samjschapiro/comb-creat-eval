# 2026-11-13 — Kombine rebuttal: three robustness analyses, abstraction-bottleneck causal test, reply drafts

## Summary

Ran the three free robustness analyses reviewer cGKy asked for, and the paid causal test of the
abstraction bottleneck (eN17, cGKy). Drafted rebuttal replies for everything with results in hand. All of
it supports the paper.

## Tasks completed

- **Leave-one-model-out / fixed-pool originality** (cGKy Q2; `robustness_originality_pool.py`, report
  `docs/reports/2026-11-13_kg_creat_rebuttal_originality_pool/`):
  - Recomputing ρ reproduces all 21,294 stored values within 3e-7.
  - Leave one model out: Spearman 1.000 in all 35 cases.
  - Leave one provider out: at most one model moves one rank.
  - Fixed external pool (other half of the models, or humans only): Spearman ≥ 0.998, top 8 unchanged.
  - Originality dimensions alone: Spearman ≥ 0.93.
- **Composite robustness** (`rebuttal_composite_robustness.py`):
  - No utility gate: 0.965, same #1. Geometric mean: 0.997.
  - 10,000 Dirichlet weightings: median 0.971, 5th percentile 0.897.
  - Caveat: the gate can't be defended as noise control, since failed and passed answers are equally original.
- **Judge self-preference** (`rebuttal_judge_self_preference.py`; report
  `docs/reports/2026-11-13_kg_creat_rebuttal_composite_and_judges/`):
  - No judge is an evaluated model.
  - Own-family difference in differences: +2.7 / −0.2 / +3.3 pp, all CIs spanning 0.
  - Leave one judge out: 0.966–0.993.
  - Factuality: Haiku agrees with gpt-oss on 88–92% of triples in every family, with no Anthropic leniency.
- **Abstraction causal test** (paid). Report: `docs/reports/2026-11-13_kg_creat_abstraction_causal/report.md`.
  - **Design:** 6 models × 30 blend items × control / content-free hint / supplied panel-validated g.
  - **Previously failed items:** success 17.2% (control) → 22.4% (hint, n.s.) → **65.5% (oracle, +59/−3, p = 1.7e-14)**.
    All six models improve.
  - **Elaboration:** given an accepted g, V 91–99% and Q≥2 about 72% whatever the source of g.
  - **Originality:** unchanged.
  - **Judge noise:** the panel re-accepts a supplied g only 72% of the time, so the oracle effect is understated.
  - **New code:**
    - `prompts.py` gains optional `blend_condition` / `given_generic_space` (default prompts byte-identical, checked);
    - `build_abstraction_causal.py`, `analyze_abstraction_causal.py`;
    - configs in `configs/kg_creat/abstraction_causal/`.
  - **Model substitutions:**
    - qwen3-max was deprecated on OpenRouter;
    - phi-4 parsed only 5–10 of 30 answers;
    - gemini-2.5-flash was refused by the provider (403);
    - replaced with deepseek-chat-v3-0324 and qwen-2.5-72b.
  - **Spend:** about $27 of the OpenRouter key; $41.41 remains.
- **Rebuttal draft** `docs/tracks/kg_creat/rebuttal_iclr_2027.md`. Filled: S1 (causal test), S3a/b/c,
  cGKy Q2 / composite / self-preference / surprise / effort, eN17 Q1.

## Decisions

- **Control re-sample as the comparison.** Under the oracle, compare against a fresh re-sample, not the paper
  run, to rule out regression to the mean.
- **The oracle g is the same for every model,** taken from a non-test model.
- **API failures are excluded** (1 cell); unparsed answers count as failures.

## Next

- Rater study results (S2): analyse when the files arrive.
- A7nz review text.
- Optional: n-gram novelty (eN17 Q1).
- Paper edits per the "Paper changes promised" list in the rebuttal doc.
