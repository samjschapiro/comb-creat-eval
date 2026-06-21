# Ablation framing & the human-comparison decision (plot_twist / T2C)

Decisions for how to present the human–LLM gap and the effort/temperature/prompting
ablations credibly. Distilled from a 2026-06-21 working session; grounded in
`docs/writing_advice.md` (Nanda).

## Main result: aggregate each system by the grand mean over temperatures

Each system's headline score is the mean over its stories across temperatures
{0.9, 1.0, 1.2}. Justifications (use in Methods, one sentence):

1. **Estimand = expected, temperature-robust quality over a realistic generation band.**
   The grid is a tight bracket around the default (~1.0), not extremes; a model that
   craters at 1.2 is legitimately less robust.
2. **Variance reduction.** 30 stories/model (3×10) is a far more stable per-model estimate
   than any single temperature's 10 — matters for ranking ~70 models.
3. **Symmetric aggregation.** Humans are pooled the same way (no temperature conditions);
   per-model best-temp would give LLMs a tuning step humans don't get.
4. **Unbiased**, unlike per-model best-temp (winner's curse: max of 3 noisy draws > mean
   even under the null; double-dipping to select-and-report on the same scores).

Caveat: grand mean is grid-dependent. Close it by also reporting a **fixed temp = 1.0**
robustness number (App.). Result is robust: humans rank #1 of 72 under averaging; even the
optimistic best-temp estimator leaves 66/69 below human.

## Ablations: lead with the null

Headline ablation result (the strong, novel contribution): **three orthogonal inference-time
levers — reasoning effort, sampling temperature, prompting strategy — do not improve the
twist-defining facets (surprise/coherence).** Sign tests across models ≈ chance. Diverse
lines of evidence → robust. Only diversity responds (to in-context regen, by construction);
"be creative" *lowers* realism.

## The existence-proof paragraph (suggestive, NOT systematic)

A handful of (model × best-config) cells reach/exceed the **best-8 human** composite
(z=1.357). Drivers, against the human facets (S=4.38, Coh=5.00, Real=5.00, Div=0.62):

| cell | z | S | Coh | Real | Div | reading |
|---|---|---|---|---|---|---|
| **sonnet-4.5 + in-context regen** | 1.364 | 4.38 | 5.00 | 5.00 | 0.62 | **ties human on all 4 facets** — the clean existence-proof |
| deepseek-v3.2-exp + in-context regen | 1.45 | 4.50 | 5.00 | 4.25 | 0.70 | wins via diversity (method-inflated), **loses realism** |
| glm-5.1 + medium effort | 1.51 | 5.00 | 5.00 | 4.00 | 0.69 | n=7 surprise spike, **loses realism** |

**Do NOT claim "scaling reasoning effort helps."** glm-5.1 is non-monotonic and
realism-degrading: low 1.32 → medium 1.51 (peak) → **high 1.07 (crash)**, realism 4.71 →
4.00 → 4.12. Over-thinking *hurts*. Treat glm-5.1 as a cautionary "effort has a model-specific
sweet spot" data point (appendix), not as support for test-time compute.

The defensible message: in-context regeneration (a crude form of test-time compute) brings
**one frontier model to parity with the best human batch across all facets** — an
existence-proof that **iterative / test-time-compute methods are a promising direction** for
transformational creativity. Tie to Wiggins (creativity as search that can *transform*, not
just explore, a conceptual space) and SBV (in-context regen ≈ deliberately re-searching for a
different axiom to flip). Future work: verifier-guided best-of-N, tree search over reveals,
explicit axiom-level planning.

### Mandatory caveats (or the claim isn't credible)
- Existence-proof, not a trend — keep the aggregate null prominent in the same breath.
- Best-config selection = winner's curse; ideally re-do with **held-out / split-half**
  config selection so the parity isn't selection-inflated.
- Realism is the human moat: 2/3 "above" cells lose on realism; only sonnet-4.5 holds it.

## Figure decisions
- Human–LLM gap: per-model barplot / over-time (human #1), + fixed-temp robustness.
- Intervention figure (`effort_temp_boxplots`): currently global-z with a best-8 human line.
  Cleanest alternative considered: within-model Δ (no human line) to avoid the
  per-condition-cell vs per-model-overall granularity confusion. Appendix `effort_temp_facets`
  (4×3 facet grid) is the mechanistic backup.
- Human reference line is red → switch to a colorblind-safe colour (it's load-bearing).
