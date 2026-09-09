# 2026-09-09 — Which cognitive facets Kombine loads on; τ-inventive multiples rebuilt

Two-day session (2026-09-08 → 09). Rebuilt the inventive-multiples definition around τ, then built the
cognitive-facet analysis: which of divergent (DAT), constrained-divergent (CDAT), convergent (RAT) and
relevance-gated-divergent (DRAT) thinking each Kombine task and facet loads on. Backfilled the three
reference tests from 18–21 models to 31–35, most of it on the user's own API keys.

## Headline: the Kombine tasks load heavily on divergent thinking

Per-model correlation, n = 31–35, facets computed **ungated**:

| | DAT | CDAT | RAT | DRAT |
|---|---|---|---|---|
| association | +0.49 | +0.08 | +0.50 | +0.06 |
| analogy | +0.60 | +0.23 | +0.41 | −0.20 |
| blending | **+0.63** | +0.44 | +0.17 | −0.33 |
| overall | **+0.66** | +0.31 | +0.39 | −0.21 |

**Twelve facet cells survive Benjamini-Hochberg** at q=0.05 over 60 tests (threshold p≤0.0129), where
at n=21 none did. DAT is the strongest and most consistent column: it clears correction on
`assoc.utility` (+0.46), `assoc.originality` (+0.44), `analo.utility` (+0.57), `analo.em_integration`
(+0.53), `analo.em_originality` (+0.53), `blend.utility` (+0.61), `blend.surprise` (+0.51),
`blend.em_integration` (+0.58), `blend.em_utility` (+0.48). Blending loads on DAT most heavily of the
three tasks, association least — the gradient the task design predicts.

**Utility and originality dissociate on the other two tests.** Utility facets load on RAT (+0.38 to
+0.48) and not CDAT; originality facets load on CDAT (+0.39 to +0.59) and go *negative* on RAT
(`blend.originality` −0.34). Convergent thinking predicts whether a model can satisfy the task gate;
constrained divergent thinking predicts whether what it produces is novel, and the two trade off.

## Three measurement bugs, each of which had inverted a result

**1. Utility gating destroyed the facet structure.** `compute_composite` zeroes surprise/originality/
emergent on any artifact failing utility — right for a leaderboard, fatal for facet correlations: it
made within-task facets near-identical (mean pairwise r **+0.95 to +0.97**), so every facet was just
re-expressing the utility rate. Recomputing ungated dropped that to **−0.05 / +0.35 / +0.25** and the
dissociation above appeared. `load_facets_ungated` in the new script computes facets from
`path_scores.json` directly and never gates.

**2. Temperature was not held fixed.** DAT/CDAT run at 1.0/1.5/2.0 and were pooled; RAT is 0.0;
DRAT 1.0; Kombine 0.9. Worse, the stored CDAT averaged only the temperatures each model *passed* the
appropriateness gate at — 49 models passed all three, 6 passed only 1.0, 11 none — so models were not
compared at a common operating point and weaker models were scored at the easier low temperatures.
Fixing both at **temperature 1.0**, with the gate applied at that same temperature, erased a "clean
dissociation" on CDAT (+0.50 to +0.70 originality loadings) that was entirely this artifact, raised
CDAT coverage 18 → 28, and cut DAT–RAT collinearity from **+0.73 to +0.40**.

**3. A 256-token cap was truncating reasoning models everywhere.** Extended-thinking models spend the
cap thinking and emit nothing. Hit DAT, CDAT, DRAT and RAT in turn. `claude-opus-5` CDAT went **2% →
100%** valid at a 4000 cap; `claude-sonnet-4-6` CDAT 0% → 100%; `claude-opus-4-6` DAT 2% → 100%.
Both models the CDAT gate had "failed" were truncation artifacts, not genuine gate failures.

## DRAT does not measure what its name says

`score_drat` computes each word's utility as **`max`** cosine over the anchors, not min — a word
survives by being near **any one** anchor. The score is then the mean pairwise distance among
survivors. So DRAT is a relevance-gated DAT, not a joint divergent-convergent test.

Its variance is dominated by the item, not the model. τ is the 90th percentile of a random-noun null
under the same max rule, so it rises with how semantically *broad* the anchor set is: τ spans
**0.246–0.486** across 194 anchor groups and correlates **−0.73** with survivor count (p=9e-34).
Broad groups (`decay, fern, vein, wind`, τ=0.486) let random nouns score high and almost nothing
clears; narrow ones (`genome, algorithm`, τ=0.258) let almost everything clear. With `n_min: 5` the
per-trial score is near-binary — 0 or ~76 — and **76% of all cells score 0** pool-wide, with a mean of
3.03 survivors of 10.

This explains every oddity in the DRAT column: nano models topping the pool (gpt-5.4-nano 5.70
survivors vs frontier models ~2), the −0.64 correlation with CDAT, and the negative loadings on every
Kombine facet. Recommendation: drop the DRAT column, or rescore with a τ fixed across items.

## Backfill: 3 routes, minimal OpenRouter

Coverage went DAT 21→33, CDAT 18→31, RAT 29→35, DRAT 27→33 (35-model Kombine pool).

- **Anthropic OpenAI-compat endpoint** (`https://api.anthropic.com/v1/`) serves all 10 Claude models —
  no code changes needed beyond parameter stripping. Free to the project.
- **The user's LiteLLM gateway** serves `gpt-5.6-sol` and `gpt-6-astra-flex`.
- **OpenRouter** only for the 13 models neither key reaches. **Total spend $27.87.**

`strip_unsupported()` in `src/dat_eval/llm.py` makes this work, keyed on the *endpoint* so the same
config routes anywhere:
- Anthropic rejects `temperature` outright on newer models; drop it, plus `top_p`/`seed`.
- `extra_body` (top_k, OpenRouter's `reasoning` block) is an OpenRouter extension; drop it elsewhere.
- The gateway's deployment for `gpt-6-astra-flex` leaks its own config keys (`mode`, `max_input_tokens`,
  `max_output_tokens`) into the upstream OpenAI request, which 400s. LiteLLM merges `extra_body` *over*
  the deployment config, so nulling those three suppresses the leak client-side. **Worth fixing at the
  source** by removing them from that deployment's `litellm_params`.
- Newer OpenAI models need `max_completion_tokens`, not `max_tokens`, and reject temperature ≠ 1.0.

## Provider data quality: several OpenRouter models are unusable

Validity (fraction of draws parsing to ≥7 words) after merge:

```
gemini-3.7-flash 100%  llama-3.3-70b 97%  deepseek-v3 96%  gpt-5 93%  kimi-k2 89%  phi-4 88%
grok-4.5 48%   qwen3-max 44%   glm-4.6 10%   glm-4.5-air (dropped, ~50% null content)
```

`glm-4.5-air` and `glm-4.6` return **null content on half or more of calls** ("provider returned null
content"); with `max_retries: 4` each failure cost 5× the work, which is why the run appeared to hang.
`qwen-2.5-72b` kept its **older** `run_v1` data because this collection was worse (69% vs 41%).
`analyze_cognitive_facets` now filters at **MIN_VALID = 0.70** of trials parsed at temp 1.0, so thin
models are excluded rather than silently averaged in on a minority of draws.

`grok-4.5` cost ~$18 and 5 hours for 4 of 6 files, then failed the validity filter anyway; `grok-4.6`
was dropped for the same reason. Both are absent from DAT/CDAT.

## RAT: gpt-5.6-sol's 43.3% is an artifact, not a result

RAT's `max_tokens: 32` is the tightest cap of the four tests, and reasoning models spend it thinking.
`gpt-5.6-sol` answered **13 of 30 and was correct on all 13** — the other 17 came back empty.
`gpt-6-astra-flex` was 29/30 with 1 empty. Neither ever gave a *wrong* answer, so both are effectively
at 100% and the recorded 43.3% / 96.7% understate them. **Not yet fixed** — needs a re-run of those
two at a larger cap (~60 calls on the gateway).

## τ-inventive multiples (2026-09-08)

- **τ is now the reported axis**, not a hidden constant: a τ-inventive multiple re-uses ≥ τ properties,
  and `tau_curve` reports the rate at every τ the data supports. Renamed `K_SHARED`→`TAU`,
  `TAU_SLOT`→`COS_SLOT`, `TAU_CON`→`COS_CON` to free the symbol.
- **The abstraction clause is gone**, matching the paper's Definition: property overlap and nothing
  else. `COS_CON` is retained as a descriptive diagnostic, never an input.
- **θ calibrated against a cross-item null** (`calibrate_theta.py`): inventions on *different* anchors
  cannot share an item-specific property, so their matches are a null. Adopted **θ = 0.545**, the
  null's 99.75th percentile (α = 0.25%, one property pair in 400). The old 0.58 had no derivation and
  implied α = 0.15%.
- **`plot_tau_theta_grid.py`** renders the joint density over (τ, θ): each pair contributes one point
  per matched property at its τ-th best match. No suptitle/caption, panel letters (a)/(b)/(c).
- **Effect sizes are much smaller than reported.** Three independent routes — size-normalised overlap,
  null-corrected embedding similarity, and exact-match per opportunity — all put blending's advantage
  over analogy at **~2×**, against the paper's 11×. Blends carry 5.01 properties to analogy's 2.60, so
  a fixed τ is a 40% bar for blending and 77% for analogy.
- **Anchor echo**: 67.5% of analogy object-collisions are the anchor entities themselves (`pi`,
  `buddhism`, `free will`) vs 2.2% for blending. Unfiltered this *inverts* #3b. The right filter is
  "the object **is** the anchor" (exact match), not word overlap — the latter wrongly drops
  `emits light` for (Mount Everest, The light bulb).
- **Exploratory**: within-item invention similarity has **no cluster structure** — the first eigenvector
  explains only 0.08–0.09 of positive variance over ~34 inventions. Multiples are not attractors.
  Property support is the better unit: 12.0% of blending property clusters are asserted by ≥2 models
  vs 1.3% by chance, max support 15 of 34 models.

## Files

New: `src/kg_creat/scripts/analyze_cognitive_facets.py`, `calibrate_theta.py`,
`plot_tau_theta_grid.py`, `make_tau_multiples_table.py`; 12 backfill/score configs under
`configs/{dat_eval,new_tests}/`.
Modified: `src/dat_eval/llm.py` (`anthropic_compat_endpoint`, `strip_unsupported`),
`src/new_tests/llm.py`, `analyze_inventive_multiples.py`.
Data: `data/dat_eval/backfill_{anthropic,gateway,openrouter}/`,
`data/new_tests/{rat,drat}/backfill_*/`, merged into `data/dat_eval/run_v1` for the pool-wide CDAT
gate (originals kept as `*.bak_pre_capfix` / `*.bak_pre_or`).

## Open

1. **RAT re-run for the two gateway models** at a larger cap — their scores are currently wrong.
2. **DRAT**: drop it, or rescore with τ fixed across items. As scored it measures item breadth.
3. Missing from DAT/CDAT: `grok-4.5`, `grok-4.6`, `glm-4.5-air`, `glm-4.6`, `qwen3-max` — a skew toward
   US labs that matters for generality more than for power.
4. DAT has little spread at the top (30 of 35 models between 82 and 92); the correlation may rest on a
   few low outliers. Worth a leave-the-bottom-three-out check.
5. Paper `#3a/#3b/#3c` still carry pre-τ numbers (27.4%, 11×, 3.5×).
