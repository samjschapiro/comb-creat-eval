# Utility: Analogy vs Blending on the Same Anchors

*2026-09-05 · kg_creat track · analysis memo*

**Question.** Utility is the gate — every later dimension is scored only on artifacts that pass it — so it decides whether the model can perform the subsequent steps at all. On the same anchor pairs, do models instantiate the analogical schema more reliably than the blend schema?

**Answer.** Yes, by **13.5 points**, and item difficulty is not the explanation. But the advantage is not uniform: the two operators find *different* items hard, and on the easiest third of pairs blending is ahead.

## Claims

1. **Analogy leads on utility by 13.5 points** — 56.4% vs 42.8%. McNemar over matched (model, item) cells: odds ratio **1.84** [1.49, 2.28], p = 3.5×10⁻⁹. Analogy is higher for **23 of 30 models** and 20 of 30 items. On the frontier subset it is +15.7 points (OR 2.06, 12 of 15 models).
2. **Item difficulty is not the explanation.** No anchor pair defeated every model on either task — the hardest still gets 23% of models through on analogy and 7% on blending, so nothing is dropped as impossible. Difficulty is also controlled by construction: McNemar conditions on the (model, item) cell, so both members of every pair are the same model on the same anchors.
3. **The two tasks find different items hard.** Per-item pass rates correlate at **r = −0.02** across the 30 pairs. A pair that resists analogy is not the pair that resists blending.
4. **So the advantage is concentrated, not general.** Splitting items into terciles by their mean pass rate across both tasks: analogy leads by **+33.4** points on the hardest third and **+17.2** on the middle third, but **trails by 9.9** on the easiest third (p = .012). Analogy is comparatively flat across difficulty; blending swings from 18.6% to 70.2%.

## The two flags are like for like

| | analogy | blending |
|---|---|---|
| **U** | `pair_sat` — the two paths share a relation sequence **and** their triples are factual | `generic_ok` — a 3-judge panel accepts the generic space as instantiated by **both** inputs |

Shared relations over factual triples *is* a valid analogy; there is nothing further to verify. That is why the check can be mechanical — the analogy operator writes its schema as an explicit structural correspondence. Blending writes its schema as a sentence, so the same question ("do both inputs instantiate it?") can only be answered by a judge. Different machinery, same job: each flag fully verifies that its operator's schema holds.

What the two flags reject:

| flag | passes | fails on structure | fails on facts |
|---|--:|--:|--:|
| U_an (n = 888) | 56.1% | 21.3% | 22.6% |
| U_bl (n = 885) | 42.7% | 57.3% | — (a blend asserts no facts to check) |

## Design

Analogy and blending run on the **same 30 anchor pairs** with the **same 30 models**, one draw each, so every (model, item) cell holds one attempt at each. Item difficulty is controlled twice: impossible items are dropped (none qualified), and McNemar conditions on the cell. Two weaker paired tests are reported alongside — per model and per item.

## Results

| subset | analogy | blending | diff | odds ratio [95% CI] | p | models | items |
|---|--:|--:|--:|---|--:|--:|--:|
| all 30 | 56.4% | 42.8% | **+13.5** | 1.84 [1.49, 2.28] | 3.5e-9 | 23/30 | 20/30 |
| frontier 15 | 68.8% | 53.1% | **+15.7** | 2.06 [1.52, 2.81] | 9.5e-7 | 12/15 | 20/30 |

By item difficulty (terciles of the **mean** pass rate across both tasks, so neither task's own ranking drives the bins):

| tercile | analogy | blending | diff | p |
|---|--:|--:|--:|--:|
| hardest 10 | 52.1% | 18.6% | **+33.4** | 7.2e-18 |
| middle 10 | 56.7% | 39.5% | **+17.2** | 2.2e-05 |
| easiest 10 | 60.3% | 70.2% | **−9.9** | .012 |

The crossover survives the neutral split and also appears when items are ranked by either task alone, so it is not an artifact of sorting.

By provider, the gap is positive for 9 of 10 (moonshotai −3.4, n = 29). Individually most are underpowered; openai (+19.4, p = 8×10⁻⁵) and x-ai (+40.7, p = 7×10⁻⁵) are clearly significant, anthropic borderline (+10.6, p = .048).

Largest per-model gaps: grok-4.6 (93.1% vs 46.7%), claude-opus-4.6 (+36.7), gpt-5 and grok-4.5 (+33.3). The five reversals are small: gemini-3.1-pro (−8.3), claude-sonnet-4.5 (−6.7), deepseek-chat (−3.3), kimi-k2 (−2.4), glm-4.5-air (−0.7).

## A judging inconsistency worth recording

The blend panel returns `generic_ok` (the utility gate) and `integration_quality` (scope 1/2/3) separately, and **they disagree on 22.9% of blends** (203 of 885): 61 where the schema is accepted but scope is 1, and 142 where it is rejected but scope is 2 or 3. A rejected schema should not be able to be double-scope. The gate is `generic_ok`.

This has a downstream consequence: `plot_abstraction_failure.py` and `catalogue_generic_space_failures.py` mark scope == 1 as "generic space rejected". On the frontier subset that is **40.4% where the gate says 46.7%**, disagreeing on 21.8% of cells. The 47% quoted in the frontier-failures report prose is correct; the figure and the catalogue are built on the wrong field.

## Limitations

- **One draw per cell** at temperature 0.9. The matched design removes model and item effects but not single-draw noise, which inflates McNemar's discordant counts symmetrically — direction unaffected, odds ratios conservative.
- **Judges, not ground truth.** The blend panel is a 3-judge majority at ICC 0.48–0.65 and is internally inconsistent on 22.9% of blends; analogy's factuality gate is a single judge with a known non-zero false-positive rate, so U_an's 22.6% factual-failure share is an upper bound.
- **30 curated, all cross-domain anchor pairs; 30 convenience-sampled models.** Rates are specific to this pool, and the tercile bins hold 10 items each.

## Reproduce

```
.venv/bin/python -m src.kg_creat.scripts.test_utility_analogy_vs_blending
```

Writes every number above to `data/kg_creat/kombine_test30/analysis/utility_analogy_vs_blending.json`.
