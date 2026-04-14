# Word-Circle Construction: A second negative result

**Date**: 2026-04-13
**Status**: Closed. Like C-PACE, no single circle metric beats PACE as a standalone creative-writing predictor at n=54. Writing it up and pivoting the research program.
**Relates to**: [C-PACE negative result](../2026-04-13_c_pace_negative_result/report.md), [PACE correlation study](../2026-04-12_preliminary_correlations/report.md), [Nagarajan et al. 2025](https://arxiv.org/abs/2504.15266)

## The hypothesis we tested

After C-PACE showed that rule-based constraints added to PACE convert the task into instruction-following rather than creativity measurement, we moved to a **structural** constraint: the model must produce an associative word-chain that *closes back* to the seed (a "word circle"). The structural requirement forces planning-for-closure without the model having to track any extra rule, borrowing directly from Nagarajan, Wu, Ding, Raghunathan (ICML 2025), which argued that constructing a novel permutation that closes is the core leap-of-thought deficit of next-token-trained models.

We ran 54 models on 20 seeds × 3 trials × 8-word circles at temperature 0.7 (so that multi-trial sampling produces variance). Each trial produces an ordered word list; we verify at scoring time whether the chain's consecutive FastText cosine similarities meet a threshold `τ_edge`, whether the closure edge (last word back to seed) meets `τ_closure`, and compute several creativity-adjacent metrics:

- `valid_circle_rate` — fraction of circles that satisfy every edge + closure + distinctness + in-vocab requirement
- `closure_rate` — fraction where the closure edge alone meets threshold
- `edge_coherence_rate` — fraction of intra-chain edges meeting threshold
- `mean_pairwise_diversity` — average FastText pairwise distance across the 8 circle words
- `mean_pace_internal_score` — PACE-style position-to-prior distance average
- `cross_trial_diversity` — average word-set Jaccard distance across the 3 trials for the same seed (Hivemind-style)
- `mean_closure_cosine` — raw FastText similarity of w₈ to the seed

Thresholds are post-hoc knobs (not in the model's prompt). Results below use `τ_edge = τ_closure = 0.2`, chosen during wire-testing to sit in a range where all models produce some valid circles.

## Headline result

**No circle metric beats PACE as a standalone creative-writing predictor on any of our four benchmarks.** PACE dominates at n=51 (Arena coverage) by a wide margin:

| Benchmark | PACE ρ | PACE r | Best circle metric | Circle ρ | Circle r |
|---|---|---|---|---|---|
| Arena creative writing | **+0.770** *** | **+0.720** *** | `mean_pairwise_diversity` | +0.638 *** | +0.594 *** |
| Arena overall | +0.724 *** | +0.667 *** | `mean_pairwise_diversity` | +0.566 *** | +0.522 *** |
| EQ-Bench creative writing | +0.756 *** | +0.710 *** | `mean_pace_internal_score` | +0.627 *** | +0.653 *** |
| Mazur creative writing v2 | +0.707 *** | +0.707 *** | `mean_pairwise_diversity` | +0.429 | +0.490 * |

At n=33 (mid-tier only), `valid_circle_rate` briefly had a higher Pearson r than PACE on Arena creative writing (+0.729 vs +0.570). This did not replicate when frontier models were added. PACE's Spearman jumped from 0.646 (n=30) to 0.770 (n=51) as GPT-5, Opus, Sonnet, and o3 joined; `valid_circle_rate` dropped from 0.614 to 0.442. Same small-n effect we saw with C-PACE's soft composite.

## What *did* replicate: incremental information

As with C-PACE, circle metrics add variance to PACE that PACE alone doesn't capture, most strongly on Arena benchmarks:

| Metric | Arena CW ΔR² | Arena overall ΔR² | EQ-Bench ΔR² |
|---|---|---|---|
| `valid_circle_rate` | **+0.156** *** | **+0.193** *** | +0.084 * |
| `closure_rate` | +0.026 | +0.046 * | +0.009 |
| `cross_trial_diversity` | +0.018 | +0.054 * | +0.021 |
| `mean_pairwise_diversity` | +0.024 | +0.011 | +0.004 |

`valid_circle_rate` adds 16 points of R² to PACE on Arena creative writing (p < 0.001). That's a real signal, slightly larger than C-PACE soft composite's +0.12 at n=44, but it doesn't compensate for the metric's weak standalone correlation.

## The capability-specificity problem, quantified

A creative-writing metric should track Arena *creative writing* more tightly than Arena *overall*. The Pearson r gap (CW minus Overall) quantifies this specificity:

| Metric | r vs Arena overall | r vs Arena CW | CW specificity |
|---|---|---|---|
| `mean_pairwise_diversity` | +0.522 | +0.594 | **+0.071** |
| **PACE** | +0.667 | +0.720 | **+0.053** |
| `mean_pace_internal_score` | +0.447 | +0.494 | +0.047 |
| `closure_rate` | +0.315 | +0.268 | -0.047 |
| `valid_circle_rate` | +0.633 | +0.609 | **-0.024** (capability-leaning) |
| `cross_trial_diversity` | +0.316 | +0.223 | -0.092 |

Only `mean_pairwise_diversity` has higher CW specificity than PACE (Δ = +0.071 vs +0.053), but its absolute correlation is substantially lower. `valid_circle_rate` — our biggest incremental-information winner — is slightly *more* capability-leaning than PACE. Benchmarks sharing 95%+ variance (Arena CW vs overall: ρ = 0.964, r = 0.976 on the full benchmark set) makes this test weak in absolute terms, but the direction still matters: `valid_circle_rate` is structurally "can the model produce a valid-output-geometry", which maps more directly onto general capability than onto creative writing.

## Why circles didn't separate from capability

Two things we can point to retrospectively:

1. **Closure is cheap when edges are lax.** At `τ_edge = 0.2`, many loosely-associated word pairs satisfy edge coherence, and closure (just one more edge) is also satisfied by most models. The planning difficulty Nagarajan et al. identify — finding a *novel* permutation — is absorbed by the generous threshold. Making τ stricter would filter more circles out but would also drop the n of "creatively closed" circles toward zero for weaker models, reintroducing the capability floor we saw in the graph-path eval.

2. **"Can produce a valid structure" is too close to instruction-following.** We thought the closure requirement was structural rather than prescriptive, but in practice the model still has to *track* the closure requirement and *satisfy* it. It's a thinner instruction than C-PACE's constraint list, but it's still an instruction the model either remembers and plans around, or doesn't. That's a capability signal, not a creativity signal — just as with C-PACE.

## Limitations

- `τ_edge` and `τ_closure` were fixed at 0.2 post-hoc. A full sweep might reveal thresholds where some metric beats PACE; we did not exhaustively scan. Based on our wire test and threshold sensitivity checks, we do not expect the story to change meaningfully.
- Partial runs: `anthropic/claude-sonnet-4.6` (25/60 parsed, 35 provider errors), `anthropic/claude-opus-4.6` (43/60), `qwen/qwen3-14b` (39/60), `qwen/qwen3-32b` (41/60). These reduce effective n for those models but don't explain the overall pattern.
- Cross-trial diversity was computed via word-set Jaccard only, not embedding-centroid distance. A more sensitive diversity measure might show more signal — though `cross_trial_diversity` already had the *worst* CW specificity (-0.092), so we doubt a measurement refinement reverses direction.
- No frontier bake-off test: we did not test whether circle metrics differ from PACE for specific model families (e.g., reasoning models vs non-reasoning). A stratified analysis might find structure.

## Cumulative learning across both negative results

Across **C-PACE** and **Circle Construction**, the same pattern held:

1. Both metrics add real incremental information to PACE on Arena benchmarks (C-PACE soft composite: +0.12 ΔR², circle's `valid_circle_rate`: +0.16 ΔR², both p < 0.01).
2. Neither beats PACE as a standalone predictor once frontier models are included.
3. In both cases, small-n (n=20 to n=33) results suggested a standalone win that vanished as n grew — a warning for anyone running creativity-metric studies on small model subsets.
4. The information each metric adds is closer to capability (Arena overall) than to creative-writing specificity (Arena CW delta).

**The generalisation we now believe**: any test-time modification to PACE that asks the model to follow a rule or produce a constrained structure will drift toward measuring capability, because instruction-following *is* a capability. PACE's unusual property — that it survives partialling and has modest CW specificity — comes from its total absence of constraints: it measures what models *naturally do* when asked to free-associate, not what they *can do* when asked to satisfy a constraint.

## Pivoting the research program

We are stopping attempts to build a standalone creativity metric that beats PACE. Instead, three directions make sense:

1. **Mechanistic analysis of PACE.** What specifically about a model's associative behaviour drives its PACE score? Decompose the chain-score into components (early vs late chain positions, common vs rare vocabulary, semantic-neighbourhood density). If we can explain *why* PACE tracks creative writing, that's a contribution without needing to replace it.

2. **Better ground-truth for "creativity".** Arena creative writing is >95% correlated with Arena overall. Every creativity-adjacent benchmark we've tested is similarly capability-confounded. A cleaner ground truth — e.g., human-judged novelty of creative outputs, or expert-rated research idea novelty (Si et al. 2024) — might reveal metric specificity that Arena cannot.

3. **Write up both negative results** (this report and the C-PACE report) as a methodological contribution: "Two failed attempts to extend PACE, and what we learned about why". The small-n failure mode in particular is worth documenting for the field.

## Artifacts

- Code: `src/comb_eval/circle.py`, `src/comb_eval/scripts/{run,score}_circle.py`
- Data: `data/comb_eval/circle_v1/` (54 models × 20 seeds × 3 trials × 8-word circles = 3,240 circle attempts)
- Configs: `configs/comb_eval/run_circle.yaml`, `configs/comb_eval/score_circle.yaml`
- Total spend: $14.01 on OpenRouter
