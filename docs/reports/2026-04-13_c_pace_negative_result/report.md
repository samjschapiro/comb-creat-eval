# Constrained PACE (C-PACE): A negative result

**Date**: 2026-04-13
**Status**: Closed. The designed metric does not beat PACE as a standalone creative-writing predictor at n=44. Writing it up as a negative result.
**Relates to**: [PACE correlation study](../2026-04-12_preliminary_correlations/report.md), [Schapiro et al. 2025 combinatorial creativity framework](https://arxiv.org/abs/2509.21043)

## The hypothesis we tested

PACE (Qiu & Hu, EMNLP 2025) measures the FastText chain score of a 20-word associative chain produced by a model. It correlates with Arena Creative Writing at Spearman ρ = +0.75 in our prior study and survives partialling for general capability. We hypothesised that *adding constraints* to PACE's chain-generation prompt — forcing the model to explore novel combinations under pressure, in the spirit of Boden's combinatorial creativity and the Schapiro et al. (2025) framework — would sharpen the creativity-specific signal.

Concretely, we extended PACE's stage-2 prompt with inclusion/exclusion constraints at four difficulty levels (L1 = none, L4 = two inclusion + two exclusion). Constraints were *lexical* (word must/mustn't start with letter L). We scored each chain with:

- `constraint_satisfaction_rate` — fraction of chains satisfying all constraints (capability channel)
- `chain_diversity` — PACE's FastText score, averaged over chains that satisfied their constraints (creativity channel)
- `composite_hard` — Schapiro's U × N: (1 + |I|)(1 + |X|) × chain_diversity, zeroed on any violation
- `composite_soft` — partial-credit (fraction of constraints satisfied) × chain_diversity

We ran this on 44 models spanning 1B–235B parameters, from Llama 3.2-1B to Claude Sonnet 4.6 (the 7 most expensive frontier models were cut when OpenRouter credits ran out — Opus 4.5/4.6, GPT-5/5.4, GPT-4-turbo, o3, Sonnet 4, and Sonnet 4.5).

## Headline result

**C-PACE's composite scores do not beat PACE as a standalone predictor of any creative-writing benchmark at n=44.** On Arena Creative Writing, Spearman ρ drops from PACE's +0.699 to the soft composite's +0.512. Pearson r drops from +0.649 to +0.596.

| Benchmark | PACE ρ | Soft composite ρ | PACE r | Soft composite r | n |
|---|---|---|---|---|---|
| Arena creative writing | **+0.699** *** | +0.512 *** | **+0.649** *** | +0.596 *** | 41 |
| Arena overall | **+0.643** *** | +0.615 *** | +0.579 *** | **+0.667** *** | 41 |
| EQ-Bench creative writing | **+0.720** *** | +0.365 | **+0.668** *** | +0.478 | 26 |
| Mazur creative writing v2 | **+0.628** ** | +0.453 | **+0.642** ** | +0.423 | 17 |

The earlier n=20 result (where the soft composite slightly beat PACE on Pearson r for Arena CW: 0.71 vs 0.63) did not replicate when we added the remaining 24 models. PACE got stronger as frontier models joined the sample; C-PACE did not.

## What *did* replicate: the composite adds incremental variance to PACE

Although it loses standalone, the soft composite carries information PACE does not, on Arena benchmarks specifically. Hierarchical regression of creative-writing score on {PACE} vs {PACE + soft_composite}:

| Benchmark | R²(PACE) | R²(PACE + soft) | ΔR² | F p |
|---|---|---|---|---|
| Arena creative writing | 0.422 | 0.544 | **+0.122** | 0.003 ** |
| Arena overall | 0.335 | 0.548 | **+0.213** | 0.0001 *** |
| EQ-Bench creative writing | 0.446 | 0.518 | +0.072 | 0.08 . |
| Mazur creative writing v2 | 0.412 | 0.413 | +0.001 | 0.87 |

Adding the soft composite doubles PACE's explained variance on Arena Overall and bumps Arena creative writing by 12 points. On EQ-Bench and Mazur, PACE carries everything.

## Why the composite doesn't capture creativity

Two observations tell us what the "incremental variance" actually is:

1. **Constraint satisfaction correlates +0.69 with Arena Overall** — a pure capability proxy. Models that satisfy constraints well are models that follow instructions well. The signal from `constraint_satisfaction_rate` × `chain_diversity` is therefore mostly instruction-following multiplied by PACE's own signal. That combination tracks Arena scores because *Arena tracks capability too* (Arena creative writing ρ with Arena overall is +0.96 — the two benchmarks share >95% of their rank-order variance). A capability proxy will predict Arena CW trivially.

2. **On Hivemind (which measures intra-model output similarity — lower means more diverse), the soft composite correlates +0.42 — the wrong direction for a creativity metric.** Higher composite = more convergent outputs, not more diverse. This is the *capability-convergence* effect known from the Hivemind paper: more capable models produce more similar outputs to themselves.

The cleaner version of the diagnosis: **PACE measures a latent tendency** — how far a model's chains naturally drift through conceptual space. The PACE prompt contains no diversity instruction, only "associate with the previous word". **C-PACE measures compliance under a rule** — can the model remember to include a word starting with 'e' while still chaining. Those are different cognitive tasks. Our constraints turned a naturalistic creativity measurement into an instruction-following task, and instruction-following is capability.

## Why lexical constraints specifically fell flat

The constraints we chose — first-letter inclusion/exclusion — are orthographic, not semantic. Satisfying "must include a word starting with 'e'" requires no genuine exploration of combinatorial space; any 'e'-word slotted at any position works. There is no coupling between the constraint and the associative trajectory the chain takes. Compared to Schapiro et al.'s graph task, where an edge-label constraint forces the path through specific graph edges, our lexical constraints are free-floating orthographic patches on top of a PACE chain. They tell us which models can follow instructions; they don't tell us which models are creative.

A natural next design — semantic constraints ("must include a word meaningfully related to 'courage'") — was considered and rejected without running. Semantic-similarity constraints are still rule-based, still measure compliance under pressure, and still conflate capability with creativity. We also considered multi-faceted cognitive constraints (WordNet supersense, concreteness, valence), which are richer, but share the same failure mode: any rule added to a free-association task shifts the measurement from natural behaviour to rule-following.

## Limitations and what we're not claiming

- n=44, frontier missing. The 7 most expensive models (Opus 4.5/4.6, GPT-5/5.4, o3, Sonnet 4/4.5) were cut by the budget cap. If the soft composite's advantage returns at frontier, the picture could change — but at n=44 PACE beats it everywhere, so "frontier rescues C-PACE" is not the base-rate expectation.
- We tested only lexical constraints. Semantic constraints might behave differently in principle; we're arguing they shouldn't, but we didn't run them.
- EQ-Bench CW has only 11 models with data and shouldn't be trusted on its own.
- The composite hard (binary-gated) score performs similarly to soft; reporting one stands in for both.

## Implication for the broader research direction

The thing that went wrong isn't fixable by tweaking the constraint formulation. Any rule we add to PACE's prompt converts it from an emergent-creativity measurement into an instruction-following measurement. If we want a creativity signal distinct from PACE, we need a task where the creative challenge is **structural** rather than **prescriptive** — the challenge should come from the geometry of the problem, not from a rule we asked the model to track.

Our next experiment (circle construction; see Nagarajan et al. 2025, "Roll the dice & look before you leap") replaces rule-based constraints with the structural requirement of closing a word-chain back to its seed. The closure requirement forces the model to plan without tracking any extra instruction. Preliminary wire tests suggest the task is tractable for most models but not trivially solvable, which is the property our C-PACE constraints did not produce.

## Artifacts

- Code: `src/comb_eval/c_pace.py`, `src/comb_eval/scripts/{run,score}_c_pace.py`
- Data: `data/comb_eval/c_pace_v1/` (44 models × 20 seeds × 3 first-associations × 4 levels = 10,560 chains)
- Configs: `configs/comb_eval/run_c_pace.yaml`, `configs/comb_eval/score_c_pace.yaml`
