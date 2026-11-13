# Rebuttal 3a: is the leaderboard robust to who is in the originality reference pool? (reviewer cGKy Q2)

2026-11-13. No API spend. Run with `.venv_mlx/bin/python -m src.kg_creat.scripts.robustness_originality_pool`
(about 2.5 minutes locally). Output: `data/kg_creat/kombine_test30/analysis/originality_pool_robustness.json`
(per-variant rows, summaries, reproduction check).

**Reproduction.** Recomputing ρ exactly as for the paper returns all 21,294 stored `originality` /
`em_originality` values within 3e-7, and every model's overall score within 8e-7 of the stored composite.

**Method.**
- **ρ as in the paper:**
  - per item, the pool is every distinct element from the 35 models plus human_kept and human_flagged;
  - k = 5 nearest neighbours, all-MiniLM-L6-v2 (MLX);
  - separate base and emergent pools for analogy and blending.
- **Composite:** via `compute_composite.artifact_dims`, with the same utility gate, fraction-of-max dimensions, task means and overall score as the paper.
- **Only the pool changes.** Humans stay in the pool for the leave-out variants, as in the paper. Only the 35 models are ranked.
- **Variants:**
  - leave one model out (35);
  - leave one provider out (10);
  - fixed pool of the other half of the models (10 random halvings × 2 directions);
  - fixed pool of human answers only. The four worked-example items, which no human saw, drop out of the originality means.

| Pool variant | n | Overall Spearman (min) | Kendall τ (min) | Max rank shift | Top 8 unchanged | Max \|Δ overall\| (pts) | Originality-dim Spearman (min / median) |
|---|---|---|---|---|---|---|---|
| Leave one model out | 35 | 1.000 | 1.000 | 0 | 8/8 | 0.14 | 0.997 / 0.999 |
| Leave one provider out | 10 | 1.000 | 0.996 | 1 | 8/8 | 0.68 | 0.992 / 0.998 |
| Fixed pool: other half of models | 20 | 0.998 | 0.985 | 1 | 8/8 | 1.18 | 0.929 / 0.990 |
| Fixed pool: humans only | 1 | 0.998 | 0.987 | 2 | 8/8 | 5.75 | 0.951 / 0.975 |

| Provider dropped | Models ranked | Spearman | Kendall | Max shift | Max \|Δ overall\| | Min originality-dim Spearman |
|---|---|---|---|---|---|---|
| Anthropic (10) | 25 | 1.000 | 1.000 | 0 | 0.68 | 0.994 |
| OpenAI (7) | 28 | 1.000 | 1.000 | 0 | 0.38 | 0.996 |
| Google (5) | 30 | 1.000 | 1.000 | 0 | 0.24 | 0.996 |
| DeepSeek (3) | 32 | 1.000 | 0.996 | 1 | 0.13 | 0.997 |
| Meta (2) | 33 | 1.000 | 1.000 | 0 | 0.10 | 0.997 |
| Qwen (2) | 33 | 1.000 | 1.000 | 0 | 0.09 | 0.999 |
| xAI (2) | 33 | 1.000 | 1.000 | 0 | 0.06 | 0.999 |
| Z.ai (2) | 33 | 1.000 | 1.000 | 0 | 0.10 | 0.992 |
| Microsoft (1) | 34 | 1.000 | 1.000 | 0 | 0.04 | 0.999 |
| Moonshot (1) | 34 | 1.000 | 1.000 | 0 | 0.06 | 0.999 |

**Rebuttal sentences**
> Originality is computed against the pool of all responses to an item, so we recomputed it under four alternative pools. These were leaving out each model (35 variants), leaving out each provider (10), and two fixed external pools that do not depend on who is evaluated: the other half of the models (20 random splits) and human answers only. Leaving out any single model leaves the ranking of the remaining 34 unchanged (Spearman = Kendall = 1.00 in all 35 cases). Dropping an entire provider, including all 10 Anthropic or all 7 OpenAI models, moves at most one model by one rank (Spearman ≥ 0.999).
>
> Scoring every model against a fixed pool it is not part of also preserves the ranking (Spearman ≥ 0.998, top 8 unchanged in every case), and the models' originality scores keep their order (Spearman ≥ 0.93 on every originality dimension; ≥ 0.99 when one model or provider is removed). The leaderboard reflects stable differences between models, not the particular pool. We will add this analysis to the appendix.

**Caveats**
- **Overall stability is partly mechanical.** Originality is 5 of the 15 composite dimensions, and the utility gate doesn't depend on the pool. The originality-dimension Spearman is the real test, and it holds.
- **External-pool variants change one rule.** In the fixed-pool variants, the scored model's answers aren't in its pool, so an identical answer in the reference pool counts as distance 0. The leave-out variants keep the paper's rule of skipping the identical surface.
- **Humans-only pool covers fewer items** (86 of 90): the worked examples have no human answers.
- **Absolute scores shift with an external pool** (up to 5.75 points for humans only), because model answers are farther from human answers than from each other. Only the rankings are claimed stable.
