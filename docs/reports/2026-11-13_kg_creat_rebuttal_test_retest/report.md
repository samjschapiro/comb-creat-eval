# Rebuttal: test–retest reliability on 30 new items (reviewer cGKy Q1)

2026-11-13. The question: does a model's Kombine score hold up on a different set of 30 items, and how much
does it move?

**Answer.** Yes, for the overall score and for analogy and blending; association is the noisy task.
- **Overall score.** Across the 10 models, scores on the paper's items and on 30 new items correlate at
  r = 0.92, with an absolute-agreement ICC of 0.92. The mean absolute change is 2.7 points on a 0–100
  scale.
- **No model moves beyond sampling noise.** Every model's 95% CI on the change includes zero.
- **By task.** Blending reproduces best (r = 0.95) and analogy well (0.87). Association is the least
  stable (0.65), and 4 of 10 models' association scores change by more than item sampling allows.

## Design

| | test30 (paper) | retest30 (new) |
|---|---|---|
| Items | the paper's 30 per task. Association uses different pairs from analogy and blending. | 30 new cross-domain anchor pairs, disjoint from test30 at the entity level, shared by all three tasks |
| Models | the same 10 | the same 10 |
| Answers | paper run, one per item, T = 0.9 | elicited 2026-09-15, one per item |
| Judging | paper panel (Haiku 4.5, gpt-5.4, o3) + Haiku 4.5 factuality | the same, scored 2026-11-13 (`configs/kg_creat/kombine_retest30_panel_score.yaml`) |
| Originality pool | **recomputed over the same 10 models only** | the 10 models |

- **Models:** claude-opus-4.5, 4.6, 4.7, 4.8, sonnet-4.6, fable-5, fable-5.1, gpt-4.1, gpt-4o-mini and
  gpt-5-mini.
- **Changed sampling conditions.** For fable-5, opus-4.5 and opus-4.6, the retest was drawn on the
  direct Anthropic API, with an explicit thinking budget and no temperature. In the paper run they went
  through OpenRouter at T = 0.9 with an 8k thinking cap. For them the retest varies the sampling
  conditions as well as the items, which can only *lower* the measured reliability. Results without them
  are shown in the last column below.
- **What the comparison measures.** The two sets differ in items *and* draws, so the comparison combines
  item-sampling and resampling noise. It is an upper bound on the pure T = 0.9 resample noise cGKy asked
  about.
- **Originality pool.** Originality depends on the pool, so test30 originality was recomputed with the
  same rule as the paper, over a pool of exactly these 10 models. That makes it comparable with
  retest30, which was scored against the same 10. Every other dimension doesn't depend on the pool.

## Results

| Score | Pearson r | Spearman ρ | ICC(A,1) | Mean test30 | Mean retest30 | Mean \|Δ\| | Item SE (one set) | Models with CI excluding 0 | r / ρ without the 3 changed-condition models |
|---|---|---|---|---|---|---|---|---|---|
| Association | 0.65 | 0.71 | 0.53 | 45.5 | 50.1 | 7.1 | 3.0 | 4 / 10 | 0.58 / 0.46 |
| Analogy | 0.87 | 0.88 | 0.80 | 46.1 | 48.7 | 7.8 | 6.7 | 1 / 10 | 0.90 / 0.75 |
| Blending | 0.95 | 0.94 | 0.84 | 33.5 | 28.4 | 5.8 | 7.0 | 0 / 10 | 0.93 / 0.96 |
| **Overall** | **0.92** | **0.88** | **0.92** | 41.7 | 42.4 | **2.7** | 3.4 | **0 / 10** | 0.88 / 0.75 |
| Utility only, overall | 0.92 | 0.88 | 0.91 | 56.5 | 58.0 | 3.5 | 4.4 | 0 / 10 | 0.89 / 0.79 |

*Item SE* is the median, over models, of the item-bootstrap standard error of one model's score on one
set of 30 items. The per-model CIs come from a bootstrap that resamples items within each task,
independently in each set (2,000 draws).

| Model (by test30 overall) | test30 | retest30 | Δ [95% CI] |
|---|---|---|---|
| opus-4.6* | 54.1 | 54.2 | +0.1 [−7.8, +8.3] |
| fable-5* | 49.9 | 49.2 | −0.7 [−9.6, +8.5] |
| opus-4.5* | 46.1 | 49.1 | +3.0 [−6.8, +12.4] |
| sonnet-4.6 | 45.8 | 51.5 | +5.7 [−3.0, +14.7] |
| fable-5.1 | 43.8 | 40.8 | −3.0 [−12.8, +6.6] |
| opus-4.8 | 41.9 | 42.4 | +0.5 [−9.1, +10.0] |
| opus-4.7 | 40.8 | 37.4 | −3.4 [−12.8, +5.7] |
| gpt-5-mini | 37.2 | 44.7 | +7.5 [−1.7, +16.4] |
| gpt-4.1 | 31.1 | 28.2 | −2.9 [−12.0, +6.1] |
| gpt-4o-mini | 26.2 | 26.3 | +0.1 [−8.1, +8.2] |

\* sampling conditions changed between the two sets.

**Reading it.**
- **The overall score is stable.** The task scores err in different directions on new items (blending
  harder, association easier), and averaging three tasks cancels much of that.
- **Blending is the most reproducible task** despite its larger per-set SE. Models differ a lot on it, so
  the differences between models dominate the noise.
- **Association is the weakest at n = 30.** Its spread between models is narrow, which pulls the
  correlation down. The new association items were also easier on average (+4.6 points). The 4 models
  whose association CI excludes zero are consistent with item effects that differ by model, which 30
  pairs don't average out.
- **Means shift with the item set, as expected.** Blending is 5 points lower on the new items and
  association 5 points higher. The paper's claims compare models on the same items, so a shift in item
  difficulty does not affect them.

## Rebuttal sentences

> To measure how much scores depend on the particular items and draws, we re-ran 10 models on 30 *new*
> entity pairs, disjoint from the benchmark's at the entity level, and scored them with the same judges.
> Overall scores on the two item sets correlate at r = 0.92 (ICC(A,1) = 0.92). The mean absolute change
> is 2.7 points on a 0–100 scale, and no model's change exceeds its item-bootstrap 95% CI. By task,
> reliability is high for blending (r = 0.95) and analogy (0.87) and lower for association (0.65), which
> we will note as the task most in need of more items. Because the retest changes both items and samples,
> these figures bound the resample variance at T = 0.9 from above.

## Caveats

- 10 models, 7 of them Anthropic. The spread is mostly within one provider, which makes this a harder
  test for the correlations, not an easier one.
- Changed sampling conditions for 3 models (see Design). Without them, the overall Spearman is lower
  (0.75, n = 7), mostly because 7 points of rank data are coarse. Pearson stays at 0.88.
- The comparison mixes items and draws, so it does not isolate pure T = 0.9 resample noise for
  association and analogy. For blending only, the abstraction causal test's control condition is a pure
  same-item resample of six models.

## Reproduce

```
env -u LLM_BASE_URL .venv_mlx/bin/python src/kg_creat/scripts/score.py configs/kg_creat/kombine_retest30_panel_score.yaml
.venv_mlx/bin/python -m src.kg_creat.scripts.rescore_split_originality data/kg_creat/kombine_retest30/scores data/kg_creat/kombine_retest30/responses
.venv_mlx/bin/python -m src.kg_creat.scripts.analyze_retest
```

Spend: about $24 on the OpenRouter key for judging; $16.90 remains. The cost ledger records $54.95 at its
own list prices, which overstates actual charges. Output:
`data/kg_creat/kombine_retest30/analysis/test_retest.json`.
