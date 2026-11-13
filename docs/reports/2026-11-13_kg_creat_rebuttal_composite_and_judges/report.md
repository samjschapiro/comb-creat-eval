# Kombine rebuttal: composite robustness and judge self-preference (reviewer cGKy)

2026-11-13. No API spend: existing paper scores and stored per-judge votes only. Both scripts first
reproduce the paper's 35-model composite exactly (max |diff| ≤ 1e-14).

```
.venv/bin/python -m src.kg_creat.scripts.rebuttal_composite_robustness
.venv/bin/python -m src.kg_creat.scripts.rebuttal_judge_self_preference
```

Outputs: `data/kg_creat/kombine_test30/analysis/rebuttal_{composite_robustness,judge_self_preference}.json`.

**Bottom line.** The leaderboard survives every reweighting and every change to the judge panel, and there
is no detectable family self-preference. One finding changes how we should defend the utility gate (see A).

## A. Composite robustness

Paper top 5: gpt-5.6-sol, gpt-6-astra-flex, opus-4.6, gemini-3.1-pro, fable-5.

| Variant vs the paper ranking | Spearman | Kendall | Same #1 | Top-5 | Top-10 | Max rank shift |
|---|---|---|---|---|---|---|
| No utility gate | 0.965 | 0.852 | yes | 4/5 | 9/10 | 9 |
| Gate, other dims averaged over passed artifacts only | 0.967 | 0.855 | no (astra-flex) | 2/5 | 8/10 | 8 |
| Utility dims only | 0.992 | 0.939 | yes | 5/5 | 10/10 | 3 |
| Originality dims only, gated | 0.988 | 0.919 | no | 4/5 | 10/10 | 4 |
| Originality dims only, ungated | 0.423 | 0.308 | no | 2/5 | 6/10 | 28 |
| Geometric mean of task scores | 0.997 | 0.973 | yes | 5/5 | 10/10 | 2 |
| Task weights ∝ 1/variance | 0.988 | 0.923 | yes | 4/5 | 9/10 | 5 |
| Single task only (association / analogy / blending) | 0.78 / 0.92 / 0.88 | — | — | 1 / 2 / 4 | — | — |

**10,000 random weightings** (Dirichlet(1) over dimensions within each task and over the three tasks):
- Spearman with the paper ranking: median 0.971 (5th percentile 0.897, minimum 0.764). Kendall: median 0.869 (5th percentile 0.731).
- Top-10 overlap: median 9/10 (5th percentile 8). Top-5 overlap: median 4/5.
- The #1 model is always one of five: gpt-5.6-sol 52%, opus-4.6 33%, astra-flex 14%, grok-4.6 0.7%, gemini-3.1-pro under 0.1%.

**How to defend the gate.** The gate does not keep failed answers from inflating originality:
- Failed and passed artifacts are equally original: association 0.337 vs 0.331, analogy 0.453 vs 0.462, blending 0.383 vs 0.382.
- Ungated originality correlates positively with utility across models (+0.36, p = 0.03).

So the gate should be defended as the definition of a creative product (novel *and* useful), not as noise
control. Pure ungated originality is a different construct, and it ranks models differently (0.42).

**Rebuttal sentences**
- "Across 10,000 random Dirichlet weightings of dimensions and tasks, the median Spearman correlation with our ranking is 0.97 (5th percentile 0.90), and a median of 9 of the top 10 models are retained. Principled alternatives give 0.99–0.997: a geometric mean of tasks, inverse-variance task weights, and utility-only scoring."
- "Removing the utility gate barely changes the ranking (Spearman 0.965, same #1, 9 of the top 10 shared). We keep the gate because it encodes the standard two-criterion definition of a creative product (novel and useful), not because the ranking depends on it. Pure ungated originality is a different construct (Spearman 0.42); we will report it as a separate column."
- "Which model is #1 depends on the weights (gpt-5.6-sol under 52% of weightings, opus-4.6 under 33%), consistent with our finding that leadership is task-specific. The set of leading models does not change."

## B. Judge self-preference

The panel is claude-haiku-4.5, o3 and gpt-5.4; Haiku is also the factuality judge F. None of the three is an
evaluated model, but the pool includes 10 Anthropic, 7 OpenAI and 18 third-party models.

**1. Leaderboards without one judge, or with a single judge.** AND means both remaining judges must pass an
answer; OR means either may. The last column is the mean rank change for Anthropic / OpenAI / third-party
models (positive = moved up).

| Panel | Spearman | Same #1 | Top-5 | Mean rank change A / O / 3rd |
|---|---|---|---|---|
| drop Haiku, AND / OR | 0.981 / 0.988 | yes / yes | 4 / 4 | −0.6 / +0.6 / +0.1 and −0.2 / +0.4 / −0.1 |
| drop o3, AND / OR | 0.966 / 0.993 | yes / yes | 3 / 4 | −0.5 / +1.9 / −0.4 and +0.3 / +0.4 / −0.3 |
| drop gpt-5.4, AND / OR | 0.992 / 0.978 | yes / no | 4 / 4 | +0.5 / +0.4 / −0.4 and +0.3 / +0.6 / −0.4 |
| Haiku only / o3 only / gpt-5.4 only | 0.978 / 0.991 / 0.982 | no / yes / yes | 4 / 4 / 4 | — |

Dropping Haiku moves Anthropic models down only 0.2–0.6 ranks. Dropping an OpenAI judge moves OpenAI models
*up*, the opposite of self-preference.

**2. Own-family leniency** (difference in differences). Per answer, the judge's vote minus the other family's
mean vote; then own-family models minus third-party models. CIs are a cluster bootstrap over models.

| Judge | Pooled estimate | 95% CI |
|---|---|---|
| Haiku | +2.7 pp | [−1.7, +7.1] |
| o3 | −0.2 pp | [−5.1, +4.3] |
| gpt-5.4 | +3.3 pp | [−3.1, +8.7] |

Of 18 judge × dimension cells, only one CI excludes zero: o3 on blend coherence, −6.6 pp [−14.0, −0.3],
which runs against its own family. The largest positive estimates are gpt-5.4 on blend emergence (+9.2,
[−1.2, +18.3]) and Haiku on analogy validity (+7.4, [−4.9, +18.6]).

**3. Each model judged only by other-family judges.** Anthropic models get o3 + gpt-5.4; OpenAI models get Haiku.
- With AND: Spearman 0.954, same #1, top-5 overlap 3. Anthropic models drop 3.5 ranks, but mostly because requiring both judges to agree is harsher than a three-judge majority.
- With OR: Spearman 0.983, top 5 unchanged as a set but #1 changes. Anthropic models rise 1.9 ranks.
- The paper's panel sits between these bounds.

**4. Factuality: Haiku vs the earlier gpt-oss-120b verdicts on the same 54,277 triples**

| Family | Haiku % true | gpt-oss % true | Agreement |
|---|---|---|---|
| Anthropic | 91.4 | 92.0 | 89.5% |
| OpenAI | 93.1 | 94.6 | 91.7% |
| Third-party | 88.8 | 90.5 | 88.3% |

Haiku is slightly harsher than gpt-oss on every family. The Anthropic difference in differences is +1.1 pp
[−0.5, +2.7]; the OpenAI one is +0.2 pp [−1.3, +1.7].

**Rebuttal sentences**
- "None of the three judges is an evaluated model. Using the stored per-judge votes, each judge's relative leniency toward its own family is +2.7 pp (Haiku 4.5, 95% CI [−1.7, 7.1]), −0.2 pp (o3, [−5.1, 4.3]) and +3.3 pp (gpt-5.4, [−3.1, 8.7]); none differs from zero."
- "Dropping any one judge leaves the ranking essentially unchanged (Spearman 0.966–0.993; same #1 in 5 of 6 configurations). Dropping Haiku moves Anthropic models down by only 0.2–0.6 ranks on average, and dropping an OpenAI judge moves OpenAI models up, the opposite of self-preference."
- "For factuality, Haiku 4.5 agrees with an independent gpt-oss-120b judge on 88–92% of 54,277 triples in every family, and is no more lenient on Anthropic models (+1.1 pp, [−0.5, 2.7])."

## Caveats

- There are only two judge families, so self-preference is identified against third-party models, and the Anthropic estimate rests on a single judge.
- Analogy has about 30 judged answers per model, so its per-dimension CIs are wide.
- gpt-oss-120b is itself an OpenAI model, and it was replaced because it often returned no verdict. The factuality comparison uses only triples both judges answered, so coverage differs by model.
- The "averaged over passed artifacts" variant is noisy for weak models, which have few passing answers.
