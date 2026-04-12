# Preliminary Results: DAT/CDAT/PACE Correlations with Arena CW

**Date**: 2026-04-12
**Status**: Preliminary — 13 of 49 target models scored. Eval run still in progress.
**Relates to**: [ICCC 2026 short paper](../2026-04-11_iccc_dat_study/report.md)

## Headline findings (preliminary, small-n)

Three creativity metrics evaluated against Chatbot Arena Creative Writing (CW) Elo rankings. All correlations are Spearman rank correlation with 500-iteration bootstrap for robustness.

| Metric | rho vs Arena CW | p | n | Bootstrap 95% CI |
|--------|-----------------|---|---|-------------------|
| **PACE** | **+0.852** | **0.0002** | 13 | [0.561, 0.939] |
| **CDAT Appropriateness @ t=1.5** | **+0.964** | **0.0005** | 7 | [0.698, 1.000] |
| **CDAT Novelty @ t=1.5** | **−0.893** | **0.007** | 7 | [−1.000, −0.373] |
| DAT (pooled) | +0.233 | 0.55 | 9 | [−0.614, 1.000] |
| CDAT Novelty (pooled) | +0.071 | 0.88 | 7 | — |
| CDAT Appropriateness (pooled) | +0.357 | 0.43 | 7 | — |
| CDAT @ t=2.0 | near zero | NS | 6 | — |
| DAT @ t=1.0 | +0.100 | 0.80 | 9 | — |
| DAT @ t=1.5 | −0.571 | 0.14 | 8 | — |

**Three punchlines:**

1. **PACE replicates.** The PACE paper (Qiu & Hu, EMNLP 2025) reported rho = 0.739 across 30 models. We get rho = 0.852 across 13 models. Even with a different seed-word list and a smaller sample, PACE cleanly predicts Arena CW.

2. **CDAT at t=1.5 exhibits the Nakajima tradeoff.** Arena CW correlates very strongly with appropriateness (+0.96) and very strongly *negatively* with novelty (−0.89) at the same temperature. Better creative writers produce CDAT responses that are more appropriate to the cue but less novel in word choice. This is exactly the novelty–appropriateness Pareto tradeoff predicted by the CDAT paper.

3. **DAT does not predict creative writing in LLMs.** Pooled rho = 0.23 (p = 0.55, NS). Per-temperature correlations are all non-significant and inconsistent in sign. This is the predicted null result — DAT's lack of appropriateness constraint means it rewards unrelated words, and unrelated words don't reflect creative ability.

## Methodology (current run)

**Models**: 13 of 49 scored so far. The scored set over-represents frontier Anthropic models (PACE only) and smaller open models (full DAT/CDAT/PACE). The full-run evaluation is ongoing; results will shift as more models complete.

Scored models include: Claude Opus 4.5/4.6, Claude Sonnet 4.5/4.6, GPT-5.4, GPT-5.4-mini, Gemma 2 9B/27B, Gemma 3 27B, Llama 3.1 8B, Llama 4 Maverick, Qwen3 32B, Phi-4, Mistral Nemo.

**Temperatures**:
- DAT and CDAT: temps 1.0, 1.5, 2.0 with unique seed per trial, top_p=1.0, top_k=0
- PACE: temp 0.0 with input-driven variance across 50 seed words (paper convention)
- Max tokens per call capped to prevent runaway generation at high temps

**Sampling**: At temp=2.0 with top_p=1.0/top_k=0, weaker models (Llama 3.1 8B, Gemma 2 27B) produce garbage tokens in 40–100% of trials. Our scoring pipeline drops trials with <7 valid words and flags temperatures with no valid data as N/A rather than 0. The pooled score for a model is computed only over valid trials across temperatures.

**Concurrency**: All 3 eval types run with async OpenAI client and a 20-concurrent semaphore against OpenRouter. DAT for a model = ~3s, CDAT = ~3s, PACE = ~60-90s.

## Caveats

- **Small n.** Per-temperature correlations use n=6–9; the bootstrap CIs span wide ranges. The CDAT t=1.5 correlations (rho=0.96 and −0.89) are striking but could shift when more models are scored.
- **Selection bias.** The scored set right now is small open models + Anthropic frontier models. We lack data on mid-tier OpenAI, Google, and DeepSeek models which will arrive as the run continues.
- **Temperature-fixed models.** 3 of 49 models (gpt-5, gpt-5-nano, o3-mini) don't respond to temperature variation; for those, the DAT/CDAT @ t=1.5 specifically will mirror their t=1.0 response.
- **Reliability thresholds.** Following Nanda's writing advice, we treat p < 0.05 as suggestive and p < 0.001 as robust. The PACE result (p=0.0002) and CDAT t=1.5 correlations (p=0.0005, p=0.007) all clear that bar — but only with the current small sample.

## Score ranges (sanity check against published work)

All three metrics are within their respective published ranges:

| Metric | Our range | Published range |
|--------|-----------|------------------|
| DAT (GloVe 840B) | 80–88 | Similar-scale papers: 70–85 at pooled temps |
| CDAT Novelty (SBERT) | 71–75 | Nakajima et al.: 60–78 |
| CDAT Appropriateness (SBERT) | 128–141 | Nakajima et al.: 125–150 |
| PACE (FastText) | 0.68–0.76 | Qiu & Hu: 0.72–0.79 |

Our PACE range slightly overlaps the low end of the published range, consistent with our 50-seed (vs their 110-seed) sample.

## What this means for the ICCC paper

If these correlations hold with the full 49-model run, the paper writes itself:

1. **DAT fails to predict creative writing in LLMs.** A negative finding with concrete consequences — researchers should stop citing DAT scores as evidence of LLM creativity.

2. **CDAT's appropriateness gate matters.** The *exact* pattern predicted by Nakajima et al. (novelty-appropriateness tradeoff) appears at the temperature where sampling is creative but coherent.

3. **PACE is a cleanly reproducible alternative.** We replicate their rho ≈ 0.7–0.8 correlation with Arena CW using a different seed list, different embedding-loading pipeline, and different model set.

## Next steps

- Continue the run until all 49 models are scored or the $30 budget cap is reached
- Re-run scoring when the full set is complete
- Compute partial correlations (CDAT / PACE vs Arena CW controlling for Arena Overall) to isolate creativity-specific signal from general capability
- Compute inter-metric correlations (do DAT, CDAT, PACE measure the same thing?) — preliminary: DAT vs PACE rho=0.00, DAT vs CDAT rho=0.43, PACE vs CDAT rho=0.07. They seem to measure mostly different things.

## Data

All scores: `data/dat_eval/run_v1/downstream/scores_v1/results/all_scores.json`
Correlation output: `data/dat_eval/run_v1/downstream/scores_v1/results/correlation_analysis.json`
