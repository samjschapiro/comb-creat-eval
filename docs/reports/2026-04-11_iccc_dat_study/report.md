# Does the Divergent Association Test Actually Measure Creativity in LLMs?

**Target venue:** ICCC 2026 (Short Paper)

## Motivation

The Divergent Association Task (DAT; Olson et al., 2021) has become a popular shorthand for measuring creativity in LLMs. The task is simple: generate 10 words as different from each other as possible, then score by average pairwise cosine distance (GloVe 840B). Higher distance = higher creativity. Several high-profile studies have used DAT to claim that GPT-4 is "more creative than most humans" (Bellemare-Pepin et al., 2025) and similar headlines.

But there's a problem. Nakajima et al. (2026) showed that DAT is fundamentally invalid for LLMs: a task-agnostic random baseline (randomly sampled nouns) outscores every model. Random words are trivially distant from each other. DAT conflates novelty with noise — it can't distinguish a model that makes genuinely creative leaps from one that outputs unrelated garbage.

This raises an interesting empirical question: **even if DAT is theoretically invalid, does it still correlate with actual creative ability?** If DAT scores predict Chatbot Arena Creative Writing rankings — where real humans judge creative output — then maybe the theoretical objection doesn't matter in practice. And if it doesn't correlate, that's a concrete demonstration that DAT is misleading.

## What We're Doing

We evaluate 49 LLMs (spanning Arena Elo 1090–1496) on three psycholinguistic creativity metrics, then correlate each with Chatbot Arena Creative Writing (CW) rankings:

1. **DAT** (Olson et al., 2021) — 120 trials per model. Generate 10 maximally different nouns. Score = mean pairwise cosine distance (GloVe 840B) of first 7 valid words × 100.

2. **CDAT** (Nakajima et al., 2026) — 50 cue words per model. Generate 10 words that are diverse from each other yet associated with a cue. Novelty scored by SBERT pairwise distance; appropriateness scored by SBERT similarity to cue. Models must pass an appropriateness gate (statistically exceed random baseline) before novelty counts.

3. **PACE** (Qiu & Hu, 2025) — 50 seed words per model, 3 parallel 20-word association chains per seed. Score = average semantic distance (FastText) across chain positions.

All three are fully automatic (no LLM-as-judge), grounded in psycholinguistic creativity theory, and have been individually validated. But nobody has run all three on the same set of models and compared their predictive power against a common external benchmark.

## The Claims

**Claim 1: DAT does not meaningfully predict creative writing ability in LLMs.**
If DAT scores show low or no correlation with Arena CW (rho < 0.3 or p > 0.05), while CDAT or PACE do correlate, this demonstrates that the theoretical invalidity identified by Nakajima et al. has practical consequences. Researchers should stop using DAT for LLM evaluation.

**Claim 2: CDAT's appropriateness constraint makes it a more valid creativity metric for LLMs.**
The CDAT adds a single modification to DAT: words must be associated with a cue (appropriateness) in addition to being diverse (novelty). If CDAT correlates more strongly with Arena CW than DAT does, this validates the dual-process view of creativity (novelty + appropriateness) and shows the gate matters.

**Claim 3: Comparing DAT, CDAT, and PACE reveals whether these metrics measure the same thing.**
If all three inter-correlate strongly AND predict Arena CW, they tap a shared creativity construct. If they diverge (e.g., PACE correlates with CW but DAT doesn't), they capture different facets. Either finding is informative.

## Why Arena Creative Writing?

Chatbot Arena CW is the closest thing we have to ground truth for LLM creative ability. It's based on head-to-head human preferences on open-ended creative writing prompts, with tens of thousands of votes per model. It's not perfect — it conflates writing quality with creativity, and preferences are noisy — but it's the best available external signal.

The PACE paper (Qiu & Hu, 2025) established the precedent: they showed PACE correlates with Arena CW at rho=0.739. We follow their methodology (Spearman rank correlation + bootstrap validation) but extend it to a three-way comparison with more models.

## Key Confound

Larger/better models may simply score higher on everything — both creativity metrics and Arena CW. If our correlations just reflect "general capability," we haven't shown anything interesting. To address this:

- We correlate with Arena Overall (not just CW). If our metrics correlate equally with Overall and CW, they're measuring general ability, not creativity.
- We compute partial correlations: metric vs Arena CW, controlling for Arena Overall. If significant signal remains after partialing out general capability, we've isolated a creativity-specific signal.
- We include reasoning-focused models (o3, o4-mini, QwQ, DeepSeek R1) that score high on Arena Overall but may not score proportionally high on CW. These models help break the general-capability confound.

## Model Selection

49 models across 10 providers, spanning the full Arena score range:

| Tier | Arena Overall | Count | Examples |
|------|-------------|-------|---------|
| Frontier | 1430+ | 8 | Claude Opus 4.6, GPT-5.4, Sonnet 4.6, Gemini 2.5 Pro |
| Strong | 1350–1430 | 12 | GPT-5, o3, GPT-4.1, DeepSeek R1, Sonnet 4 |
| Mid | 1280–1350 | 17 | Qwen3-235B, Gemma 3, Llama 4, Claude 3.5 Haiku |
| Lower | 1200–1280 | 8 | Qwen 2.5 72B, Llama 3.1 70B, Phi-4 |
| Weak | <1200 | 4 | GPT-3.5 Turbo, Llama 3.1 8B, Mistral 7B |

This is 19 more models than PACE evaluated (30) and with a wider score range.

## What Would Make This Paper Interesting

**Best case:** DAT doesn't correlate with Arena CW (or correlates negatively), CDAT does, and PACE does. Clean story: DAT is broken for LLMs, CDAT fixes it with one simple modification, PACE captures something complementary. The appropriateness constraint is the key insight.

**Interesting alternative:** All three correlate with CW, but after controlling for Arena Overall, only CDAT and PACE retain signal. DAT's correlation was just a proxy for model size.

**Also interesting:** PACE and CDAT correlate with each other but capture different things — one predicts CW better for certain model families (e.g., PACE is better for reasoning models, CDAT for chat models). This would suggest creativity is multi-faceted and no single metric suffices.

**Null result (still publishable for ICCC):** None of them correlate with CW after controlling for general capability. This would suggest that word-level associative tasks are fundamentally disconnected from paragraph-level creative writing — a meaningful negative finding for the computational creativity community.

## Experimental Setup

- All models queried via OpenRouter (single API, consistent interface)
- **Main run (CDAT, PACE):** Temperature 0.0 (following PACE methodology — captures intrinsic associative behavior; variance comes from cue/seed diversity, not stochastic sampling)
- **DAT run:** Temperature 1.0 with 120 trials per model — DAT lacks built-in variance (one prompt only), so we need stochastic sampling. Temp=1.0 is what Bellemare-Pepin et al. (2025) used for their GPT-4 DAT benchmark.
- CDAT: 50 cue words spanning diverse semantic domains
- PACE: 50 seed words, 3 chains of 20 words each per seed
- Scoring: GloVe 840B (DAT, per original Olson protocol), SBERT all-mpnet-base-v2 (CDAT, per Nakajima protocol), FastText crawl-300d-2M (PACE, per Qiu & Hu protocol)
- Correlation: Spearman rho + 500-iteration bootstrap for CIs and significance ratios
- Partial correlations controlling for Arena Overall

### Temperature Limitations (Methodological Note)

Some models do not respect temperature settings via OpenRouter and produce deterministic output regardless of the requested value. We tested every model in our set across temperatures 1.0, 1.5, and 2.0 with the full DAT prompt:

**Models that produce varied output at temp=1.0** (44/49): all Claude 4.x models, GPT-5.4 family, GPT-4.x family, Gemini 2.x, DeepSeek V3/R1, most Qwen, Llama, Mistral, Cohere, Gemma, etc. These are run with 120 trials at temp=1.0 for the DAT.

**Models that vary only at higher temperatures** (2/49):
- `openai/gpt-5-mini` — varied at temp=1.5, deterministic at 1.0 and 2.0
- `qwen/qwen3-32b` — varied at temp=1.5, deterministic at 1.0 and 2.0

**Effectively fixed-temperature models** (3/49) — return identical output across all tested temperatures (0, 1, 1.5, 2):
- `openai/gpt-5`
- `openai/gpt-5-nano`
- `openai/o3-mini`

For the fixed-temperature models, DAT yields a single deterministic point estimate per model. This is consistent with how the PACE paper handled o3-mini (they noted its API fixes temperature at 1). These models retain a single DAT score in the analysis but cannot contribute to within-model variance estimates. We flag this as a limitation in the paper.

This is also a good reminder of why DAT-as-an-LLM-eval is methodologically fragile: a metric that depends on stochastic sampling can't be applied uniformly when models have heterogeneous temperature behavior. CDAT and PACE sidestep this issue by deriving variance from input diversity (50 cues, 50 seeds) rather than stochastic sampling.

## Timeline

- 2026-04-11: Implementation complete, full eval running (49 models × all 3 tasks)
- Next: Score responses, run correlation analysis, draft paper
