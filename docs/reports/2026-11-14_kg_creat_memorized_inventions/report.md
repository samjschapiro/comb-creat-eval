# Memorised inventions in Kombine

*2026-11-14 · kg_creat track · rebuttal analysis (reviewer En17 Q1) · companion to the inventive-multiples
memo (2026-09-01) and the corpus contamination check (2026-11-14)*

**The question.** An *inventive multiple* is two models independently inventing the same concept. A
*memorised invention* is a model "inventing" a concept that already exists in human text. How often does
it happen? Which models, which inputs and which task? Does it explain the multiples?

**Answer.**
- **Rare.** About 1 in 50 inventions is a memorised concept (2.1%, against a 0.4% cross-item null).
- **Equally common on both tasks.**
- **Concentrated in older open-weight models.** Llama-3.3-70B, Qwen-2.5-72B, Phi-4 and Llama-4-Maverick
  lead, with Claude Sonnet 4.5 the exception among closed models. 15 of the 35 models never do it.
- **More likely when the inputs co-occur in text.**
- **Unrelated to convergence.** Memorised inventions are *less* often part of an inventive multiple, so
  models converging on the same invention is not shared recall of a known concept.

## Definition

An analogy invention h or blend c' is **memorised** when, in the 4.6T-token OLMo 2 pretraining mix
(infini-gram):
1. **attested:** its name occurs verbatim at least 10 times;
2. **in context:** the name co-occurs within 100 tokens with at least one of the item's two inputs;
3. **described:** at least 2 of the invention's property objects co-occur with the name within 100
   tokens. Property objects come from the projection images for analogy, and from triples whose subject
   is the concept for blending, excluding anchor echo.

In other words, the concept exists in text, near its inputs, with the properties the model gave it.

**Strict:** condition 2 holds for *both* inputs.

**Cross-item null**, as in the multiples memo: the same test, but each name is paired with a *different*
item's inputs and a different invention's properties (same task, seed 0). At 4.6T tokens almost any two
common strings share a paragraph somewhere, and the null measures that.

**Specific:** memorised with its own context, but not under the null. This removes stock words that
co-occur with anything.

**Counting** follows the contamination check: nonzero on the 4.6T index means documented; an exact zero
counts; a sampled zero falls back to the exact 9.2B Dolma sample. Script:
`src/kg_creat/scripts/memorized_inventions.py`. Output: `data/kg_creat/corpus_novelty/memorized_inventions.json`.
About 6,600 new queries; no API spend.

## Claims

1. **Memorised inventions are rare.** 43 of 2,070 inventions are memorised: 2.1%, against 0.4% under
   the cross-item null. 40 are specific, and 1.3% meet the strict definition.
2. **Task makes no difference.** Analogy 2.1% vs blending 2.0%, paired over the 30 shared anchor pairs
   (Wilcoxon p = 0.96).
3. **Provider and model matter.**
   - The spread between providers exceeds a model-label permutation (p = 0.016).
   - Per model, the memorised rate correlates negatively with the paper composite (ρ = −0.31,
     p = 0.07), and on blending more strongly (ρ = −0.41, p = 0.015).
   - So weaker models fall back on stock concepts.
4. **Familiar inputs invite stock concepts.** Items whose inputs co-occur more in text have more
   memorised inventions (ρ = +0.37 analogy, +0.38 blending; p ≈ 0.04). Neither the entities' prominence
   nor their semantic distance predicts it.
5. **Memorisation does not explain multiples.**
   - No memorised analogy invention is part of an inventive multiple, against 2.7% of other inventions.
   - Memorised blends are in a multiple 19% of the time, against 38% for other blends.
   - Of the 385 blends that are in a multiple, only 1.0% are memorised.
   - Items with more memorised blends have *fewer* multiples (ρ = −0.37, p = 0.04).
   - Convergence happens on new concepts, not on recalled ones.

## Headline

| | n | Name attested (≥ 10×) | In context (null) | **Memorised** (null) | Strict | Specific |
|---|---|---|---|---|---|---|
| Analogy | 1,037 | 48.7% | 12.1% (4.8%) | **2.1%** (0.3%) | 1.2% | 2.1% (22) |
| Blending | 1,033 | 53.9% | 15.2% (2.5%) | **2.0%** (0.5%) | 1.4% | 1.7% (18) |
| All | 2,070 | 51.3% | 13.6% (3.7%) | **2.1%** (0.4%) | 1.3% | 1.9% (40) |

Each condition removes most of what passed the one before:
- Half the names exist as strings.
- About one in seven appears near its inputs, against a 4% chance rate.
- About one in fifty also carries its stated properties in text.

## By provider and model

Mean of per-model rates; providers with at least 2 models enter the permutation test.

| Provider (models) | Analogy | Blending | All |
|---|---|---|---|
| Meta (2) | 16.0% | 6.7% | **11.1%** |
| Microsoft (1) | 6.7% | 5.6% | 6.2% |
| Qwen (2) | 3.3% | 5.0% | 4.2% |
| Moonshot (1) | 3.4% | 3.3% | 3.4% |
| Anthropic (10) | 1.7% | 2.4% | 2.0% |
| DeepSeek (3) | 1.1% | 1.1% | 1.1% |
| OpenAI (7) | 0.5% | 1.4% | 1.0% |
| Google (5) | 0.7% | 0.7% | 0.7% |
| xAI (2) | 0 | 0 | 0 |
| Z.ai (2) | 0 | 0 | 0 |
| Permutation p (between-provider spread) | 0.029 | 0.039 | 0.016 |

**Highest-rate models:** llama-3.3-70b 17% (10 of 58), claude-sonnet-4.5 8% (5 of 60), qwen-2.5-72b 7%,
phi-4 6%, llama-4-maverick 5%.

**Lowest:** 15 of 35 models never produce one. They include the current frontier: opus-5, fable-5,
sonnet-5, sonnet-4.6, gpt-5.6-sol, gpt-5.2, gemini-3.1-pro, gemini-2.5-pro, deepseek-r1, and both Grok and
both GLM models.

**Shared memorised concepts across models:** only two.
- *Moral autonomy* on *Ten Commandments + Free will*, from llama-4-maverick, gpt-4o-mini and qwen-2.5-72b.
- *Revolutionary discourse* on *Language + French Revolution*, from deepseek-chat and llama-3.3-70b.

## By inputs

| Per item (n = 30) | Analogy ρ (p) | Blending ρ (p) |
|---|---|---|
| Inputs' co-occurrence in text (log count) | **+0.37** (0.048) | **+0.38** (0.036) |
| Co-occurrence lift over chance | −0.09 (0.63) | +0.05 (0.79) |
| Prominence (mean log sitelinks) | +0.08 (0.67) | +0.14 (0.46) |
| Semantic distance between inputs | +0.08 (0.70) | −0.14 (0.48) |
| The item's inventive-multiple rate (τ = 2) | +0.06 (0.75) | **−0.37** (0.042) |

**Most-affected items:**
- *Justice + Electricity* (analogy 5, blending 2);
- *Ten Commandments + Free will* (blending 5);
- *Networks + Salt* (analogy 3, blending 2);
- *Christianity + Beauty* (blending 3);
- *Language + French Revolution* (blending 3).

These are pairs with an obvious stock bridge: *regulator*, *moral compass*, *flavour enhancer*, *icon*.

**By the inputs' domain** (an invention counts for both of its inputs' domains):
- **Highest:** ideas 6.2%, politics 4.3%, philosophy 3.6%.
- **Lowest:** biology 0.4%, nature 0.

Abstract-concept inputs invite existing abstract nouns. The domain cells are small (n = 68–417).

## Outcomes of memorised vs other inventions

| | Passes utility | Emergent originality (within-item difference) | In an inventive multiple |
|---|---|---|---|
| Analogy, memorised (n = 22) vs other | 54.5% vs 54.1% | 0.483 vs 0.506 (−0.024, 15 items, p = 0.12) | **0% vs 2.7%** |
| Blending, memorised (n = 21) vs other | **28.6% vs 44.5%** | 0.461 vs 0.470 (−0.006, 10 items, p = 0.38) | **19.0% vs 37.6%** |

- **Memorised blends fail the generic-space check more often.** A stock concept (*Moral Compass*,
  *Iconography*) usually fits one input's frame rather than both.
- **Memorised inventions score slightly lower on originality,** but not significantly, with this few
  of them.

## Examples (specific memorised inventions, most frequent names first)

| Inputs | Task | Name (occurrences in 4.6T) | Model | Properties documented with the name |
|---|---|---|---|---|
| Opera + Documentary film | analogy | *Treatment* (377M) | llama-3.3-70b | 2/2: interviews, screenwriters |
| Christianity + Beauty | blend | *Icon* (102M) | sonnet-4.5 | 2/5: consecration, radiant form, … |
| Christianity + Beauty | blend | *Iconography* (2.7M) | opus-4.7 | 4/5: golden ratio, sacrament, harmonious form, faces |
| Ten Commandments + Free will | blend | *Moral Compass* (2.4M) | llama-3.3-70b | 5/5: divine law, personal autonomy, … |
| Cinema + Evolution | analogy | *Mimicry* (2.2M) | phi-4 | 2/2: predator–prey dynamics, survival advantage |
| Frida Kahlo + Bob Dylan | analogy | *Concept Album* (975k) | llama-3.3-70b | 2/2: poetic lyrics, social justice |
| Justice + Electricity | analogy | *voltage regulator* (785k) | deepseek-v3 | 2/2: power flow, stability |
| Charlie Chaplin + Surrealism | analogy | *The Sleepwalker* (369k) | opus-4.8 | 2/2: automatism, unconscious |
| Constellations + Cricket | blend | *Star Field* (201k) | sonnet-4.5 | 4/6: innings, runs, named formations, seasonal cycles |
| Networks + Salt | analogy | *Flavor Enhancer* (176k) | llama-3.3-70b | 2/2: flavor, taste |

**For contrast, the paper's showcase inventions are not memorised.** *Stellar moksha* never appears
verbatim. *Liquid franchise* (38–45 occurrences) and *narrative species* (912) do occur as strings, but not
near their inputs together with the properties the models gave them. This is exactly the distinction the
definition draws.

A typical memorised invention is not a creative blend. It is an existing concept from the vicinity of one
input (*voltage regulator* for Justice + Electricity), relabelled as the answer.

## Caveats

- **Proxy corpus and exact matching.** The corpus stands in for the models' unknown training data, and
  exact string matching misses a familiar concept under a new name. The rates are lower bounds on
  "reusing a known concept", not on "relying on prior knowledge", which every answer does.
- **The null is one random pairing per invention.** The excess over the null (1.7 points) is the robust
  number. Individual flags carry some noise; *Treatment* passes as a film term, correctly but
  generically.
- **Few positives** (43), so the per-provider and per-domain rates are coarse. The Meta lead rests on two
  models: llama-3.3-70b alone has 10 of the 43.
- **Model-level p-values come from a permutation over 35 models.** Providers are confounded with model
  generation and capability: the memorised rate falls with the composite.

## Rebuttal sentence

> We also searched for *memorised inventions*: invented concepts already documented near their inputs,
> with the properties the model assigned, in 4.6T tokens of pretraining text.
> - Only 2.1% of inventions qualify, against 0.4% under a cross-item null, at the same rate in analogy
>   and blending.
> - They come mostly from older open-weight models (Llama-3.3-70B 17%, Qwen-2.5-72B 7%), and the rate
>   falls with a model's Kombine score (blending ρ = −0.41).
> - They are *less* likely to be part of an inventive multiple (blending 19% vs 38%; analogy 0% vs
>   2.7%), so the convergence we report is on new concepts, not shared recall of existing ones.
