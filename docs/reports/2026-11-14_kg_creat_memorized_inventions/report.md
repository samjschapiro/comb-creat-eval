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
   - **Hand audit** (see "Audit of all 43"): 35 are reuses of a known concept, 8 of them a concept from
     one input's domain relabelled. 8 are string coincidences.
   - **Audited rate:** about 1.7%.
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

## Examples in detail

**Notation.**
- **Corpus columns:** ✓ = the property's object co-occurs with the name within 100 tokens in the 4.6T
  corpus; ✗ = it does not; ? = inconclusive (sampled zero, and absent from the 9.2B sample).
- **Counts** are verbatim occurrences of the name.
- **Analogy inventions** are shown as source triple → image triple. The relation is copied from the
  source by construction, so only the object is the model's choice.
- **Blends** show the tag the model gave each triple: *u* / *v* = inherited from one input, *uv* = from
  both, *emergent* = new to the blend.

### 1. A known concept reused as the blend: *Icon*

*Christianity + Beauty* → **Icon** (102M occurrences), claude-sonnet-4.5. It passes the panel.

Generic space: *"a transformative revelation of transcendence through contemplatable manifest form"*.

| Tag | Triple | Corpus |
|---|---|---|
| u | Icon — requires — consecration | ✓ |
| v | Icon — exhibits — compositional harmony | ? |
| uv | Icon — reveals through — radiant form | ? |
| uv | Icon — transforms — beholders | ✓ |
| emergent | Icon — accrues sanctity through — visible aging | ? |

- The religious icon *is* already the place where Christianity and beauty meet, so the model retrieved a
  ready-made blend rather than building one.
- claude-opus-4.7 produced its sibling *Iconography* (2.7M) on the same item: *summons devotion through
  harmonious form* ?, *promises salvation* ✓, *obeys golden ratio* ✓, *treats symmetry as sacrament* ✓,
  *canonizes faces* ✓. The panel *rejected* it.
- Both are judged valid or invalid on their merits. Memorisation is a different axis from validity.

### 2. A stock concept that fits one input: *Moral Compass*

*The Ten Commandments + Free will* → **Moral Compass** (2.4M), llama-3.3-70b. It fails the panel.

Generic space: *"a guiding principle for decision making"*.

| Tag | Triple | Corpus |
|---|---|---|
| uv | Moral Compass — influences — behavior | ✓ |
| u | Moral Compass — is based on — divine law | ✓ |
| v | Moral Compass — respects — personal autonomy | ✓ |
| emergent | Moral Compass — evolves through — self-reflection | ✓ |
| uv | Moral Compass — balances — freedom and responsibility | ✓ |

- **Every property is documented** with the name: the textbook memorised invention.
- **The generic space is too thin** to be instantiated by both inputs, which is why it fails.
- **The item attracts this failure.** Five models land in the same neighbourhood: *Ethical Compass*
  (gemini-2.5-flash), and *Moral Autonomy* three times (llama-4-maverick, qwen-2.5-72b, gpt-4o-mini).
  The only one that passes is gpt-4o-mini's *moral autonomy*, which *prescribes ethical guidelines* ✓,
  *exercises individual choice* ✓, *balances duty and freedom* ✓ and *enhances personal
  responsibility* ✓.

### 3. An analogy that maps to an existing concept in the target domain: *voltage regulator*

*Justice + Electricity* → **voltage regulator** (785k), deepseek-chat-v3. Projected source: *appeals
court*.

| Source triple | → | Image triple | Corpus |
|---|---|---|---|
| appeals court — reviews — verdict | → | voltage regulator — reviews — power flow | ✓ |
| appeals court — ensures — fairness | → | voltage regulator — ensures — stability | ✓ |

- **The mapping is sensible.** But its image is a component that already exists, so the "invention"
  is a relabelled real device.
- **The same item produced four more** existing electrical concepts: *Dielectric Breakdown*
  (gemini-3-flash; from *Perjury*: violates Insulation ✓, corrupts Conduction ✓), *Power Surge
  Protector* (llama-3.3), *Negative Power* (kimi-k2) and *Recalibration* (phi-4).
- **Justice + Electricity is the most-affected item.** The target domain has a ready vocabulary of
  control devices onto which "fairness / correction" maps.

### 4. A genuine cross-domain analogy that lands on an existing term: *Concept Album*

*Frida Kahlo + Bob Dylan* → **Concept Album** (975k), llama-3.3-70b. Projected source: *Self Portrait*.
It passes.

| Source triple | → | Image triple | Corpus |
|---|---|---|---|
| Self Portrait — explores — Identity | → | Concept Album — explores — Social Justice | ✓ |
| Self Portrait — features — Vibrant Colors | → | Concept Album — features — Poetic Lyrics | ✓ |

The mapping (a painter's self-portrait ↦ a songwriter's concept album) is a reasonable analogy. Its
image is simply an existing genre, with generic one- or two-word properties.

### 5. A familiar name with new content: *Dream factory*

*Charlie Chaplin + Surrealism* → **Dream factory** (58k), gpt-6-astra-flex. It passes.

| Tag | Triple | Corpus |
|---|---|---|
| u | Dream factory — employs — The Tramp | ✓ |
| u | Dream factory — uses — assembly lines | ✓ |
| uv | Dream factory — drives production through — involuntary gestures | ✗ |
| uv | Dream factory — mass produces — dream objects | ✗ |
| emergent | Dream factory — turns sleep into — overtime | ? |
| emergent | Dream factory — turns repression into — machine breakdowns | ✗ |

- *Dream factory* is a long-standing nickname for Hollywood, so the name and the Chaplin side are
  documented.
- **The blend itself is not documented:** none of the double-scope or emergent properties are.
- It passes the definition on its two *u*-side properties. This is the kind of case where
  name-plus-properties overstates memorisation. The model took a known label and built something new
  under it.

### 6. False positives: the name exists, the concept doesn't

| Inputs | Invention (count) | Model | What the string means in the corpus | Properties that matched |
|---|---|---|---|---|
| Justice + Electricity, blend | *current court* (106k) | qwen3-max | "the current court" | adjudicates disputes ✓, conducts charge ✓; emergent properties ✗ / ? |
| Mount Everest + The light bulb, analogy | *power window* (130k) | gpt-5 | car power windows | enables illumination ✓, used after connection ✓, signaled at lamp base ✓ (from *weather window*) |
| Justice + Electricity, analogy | *Recalibration* (552k) | phi-4 | the ordinary word | powers Machinery ✓, transforms Energy ✓ |
| Hinduism + Gravity, blend | *Karmic Gravity* (347) | opus-4.8 | a rare New-Age phrase | cycles souls through samsara ✓, pulls soul by karmic mass ✓ |

These pass the definition because their names are ordinary strings and their properties are common
nouns. The analogy *weather window → power window* is the model's own construction; the car part
already has that name by coincidence.

## Audit of all 43

Our reading of what each flagged invention is:
- **K**: a known concept reused as the answer (27);
- **K-1**: a known concept from *one* input's domain, relabelled (8);
- **F**: a false positive, i.e. the string exists but the concept is the model's own (8).

The counts are one author's judgement, to be checked by a co-author before quoting.

| Item | Task | Model | Name (count) | Properties (relation — object, corpus) | Label |
|---|---|---|---|---|---|
| Christianity + Beauty | blend | sonnet-4.5 | Icon (102M) | requires consecration ✓; exhibits compositional harmony ?; reveals through radiant form ?; transforms beholders ✓; accrues sanctity through visible aging ? | K |
| Christianity + Beauty | blend | opus-4.7 | Iconography (2.7M) | summons devotion through harmonious form ?; promises salvation ✓; obeys golden ratio ✓; treats symmetry as sacrament ✓; canonizes faces ✓ | K |
| Christianity + Beauty | blend | opus-4.6 | Gloria (7.8M) | transforms perceiver through devoted attention ?; offers redemption ✓; manifests in sensory form ?; makes right perception salvific ✓; makes kitsch heresy ✓ | K (theological "glory") |
| Christianity + Beauty | analogy | llama-3.3 | Aesthete (283k) | embodies Taste ✓; serves Culture ✓ (from *Saint*) | K |
| Social contract + Prayer | blend | sonnet-4.5 | Covenant (21M) | binds through mutual obligations ✓; legitimates authority ✓; specifies explicit terms ✓; requires faith ✓; treats violations as crime and sin ?; established by sworn oath ✓ | K |
| Social contract + Prayer | analogy | llama-3.3 | Litany (3.6M) | outlines Blessings ✓; establishes Rituals ✓ (from *Constitution*) | K |
| Ten Commandments + Free will | blend | llama-3.3 | Moral Compass (2.4M) | all five ✓ (see above) | K |
| Ten Commandments + Free will | blend | gemini-2.5-flash | Ethical Compass (58k) | possesses inherent directives ✗; allows alternative paths ✗; guides towards moral integrity ✓; generates self-imposed accountability ✗; calibrates based on personal conviction ✓ | K |
| Ten Commandments + Free will | blend | gpt-4o-mini | moral autonomy (68k) | prescribes ethical guidelines ✓; exercises individual choice ✓; balances duty and freedom ✓; enhances personal responsibility ✓ | K |
| Ten Commandments + Free will | blend | qwen-2.5-72b | Moral Autonomy (69k) | provides guidelines ✓; enables choice ✓; balances commandments and freedom ✗; adapts to situational context ✗ | K |
| Ten Commandments + Free will | blend | llama-4-maverick | Moral Autonomy (69k) | is guided by principles ✓; involves choice ✓; balances obligation and desire ✗; develops personal responsibility through moral action ✗ | K |
| Language + French Revolution | blend | llama-3.3 | Revolutionary Discourse (12k) | uses semantics ?; involves power struggle ?; shapes public opinion ✓; creates new social norms ✗; employs rhetorical devices ✗; leads to social upheaval ✓ | K |
| Language + French Revolution | blend | deepseek-chat | Revolutionary Discourse (12k) | is composed of signs ?; targets oppressive regimes ✓; transforms meaning ✓; propagates through collective action ? | K |
| Language + French Revolution | blend | llama-4-maverick | Revolutionary Rhetoric (66k) | uses metaphor ✓; mobilizes masses ?; frames social identity ?; reshapes public discourse ✓; creates new solidarity ✗ | K |
| Language + French Revolution | analogy | qwen-2.5-72b | Revolutionary Manifesto (12k) | structures Rebellion ✓; guides Liberty ?; enables Progress ✓ (from *Grammar*) | K |
| Inflation + The Constitution | blend | gpt-5-mini | Monetary Constitution (7.6k) | defines unit of account ✓; sets legitimate change procedure ✗; alters money supply ✓; allocates amendment authority ✗; entrenches non-dilutable purchasing rights ✗ | K (Buchanan) |
| Inflation + The Constitution | blend | qwen-2.5-72b | Economic Constitution (11k) | regulates value ?; ensures rights ✓; distributes purchasing power ✓; adapts to economic shifts ✗ | K (ordoliberal) |
| Inflation + The Constitution | analogy | llama-3.3 | Constitutional Safeguard (12k) | is implemented by Supreme Court ✓; aims to stabilize Civil Rights ✓ (from *QE*) | K |
| Vaccines + Ethics | analogy | llama-3.3 | Collective Conscience (195k) | require Moral Education ✓; protect Minority Rights ✓ (from *Herd Immunity*) | K (Durkheim) |
| Frida Kahlo + Bob Dylan | analogy | llama-3.3 | Concept Album (975k) | explores Social Justice ✓; features Poetic Lyrics ✓ | K |
| Opera + Documentary film | analogy | llama-3.3 | Treatment (377M) | informs Interviews ✓; written by Screenwriters ✓ (from *Libretto*) | K (film treatment) |
| Charlie Chaplin + Surrealism | blend | gpt-6-astra-flex | Dream factory (58k) | see above | K, new content |
| Charlie Chaplin + Surrealism | analogy | opus-4.8 | The Sleepwalker (369k) | created by Surrealism; performed via automatism ✓; conveys unconscious ✓ (from *The Tramp*) | K |
| Networks + Salt | blend | opus-4.7 | Ionic Network (1.4k) | has nodes and edges ?; forms crystal lattice ✓; connects nodes via ionic bonds ✓; tunes percolation by solvent dielectric ✗; dissolves into disconnected ion cloud ✗; recrystallizes on evaporation ✓ | K (chemistry) |
| Networks + Salt | blend | kimi-k2 | crystalline lattice (64k) | has nodes ✓; forms via ionic bonding ✓; maintains structure through local neighbor interactions ✗; dissolves at specific threshold density ✗ | K |
| Cinema + Evolution | analogy | phi-4 | Mimicry (2.2M) | is used for Survival Advantage ✓; transforms Predator-Prey Dynamics ✓ (from *CGI*) | K |
| The blue whale + The mattress | analogy | llama-4-maverick | rotation schedule (48k) | is influenced by comfort preference ✗; affects mattress wear ✓; is tracked by users ✓ (from *migration pattern*) | K (mattress rotation) |
| Justice + Electricity | analogy | deepseek-v3 | voltage regulator (785k) | see above | K-1 |
| Justice + Electricity | analogy | gemini-3-flash | Dielectric Breakdown (65k) | violates Insulation ✓; corrupts Conduction ✓ (from *Perjury*) | K-1 |
| Justice + Electricity | analogy | llama-3.3 | Power Surge Protector (7.6k) | prevents Overload ✓; is installed by Electricians ✓ (from *Vaccine*) | K-1 |
| Justice + Electricity | analogy | kimi-k2 | Negative Power (191k) | restores Load ✓; requires Sink ?; measures Demand ✓ (from *Restitution*) | K-1 |
| Mount Everest + The light bulb | analogy | qwen-2.5-72b | power pack (604k) | supplies electricity ✓; enables illumination ✓ (from *oxygen tank*) | K-1 |
| Networks + Salt | analogy | llama-3.3 | Flavor Enhancer (176k) | directs Flavor ✓; optimizes Taste ✓ (from *Router*) | K-1 |
| Don Quixote + Pi | analogy | sonnet-4.5 | rational approximation (52k) | grounds Pi; simplifies calculation ✓; provides finite precision ✓ (from *Sancho Panza*) | K-1 |
| Rice + Radio | analogy | sonnet-4.5 | Premodulation (1.7k) | precedes Modulation ✓; drives Interference ✓; migrates into Carrier ✓; prevents Reception ? (from *Parboiling*) | K-1 |
| Justice + Electricity | blend | qwen3-max | current court (106k) | see above | F |
| Mount Everest + The light bulb | analogy | gpt-5 | power window (130k) | see above | F |
| Justice + Electricity | analogy | phi-4 | Recalibration (552k) | powers Machinery ✓; transforms Energy ✓ (from *Reform*) | F |
| Hinduism + Gravity | blend | opus-4.8 | Karmic Gravity (347) | cycles souls through samsara ✓; curves bodies along geodesics ?; pulls soul by karmic mass ✓; (+ moksha is escape velocity, emergent) | F |
| Constellations + Cricket | blend | sonnet-4.5 | Star Field (201k) | organizes players into named formations ✗; reads positions for tactical advantage ✗; rotates formation with innings ✓; scores runs ✓; marks seasonal cycles ✓; navigates play by stellar positions ✓ | F (astronomy term) |
| Justice + Electricity | blend | phi-4 | Circuity (62k) | transmits electrical current ✓; ensures equitable access ✗; provides power ✓; balances network stress ✗ | F (the word "circuity") |
| Networks + Salt | analogy | fable-5.1 | Seed cluster (7.0k) | reverses Percolation ✓; attracts Nodes ✓; nucleates Edges ?; regrows Networks (from *Seed crystal*) | F |
| Networks + Salt | analogy | opus-4.5 | seed network (4.5k) | templates network growth ✓; determines network topology ✓; placed in connection rich environment ✗ (from *seed crystal*) | F |

## What the properties and relations look like

Comparing the 43 memorised inventions with all the others:

| | Properties per invention | Words per object | Words per relation | Words per name |
|---|---|---|---|---|
| Analogy, memorised (22) | 2.55 | **1.30** | 1.38 | 1.77 |
| Analogy, other (1,015) | 2.60 | 1.79 | 1.45 | 2.15 |
| Blending, memorised (21) | 4.95 | **1.85** | 1.66 | 1.76 |
| Blending, other (1,012) | 5.01 | 2.13 | 1.83 | 2.01 |

**Relations.**
- **In blends, the most frequent are generic:** *balances* (5), *uses* (3), *has*, *requires*,
  *transforms*, *involves*, *creates*, *employs*, *ensures*, *provides*, *adapts to* (2 each).
- **For comparison:** the paper's showcase blends use specific, compound relations, e.g. *tunes
  percolation by*, *turns repression into*, *accrues sanctity through*.
- **In analogies, the relation is inherited from the source path,** so it says more about the source
  than the invention. The memorised ones come from generic sources too: *enables* (3), *prevents*,
  *transforms* (2 each).

**What this means.**
- **The memorised profile:** a short, often one-word name; generic relations; one-word objects
  (*Flavor*, *Taste*, *Culture*, *Rituals*, *Energy*).
- **Partly mechanical.** Short, common strings are more likely to co-occur in text, so the detector is
  biased toward finding them.
- **Partly substantive.** The analogy inventions that cleared the bar mostly have the minimum two
  properties, stated as bare nouns. They are retrievals, not constructions.

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
