# How Blends Fail: Faking the Shared Slot

*2026-09-03, updated 2026-09-05 for the 30-model pool · kg_creat track · analysis memo*

**Question.** A conceptual blend should *fuse* two inputs into one new concept. What separates a real
fusion from a fake one, how do frontier models fake it, and is the failure the model's or the anchors'?

**Sample frame.** 885 scored blends: 30 models × 30 anchor pairs, one blend per pair, temperature 0.9 (15 draws are missing or unparseable). Each is
judged by a 3-judge panel (Claude Haiku 4.5, GPT-5.4, o3), majority vote. Numbers are corpus-wide unless
a smaller *n* is given; the paired examples illustrate, and are not a random sample.

## The criterion: one property both inputs organize

A genuine blend maps *both* input structures onto the **same property** of the invention — one slot
carries both frames. The benchmark's worked example (\Cref{tab:examples}) is the *liquid franchise*
(democracy + banking): its one "allocates" slot is fed by BOTH democracy's "allocates votes" AND
banking's "allocates credit," fusing into "allocates transferable vote-shares." Listing democracy's
properties beside banking's is concatenation, not fusion.

The task makes this explicit. Each emitted triple is tagged `u` (from the first input only), `v` (from
the second only), `uv` (one property both inputs organize), or `emergent` (true of the blend, of neither
input alone). A panel verifies each `uv` **by its content, not its label**, and assigns a scope:
**1** = no genuine shared slot, **2** = a real shared slot, **3** = a shared slot plus emergent structure.
Coherence is scored *separately* (does the blend read as one usable concept?), so a blend can be
double-scope yet incoherent; the examples below are chosen to pass both.

**Every blend claims a shared slot** — 884 of 885 emit at least one `uv` triple. But only **52% survive
verification** (345 at scope 3, 114 at scope 2); the other **48% (426) are scope 1** — a `uv` slot only
one input actually organizes. The whole game is whether the claimed slot is real.

## What a real fusion looks like

**Adam Smith + Bacteria → fable-5, "quorum market" (scope 3, coherent).**
Generic space: *a decentralized population of self-interested agents whose local exchanges produce global
order without any central planner.*

```
u        (quorum market, guided by, invisible hand)
v        (quorum market, grows by, binary fission)
uv       (quorum market, sets prices via, quorum signals)        <- both inputs organize this slot
uv       (quorum market, divides labor among, specialized cells) <- both inputs organize this slot
emergent (quorum market, spreads contracts via, gene transfer)
emergent (quorum market, evolves prices by, natural selection)
```
Two slots carry both frames: price-setting is fed by BOTH the invisible hand AND bacterial
quorum-sensing; division of labor by BOTH Smith's specialization AND cellular differentiation. The
generic space genuinely organizes both inputs, and the result is a coherent idea — a bacterial colony
run as a market. This pair is the easiest in the set: 29 of 30 models fuse it.

## Two ways to fail, each holding one thing fixed

**Same anchors, different model.** On *Cinema + Evolution*, gpt-5.2 fuses and gpt-5-mini fakes it.

*gpt-5.2, "film species" (scope 3, coherent)* — generic space *a lineage of replicable patterns
expressed in instances and shaped by selection among variants:*
```
u        (film species, shown in, theater)
v        (film species, has, fitness)
uv       (film species, mutates via, editing)      <- variation: editing IS mutation
uv       (film species, selected by, audience)     <- selection: box office IS fitness
emergent (film species, reproduces as, sequels)
emergent (film species, goes extinct as, lost cut)
```
Both slots are genuinely double: editing plays the role of mutation, the audience the role of the
selecting environment. A coherent concept — a film as a species.

*gpt-5-mini, "cinevolution" (scope 1)* — same pair, generic space *a population of discrete heritable
variants selected by external evaluators:*
```
u        (cinevolution, uses, shot montage)
v        (cinevolution, evolves via, mutation and recombination)
uv       (cinevolution, selects variants for, cultural transmissibility)  <- claimed shared
emergent (cinevolution, produces, adaptive franchise genomes)
```
The panel returns `shared_properties: []`: *"the generic space is unbalanced — 'external evaluators' is
natural for cinema (audiences, critics), but evolution is driven by environmental fitness, not external
evaluation."* The one `uv` slot is organized by cinema and merely asserted of evolution. Same anchors as
"film species"; the difference is the model.

**Same model, different anchors.** The very model that fused "film species" fakes a harder pair.
*gpt-5.2, "seed station" on Rice + Radio (scope 1)* — generic space *a carrier medium modulated to
deliver content to receivers:*
```
u        (seed station, uses, rice grain)
v        (seed station, uses, radio carrier)
uv       (seed station, modulates, carrier medium)   <- only radio modulates; rice does not
emergent (seed station, stores broadcasts in, DNA)
emergent (seed station, duplicates broadcasts by, replanting)
```
The panel: *"radio perfectly exemplifies a modulated carrier; but rice does not modulate — it reproduces
through inheritance."* A one-sided slot dressed as a shared one — the same failure as "cinevolution,"
now from a strong model, because the anchors resist a real slot.

## The anatomy of a failure

Across the 426 scope-1 blends, the panel's stated reason is a **one-sided / unbalanced schema in 266
(62%)** — the slot is natural for one input and forced onto the other, as in "cinevolution" and "seed
station." **Categorical absurdity** — grafting one input's literal properties until the artifact stops
being what it is — is a distinct and smaller mode, **190 (45%)** (categories overlap; these are themes in
the judges' free-text reasons, not a separate labeled pass). These two figures are **not comparable to
the 94% / 11% this memo reported before**: that split came from an unrecorded one-off scan, while these
come from keyword patterns written down in `analyze_blend_integration.py`. Faking the slot, not grafting nonsense, is
the common way to fail.

## The failure is the model's, not the anchors'

**All 30 anchor pairs contain both a genuine fusion (some model at scope 3) and a fake (some model at
scope 1).** On identical inputs the outcome flips with the model, so scope measures a model's fusion
skill rather than the pair's difficulty. Difficulty still sets the rate: the hardest pairs share little
structure (X-rays + Nuclear fission, 19/21 fake; Photosynthesis + Bread and Mount Everest + The light
bulb, 17 fake), the easiest invite a real slot (Adam Smith + Bacteria, 1 fake; Cinema + Evolution, 6),
but no pair is a pure trap or a pure gift.

Genuine-fusion rate (scope ≥ 2), best to worst: **gpt-5.6-sol 87%**, claude-fable-5 73%,
gemini-3.7-flash 67%, gpt-5 67%, gpt-5.2 67%, … , gpt-4o-mini 33%, qwen-2.5-72b 33%, **phi-4 17%**.
The 17–87% spread does not track raw capability — blending is a distinct skill.

## Limitations

- **Scope and coherence are 3-judge majority votes**, not ground truth. Panel agreement is
  fair-to-good (ICC 0.48–0.65, \Cref{tab:icc}) but has not been checked against human raters on this
  tagged format; the panel's coherence bar admits fantastical-but-consistent blends.
- **The 62% / 45% failure split is from keyword themes** in the judges' explanations, not an independent
  labeling pass; read it as indicative, and note the two modes overlap. The patterns are in the script.
- **The examples are chosen to show the contrast**, not sampled at random; the 52% / 48% split is the
  corpus-wide measure behind them.

## Reproduce

```
.venv/bin/python -m src.kg_creat.scripts.analyze_blend_integration
```

Every corpus-wide number above comes from that script and is written to
`data/kg_creat/kombine_test30/analysis/blend_integration.json`.

Blends and their tagged structure are in `data/kg_creat/kombine_test30/responses/*/responses.json`; the
scored `path_scores.json` under `data/kg_creat/kombine_test30/scores/*/` carries `blend_integration`
(scope 1/2/3), `blend_utility` (coherence), `shared_properties` (the panel's named slots), and per-judge
explanations. The per-pair genuine-vs-fake dumps are one-off scans over those files.
