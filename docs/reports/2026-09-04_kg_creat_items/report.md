# What the Item Decides — and One Finding That Did Not Survive

*2026-09-04, updated 2026-09-05 for the 30-model pool · kg_creat track · analysis memo · **Claim 1 retracted**, see the box below*

> **Retraction.** This report originally led with "distant anchor pairs produce less original blends"
> (r = −0.47 under `all-MiniLM-L6-v2`). Re-running the identical pipeline under two other sentence
> encoders **flips the sign**. On the 30-model pool: **−0.36** (MiniLM), **+0.06** (bge-small-en-v1.5),
> **+0.34** (multilingual-e5-small) — the same disagreement, on more data.
> Both d(u,v) and originality are cosines in the *same* embedding space, and the result turns out to be
> a property of that space rather than of the models. The mechanism analysis in Finding 2 explained an
> effect that is not there, so it goes with it. Kept below, struck through, because the way it failed is
> the useful part. The leaderboard itself is unaffected — see "Does the encoder decide anything?".

**Question.** Every Kombine number so far has been about models. Turn the table on its side: hold the model out and let the 30 anchor pairs vary. What does the *item* decide, and what is left for the model?

## Claims

1. ~~**Distant anchor pairs produce less original blends.**~~ **Retracted** — see the box above. Under the scoring encoder the correlation is −0.42 (p = .022) and survives 28 of 30 leave-one-out fits, but it does not survive changing the encoder, which is the check that mattered.
2. ~~**The mechanism is polarisation of the element pool.**~~ **Retracted with it.** The mediation was measured in the same space that produced the effect.
3. **Blending belongs to the item; association belongs to the model.** Of the variance in blending surprise, **48% is the anchor pair and 10% is the model**; blending's emergent utility is 7% model. Association inverts it: **59% of its originality variance is the model**, 8% the item. A benchmark table reporting both as "model scores" is reporting two different things.

## Data & sampling

**Frame.** The 30-model Kombine run at temperature 0.9, one draw per model per item: 30 blending items and 30 analogy items (the same anchor pairs), 29 association items (a disjoint set of pairs). Every value below is an **item mean over the 30 models**, so the replication unit is the anchor pair and every correlation has **n = 30** (n = 29 for association); **|r| ≥ 0.36 is p < .05**. Element pools are built from the 900 blends' non-anchor triple elements. d(u,v) is cosine distance between the two anchor strings under `all-MiniLM-L6-v2`.

**What this frame is not evidence about.** 30 pairs drawn from a curated, domain-balanced pool, all cross-domain by construction. The distance range they span is narrow (0.58 to 0.88), so this is a claim about *variation within a set of already-remote pairs*, not about near versus far in general.

## Does the encoder decide anything? (the check that retracted Claim 1)

Surprise and originality are the two embedding-derived dimensions, and both feed the composite, so the whole leaderboard could in principle be an artifact of one encoder's geometry. Recomputing both dimensions from the raw responses under three encoders — `all-MiniLM-L6-v2` (the scoring encoder), `bge-small-en-v1.5`, `multilingual-e5-small` — with judge verdicts held fixed:

| comparison | Spearman | Kendall | top-5 overlap |
|---|--:|--:|--:|
| MiniLM ~ bge-small | **+1.000** | +0.995 | 5/5 |
| MiniLM ~ e5-small | **+0.999** | +0.986 | 5/5 |
| bge-small ~ e5-small | **+0.999** | +0.991 | 5/5 |

No model moves more than one rank — 25 of the 30 do not move at all — and per-task agreement is +0.998 (analogy, blending) to +0.999 (association). **The ranking is not an artifact of the encoder.**

The item-level effect is a different story:

| encoder | d(u,v) ↔ blend originality |
|---|--:|
| all-MiniLM-L6-v2 (scoring encoder) | **−0.36** (p = .051) |
| bge-small-en-v1.5 | +0.06 (n.s.) |
| multilingual-e5-small | **+0.34** (p = .062) |

Same pipeline, same artifacts, same judge verdicts — only the encoder changes, and the sign reverses. A per-model aggregate over 900 blends is robust to the encoder; a 30-point correlation between two cosines in the same space is not.

The variance decomposition (Finding 3) was put through the same check and holds. Item%/model% under each encoder:

| dimension | MiniLM | bge-small | e5-small |
|---|--:|--:|--:|
| association surprise | 15 / 27 | 23 / 32 | 37 / 30 |
| **association originality** | 7 / **59** | 13 / **60** | 16 / **58** |
| analogy surprise | 27 / 13 | 22 / 13 | 24 / 14 |
| analogy originality | 15 / 27 | 15 / 28 | 24 / 25 |
| **blending surprise** | **48** / 10 | **52** / 10 | **64** / 7 |
| blending originality | 8 / 17 | 14 / 20 | 21 / 20 |

The **model** share is essentially encoder-invariant (association originality 58–60%, blending surprise 7–10%, analogy surprise 13–14%). The **item** share drifts upward under e5-small, which attributes more variance to items throughout, so the item numbers are quoted as ranges below rather than as point estimates. The claim that survives in all three spaces is the contrast: association originality is model-dominated, blending surprise is item-dominated.

## ~~Finding 1~~ (retracted) — remoteness costs originality, in blending only

| task, dimension | r with d(u,v) | Spearman | leave-one-out | item-mean range |
|---|--:|--:|---|---|
| analogy surprise | **+0.73** (p < .0001) | +0.71 | 30/30 | 0.672 – 0.852 |
| blending surprise | **+0.41** (p = .025) | +0.44 | 28/30 | 0.682 – 0.917 |
| **blending originality** | **−0.42** (p = .022) | −0.38 | 28/30 | 0.416 – 0.472 |
| blending emergent originality | −0.31 (n.s.) | −0.28 | 2/30 | 0.444 – 0.521 |
| analogy originality | −0.20 (n.s.) | −0.13 | 0/30 | 0.445 – 0.524 |

Distance buys surprise everywhere, as the metric is built to do: a blend of far-apart inputs has a generic space far from both. But in blending it simultaneously *costs* originality, and the two are not in tension by construction — surprise measures distance from the anchors, originality measures distance from what the other models wrote.

The effect is not an outlier artifact: 28 of 30 single-item deletions keep p < .05. It is, however, small in absolute terms — item means run 0.416 to 0.472 on a 0–1 scale.

**Analogy is exempt.** Same 30 pairs, same measurement, r = −0.20 with no leave-one-out fit reaching significance. Whatever this is, it belongs to fusion, not to projection.

## ~~Finding 2~~ (retracted) — why: the pool splits into two lobes

Originality is the mean distance from an artifact's elements to their five nearest neighbours *among all models' elements for the same item*. So anything that changes the shape of that pool changes the score. Four candidate mechanisms, one survives:

| candidate | d(u,v) → mediator | mediator → originality | partial r(d, orig \| mediator) |
|---|--:|--:|--:|
| **element polarisation** | **+0.45** (p = .014) | **−0.46** (p = .011) | **−0.27 (n.s.)** |
| pool sparsity | −0.18 (n.s.) | +0.44 (p = .015) | −0.38 (p = .038) |
| anchor proximity | +0.14 (n.s.) | +0.16 (n.s.) | −0.45 (p = .013) |

**Polarisation** is the mean of |d(e,u) − d(e,v)| over the item's elements: how strongly each element commits to one side. When the anchors are far apart, models stop producing middle-ground elements and produce *u*-elements and *v*-elements — the pool becomes two tight clusters, every element sits near its own kind, and nearest-neighbour originality drops. It is the only candidate that both rises with distance and predicts originality, and it is the only one whose removal kills the effect.

Two things it is **not**. It is not models retreating to the anchors' own vocabulary — distance is uncorrelated with how close elements sit to the nearer anchor (+0.14, n.s.). And it is not models giving up on fusion: verified scope stays flat with distance (blending utility ~ d(u,v) r = +0.00, n.s.). Models keep claiming the shared slot; what changes is that the material they build with has separated into two piles.

## Finding 3 — which task is about the model at all

Two-way variance decomposition of each dimension (models × items, one draw per cell):

| dimension | item | model | residual |
|---|--:|--:|--:|
| association originality | 7–16% | **59%** | 34% |
| association utility | 5% | 40% | 55% |
| analogy originality | 15–24% | 27% | 58% |
| analogy emergent originality | 15% | 21% | 64% |
| **blending utility** | **24%** | 14% | 62% |
| **blending surprise** | **48–64%** | **7–10%** | 42% |
| blending emergent utility | 11% | **7%** | 82% |

(Ranges are across the three encoders where the dimension is embedding-derived; utility is judge-derived and encoder-independent. Residuals are from the scoring encoder.)

Association separates models; blending separates items. Roughly half the variance in a blend's surprise is *which pair you asked about*, and its emergent utility is almost entirely not about the model — so a leaderboard column built from blending surprise is closer to a description of the anchor set than of the pool. The residual (model × item interaction plus single-draw noise) runs 34–86% throughout, which is the precision ceiling at one draw per cell.

**Item difficulty is operator-specific.** On the same 30 pairs, analogy and blending item means correlate at **surprise +0.70** and **emergent utility +0.46**, but **utility −0.01 (n.s.)**: a pair that readily admits a valid structure mapping is no more likely to admit a genuine shared generic space. Remoteness is a property of the pair; difficulty is a property of the pair *and the operator together*.

## Finding 4 — anchor domain (descriptive only)

Item means by the domain tag of either anchor, 3–6 anchors each — too few to test, reported because the utility spread is large enough to be worth a look:

| domain | n | utility | originality | surprise |
|---|--:|--:|--:|--:|
| art | 3 | **0.81** | 0.455 | 0.774 |
| film | 5 | 0.65 | 0.444 | 0.763 |
| music | 3 | 0.52 | 0.435 | 0.812 |
| religion | 6 | 0.51 | 0.444 | 0.763 |
| history | 3 | 0.50 | 0.447 | 0.786 |
| economics | 3 | 0.46 | 0.440 | 0.816 |
| ideas | 3 | 0.46 | 0.451 | 0.796 |
| science | 5 | 0.43 | 0.439 | 0.808 |
| biology | 4 | 0.41 | 0.435 | 0.849 |
| philosophy | 6 | 0.36 | 0.446 | 0.756 |
| food | 3 | 0.25 | 0.432 | 0.812 |
| technology | 3 | **0.18** | 0.446 | 0.819 |

Blends anchored on art or film clear the generic-space gate three times as often as blends anchored on technology or food, while originality barely moves across domains (0.444–0.486). If this holds on a larger pool it would say the gate is easier for domains whose entities are described in terms of roles and effects than for those described in terms of parts and mechanisms — but at n = 3 per domain it is a hypothesis, not a finding.

## Limitations and red-team

- **One embedding model defined both variables, and that is exactly what killed Finding 1.** d(u,v) and originality are both cosines in the scoring encoder's space; two other encoders reverse the sign. The mediation result did not protect against this — it was computed in the same space. Any future claim relating two embedding-derived quantities on 30 points has to clear this check first.
- **Small effect, narrow range.** Item means span 0.085 on a 0–1 scale, and the anchor pairs span only 0.58–0.88 in distance. Robust to leave-one-out, but this is a gradient inside a narrow band, not a large effect.
- **n = 30 items, and every pair is cross-domain.** Nothing here speaks to within-domain blends.
- **The variance decomposition assumes one draw per cell**, so the residual mixes true model × item interaction with sampling noise and cannot separate them.
- **The domain table is not a test.** 3–6 anchors per domain, no correction for anything, and domain tags come from the curated pool's own labels.
- **Post-hoc.** The distance–originality effect was found by scanning correlations, not predicted in advance. The mechanism test was pre-specified once the effect was in hand; the four candidates were chosen before their numbers were computed, and three failed.

## Reproduce

```
.venv_mlx/bin/python -m src.kg_creat.scripts.analyze_item_effects        # Findings 1, 2, 4 (1-2 retracted)
.venv/bin/python -m src.kg_creat.scripts.analyze_facet_correlations      # Finding 3 (variance shares)
.venv_mlx/bin/python -m src.kg_creat.scripts.embedding_robustness        # the encoder check
```

Both write every number quoted here to `data/kg_creat/kombine_test30/analysis/` (`item_effects.json`, `facet_correlations.json`).
