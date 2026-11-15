# Rebuttal: are Kombine's combinations memorised from pretraining text? (reviewer En17 Q1)

2026-11-14. No API spend: the public infini-gram API. Script `src/kg_creat/scripts/corpus_novelty.py`;
output `data/kg_creat/corpus_novelty/corpus_novelty.json`. Every query is cached in `cache.jsonl`, about
6.6k queries.

**The question.** Could models solve Kombine by recalling relationships or inventions they saw in
pretraining?

**Answer.** Partly for *validity*, not for *creativity*.
- **Inputs:** the two input entities co-occur in text no more than their frequencies predict.
- **Steps:** models build association chains from documented facts, which is the intended retrieval
  component.
- **Validity:** how often two inputs co-occur in text predicts how easily models find a valid
  association or analogy.
- **Creativity:** it predicts neither originality nor cross-model convergence in any task, and nothing at
  all in blending.
- **Names:** most invented names are new phrases or ordinary words. Familiar ones score modestly lower on
  our originality metric.

## Method

We can't search closed models' training data, so two open corpora stand in for it:

| Index | Size | Used for |
|---|---|---|
| `v4_olmo-mix-1124_llama` (OLMo 2 pretraining mix, includes DCLM web text) | 4.6T tokens | does a link exist at all; verbatim name counts |
| `v4_dolma-v1_6-sample_llama` (Dolma sample) | 9.2B tokens | exact co-occurrence counts |

**Matching.**
- Strings are matched exactly, as an OR over case variants, with a leading "The " stripped.
- Co-occurrence means the two strings appear within 100 tokens of each other.
- Exact matching misses paraphrases, so it *under*states documentation. That direction is conservative for
  "documented hops" and generous for "novel names".

**Counting rule.** Co-occurrence counts on the 4.6T index are *sampled* when a term has more than 500k
matches. A sampled nonzero is a real match, but a sampled zero is inconclusive. So:
- "documented" means nonzero on the 4.6T index, falling back to the exact 9.2B count when the big index
  sampled;
- strength of association (lift) uses the exact 9.2B counts.

**Lift** = observed co-occurrences ÷ the number expected if the two terms were placed independently,
given their frequencies. It is approximate.

## 1. Are the inputs linked in text?

60 anchor pairs: 30 association, plus 30 shared by analogy and blending.

| | |
|---|---|
| Never co-occur within 100 tokens in the 9.2B sample | 15% (e.g. *Mount Everest / Winston Churchill*, *Frida Kahlo / Bob Dylan*) |
| Fewer than 10 co-occurrences in the 9.2B sample | 55% |
| Lift: quartiles | 0.20 / 0.45 / 1.21 |
| Pairs co-occurring *less* often than chance (lift < 1) | 68% |
| Co-occur at least once somewhere in 4.6T tokens | 57 of 60 (3 inconclusive) |

The pairs are not textually associated. The typical pair appears together *less than half* as often as
chance predicts. But at 4.6T tokens nearly any two common entities share a paragraph somewhere, so "never
co-mentioned" is not a claim we can make.

## 2. Each step known, the combination assembled

**Association chains.** We sampled 990 chains that passed utility, stratified over the 30 items, giving
4,162 unique hops.
- 87.3% of hops are documented in the 4.6T corpus, 6.1% are not, and 6.7% are inconclusive.
- 64.3% of chains have every hop documented.

The chains are built from known facts. That is by design, since factuality is checked hop by hop. The
creative part is the *route*.

**What failed.** We had planned to show Swanson-style undiscovered links (every hop documented, endpoints
never co-mentioned). The endpoint test failed: every endpoint pair turned up somewhere in 4.6T tokens
(previous section). Within the 9.2B sample, 22.9% of fully documented chains connect endpoints that never
co-occur there.

**Invention names**, counted verbatim in 4.6T tokens:

| | n | Never | 1–9× | ≥ 10× | ≥ 1000× |
|---|---|---|---|---|---|
| Analogy invention h | 1,037 | 40.3% | 11.0% | 48.7% | 17.8% |
| Blend c' | 1,029 | 31.9% | 14.0% | 54.1% | 14.4% |

- **Never-seen names:** *solar leaven*, *suspense sculptor*, *glass tribunal*, *votive ballad*,
  *self-legislated decalogue*.
- **Frequent names are ordinary words:** *Treatment*, *David*, *Covenant*, *Icon*, *Lattice*. A name's
  existence does not mean the *concept* existed, so this table bounds verbatim naming, not memorisation.
- **Familiar names score modestly lower on our originality metric.** The metric never sees the corpus.
  It only compares an invention with the other models' inventions for the same item. Even so, emergent
  originality falls slightly as the name gets more frequent in pretraining text.
  - **Within item:** Spearman −0.15 for analogy (permutation p = 0.0005) and −0.10 for blending
    (p = 0.0015). Correlations are computed after centring each item, so this is not "easier items have
    both commoner names and lower originality".
  - **Effect size:** small, about half an SD between never-seen and very common names, and the four bands
    fall in order. Mean emergent originality by verbatim count (SD about 0.06):

    | Task | Never | 1–99× | 100–100k× | ≥ 100k× |
    |---|---|---|---|---|
    | Analogy | 0.513 (n = 418) | 0.510 (278) | 0.494 (303) | 0.482 (38) |
    | Blending | 0.475 (327) | 0.469 (327) | 0.464 (361) | 0.457 (11) |

  - **Examples from one item each:**
    - analogy E5: *Phagosomal emission* (never seen) 0.579 vs *event horizon* (1.9M occurrences) 0.436;
    - blending F1: *grief cipher* (never seen) 0.513 vs *The Blue Key* (192k occurrences) 0.409.
  - **Mechanism:** the name is one of the invention's elements. It is always included for analogy, and
    for blending it enters as the subject of the emergent triples. Stock phrases such as *event horizon*
    tend to sit near what other models write for the item, so their nearest-neighbour distance is
    smaller.
  - **What this supports:** only weakly that pool-relative originality *tracks* corpus novelty
    (convergent validity with an H-novelty reference). It does not support "the metric discounts familiar
    concepts". The relationship explains about 1–2% of the variance in ranks.

- **Convergence is not recall of a named concept.** An invention is no more likely to be part of an
  inventive multiple when its name already exists:

  | | Existing name | New name |
  |---|---|---|
  | Analogy | 1.8% | 3.8% |
  | Blending | 38.4% | 35.1% |

## 3. Does textual familiarity predict performance?

Per item (n = 30 per task): Spearman correlation between log anchor co-occurrence and each outcome, with a
permutation p-value and a partial correlation controlling for entity prominence (mean log Wikipedia
sitelinks).

| Task | Outcome | ρ (co-occurrence) | p | Partial on prominence | ρ (lift) | p |
|---|---|---|---|---|---|---|
| Association | pass rate | **+0.47** | 0.010 | +0.45 | +0.37 | 0.046 |
| Association | originality | +0.05 | 0.77 | +0.09 | +0.09 | 0.63 |
| Analogy | pass rate | +0.30 | 0.11 | +0.30 | **+0.49** | 0.006 |
| Analogy | originality / emergent | +0.14 / −0.26 | 0.45 / 0.16 | | +0.22 / −0.22 | 0.22 / 0.24 |
| Analogy | multiples (τ ≥ 1 / per property) | −0.07 / −0.07 | 0.74 / 0.70 | | −0.01 / 0.00 | 0.95 / 0.99 |
| Blending | pass rate | −0.07 | 0.71 | −0.11 | +0.03 | 0.89 |
| Blending | originality / emergent | +0.13 / 0.00 | 0.48 / 0.99 | | +0.07 / +0.14 | 0.72 / 0.47 |
| Blending | multiples (τ ≥ 1 / per property) | −0.13 / −0.11 | 0.48 / 0.56 | | −0.12 / −0.12 | 0.54 / 0.53 |

**Reading it.**
- **Familiarity helps validity, where expected.** Inputs that co-occur more are easier to connect by a
  factual chain, or to map analogically. This survives controlling for prominence.
- **It doesn't drive the creativity measures.** Familiarity predicts no originality dimension in any task
  (all |ρ| ≤ 0.26, n.s.), and does not predict convergence between models (|ρ| ≤ 0.13).
- **Blending, the task behind our main findings, is unrelated to corpus familiarity on every outcome.**

## Rebuttal sentences

> We tested memorisation directly against open pretraining corpora: the 4.6T-token OLMo 2 mix and a
> 9.2B-token Dolma sample, searched with infini-gram.
> - **Inputs.** Kombine's input pairs are not textually associated: the median pair co-occurs within
>   100 tokens less than half as often as its frequencies predict (median lift 0.45; 68% below chance).
> - **Steps.** Models do assemble association chains from documented facts (87% of hops appear in the
>   corpus), which is the intended retrieval component.
> - **Effect of familiarity.** Familiarity of the inputs makes a *valid* connection easier to find
>   (association pass rate ρ = +0.47, robust to entity prominence). It predicts neither originality
>   (|ρ| ≤ 0.26, n.s., all tasks) nor convergence between models (|ρ| ≤ 0.13). In blending it predicts
>   nothing.
> - **Inventions.** 32–40% of invented names never appear verbatim in 4.6T tokens. Inventions with more
>   familiar names receive modestly lower originality from our metric (within item ρ = −0.15 / −0.10),
>   although it never sees the corpus, and convergent inventions are not
>   more likely to carry existing names.
>
> Following Saakyan et al. (2025), we use n-gram statistics only as a contamination check, not as a
> creativity measure.

## Caveats

- **Proxy corpora.** OLMo-mix and Dolma stand in for proprietary training data, which is larger and
  unknown.
- **Exact matching** misses paraphrases.
- **Lift is approximate.** It is computed from clause counts with a ±100-token window, and is a relative
  measure only.
- **The name check is about verbatim naming.** A new name can label a familiar idea.
- **The association hop sample** is 990 of 9,386 passing chains, stratified by item.
