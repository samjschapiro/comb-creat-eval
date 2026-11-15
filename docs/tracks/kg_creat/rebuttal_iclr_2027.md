# ICLR 2027 (Kombine) — Rebuttal drafts

Status: drafting (2026-11-13). Rebuttals due **2026-11-18**. Reviewers: cGKy, En17, A7nz (all weak accept, 3/3
confidence; full thread in hand as of 2026-11-13).

Legend: **[PENDING: …]** marks a number that an analysis still running will fill in. Everything else is
backed by results already in the repo; each section names its source.

---

## Shared points (2+ reviewers)

### S1 — Is the abstraction bottleneck causal? (En17, cGKy)

> En17: An experiment supplying validated generic spaces and measuring improvements in downstream blending
> performance would provide stronger evidence for this claim.
> cGKy: The proposed explanation (missing experiential knowledge) is speculation with no test, e.g.
> providing the generic space or a hint.

**Draft response.** Thank you both; we ran exactly this intervention.
- **Models:** six, spanning frontier to small (claude-opus-5, gpt-5.2, deepseek-chat-v3, qwen-2.5-72b,
  llama-3.3-70b, gpt-4o-mini).
- **Items:** all 30 blending items.
- **Conditions:**
  - **control:** the original prompt, freshly re-sampled. This is the baseline any gain must beat, since
    a model that failed once may succeed by chance on a second draw.
  - **hint:** the original prompt plus a content-free procedural hint, verbatim: *"first list, for EACH
    input separately, its 3-4 core functional roles (what acts, on what, by what means, toward what end).
    Then look for roles that CORRESPOND across [u] and [v] (they play the same part in each). The generic
    space is the abstract structure those corresponding roles share -- state it so that it is true of BOTH
    inputs."*
  - **oracle:** the generic space is supplied. It is a panel-validated one written by a model outside the
    test set, and the same one is used for every model on a given item.

Everything is scored by the same three-judge panel.

**Supplying the abstraction is what unlocks blending.** On items each model *failed* in the paper run
(n = 116 model × item cells, exact paired McNemar):

| Condition | Full blend success (valid g ∧ coherent) |
|---|---|
| control (re-sample) | 17.2% |
| hint (content-free) | 22.4% (n.s., p = 0.31) |
| **oracle (g supplied)** | **65.5%** (59 gains vs 3 losses, p = 1.7 × 10⁻¹⁴) |

- **Every one of the six models improves**, from +31 pp (llama-3.3-70b) to +60 pp (deepseek-chat).
- **Elaboration is intact.** Given an accepted generic space, models write a coherent blend 91–99% of the
  time, and a double-scope one about 72% of the time, whether they found the abstraction themselves or
  were handed it. This matches the 96.9% the same models reach on their own valid generic spaces in the
  paper run.
- **Originality is unaffected** (emergent originality 0.525 oracle vs 0.526 control).
- **The oracle effect is if anything understated.** Success also requires the panel to re-accept the
  supplied generic space, which it does for only 72% of verbatim copies.
- **The content-free hint** (map each input's roles, then abstract) does not close the gap.

The failure therefore lies in *finding* the shared schema, not in elaborating a blend once the schema is
in hand. We will add this experiment to the paper.

On "experiential knowledge": we agree it is a hypothesis, and we will label and position it as such. The
intervention locates the bottleneck causally. The failure of a purely procedural hint is consistent with a
knowledge-based rather than a procedural account, but does not by itself establish which knowledge is
missing. Source: `docs/reports/2026-11-13_kg_creat_abstraction_causal/report.md`.

### S2 — Validity of the LLM judges against humans (A7nz, En17)

> A7nz: correlation of judge to human connection is thin.
> En17: Expanding the human evaluation to include multiple independent annotators and more samples would
> strengthen confidence in the benchmark's scoring methodology.

**Draft response.** We agree. We have run a new multi-annotator study.
- **Raters:** [N] co-authors rated independently, blind to source and to the judges' verdicts.
- **Answers:** the same 40, from 12 entity pairs, half human and half model, with half passing the panel.
- **Questions:** the judge's question for each dimension (factuality; valid analogy, mapping used,
  coherence; generic space, coherence, shared slot, emergence), plus 1–5 originality.

We report, per dimension:
1. Krippendorff's α between the human raters.
2. The key test of judge validity: agreement between the panel and each held-out rater, compared with
   agreement between the *other raters' majority* and that rater. This asks whether the panel is as
   close to a human as another human is.
3. The Spearman correlation between mean human originality and our embedding originality ρ, both pooled
   and within entity pair.

[PENDING: table from `analyze_rater_study.py`.]

This adds to the existing single-author check: 75% agreement on unanimous-panel items, with the leaderboard
nearly unchanged when restricted to unanimous items (Spearman ρ = 0.945).

### S3 — Robustness to design choices and parameters (cGKy; common point)

**(a) Leave-one-model-out originality and a fixed reference pool.** See cGKy Q2 below. In brief:
leave-one-model-out Spearman = 1.00 in all 35 cases; a fixed external pool gives Spearman ≥ 0.998 with
the top 8 unchanged.

**(b) Inventive-multiple threshold θ.**
- **How θ was set.** θ = 0.674 is where matched property pairs stop sharing just a noun and become
  paraphrases of the same claim. We set it by inspecting sampled matches in each cosine band, e.g.
  *cushions sleeper / conforms to sleepers* at 0.674, and *transforms / splits atomic nuclei* at 0.70.
  Between 0.50 and 0.60, matches share a single noun and little else.
- **False positives.** Against a cross-item null, where two inventions answer *different* entity pairs
  and so cannot share an item-specific property, only 0.03% of 185,480 matched pairs clear 0.674.
- **Sensitivity.** We swept θ from 0.55 to 0.75, rerunning the full multiples pipeline at each value
  (name-free properties, anchor echo removed, one-to-one matching).

  Columns, all at τ = 2:
  - **All / Blends / Analogies:** share of inventions that are part of at least one multiple (all
    tasks; blending only; analogy only).
  - **Ratio:** Blends ÷ Analogies, the paper's 14.3×.
  - **Per property:** the same ratio after normalising for properties per invention (shared properties
    ÷ the smaller invention's property count, averaged within item).
  - **Provider:** share of concepts with a multiple from a same-provider model, ÷ the share expected
    from an equal number of other-provider models. This is the paper's 2.3× (9.3% vs 4.1%).

  | θ | All | Blends | Analogies | Ratio | Per property | Provider |
  | --- | --- | --- | --- | --- | --- | --- |
  | 0.55 | 45.3% | 83.7% | 6.9% | 12.1× | 3.3× | 1.6× |
  | 0.60 | 34.9% | 65.1% | 4.9% | 13.2× | 3.2× | 1.9× |
  | 0.62 | 30.8% | 57.3% | 4.3% | 13.2× | 3.3× | 2.1× |
  | 0.65 | 24.8% | 45.9% | 3.8% | 12.2× | 3.3× | 2.3× |
  | **0.674 (paper)** | **19.9%** | **37.3%** | **2.6%** | **14.3×** | **3.4×** | **2.3×** |
  | 0.70 | 14.8% | 27.5% | 2.2% | 12.4× | 3.5× | 2.1× |
  | 0.72 | 11.7% | 21.9% | 1.5% | 14.2× | 3.5× | 2.0× |
  | 0.75 | 8.2% | 15.2% | 1.2% | 13.1× | 3.6× | 1.8× |

  - **Absolute rates move with θ, as they must.** Each finding holds in direction and roughly in size at
    every θ:
    - **Blends converge more than analogies.** By the paper's measure (share of inventions in a
      multiple), 12.1–14.3× (14.3× at the paper's θ). Per property, the ratio robust to
      properties per concept, 3.2–3.6×, nearly flat across θ.
    - **Same-provider models converge more than cross-provider models:** 1.6–2.3× (2.3× at the paper's
      θ).
    - **Convergence collapses from τ = 2 to τ = 3:** by 6–27×, depending on θ.
  - **The cross-item null almost never produces a τ = 2 multiple** (≤ 0.015% at any θ ≥ 0.55; none at
    θ ≥ 0.65). So multiples are convergence on the item, not shared phrasing.
  - A joint density over (τ, θ) shows the full picture rather than one cell. We will add it to the
    appendix.

  Source: `src/kg_creat/scripts/multiples_theta_sweep.py` →
  `data/kg_creat/kombine_test30/analysis/multiples_theta_sweep.json`.

  *(Note on the ratios. Both columns use the paper's definitions, and both are reproduced exactly at
  θ = 0.674:
  - 14.3× is the share of inventions in a multiple, 37.3% vs 2.6%;
  - 2.3× is the share of concepts with a same-provider multiple vs the expected share from an equal
    number of other-provider models, 9.3% vs 4.1%.

  Pair-level rates (27× and 2.5×) are different quantities. Don't quote them.)*

*(Paper fix: the text calls 0.674 "the 99th percentile of cosine similarity over all property pairs".
It was chosen by inspection; 0.674 happens to be the 99.2nd percentile of within-item similarities.
Correct the wording.)* Source: `docs/reports/2026-09-01_kg_creat_inventive_multiples/report.md`.

**(c) Reasoning effort.** See cGKy W2 below.

---

## Reviewer cGKy

*Triage (2026-11-13): this review reads as largely LLM-written. It is long, every claim is hedged, and it
asks for every standard robustness check. Answer the points the AC will weigh, which are also the ones
other reviewers raise: pool robustness, the causal test, human validation, the composite and gate, and
self-preference. Answer the rest in one or two sentences, and don't commit new experiments to them. The
per-point replies below are a menu, not a checklist; cut them to fit the length limit.*

### W1 / Q2 — Stability of the leaderboard, especially originality, under a changed reference pool

> How stable is the leaderboard, originality especially, if you remove a model or a whole provider, or
> use a fixed reference pool?

**Draft response.** Thank you; this is an important check, since originality is computed against the pool
of all responses to an item. We recomputed ρ under four alternative pools. Our recomputation first
reproduces every stored originality value to within 3e-7.

| Pool variant | n | Overall Spearman (min) | Max rank shift | Top 8 unchanged | Originality-dim Spearman (min) |
|---|---|---|---|---|---|
| Leave one model out | 35 | 1.000 | 0 | 8/8 | 0.997 |
| Leave one provider out | 10 | 0.9996 | 1 | 8/8 | 0.992 |
| Fixed pool: the other half of the models | 20 | 0.998 | 1 | 8/8 | 0.929 |
| Fixed pool: human answers only | 1 | 0.998 | 2 | 8/8 | 0.951 |

- **Removing models.** Leaving out any single model leaves the ranking of the remaining 34 unchanged
  (Spearman = Kendall = 1.00 in all 35 cases). Removing a whole provider, including all 10 Anthropic or
  all 7 OpenAI models, moves at most one model by one rank.
- **Fixed pools.** Scoring every model against a pool it is not part of (the other half of the models over
  20 random splits, or human answers only) also preserves the ranking. The top 8 are unchanged in every
  case.
- **Originality dimensions alone,** the stricter test since utility does not depend on the pool, keep
  their order at Spearman ≥ 0.93 in every variant, and ≥ 0.99 when one model or provider is removed.
- **What does move.** Absolute originality shifts with an external pool (up to 5.75 points for the
  humans-only pool, since model answers are farther from human answers than from each other). The
  ordering does not.

We will add this as an appendix. Source: `docs/reports/2026-11-13_kg_creat_rebuttal_originality_pool/`.

### W1 (cont.) — "Surprise is embedding distance … semantic distance between entities is not surprise."

**Draft response.**
- **What we measure.** We operationalise surprise as *semantic remoteness* between combined elements. This
  follows a long psychometric line, from Mednick's associative theory (remote associations underlie
  creative combination) to the Divergent Association Test and SemDis-style scoring. It also fits
  computational-creativity accounts of surprise as low expectedness of a combination.
- **What we are not claiming.** That is not the same as surprise in the Bayesian sense, and we will say so
  and call the dimension "semantic surprise (remoteness)" in the text.
- **Two safeguards against conflating it with originality:**
  1. Surprise is *within-artifact* (how far apart the combined elements are), while originality is
     *across-artifact* (how far the answer is from everyone else's to the same item). The two can and do
     dissociate.
  2. The embedding choice does not drive the results. Re-scoring every artifact under two encoders from
     different model families (BGE, E5) leaves the 35-model ranking essentially unchanged (Spearman
     ρ ≥ 0.998 overall, ≥ 0.996 per task; App. G).
- **External validity.** The Kombine composite correlates with independent creativity tests across models
  (DAT r = 0.66).

### W1 (cont.) — Equal-weight composite and the hard zero on utility failure

**Draft response.** We agree both choices should be tested, and the ranking does not depend on them.

| Variant vs the paper ranking | Spearman | Same #1 | Top-10 overlap |
|---|---|---|---|
| No utility gate | 0.965 | yes | 9/10 |
| Geometric mean of task scores | 0.997 | yes | 10/10 |
| Task weights ∝ 1/variance | 0.988 | yes | 9/10 |
| Utility dimensions only | 0.992 | yes | 10/10 |
| 10,000 random Dirichlet(1) weightings over dimensions and tasks | median 0.971 (5th pct 0.897) | 52% of draws | median 9/10 |

**Why we gate on utility.** The gate encodes the standard two-criterion definition of a creative product,
which must be novel *and* useful. A non-factual chain or a generic space that fits only one input is not a
creative product, however remote its parts are. The ranking does not hinge on the gate (0.965 without it).
Pure ungated originality is a different construct, which ranks models differently (Spearman 0.42); we will
report it as its own column, so readers can apply their own weighting.

**Why equal weights.** They are the neutral choice without a principled weighting. Which model is #1 does
depend on the weights (gpt-5.6-sol under 52% of random weightings, opus-4.6 under 33%), which is consistent
with our finding that leadership is task-specific. The set of leading models does not change.

Source: `docs/reports/2026-11-13_kg_creat_rebuttal_composite_and_judges/`.

*(Internal note: don't argue that the gate "removes noise". Failed and passed artifacts are equally
original: association 0.337 vs 0.331, analogy 0.453 vs 0.462, blending 0.383 vs 0.382.)*

### W1 (cont.) — "The panel includes evaluated models (self-preference?)"

**Draft response.**
- **No judge is evaluated.** None of the three panel judges (claude-haiku-4.5, gpt-5.4, o3) is among the
  35 evaluated models. The panel was chosen to be non-subject, and the factuality judge (claude-haiku-4.5)
  is not evaluated either.
- **Family-level bias, tested.** Judges do share providers with evaluated models (10 Anthropic and
  7 OpenAI models among the 35), so we tested this directly from the stored per-judge votes:
  1. **Own-family leniency.** For each judge: its vote minus the other family's judges', on its own
     provider's models vs third-party models (cluster bootstrap over models). The estimates are +2.7 pp
     for Haiku 4.5 (95% CI [−1.7, 7.1]), −0.2 pp for o3 ([−5.1, 4.3]) and +3.3 pp for gpt-5.4
     ([−3.1, 8.7]). None differs from zero. Of 18 judge × dimension cells, the only significant one goes
     against the judge's own family.
  2. **Leave one judge out.** The ranking is essentially unchanged (Spearman 0.966–0.993; same #1 in 5 of
     6 configurations). Dropping Haiku moves Anthropic models *down* by only 0.2–0.6 ranks on average,
     and dropping an OpenAI judge moves OpenAI models *up*, the opposite of what self-preference predicts.
  3. **Factuality.** Haiku 4.5 agrees with an independent judge (gpt-oss-120b) on 88–92% of 54,277
     triples in every provider family. It is slightly *stricter* everywhere, and no more lenient on
     Anthropic models (difference in differences +1.1 pp, [−0.5, 2.7]).

  Source: `docs/reports/2026-11-13_kg_creat_rebuttal_composite_and_judges/`.

### W2 — Reasoning effort

> Since reasoning level is left at default and differs across models, how much of the ranking reflects
> thinking budget rather than ability?

**Draft response.** We ran a controlled effort study.
- **Models:** two frontier reasoning models (gpt-5.6-sol, gpt-6-astra-flex), each at low, medium and
  high reasoning effort.
- **Items:** all 90.
- **Scoring:** all six configurations in one pooled pass, so originality is comparable across effort
  levels.

| Model · effort | Mean reasoning tokens | Association | Analogy | Blending | **Overall** [95% CI] |
|---|---|---|---|---|---|
| gpt-5.6-sol · low | 1,786 | 49.1 | 78.6 | 63.8 | **63.8** [59.1, 67.9] |
| gpt-5.6-sol · medium | 4,981 | 46.7 | 71.8 | 67.2 | **61.9** [56.7, 66.5] |
| gpt-5.6-sol · high | 13,076 (7.3×) | 44.5 | 73.9 | 71.9 | **63.4** [58.3, 67.8] |
| gpt-6-astra-flex · low | 1,011 | 44.9 | 70.0 | 64.9 | **59.9** [54.3, 65.0] |
| gpt-6-astra-flex · medium | 6,526 | 44.4 | 66.4 | 53.8 | **54.8** [48.6, 60.3] |
| gpt-6-astra-flex · high | 14,779 (14.6×) | 46.6 | 70.3 | 61.6 | **59.5** [53.6, 64.5] |

- **Effort had no effect on the overall score.** Paired high − low difference: sol −0.4 [−6.2, +5.4];
  astra −0.4 [−8.1, +7.0].
- **Only one effect, and it is mechanical.** Of 38 per-dimension contrasts, only three exclude zero.
  They are a single effect, sol's association utility: surprise and originality are gated on utility,
  so a utility drop pulls both down.

**That effect is a path-length artifact.** More effort produces longer chains, and a chain is factual
only if *every* link is. Per-link factuality barely moves, and per-link factuality raised to the mean
chain length predicts each configuration's path utility within 1–3 points:

| Model · effort | Association paths | Mean hops | Per-link factuality | Path utility (observed) | Predicted = p_link ^ hops |
|---|---|---|---|---|---|
| sol · low | 586 | 4.81 | 93.9% | 73.0% | 73.9% |
| sol · medium | 804 | 4.95 | 92.3% | 69.2% | 67.3% |
| sol · high | 1,220 | 5.52 | 92.2% | 65.7% | 63.9% |
| astra · low | 1,187 | 6.79 | 93.8% | 65.6% | 64.8% |
| astra · medium | 1,620 | 7.29 | 93.7% | 64.5% | 62.2% |
| astra · high | 1,572 | 7.29 | 94.2% | 67.6% | 64.7% |

Thinking an order of magnitude longer does not buy combinatorial creativity on Kombine, so the ranking
reflects what models do with their default budget, not the budget itself. We will add this as an
appendix. Source: `data/kg_creat/effort_study/`, log 2026-09-07.

---

## Reviewer En17

### W1 — LLM-judge validity → see S2.

### Q1 — Data contamination and memorisation

> How do the authors distinguish genuinely novel combinations from associations or concepts that models
> may have memorized during pretraining? ... Have the authors considered additional controls?

**Draft response.** Three features of the design address this, plus a clarification we will make.
1. **The inputs are unrelated by construction.** Entity pairs are sampled at random from a curated
   Wikidata pool spanning domains (e.g. *Photosynthesis + Bread*, *Networks + Salt*), not chosen for
   known connections. This is the opposite of benchmarks that pre-filter for pairs with documented links,
   so a pretraining document stating the answer is unlikely.
2. **Memorisation is penalised by the originality metric.** Originality is pool-relative: an answer scores
   high only if nothing semantically close was produced by any other model or human for the same item.
   A memorised connection that many models share is exactly what scores *low*. Our inventive-multiples
   analysis quantifies this convergence directly (one in five inventions shares ≥ 2 properties with
   another model's).
3. **Inventions are, by definition, not in the training data as such.** For analogy and blending, the
   judged object is an invented concept (e.g. *whale groomer drone*, *market colony*), and the judges
   check that it was built by the mapping, not recalled.
4. **The clarification (Boden's P- vs H-creativity; Swanson's "undiscovered public knowledge").**
   Novelty is always relative to a reference class. We measure novelty relative to what other systems
   and people produce for the same item, not relative to all human knowledge. A model can produce a
   combination new to every respondent that is nonetheless implicit in the literature, which is exactly
   Swanson's point about discoverable-but-undiscovered connections. We will state this scope explicitly
   in the paper.

5. **A direct contamination test (new).** We searched two open pretraining corpora with infini-gram: the
   4.6T-token OLMo 2 mix and a 9.2B-token Dolma sample.
   - **Inputs.** The input pairs are not textually associated. The median pair co-occurs within 100
     tokens less than half as often as its frequencies predict (lift 0.45; 68% below chance).
   - **Steps.** Association chains are built from documented facts (87% of hops appear in the corpus),
     which is the intended retrieval component.
   - **Familiarity and validity.** Familiarity of the inputs makes a *valid* connection easier to find
     (association pass rate ρ = +0.47, robust to entity prominence).
   - **Familiarity and creativity.** It predicts neither originality (|ρ| ≤ 0.26, n.s., every task) nor
     convergence between models (|ρ| ≤ 0.13). In blending it predicts nothing at all.
   - **Inventions.** 32–40% of invented names never appear verbatim in 4.6T tokens. Familiar names
     receive modestly lower originality from our metric (within item ρ = −0.15 / −0.10), although it
     never sees the corpus, and convergent inventions are not more likely to
     carry existing names.

   - **Memorised inventions.** We define these as invented concepts already documented near their
     inputs, with the properties the model assigned. Only 2.1% of inventions qualify, against 0.4% under
     a cross-item null, at the same rate for analogy and blending.
     - They come mostly from older open-weight models (Llama-3.3-70B 17%), and their rate falls with a
       model's Kombine score (blending ρ = −0.41). 15 of the 35 models never produce one.
     - They are *less* often part of an inventive multiple (blending 19% vs 38%), so the convergence we
       report is on new concepts, not shared recall. Source:
       `docs/reports/2026-11-14_kg_creat_memorized_inventions/`.

   Following Saakyan et al. (2025, "Death of the Novel(ty)"), we use n-gram statistics only as a
   contamination check, not as a creativity score. Source:
   `docs/reports/2026-11-14_kg_creat_rebuttal_corpus_contamination/`.

---

### Q2 — Human-validated generic spaces

> Have the authors considered evaluating blending performance when models are explicitly provided with
> human-validated generic spaces?

**Draft response.** Yes; see S1. The supplied generic spaces were panel-validated. [OPTIONAL, ~15 min of
co-author time: have 2–3 co-authors accept or reject the 30 oracle generic spaces
(`data/kg_creat/abstraction_causal/oracle_g.json`) and report the oracle effect restricted to
human-accepted ones. This would make the answer "human-validated" literally.]

---

## Reviewer cGKy: lower-priority points (one or two sentences each in the final reply)

*Not worth new analyses: the same-provider version split and the same-model multiples baseline. Q1
resample variance and the Q5 tier split are free and quick, so run them only if there's spare time; the
replies stand without them.*

### Q1 — Size, samples per item, resample variance

> How many entity pairs per task, and how many samples per model per item? How much does a model's score
> vary across resamples at T = 0.9?

**Draft response.**
- **Size.** We confirm: 30 entity pairs per task. Analogy and blending share the same 30 pairs;
  association uses a different 30. That gives 90 items per model, with one sample per model per item at
  T = 0.9: 3,150 scored artifacts.
- **Why one sample is enough here.** Per-model conclusions rest on 30 items per task with bootstrap CIs
  over items, not on single answers, and the paper's claims are about the population of models.
- **Resample and item variance (new).** We re-ran 10 models on 30 *new* entity pairs, disjoint at the
  entity level, and scored them with the same judges. Fresh items and fresh draws together bound the
  T = 0.9 resample variance from above. Agreement across the 10 models between the two item sets:

  | Score | Pearson r | Spearman ρ | ICC(A,1) | Mean \|Δ\| (points) | Item SE, one set | Models with Δ outside 95% CI |
  |---|---|---|---|---|---|---|
  | Association | 0.65 | 0.71 | 0.53 | 7.1 | 3.0 | 4 / 10 |
  | Analogy | 0.87 | 0.88 | 0.80 | 7.8 | 6.7 | 1 / 10 |
  | Blending | 0.95 | 0.94 | 0.84 | 5.8 | 7.0 | 0 / 10 |
  | **Overall** | **0.92** | **0.88** | **0.92** | **2.7** | 3.4 | **0 / 10** |

  Per model, overall score on the original vs the new items:

  | Model | Original 30 | New 30 | Δ [95% CI] |
  |---|---|---|---|
  | claude-opus-4.6* | 54.1 | 54.2 | +0.1 [−7.8, +8.3] |
  | claude-fable-5* | 49.9 | 49.2 | −0.7 [−9.6, +8.5] |
  | claude-opus-4.5* | 46.1 | 49.1 | +3.0 [−6.8, +12.4] |
  | claude-sonnet-4.6 | 45.8 | 51.5 | +5.7 [−3.0, +14.7] |
  | claude-fable-5.1 | 43.8 | 40.8 | −3.0 [−12.8, +6.6] |
  | claude-opus-4.8 | 41.9 | 42.4 | +0.5 [−9.1, +10.0] |
  | claude-opus-4.7 | 40.8 | 37.4 | −3.4 [−12.8, +5.7] |
  | gpt-5-mini | 37.2 | 44.7 | +7.5 [−1.7, +16.4] |
  | gpt-4.1 | 31.1 | 28.2 | −2.9 [−12.0, +6.1] |
  | gpt-4o-mini | 26.2 | 26.3 | +0.1 [−8.1, +8.2] |

  \* Sampling conditions also changed for these three (direct API with a thinking budget vs OpenRouter
  at T = 0.9), which can only lower agreement.

  - **The overall score reproduces.** No model's change exceeds what item sampling alone would produce.
  - **Association is the least reliable task.** Its spread between models is narrowest (SD 5.7 vs about
    10 for the other tasks), so it is the task that most needs more items. We will say so.
  - **What the statistics measure.** r and ρ ask whether the leaderboard reproduces. ICC(A,1) also
    penalises a shift common to all models. Blending's ICC (0.84) is below its r (0.95) because the new
    blending items were about 5 points harder for every model.

  Source: `docs/reports/2026-11-13_kg_creat_rebuttal_test_retest/`.

### W1 (judges) / Q5 — Validation of the Haiku 4.5 factuality judge

> Path factuality is judged by Claude Haiku 4.5, plausibly weaker than many models it judges, with no
> retrieval. ... What is its error rate on paths from stronger models?

**Draft response.**
- **The task is easy for the judge.** It checks single Wikidata-style triples about well-known entities
  (each has ≥ 78 sitelinks), one at a time. It never judges a whole creative answer.
- **Agreement with an independent judge.** Haiku 4.5 agrees with gpt-oss-120b on 88–92% of 54,277
  triples in every provider family. It is slightly stricter, with no family leniency (see self-preference
  above).
- [PENDING, free: the same agreement split by evaluated-model tier (frontier vs the rest), to answer "on
  paths from stronger models".]
- [OPTIONAL: humans also rate factuality in the S2 study, which gives a human error rate for the panel's
  factuality on association.]

### W1 (judges) — ICC 0.48–0.67 and quoting 75% rather than 66%

**Draft response.**
- We agree the main text should report both numbers. We will quote 66% overall agreement alongside the
  75% on unanimous items.
- Fair-to-good reliability between *individual* judges is why we use a three-judge majority. The new
  multi-annotator study (S2) shows how the panel compares with the agreement *between humans* on the
  same items.

### W1 — "No human baseline or human creativity ratings"

**Draft response.**
- **Human baseline.** Since submission, we have collected a human generation study on the same items:
  94 kept sessions; 375 association, 298 analogy and 201 blending answers.
  - Humans pass the validity gate less often than the typical model, mostly by breaking task rules.
  - Their valid answers are more original than most models' valid answers, and less surprising.
  - Human answers sit farther from model answers than from each other.
  - [Decide how much of this to put in the rebuttal. The full composite places kept humans last of 37
    entries, which needs careful framing: it is a rules-compliance effect.]
- **Human creativity ratings.** The S2 study collects them: 1–5 originality from multiple raters,
  correlated with ρ.

### "Blending is 10–15 points harder": non-comparable rubrics across tasks

**Draft response.**
- **We agree the three rates use different criteria.** We will soften the cross-task wording to "models
  pass the blending gate less often". The claim we make is about the gate each task defines.
- **The comparable contrast is analogy vs blending.** Both are posed on the *same* 30 entity pairs, and
  both gates are three-judge panel verdicts on a structural claim (shared relational structure; a
  generic space instantiated by both inputs). Paired by (model, item), analogy leads by 13.5 points
  (McNemar OR 1.84 [1.49, 2.28], p = 3.5 × 10⁻⁹; 23 of 30 models).
- **The abstraction bottleneck no longer depends on a cross-task comparison.** It is now established
  *within* blending by the intervention in S1: the same models on the same items go from 17% to 66%
  success when given the generic space.

### Q4 — Multiples: same-model baseline, threshold, per-property normalisation

**Draft response.**
- **Threshold.** See S3b. Every finding keeps its direction across θ ∈ [0.62, 0.72], and the
  cross-item null false-positive rate is 0.03%.
- **Per-property normalisation.** Already in the multiples analysis: the fixed-τ blend/analogy ratio is
  mostly because blends assert about twice as many properties (5.0 vs 2.4). Normalised per property,
  blends still converge 3.3× more often (paired Wilcoxon p = 5 × 10⁻⁸; 2.0× with verbatim,
  encoder-free matching). We will lead with the per-property figure.
- **Same-model baseline.** [PENDING, free: the causal-test control is a second T = 0.9 blend from six
  models on all 30 items. The rate of τ = 2 multiples between a model's two samples gives the
  within-model ceiling, to set against the cross-model rate.]

### W — Same-provider effect could be version similarity

**Draft response.** [PENDING, free: split same-provider pairs into successive versions of one line
(e.g. opus-4.5 / opus-4.6) and different lines from the same provider (e.g. haiku / opus, gpt-4o-mini /
o3), then re-test both against cross-provider pairs. If the effect holds for different lines, it is not
only version similarity. If it doesn't, we report it as lineage similarity, which is still a finding
about homogeneity.]

---

## Reviewer A7nz

Rating 3 (weak accept), soundness 3. The review is in bullet points with no questions. Each point is
answered briefly, mostly by pointing to shared sections.

### "How do you differentiate creativeness against factual incorrectness, truthfulness?"

**Draft response.**
- **Factuality is a precondition, not part of creativity.**
  - In association, every link in a chain must be a true fact (checked triple by triple). A chain with a
    false link scores zero, however remote its endpoints.
  - In analogy and blending, the *source* structure (the paths, the generic space) must be true of the
    real entities.
  - Only the *invented* concept is allowed to be counterfactual, since it is new by construction. It is
    judged for coherence (is it built from the mapping or blend?), not truth.
- **So the two are separated by design.** Creativity is scored only on factual, valid combinations, and
  the invention is the one place where non-existence is the point.

### "Score is driven by validity: if not valid then 0, which doesn't align to interpret creativity"

**Draft response.**
- **Why the gate.** It encodes the standard two-criterion definition of a creative product, which must
  be novel *and* useful/appropriate (Runco & Jaeger, 2012). A remote but invalid combination is novel
  without being creative.
- **The ranking does not depend on it.** Removing the gate gives Spearman 0.965 with the paper ranking
  (same #1, 9 of the top 10).
- **Failed answers are not more original.** Failed and valid artifacts are equally original
  (association 0.337 vs 0.331; analogy 0.453 vs 0.462; blending 0.383 vs 0.382), so the gate does not
  hide creative-but-invalid work.
- **We will report ungated originality as its own column.** It is a distinct construct (Spearman 0.42
  with the composite), and readers can weight it themselves. See cGKy W1, composite.

### "Abstraction bottleneck may come from the judge end, which hasn't been explored"

**Draft response.** This is a fair concern, and the new experiment (S1) tests it.
- **If the bottleneck were judge strictness on generic spaces,** supplying a generic space should not
  help much: the same judges evaluate it.
- **Instead, success goes from 17% to 66%** on previously failed items when a validated generic space is
  supplied. A content-free hint does not help (22%, n.s.).
- **Elaboration is not judged harshly.** Models produce coherent blends 91–99% of the time once they have
  an accepted generic space.
- **The judges are not simply lenient on supplied abstractions either.** They re-accept a supplied
  generic space in only 72% of verbatim copies.
- **Humans check the judges directly** on generic-space validity and blend coherence in the new
  multi-annotator study (S2).

### "Correlation of judge to human connection is thin" → see S2.

### Missing related work

**Draft response.** Thank you; we will add all of these.
- **Creativity tests on LLMs:**
  - Stevenson et al. (2022): GPT-3 on the Alternate Uses Task.
  - Koivisto & Grassini (2023): AI beat average humans but not the best.
  - Guzik et al. (2023): Torrance tests.
  - Organisciak et al. (2023): automated creativity scoring.

  These test divergent thinking on standard prompts. Kombine instead tests whether models can *combine*
  given, unrelated concepts into a valid structure and use it to invent. Our external-validity analysis
  already relates Kombine to such tests across the 35 models (composite vs DAT r = 0.66).
- **Judging creative quality:**
  - Chakrabarty et al. (CHI 2024, TTCW): expert-derived rubrics for creative writing, which supports our
    rubric-based panel.
  - Lu et al. (2024, Creativity Index): measures novelty against web text, a complementary *H-novelty*
    reference to our *P-novelty* pool. We will discuss it as the natural extension (see En17 Q1) and
    note why it fits our setting poorly: our outputs are short structured triples about well-known
    entities, where n-gram overlap with the web is high whether or not the combination is new.
- **AIs sounding alike:**
  - Doshi & Hauser (2024);
  - Padmakumar & He (ICLR 2024);
  - Anderson et al. (2024);
  - Kirk et al. (2024).

  These document homogenisation in human–AI writing and after RLHF. Our inventive-multiples result
  extends it from surface text to *invented concepts*: different models independently invent concepts
  with the same properties, and more so within a provider.

*(Check before sending: the Lu et al. claim that n-gram novelty fits short triples poorly is our
argument, not theirs. Verify against the paper or soften it. Our own corpus check supports it: many
invented names are ordinary words that occur millions of times, e.g. "Treatment", "Covenant". Better
still, point to the corpus contamination test (En17 Q1) as our engagement with H-novelty. "Death of the
Novel(ty)" is checked: Saakyan et al., ICLR 2026, arXiv 2509.22641.)*

---

## Paper changes promised (track)

- Label the experiential-knowledge explanation as a hypothesis; add the abstraction intervention (S1).
- Add the multi-annotator validity study (S2).
- θ: correct the "99th percentile" wording; add the (τ,θ) density figure and null FPR (S3b).
- Add appendices: effort study (cGKy W2), composite / judge robustness (cGKy W1), pool robustness (cGKy Q2).
- Rename surprise to "semantic surprise (remoteness)" and clarify within- vs across-artifact.
- State P- vs H-creativity scope; add the corpus contamination appendix (En17 Q1).
- Formalism fixes from 2026-11-11 (association surprise denominator, domain of M, Q with an empty
  projection, direction of the analogy example).
- Add the test–retest appendix (10 models, 30 new items; cGKy Q1).
- Report 66% overall human–panel agreement alongside 75% on unanimous items (cGKy W1).
- Soften the cross-task "blending is harder" wording; lead with the same-anchor analogy vs blending
  contrast and the within-blending intervention (cGKy).
- Lead the multiples comparison with the per-property ratio (3.3×), not the fixed-τ ratio (cGKy Q4).
- Related work: the 12 papers A7nz lists, plus a P- vs H-novelty discussion around Lu et al. (2024).
- Clarify that only the invented concept may be counterfactual; source structures must be factual (A7nz).
