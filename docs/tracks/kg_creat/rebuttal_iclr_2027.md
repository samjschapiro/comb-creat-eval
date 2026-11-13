# ICLR 2027 (Kombine) — Rebuttal drafts

Status: drafting (2026-11-13). Rebuttals due **2026-11-18**. Reviewers: cGKy, eN17, A7nz (A7nz's full
review text not yet in hand).

Legend: **[PENDING: …]** marks a number that an analysis still running will fill in. Everything else is
backed by results already in the repo; each section names its source.

---

## Shared points (2+ reviewers)

### S1 — Is the abstraction bottleneck causal? (eN17, cGKy)

> eN17: An experiment supplying validated generic spaces and measuring improvements in downstream blending
> performance would provide stronger evidence for this claim.
> cGKy: The proposed explanation (missing experiential knowledge) is speculation with no test, e.g.
> providing the generic space or a hint.

**Draft response.** Thank you both; we ran exactly this intervention.
- **Models:** six, spanning frontier to small (claude-opus-5, gpt-5.2, qwen3-max, llama-3.3-70b,
  gpt-4o-mini, phi-4).
- **Items:** all 30 blending items.
- **Conditions:**
  - **control:** the original prompt, freshly re-sampled. This is the baseline any gain must beat, since
    a model that failed once may succeed by chance on a second draw.
  - **hint:** the original prompt plus a content-free procedural hint: list each input's functional
    roles, find the corresponding ones, and abstract them.
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

### S2 — Validity of the LLM judges against humans (A7nz, eN17)

> A7nz: correlation of judge to human connection is thin.
> eN17: Expanding the human evaluation to include multiple independent annotators and more samples would
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
- **Sensitivity.** We swept θ.
  - At τ = 2, the rate of multiple pairs (overall / blending / analogy) is 2.2 / 4.3 / 0.1% at θ = 0.62,
    1.1 / 2.2 / 0.1% at 0.674, and 0.6 / 1.1 / 0.0% at 0.72.
  - Absolute rates move with θ, as they must, but **the direction of every finding is unchanged at every
    θ**: blends converge more than analogies, same-provider pairs converge more than cross-provider pairs,
    and convergence collapses from τ = 2 to τ = 3.
  - A joint density over (τ, θ) shows the full picture rather than one cell. We will add it to the
    appendix.

*(Paper fix: the text calls 0.674 "the 99th percentile of cosine similarity over all property pairs".
It was chosen by inspection; 0.674 happens to be the 99.2nd percentile of within-item similarities.
Correct the wording.)* Source: `docs/reports/2026-09-01_kg_creat_inventive_multiples/report.md`.

**(c) Reasoning effort.** See cGKy W2 below.

---

## Reviewer cGKy

### W1 / Q2 — Stability of the leaderboard, especially originality, under a changed reference pool

> How stable is the leaderboard, originality especially, if you remove a model or a whole provider, or
> use a fixed reference pool?

**Draft response.** Thank you; this is an important check, since originality is computed against the pool
of all responses to an item. We recomputed ρ under four alternative pools. Our recomputation first
reproduces every stored originality value to within 3e-7.

| Pool variant | n | Overall Spearman (min) | Max rank shift | Top 8 unchanged | Originality-dim Spearman (min) |
|---|---|---|---|---|---|
| Leave one model out | 35 | 1.000 | 0 | 8/8 | 0.997 |
| Leave one provider out | 10 | ≥ 0.999 | 1 | 8/8 | 0.992 |
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
- **Design:** two frontier reasoning models (gpt-5.6-sol, gpt-6-astra-flex) at low / medium / high effort.
  All six configurations were scored in one pooled pass, so originality is comparable across levels.
- **Manipulation check:** the models genuinely thought more. Mean reasoning tokens went 1,786 → 4,981 →
  13,076 (sol, 7.3×) and 1,011 → 6,526 → 14,779 (astra, 14.6×).
- **Result: the overall score did not move.** Paired high − low: sol −0.4 points, 95% CI [−6.2, +5.4];
  astra −0.4, [−8.1, +7.0]. Of 38 per-dimension effort contrasts, only three exclude zero, and they are
  one effect: sol's association utility.
- **That one effect is a path-length artifact.** Higher effort produces longer chains (4.8 → 5.5 hops),
  and a chain is factual only if every link is. Per-link factuality is nearly flat (93.9 → 92.2%), and
  per-link factuality raised to the mean chain length predicts each configuration's path utility within
  1–3 points.

Thinking an order of magnitude longer does not buy combinatorial creativity on Kombine, so the ranking
reflects what models do with their default budget, not the budget itself. We will add this as an
appendix. Source: `data/kg_creat/effort_study/`, log 2026-09-07.

---

## Reviewer eN17

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

*(Optional, not yet run: n-gram novelty of answers against a pretraining corpus index. Check the "death of
the n-gram" critique first; only offer it if run.)*

---

## Reviewer A7nz

[Need the full review text.] Known point: judge–human correlation is thin → see S2.

---

## Paper changes promised (track)

- Label the experiential-knowledge explanation as a hypothesis; add the abstraction intervention (S1).
- Add the multi-annotator validity study (S2).
- θ: correct the "99th percentile" wording; add the (τ,θ) density figure and null FPR (S3b).
- Add appendices: effort study (cGKy W2), composite / judge robustness (cGKy W1), pool robustness (cGKy Q2).
- Rename surprise to "semantic surprise (remoteness)" and clarify within- vs across-artifact.
- State P- vs H-creativity scope (eN17 Q1).
- Formalism fixes from 2026-11-11 (association surprise denominator, domain of M, Q with an empty
  projection, direction of the analogy example).
