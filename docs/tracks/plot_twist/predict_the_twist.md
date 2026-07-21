# Predicting the Twist: inter-subjective surprise via a similarity gradient

**Status:** design memo, 2026-07-18. A proposed extension of TwistBench (Schapiro et al.,
*TwistBench: Benchmarking Transformational Creativity in LLMs via Literary Plot Twists*).
Self-contained — readable without prior context.

---

## The gap this addresses

TwistBench scores each story on **surprise, coherence, realism, diversity**. Surprise —
"how drastically must one reinterpret earlier story elements after the twist?" — is currently
a Likert rating from a 3-judge LLM ensemble. That is the benchmark's softest, most subjective
dimension: it asks a judge to introspect on how surprised a reader *would* be.

**The idea (Game 1): replace judged surprise with a behavioral measure — how well other models
can *predict* the twist.** A twist that everyone anticipates is, by definition, not surprising;
a twist no one predicts is. This makes surprise an *information-theoretic* quantity
(improbability under a predictor) rather than a rating, and it sits squarely in the
Bayesian-surprise lineage TwistBench already cites (Chieppe et al. 2022; cf. Itti & Baldi).

The operational definition of creativity that falls out is clean and defensible:

> **A model is creative to the degree that a population of other minds cannot anticipate its
> twists.** Surprise is not a property of a story in the abstract — it is improbability relative
> to a *named reference population*. Specify the population and surprise becomes measurable.

That last point is also the entire answer to the same-corpus confound (below).

## Game 1: predict the twist

- **Setup / reveal split.** Every TwistBench story has a setup (everything the reader is led to
  believe) and a reveal (the late twist that recasts it). Cut each story at the reveal boundary.
  TwistBench's per-story annotations (it extracts a one-sentence `setup` and `reveal`, and records
  the twist's location in the text) give this split.
- **The prediction task.** Give **predictor** model A the setup written by **generator** B, and
  have A predict B's twist. Two elicitation modes, run both:
  1. **Free generation** — A writes the twist it expects; score by embedding similarity to B's
     actual twist (reuse TwistBench's `all-mpnet-base-v2` twist embeddings).
  2. **Forced choice / ranking** — present B's real twist among distractor twists (other models'
     twists for the same setup, plus decoys); A ranks or picks the most likely continuation.
     Cleaner signal than free-gen, and works for closed models (no logprobs needed).
- **Surprise.** `surprise(B's twist | A) = 1 − predictability(A → B's twist)`. Predictability is
  embedding similarity (mode 1) or top-k / reciprocal-rank / choice probability (mode 2).
- **The object of study is the full N×N predictor×generator matrix** — a round-robin design where
  every model predicts every model (and itself).

## The core design: stratify along a similarity gradient

The problem the user flagged: **models from the same family share pre-/post-training corpora, so
they will predict each other's twists trivially** — not because the twist is obvious, but because
the two models are near-copies. A single pooled "mean predictability" is therefore confounded.

**Fix: do not average over predictor–generator similarity — make it the independent variable.**
Order every (predictor, generator) dyad along a similarity gradient and measure predictability as
a function of position on it:

| Tier | Predictor vs generator | What shared |
|---|---|---|
| T0 | **self** (same checkpoint, resampled) | everything — upper bound on predictability |
| T1 | same family, different size (e.g. Opus vs Haiku) | architecture + most training data |
| T2 | same base, different post-training (e.g. two RLHF variants of one base) | pretraining corpus |
| T3 | **cross-family** (different lab, different base) | little beyond the public web |
| T4 | **human** predictors | none of the model training corpora |

Read surprise off the **shape of the curve**, not a single number:

- The **slope** of predictability vs. distance tells you how much apparent surprise was merely
  distributional overlap between predictor and generator.
- The **intercept at maximum distance** (T3/T4) estimates *intrinsic* surprise — surprise that
  survives even when the predictor shares nothing with the generator.
- **Human predictors (T4) are the anchor** for "surprising in an absolute sense," because humans
  share none of the model corpora. TwistBench already has human-story and human-eval
  infrastructure to extend here.

A twist that is unpredictable even at T3/T4 is genuinely novel; one that is predictable at T1 but
not T3 was only "surprising" to strangers, i.e. its novelty is a family artifact.

## Decompose the round-robin matrix (Social Relations Model)

Round-robin dyadic data (everyone rates/predicts everyone) is exactly what the **Social Relations
Model** (Kenny & La Voie) is built to decompose. Fit predictability of dyad (A predicts B on
story s) as a crossed random-effects / bilinear model:

```
predictability(A→B, s) = μ
                        + generator_effect(B, s)     # intrinsic surprisingness of B's twist  <-- the quantity we want
                        + predictor_skill(A)          # A's general ability to anticipate others
                        + dyadic_similarity(A, B)     # the confound: corpus/lineage overlap
                        + error
```

Estimate `dyadic_similarity` explicitly (seed it with the gradient tier, or with an empirical
behavioral distance between A and B) and **regress it out**; the residual `generator_effect` is
your **corpus-controlled surprise**. This gives the confound a named home instead of hoping it
averages away, and it separates a good *predictor* (high `predictor_skill`) from a distinctive
*generator* (low predictability-to-others after controlling similarity).

## Within-family predictability *is* a finding, not just noise

TwistBench documents mode collapse and the "artificial hivemind" qualitatively (10 of one model's
30 stories collapse onto a single "dead-spouse grief-denial" reveal). The predictability matrix
makes this **quantitative**: dense within-family blocks and sparse cross-family blocks are a direct
measure of how homogeneous the model population is. Reported as a **hivemind index** (mean
within-family predictability − mean cross-family predictability), it is the first number on "how
much of an LLM's apparent creativity is a shared-training-data artifact." The confound and the
result are the same phenomenon viewed two ways.

## Gating: unpredictability alone rewards randomness

Surprise-as-unpredictability has TwistBench's exact "breaking the world model" failure baked in:
a **novel-but-incoherent** twist is also hard to predict, so raw unpredictability would score
word-salad as maximally creative. Therefore the predict-game replaces **only the surprise
dimension**; keep TwistBench's **coherence and realism as gates**. A twist earns surprise credit
only when it is coherent (prepared by the setup, nothing retracted) and realistic (no ghosts /
dreams / simulation escape hatch). Formally, mirror TwistBench's gating: count surprise only when
`realism = max` and `coherence` clears threshold.

Note a subtlety this forces you to respect: a *well-made* twist is coherent — i.e. foreshadowed —
yet still not the *modal* continuation. So predictability must be measured against the
**distribution of plausible continuations**, not against "was it derivable in hindsight." A twist
can be fully prepared (high coherence) and still improbable a priori (high surprise); those are not
in tension. This is the "well-made surprise" (Chieppe et al.).

## Metrics

- **Per generator (creativity):** mean surprise to each tier; headline = **distinctiveness** =
  gated surprise at maximum predictor distance (T3/T4).
- **Per predictor (theory-of-others):** `predictor_skill` — how well it anticipates others'
  twists. (This is a small window onto Game 2, metacognition, without committing to it.)
- **Population:** hivemind index (within − cross family predictability); the predictability matrix
  itself, clustered.
- **Per generator, the surprise-vs-distance slope** — how much of its "surprise" is family-local.
- **Validation:** correlate behavioral surprise (T4 human-predictability) against TwistBench's
  existing judged-surprise Likert. Agreement validates the cheaper metric; divergence localizes
  where the LLM judge was wrong.

## Confounds and risks (read before building)

- **Predictor ability is a nuisance dimension.** A weak predictor makes everything look
  surprising. Handled by the SRM `predictor_skill` term; do not read raw predictability without it.
- **Embedding-match noise (free-gen mode).** Two models may produce the *same* twist worded
  differently (false low-similarity) or *different* twists that embed close (false high-similarity).
  Mitigate with the forced-choice mode and/or a human/LLM "same-twist?" adjudication on a sample.
- **Setup leakage.** A heavily foreshadowed setup can make the twist near-deducible; that is
  coherence, not lack of creativity (see the well-made-surprise note). Do not treat "predictable
  from a rich setup" as automatically uncreative — control for setup informativeness.
- **Similarity metric for the gradient.** Tier labels (family/base/post-training) are a proxy;
  lineages are often undisclosed. Consider an *empirical* behavioral distance between models
  (e.g. divergence of their twist distributions on a shared setup set) as a continuous alternative.
- **Do not define creativity as unpredictability alone.** Gate on coherence/realism and anchor on
  humans, or the metric collapses into "rewards noise."

## Relationship to the science program (why this matters beyond literature)

Framed narrowly, this is a better TwistBench. Framed usefully, it develops a **judge-free,
domain-general surprise mechanism** — surprise = inter-subjective unpredictability, gated by
coherence/realism — on a validated literary testbed. That mechanism is exactly what a *scientific*
frame-replacement benchmark needs to escape its reliance on an LLM judge: build the measurement
where there is a clean testbed, then port it to science. Keep that transfer in view.

## Next steps

1. Reuse the TwistBench corpus: 71 models × up to 30 stories + 18 expert-human stories, each with
   an extracted setup/reveal and twist embedding.
2. Choose a predictor set that deliberately spans the gradient (several within-family clusters +
   cross-family + a human batch). Under-sampling any tier flattens the curve you are trying to read.
3. Run Game 1 in both modes (free-gen embedding-similarity; forced-choice ranking). Build the
   N×N matrix.
4. Fit the SRM; report the corpus-controlled `generator_effect`, the hivemind index, and the
   per-generator surprise-vs-distance slope.
5. Validate against TwistBench's judged surprise and the human anchor.

## References

- Schapiro et al., *TwistBench* (transformational creativity via literary plot twists).
- Chieppe et al. 2022, *Bayesian modelling of the well-made surprise* (ICCC); Itti & Baldi,
  Bayesian surprise.
- Kenny & La Voie, the **Social Relations Model** for round-robin dyadic data.
- Jiang et al. 2025, *Artificial hivemind*; Zhang et al. 2025, *NoveltyBench* — the homogeneity
  this quantifies.
