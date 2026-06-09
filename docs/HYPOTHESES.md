# Hypotheses

A project-wide log of falsifiable research hypotheses, with the
predictions and tests that would resolve them. Each entry follows
this template:

```
## H<N> — <short title>

**Track.** <track name>
**Status.** open | testing | supported | falsified | abandoned
**Logged.** <YYYY-MM-DD>

**Statement.** One sentence; falsifiable.
**Motivation.** Why we believe it; what observations led to it.
**Prediction.** Concrete numbers / signs / orderings on specific
benchmarks or experiments.
**Falsification.** What outcome would resolve it as false.
**Tests.** Pointer to scripts/configs that run the test.
**Result.** Filled in when resolved.
```

A hypothesis logged here is a commitment: we run the test, we report
the result, and we record whether the prediction held — even (especially)
if it did not. The point is to make our research path traceable rather
than retroactively narrated.

---

## H1 — CrPO does not improve creativity

**Track.** new_tests
**Status.** open
**Logged.** 2026-05-03

**Statement.** Training a base instruction-tuned LLM on MuCE-Pref via
CrPO ([Ismayilzada et al., 2025](https://arxiv.org/abs/2505.14442))
does not produce a generalizable improvement in creativity. Apparent
gains on MuCE-style held-out tasks reflect distributional fit to the
psychometric-prompt template, not a transferable creative competence.

**Motivation.**
- Creativity is, by construction, the ability to produce novel and
  appropriate responses *outside the training distribution*. A method
  that improves performance only on its training-distribution shape
  has not taught creativity; it has taught distributional fit.
- MuCE-Pref is built from ~12 templated psychometric tasks
  ("Come up with an original and creative {X} for the following
  {Y}: {Z}") with short undergraduate-pool free-text responses. The
  prompt structure, response length, response register, and rating
  rubric are tightly coupled.
- The CrPO paper itself reports that *vanilla DPO on MuCE-Pref* — no
  creativity injection — already beats GPT-4o, Claude-3.7-Sonnet, and
  Gemini-2.0-Flash on the held-out MuCE suite. This is implausible as
  a general creativity result and parsimoniously explained by the
  held-out suite sharing the MuCE-Pref distribution.
- The CrPO paper reports a quality regression on NoveltyBench (SFT
  beats CrPO) — exactly what would be predicted if MuCE training
  fits the psychometric distribution at the cost of behaviour on
  real-world prompts.
- The CrPO paper does not evaluate any of the construct benchmarks
  used in dat_eval (Arena CW, EQ-Bench CW, Mazur CW, Hivemind,
  NoveltyBench Utility, LiveIdeaBench). The transfer claim is
  untested.

**Prediction.** Score the released CrPO checkpoints (Llama-3.1-8B-
Instruct and Mistral-7B-Instruct-v0.3 variants from the
[CNCL-Penn-State HF collection](https://huggingface.co/collections/CNCL-Penn-State/crpo-67d0b11ff358430823dbb3df))
and their SFT-only and DPO-only ablations on the six benchmarks above,
plus a held-out MuCE slice as a control. Predicted result:

- **MuCE held-out (control):** CrPO and DPO show large positive
  deltas vs base.
- **Arena CW, EQ-Bench CW, Mazur CW, Hivemind, NoveltyBench Utility,
  LiveIdeaBench:** CrPO deltas vs the base model are small (within
  ±2σ of zero on the per-benchmark score scale) and inconsistent in
  sign across benchmarks. Specifically:
  - No benchmark shows a CrPO delta ≥ +0.5σ that is also matched by
    a positive DPO delta of similar size — i.e. anything CrPO gains
    vanilla DPO gains too.
  - On at least one benchmark (most likely NoveltyBench Utility or a
    creative writing benchmark), CrPO scores *lower* than the base
    model.

**Falsification.** Hypothesis is falsified if CrPO produces a
consistent positive benchmark delta of ≥ +0.5σ across at least four
of the six external benchmarks, or a single ≥ +1σ gain on
LiveIdeaBench, vs the matched SFT-only baseline.

**Tests.** *(pending implementation — owner to add config / script
paths once written.)* Scoring pipeline is the existing
[src/dat_eval/scripts/score_evals.py](../src/dat_eval/scripts/score_evals.py)
applied to CrPO checkpoint outputs in
`data/new_tests/crpo_replication/`.

**Result.** *(unfilled, awaiting test.)*

---

## H2 — Semantic-distance reward channels inherit dat_eval's construct-binding

**Track.** new_tests
**Status.** open
**Logged.** 2026-05-03

**Statement.** Two of CrPO's four reward channels —
**novelty** (DSI: a single-response semantic-integration measure
computed in embedding space) and **diversity** (mean pairwise
semantic distance across responses to a prompt) — are structurally
the same family of signal that DAT, CDAT-N, and PACE compute. The
[dat_eval paper](../papers/iccc-2026/sections/) shows that this
family of signal is construct-bound: it has predictive power for
creative writing (DAT) and for divergent thinking (CDAT) but no
predictive power for scientific ideation. Optimizing toward a signal
that does not correlate with a construct cannot move the construct.
Therefore CrPO variants whose reward composition is dominated by
DSI and pairwise-distance cannot improve scientific ideation, no
matter how the loss is shaped.

**Motivation.**
- DAT and CDAT-N reduce to averaged pairwise cosine distances between
  embedded words. DSI is the same operation applied across the
  semantic units within a single response. Pairwise-diversity in
  CrPO is the same operation applied across responses. They are the
  same signal at three different granularities.
- dat_eval reports, on LiveIdeaBench (n = 17), Pearson r between
  every semantic-distance test and the benchmark in [-0.11, +0.20]
  with no test reaching p < 0.05 ([results §05](../papers/iccc-2026/sections/05_results.tex)).
  This is a measurement-side null result: the signal does not track
  the construct in this population of models.
- A reward signal that does not correlate with a construct in a
  fixed-policy population cannot, by gradient ascent, move the
  construct in a trained policy. The optimizer can only push the
  policy along directions the reward can distinguish.
- This argument is independent of H1. Even if MuCE-Pref were drawn
  from a perfectly construct-aligned prompt distribution, the
  *channels* CrPO optimizes are themselves construct-bound.

**Prediction.** Score the per-channel CrPO variants — CrPO-nov,
CrPO-div, CrPO-nov-div-sur, CrPO-cre — on LiveIdeaBench. Predicted
result:

- LiveIdeaBench score deltas vs base for CrPO-nov, CrPO-div, and
  any combination of the two are within ±0.3σ of zero on the
  benchmark's per-model score scale.
- The same variants on creative writing benchmarks (Arena CW,
  EQ-Bench CW, Mazur CW) show small positive deltas — consistent
  with DSI-family signals having modest construct-validity for
  creative writing per dat_eval.
- The same variants on divergent thinking benchmarks (Hivemind,
  NoveltyBench Utility) show small-to-moderate positive deltas
  for the diversity-channel variants — consistent with
  pairwise-distance signals having modest construct-validity for
  divergent thinking per dat_eval.

**Falsification.** Hypothesis is falsified if any single CrPO
variant whose reward composition is dominated by DSI or pairwise-
distance produces a LiveIdeaBench delta ≥ +0.5σ vs the matched base
model.

**Tests.** Same pipeline as H1, partitioned by per-channel CrPO
variant. The pattern across (variant × benchmark) is the test, not
any single cell.

**Result.** *(unfilled, awaiting test.)*

---

## H3 — Per-channel CrPO gains mirror the dat_eval per-test
construct-validity pattern

**Track.** new_tests
**Status.** open
**Logged.** 2026-05-03

**Statement.** The construct-binding of a CrPO variant is inherited
from its dominant reward channel rather than from any property of
the loss form, the optimizer, or the dataset. Concretely: ranking
CrPO variants by their per-channel composition will reproduce, on
the construct-benchmark axis, the dat_eval ranking of the analogous
semantic-distance tests on the same axis.

**Motivation.** This is the operational version of H2 and the
mechanism that explains H1's distributional-fit story at the channel
level. dat_eval established a per-test, per-construct ranking:
- DAT (DSI-family) → best for creative writing.
- CDAT (constrained pairwise distance) → best for divergent thinking.
- No semantic-distance test → predictive of scientific ideation.

If reward channels inherit construct-binding from the signal they
implement, the per-variant CrPO benchmark deltas should reproduce
this ranking.

**Prediction.**
- CrPO-nov (DSI-only) shows the largest positive delta among CrPO
  variants on creative writing benchmarks (Arena CW / EQ-Bench CW /
  Mazur CW), and is the strongest creative writing variant relative
  to vanilla DPO.
- CrPO-div (pairwise-distance-only) shows the largest positive
  delta on divergent thinking benchmarks (Hivemind / NoveltyBench
  Utility) among CrPO variants.
- No CrPO variant — including CrPO-cre, which combines all four
  channels — shows a meaningful (≥ +0.3σ) delta on LiveIdeaBench.

**Falsification.** Different ordering of best-channel per construct,
or any channel showing a meaningful LiveIdeaBench gain.

**Tests.** Same pipeline as H1 / H2. Output is a (variant × benchmark)
matrix of deltas vs the matched SFT baseline.

**Result.** *(unfilled, awaiting test.)*

---

## H4 — Embedding semantic-distance metrics are blind to transformational creativity

**Track.** plot_twist
**Status.** open
**Logged.** 2026-06-08

**Statement.** Divergent Semantic Integration (DSI,
[Johnson et al. 2022](https://doi.org/10.3758/s13428-022-01902-8)) — and the
embedding semantic-distance family more broadly (DAT/CDAT) — does **not** predict
(i) the *presence* or (ii) the *quality* of a plot twist, even though it still
predicts general creative-writing quality on the same stories. Embedding distance
captures *exploratory* creativity but is blind to *transformational* creativity.

**Motivation.**
- DSI scores a narrative by mean pairwise contextual-embedding (BERT) distance
  among its words; it is validated to predict human creativity ratings of short
  narratives. It is the narrative-level cousin of DAT/CDAT, which dat_eval showed
  are construct-bound to exploratory/divergent creativity and null for scientific
  ideation. This hypothesis extends that line to transformational creativity. [[H2]]
- A plot twist is transformational: a good twist *re-contextualizes existing
  elements* (high preservation = staying inside the established semantic space), so
  it adds little semantic spread. A metric that measures semantic spread should
  therefore be null — or even slightly negative — w.r.t. twist quality.
- This is the measurement-side claim motivating T2C's structural metric and rubric:
  existing automated creativity metrics cannot see transformational creativity.

**Prediction.** On the T2C contrast triples (twist / predictable / random) and the
human-rated twist set:
- (a) **Presence:** DSI does not separate twist-present from predictable stories —
  classification AUC ∈ [0.45, 0.58] (≈ chance).
- (b) **Quality:** DSI does not correlate with twist quality (rubric or human) —
  |Pearson r| < 0.15, n.s.
- (c) **Dissociation control:** on the *same* stories DSI *does* predict general
  creative-writing quality (r ≳ 0.30, replicating its validated use) — so the null
  is twist-specific, not DSI being broken.
- (d) **Positive contrast:** the structural `T_mod × preservation` (and the rubric)
  *do* predict twist presence (AUC ≥ 0.75) and quality (r ≥ 0.40) on the same data
  → a double dissociation (embedding ↔ exploratory, structure ↔ transformational).

**Falsification.** DSI predicts twist presence (AUC ≥ 0.65) or twist quality
(|r| ≥ 0.30, p < .01) on the held-out set; or the structural metric fails to beat
DSI on the same data.

**Tests.** *(pending)* Experiment 1 in
[tracks/plot_twist/experiments.md](tracks/plot_twist/experiments.md). Reuse the
dat_eval correlation pipeline; DSI computed over T2C stories.

**Result.** *(unfilled, awaiting test.)*

---

## H5 — Human plot twists beat LLM plot twists (and the gap is in coherence)

**Track.** plot_twist
**Status.** open
**Logged.** 2026-06-08

**Statement.** Under the fixed-rubric LLM judge, human-written plot twists score
higher than frontier-LLM-written twists on the twist-specific dimensions (surprise,
coherence); the gap is **concentrated in coherence/preservation**, survives a
prose-quality control, and holds in a matched-prompt head-to-head (not only on
famous stories).

**Motivation.**
- LLM stories are documented as low plot-diversity and as foreshadowing twists
  poorly ([Echoes, 2501.00273](https://arxiv.org/abs/2501.00273); [Human-Level
  Narratives, 2407.13248](https://arxiv.org/abs/2407.13248)). Foreshadowing *is*
  preservation/coherence — the term models should struggle with most (they can
  be surprising, but cannot make the surprise *retroactively fit*).
- This is the human-anchored version of the §4 frontier gap: it sets a ceiling and
  localizes the deficit, motivating CSAM (built to raise preservation).

**Prediction.**
- Famous human twists > frontier-LLM twists on the overall rubric (Δ ≥ 0.5 on the
  rubric scale) and on surprise **and** coherence separately.
- The gap is **larger on coherence than on surprise**.
- **Prose-quality control:** with a prose-quality dimension covaried out, the
  twist-dimension gap remains significant — i.e. it is not just "humans write better
  prose."
- **Matched-prompt head-to-head** (humans + LLMs write twists to the same
  WritingPrompts prompts): humans still > LLMs, by a smaller margin than the famous
  comparison.

**Falsification.** Frontier LLMs match or exceed human twists on the twist
dimensions in the matched-prompt comparison (Δ within ±0.2, n.s.), or the famous-set
gap disappears once prose quality is controlled.

**Tests.** *(pending)* Experiment 2 in
[tracks/plot_twist/experiments.md](tracks/plot_twist/experiments.md). Human-twist
corpus (famous public-domain + WritingPrompts twist-tagged) vs LLM-generated twists,
scored by the fixed-rubric judge blind to authorship.

**Result.** *(unfilled, awaiting test.)*
