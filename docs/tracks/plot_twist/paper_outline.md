# Planned submission — PT²CB (single benchmark + method paper)

**PT²CB — Plot Twist for Transformational Creativity Benchmark** (ASCII: `PT^2CB`).
Working title: *Surprising Yet Coherent: Benchmarking and Eliciting Transformational
Creativity in LLMs via Conceptual-Space Axiom Modification.*

Target: **ARR August 2026 cycle** (submission deadline **Aug 3, 2026**; commits to
**EACL 2027**). ACL Rolling Review template, 8pp long paper + unnumbered Limitations.
**One paper, three claims** — the benchmark, the frontier-model gap, and the method that
closes it. This doc is the section-by-section plan; the technical spec is in
[design.md](design.md); status/roadmap in [progress.md](progress.md). Prose follows
[writing_advice.md](../../writing_advice.md).

---

## The arc in one paragraph

Transformational creativity — restructuring a conceptual space, Boden's third and hardest
mode — is unmeasured for LLMs; the everyday instance is the **plot twist** (the reveal
forces the reader to *reinterpret* prior events). **(§3)** We introduce **PT²CB**, a
benchmark that scores a twist structurally and automatically using the lab's own graphical
theory of transformational creativity
([SBV, ICCC 2025, arXiv 2504.18687](https://arxiv.org/abs/2504.18687)): a twist *is* an
**axiom modification** of the reader's story-DAG, so quality = **surprise** `T_mod(a*)` ×
**coherence** `preservation(a')` — "surprising yet coherent" as novelty ×
appropriateness on the graph. **(§4)** Frontier models — *even with thinking and high token
budgets* (CREATE's finding, replicated here) — produce weak twists; PT²CB has large
headroom and its score is creativity-specific (survives capability control). **(§5)** We
then close the gap with **CSAM**, an inference-time method that makes the model externalize
the story-DAG `G`, perform a controlled axiom flip `G → G'`, and continue — beating frontier
models (incl. thinking) on PT²CB and in blinded human eval, with `T_mod`/`preservation`
explaining *why*.

## Three claims (per writing_advice.md)

1. **PT²CB measures transformational creativity** — a theory-grounded, automatically scored
   benchmark whose metric separates true/predictable/random twists, tracks human
   twist-quality, and is **creativity-specific** (predicts creative writing with specificity
   after capability control; incremental over exploratory metrics DAT/CDAT).
2. **Frontier models have a transformational-creativity gap** — large headroom on PT²CB, and
   **thinking/scale does not close it** (CREATE-style probe). Motivates a method.
3. **CSAM closes the gap** — explicit story-DAG + controlled axiom modification beats
   frontier models (incl. thinking, compute-matched) on PT²CB *and* in blinded human eval;
   `T_mod(a*)` predicts human-rated surprise, `preservation(a')` predicts coherence.

Lead with the arc 1→2→3; it is one cohesive narrative, not a grab-bag.

## §1 — Introduction

Context (creativity in LLMs, Boden's three modes, the unmeasured transformational one);
the construct (plot twist = reader-axiom modification, "surprising yet coherent"); the
documented deficiency (LLM twists poorly foreshadowed, low plot diversity —
[Echoes, 2501.00273](https://arxiv.org/pdf/2501.00273);
[Human-Level Narratives?, 2407.13248](https://arxiv.org/html/2407.13248v1)); the three
contributions; a Figure 1 that carries the arc (the surprise × coherence plane with
frontier models clustered low and CSAM pushing into the high-high corner).

## §2 — Background & theory

- **Boden's three modes**; the lab's portfolio covers exploratory (DAT/CDAT/PACE, DRAT) and
  combinatorial (comb_eval, kg_creat) — transformational is the gap this paper fills.
- **SBV graphical theory** (the load-bearing background): conceptual space = finite DAG;
  axioms = sinks; `depends(v)`, `T_mod(v)=|depends(v)|`; Thm 4 (axiom modification is
  maximally transformative). Full restatement in [design.md](design.md).
- **The story-DAG mapping** (axioms = reader assumptions, artifacts = narrated events with
  support sets, a twist = axiom flip). The unifying point: all three creativity modes are
  operations on one graphical substrate.
- **Related work**: CREATE ([2603.09970](https://arxiv.org/abs/2603.09970)) — methodological
  cousin (open-ended generative task, objective automatic grading, specificity × diversity,
  "thinking doesn't always help"), *different domain* so inspiration not baseline; creative-
  writing benchmarks (Arena-CW, EQ-Bench-CW, Mazur-CW); narrative datasets (WritingPrompts,
  STORIUM, WHODUNIT, Flawed Fictions); story-planning / plan-and-write methods (the baseline
  family CSAM must beat).

## §3 — The PT²CB benchmark

**Task.** Given a writing prompt, generate a short story with a plot twist (optionally `k`
distinct twists for the diversity term). Items: `(S, r)`, story + reveal point.

**Primary metric — LLM-as-judge, fixed rubric (decided 2026-06-07).** The scalable
benchmark score is an **LLM judge applying a fixed rubric** to each twist, scoring the
theory-derived dimensions: **surprise** (how much of the prior story must be reinterpreted),
**coherence** (do prior events still cohere — ideally get re-explained — after
the reveal), the **joint "surprising-yet-coherent"**, and **overall**. The rubric is
*operationalized directly from the SBV theory* (surprise ≈ `T_mod`, coherence ≈
`preservation`), so the theory grounds the rubric even though scoring is judge-based —
the same pragmatic choice CREATE and EQ-Bench-CW make, and what makes the benchmark cheap
and saturation-resistant. **Diversity** `D` over `k` twists (CREATE-style greedy
quality×diversity) is the set leg.

**Secondary metric — the structural SBV score (theory-pure, judge-free).** An extractor
reconstructs the reader-DAG `G`, locates the flipped axiom `a*→a'`, and computes
`T_mod(a*) × preservation(a')`. Role here: (i) a judge-free corroboration that the rubric
tracks the *structure* it claims to, and (ii) the **mechanism analysis** for the method
(§5). It is *not* the make-or-break path — demoting it to secondary de-risks the benchmark
(no whole-corpus extractor-reliability dependency).

Why it is *creativity* and not quality: both metrics are explicit novelty(surprise) ×
appropriateness(coherence), with coherence defined by *reinterpretation*, not fluency.

**Data (validation of the metric).** Contrast triples (true/predictable/random ending), a
human-rated validation subset, and a **synthetic controlled leg** with known `a*`/`T_mod`
(the pre-registered, hard-to-deny test). Seed prompts = WritingPrompts.

**Judge reliability — the make-or-break number (CREATE standard, 0.94/0.52 bar).**
LLM-judge agreement with the human-rated subset (and rubric inter-judge agreement across
judge models / ranking-invariance). Gates the benchmark; stated as the lead limitation.
(The secondary structural metric's extractor reliability is reported but not load-bearing.)

**Dual virtue to foreground:** theory-grounded **and** objective/automatic/saturation-
resistant (CREATE-style) — humans validate the rubric once, the judge scores any model
cheaply. That is what makes it a benchmark, not a human-eval study.

## §4 — Results: the frontier-model gap

Score the ≈31-model creative-writing/LIB pool (shared with dat_eval/kg_creat) with the
**fixed-rubric LLM judge** (structural metric as judge-free corroboration):
- **Metric validation**: rubric scores order true>predictable>random (effect size on the
  synthetic leg); rubric correlates with the human-rated subset, and the structural
  `T_mod×preservation` corroborates judge-free (claim 1).
- **Validity/specificity** (reuse dat_eval): `r(PTC, creative-writing)`; semi-partial over
  `(Arena-Overall, MMLU-Pro)`; incremental-R² over DAT/CDAT → transformational ≠ exploratory
  (claim 1).
- **The gap + thinking probe** (claim 2): absolute PTC is low across frontier models;
  **thinking/reasoning mode at matched token budget does not lift it** (CREATE-style); scale
  curve is flat-ish — capability alone does not buy transformational creativity. This is the
  headroom the method exploits.

## §5 — Method: CSAM closes the gap

**Procedure.** Build `G` (rich axioms) → narrate `≤t` → controlled axiom flip `G→G'` →
narrate `t'>t`. Inference-time, no training. Full spec in [design.md](design.md).

**Fair scoring.** CSAM outputs are scored by the **same fixed-rubric LLM judge** as the
frontier baselines (CSAM's self-emitted `G` is used only for the mechanism analysis, not for
its benchmark score) — apples-to-apples on PT²CB.

**Baselines (compute/token-matched — the make-or-break confound).** Free-form plan→write→
twist (no graph), thinking mode (equal budget), direct + self-refine, temperature sweep.
Headline: structured axiom-modification beats compute-matched unstructured elicitation.

**Results.** (claim 3) CSAM > frontier (incl. thinking) on PT²CB (rubric judge). Because the
method claim is the load-bearing one, we add — *on top of the benchmark* — a **short blinded
human study** on a subset: pairwise CSAM-vs-baseline (surprise / coherence / joint /
preference, length-matched, IAA, Bradley–Terry, p<.001). It does double duty: confirms the
method win with real humans **and** validates that the rubric judge agrees with humans.
**Mechanism** — `T_mod(a*)`→human-surprise, `preservation(a')`→human-coherence (the
secondary structural metric earns its keep here). **Ablations** isolate the active
ingredient (graph vs prose plan; explicit flip vs "just add a twist"; model-chosen vs
high-`T_mod` vs random axiom; single vs multi; combiner).

## §6 — Discussion & limitations

Unifying claim (all three Boden modes on one graphical substrate); the surprise–coherence
tradeoff mirrors the combinatorial novelty–utility tradeoff
([Schapiro et al. 2025, 2509.21043](https://arxiv.org/abs/2509.21043)). **Limitations up
front:** extractor reliability (metric is extraction-based); contamination (famous twists in
pretraining); small n for specificity; gold-DAG subjectivity; scope = plot twists, not all
transformational creativity; reveal-point detection adds a degree of freedom; CSAM gains may
trade against fluency (measured via the fluency control).

## Evidence map (what each claim needs)

- **Claim 1** — discrimination + human-correlation legs + extractor reliability (§3/§4);
  validity/specificity + incremental-R² (§4); synthetic leg as ground truth. Effect-size
  framing, bootstrap CIs (small n; not p<.05).
- **Claim 2** — frontier PTC distribution + thinking-budget probe (§4), compute-matched.
- **Claim 3** — CSAM vs compute-matched baselines on PTC + blinded human eval (p<.001) +
  mechanism regressions + ablations (§5).

## Open decisions (resolve in pilot)

- DAG schema + extractor (serialization, one-shot vs staged, which extractor model — pre-
  register to avoid DRAT anchor-bank degrees of freedom).
- `preservation` operationalization (NLI / judge / support-set re-derivation); `T_mod`
  normalization; PTC combiner; diversity aggregator.
- Reveal-point detection (self-marked vs extractor segmentation).
- Whether CSAM and the frontier eval **share the seed-prompt pool** (efficient) or stay
  disjoint (clean claim separation).
- Gold-DAG annotation protocol; expert-writer recruitment for human eval.
- Optional: a per-model validity/specificity leg as a secondary table (reuse dat_eval).
