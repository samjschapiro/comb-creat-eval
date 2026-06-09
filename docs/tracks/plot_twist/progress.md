# plot_twist — progress

## Goal

A **methods paper**: an inference-time procedure — **conceptual-space axiom
modification (CSAM)** — that elicits *transformational* creativity (plot twists) from
LLMs by making them externalize a story's conceptual-space DAG `G` and perform a
controlled axiom flip `G → G'`. The empirical claim: CSAM produces measurably more
**"surprising-yet-coherent"** twists than *compute-matched* prompting / reasoning /
temperature baselines, in **blinded human evaluation** — and the transformative
potential `T_mod` of the flipped axiom predicts human-rated surprise.

Grounded in the lab's own
[Transformational Creativity in Science: A Graphical Theory (Schapiro, Black,
Varshney, ICCC 2025, arXiv 2504.18687)](https://arxiv.org/abs/2504.18687): a plot twist
*is* SBV's Thm-4 axiom-modification operation applied to the reader's world-model.
Full spec in [design.md](design.md).

## One paper — PT²CB (benchmark + method), three claims

**Single paper, ARR Aug 2026 cycle** (retargeted 2026-06-08 from ICLR 2027): the benchmark and the method are one
arc — the benchmark creates the headroom, the "thinking doesn't help" result sharpens
the gap, and the method (CSAM) closes it; benchmark and method validate each other.
Section-by-section plan in [paper_outline.md](paper_outline.md):

- **§3 Benchmark — PT²CB** (Plot Twist for Transformational Creativity Benchmark): an
  independent extractor reconstructs the reader story-DAG post-hoc from *any* story and
  scores `PTC = T_mod × preservation` (+ a CREATE-style diversity term); theory-grounded
  *and* automatically scored. Metric validated (true>predictable>random, human-rated,
  creativity-specific under the dat_eval validity/specificity framework).
- **§4 Results — the frontier gap**: frontier models score low on PT²CB and
  **thinking/scale does not close it** (CREATE-style probe).
- **§5 Method — CSAM**: the model externalizes `G`, performs a controlled axiom flip
  `G→G'`, continues; **beats frontier models (incl. thinking, compute-matched)** on PT²CB
  and in blinded human eval; `T_mod`/`preservation` explain why. Spec in [design.md](design.md).

Fairness note: CSAM (§5) is scored by the **same independent post-hoc extractor** as the
frontier baselines (§4), not its self-emitted `G` — apples-to-apples on PT²CB.
**Target: ARR August 2026 cycle — deadline Aug 3, 2026; commits to EACL 2027.**
ACL Rolling Review template, 8pp long paper + unnumbered Limitations.

## Why a separate track (not new_tests / kg_creat)

Per [repo_usage.md](../../repo_usage.md), tracks separate *fundamentally different
approaches*:
- `dat_eval` / `new_tests` — *exploratory* creativity, semantic-distance metrics.
- `comb_eval` / `kg_creat` — *combinatorial* creativity, constrained pathfinding.
- `plot_twist` (this track) — *transformational* creativity: a **benchmark** (PT²CB,
  story-DAG axiom-modification scoring) **+ method** (CSAM) in one paper. Reuses the SBV
  theory and the verification ethos, but the research questions (can we *measure* and
  can we *elicit* transformational creativity?) and the substrate (narrative, human-eval)
  are distinct.

## Headline contributions (target — per writing_advice.md, 1–3 claims)

Theme: *transformational creativity is a distinct, measurable axis of LLM creativity that
frontier models (even with thinking) lack — and an explicit DAG axiom-modification method
supplies it.* Full section-by-section plan: [paper_outline.md](paper_outline.md).

1. **PT²CB measures it** (§3) — a theory-grounded, automatically scored benchmark whose
   metric separates true/predictable/random twists, tracks human twist-quality, and is
   **creativity-specific** (validity/specificity over capability; incremental over DAT/CDAT).
2. **Frontier models have a gap** (§4) — low PT²CB scores, and **thinking/scale does not
   close it** (CREATE-style probe). Motivates a method.
3. **CSAM closes it** (§5) — explicit story-DAG + controlled axiom flip beats frontier
   models (incl. thinking, compute-matched) on PT²CB *and* in blinded human eval;
   `T_mod`→human-surprise, `preservation`→human-coherence explain why.

The arc 1→2→3 is the narrative. (Unifying flourish: the surprise–coherence tradeoff
mirrors the combinatorial novelty–utility tradeoff,
[Schapiro et al. 2025, 2509.21043](https://arxiv.org/abs/2509.21043).)

## Status — 2026-06-07 (Phase 0 — track scaffolded)

Design spec only. [design.md](design.md) written: SBV→story-DAG mapping, the CSAM
procedure, compute-matched baselines, ablation grid, datasets, human-eval protocol,
risks, reuse map. No code yet.

Locked decisions:
- **One paper** (benchmark + method), **ARR Aug 2026 cycle → EACL 2027** (retargeted
  2026-06-08 from ICLR 2027): PT²CB benchmark (§3) → frontier gap (§4) → CSAM method (§5).
  Benchmark and method validate each other.
- **PTC is the benchmark metric** (extracted post-hoc, scores any model) *and* the
  analysis instrument that explains why CSAM works.
- **Compute/token-matched baselines are mandatory** (the make-or-break confound for §5).
- **Extractor reliability is the make-or-break number for §3** (CREATE 0.94/0.52 standard).
- **CSAM scored by the same independent extractor** as frontier baselines (fair §4 vs §5).
- **Primary §5 evidence = blinded human eval**; PT²CB automatic score is the scalable backbone.
- **Seed prompts = WritingPrompts**; synthetic controlled leg for ground-truth `T_mod`;
  Flawed-Fictions-style perturbation as the incoherent negative control.
- **Scope = plot twists / transformational creativity**, not general creativity.
- **Seed prompts = WritingPrompts**; synthetic controlled leg for ground-truth `T_mod`;
  Flawed-Fictions-style perturbation as the incoherent negative control.
- **Scope = plot twists / transformational creativity**, not general creativity.

## Phased roadmap (benchmark-first — it is the prerequisite)

1. **Phase 1 — PT²CB metric core (§3).** `src/plot_twist/` — DAG schema (`𝒜`, rules,
   artifacts + edges), the **post-hoc extractor** (`story → G, a*, a'`), and
   `T_mod`/`preservation`/diversity scoring. Smoke on hand-written contrast triples
   (true/predictable/random, e.g. the detective-narrator example) to confirm ordering.
2. **Phase 2 — metric validation + reliability (§3).** Contrast triples + synthetic
   controlled leg (ground-truth `T_mod`) + human-gold subset; **extractor reliability**
   (recall/precision/agreement, CREATE 0.94/0.52 standard) — the make-or-break number.
3. **Phase 3 — frontier eval / the gap (§4).** Score the ≈31-model creative-writing/LIB
   pool; validity + specificity (reuse dat_eval) + incremental-R² over DAT/CDAT; the
   **thinking/scale probe** (CREATE-style) — establish the headroom.
4. **Phase 4 — CSAM method (§5).** 4-step runner (build `G` → narrate `≤t` → flip `G→G'` →
   narrate `t'`) + compute-matched baselines + faithfulness checks. Scored by the **same
   independent extractor** as §4. Show CSAM > frontier (incl. thinking) on PT²CB.
5. **Phase 5 — human eval + mechanism + ablations.** Pre-registered blinded pairwise study
   (crowd + expert writers; Bradley–Terry, IAA, p<.001); `T_mod`→surprise /
   `preservation`→coherence regressions; graph-vs-prose, flip-vs-no-flip, axiom-choice,
   single-vs-multi, combiner, extractor ablations.
6. **Phase 6 — write-up.** ARR Aug 3 2026 cycle (Overleaf: `papers/pt2cb-iclr-2027/`,
   folder name predates the venue switch; ACL/ARR template).

## Hypotheses & experiments

Two falsifiable hypotheses logged in [HYPOTHESES.md](../../HYPOTHESES.md), with
experiment scaffolds in [experiments.md](experiments.md):

- **H4** — embedding semantic-distance metrics (DSI, DAT/CDAT) are *blind to
  transformational creativity*: DSI predicts neither twist presence nor twist
  quality, yet still predicts general creative-writing quality, while the structural
  `T_mod × preservation` predicts twists — a double dissociation. (Exp 1; feeds §4
  Claim 1, the "embedding baselines fail" result.)
- **H5** — *human twists beat LLM twists* under the fixed-rubric judge, with the gap
  concentrated in **coherence/preservation**, surviving a prose-quality control
  and a matched-prompt head-to-head. (Exp 2; feeds §4 Claim 2, the frontier gap, and
  motivates CSAM.)

## Next steps

1. ~~Set up the LaTeX skeleton~~ DONE (2026-06-08): ACL/ARR skeleton + Figure 1 in
   `papers/pt2cb-iclr-2027/` (Overleaf). Next: fill the rubric text and §4/§5 tables.
2. Design the **DAG schema** (parseable axioms/rules/artifacts + edges) + the **extractor**
   prompt; decide `G` serialization. Build `src/plot_twist/` scorer; smoke on contrast triples.
3. Pre-register the extractor-reliability protocol, the synthetic-leg generator, and the
   human-eval dimensions/analysis.

## Open decisions (tracked)

- **`G` serialization** — JSON adjacency vs DSL vs natural-language-with-tags; must be
  reliably parseable *and* not cripple narrative fluency.
- **Axiom-flip selection** as the headline lever — model-chosen vs forced high-`T_mod`.
- **`preservation` operationalization** — NLI non-contradiction of artifacts vs
  LLM-judge "still consistent / better explained" vs support-set satisfiability.
- **PTC combiner** — product / geom-mean / min (Phase 5 ablation).
- **Expert-writer recruitment** for human eval (cost/scale vs construct validity).
- Whether to add a **validity/specificity leg** (per-model CSAM-twist score vs creative-
  writing benchmarks, capability-controlled) as a secondary result reusing `dat_eval`.
