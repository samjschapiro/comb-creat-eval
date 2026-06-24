# plot_twist — progress

## Current status (2026-06-24) — SUBMITTED

**The paper shipped as a benchmark paper, "TwistBench", to the Scientific Understanding of
Foundation Models (Sci-FM) workshop @ COLM 2026.** The original CSAM-*method* plan below was
dropped; the paper is purely the **benchmark + analysis**:

- **Task:** 71 LLMs (+ 18 expert-human stories) write plot-twist short stories.
- **Metric (headline):** realism-**gated** equal-weight z-composite — surprise/coherence count only
  for fully realistic stories (realism==5; the "fair-play" gate), plus reveal diversity. Realism is
  the gate, not a 4th facet. Centralized in `src/plot_twist/join.py`. **Humans rank #1/72** (z≈+2.0).
- **Findings:** LLMs underperform humans; two failure modes — **mode collapse** (frontier models lack
  diversity; e.g. Opus-4.5 collapses to the dead-spouse cluster c9) and **breaking the world model**
  (surprising+coherent but unrealistic twists; e.g. Gemini-2.5-Pro/DeepSeek "synthetic being"). Neither
  reasoning-effort scaling nor prompting (be-creative, in-context-regen) lifts the human ceiling
  (in-context-regen ties it for Sonnet-4.5 via diversity). Reasoning traces show process-level
  homogeneity (twist-first, then retrofit the plot).
- Paper repo: `papers/pt2cb-iclr-2027/` (Overleaf, branch master). Latest full-session writeup:
  [logs/2026-06-24/1106_twistbench-realism-gate-and-submission.md](../../logs/2026-06-24/1106_twistbench-realism-gate-and-submission.md).

The sections below document the earlier (superseded) method-paper framing and remain for history.

---

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

## Status — 2026-06-11 (Phases 1–3 substantially done — benchmark + frontier gap)

The **benchmark (§3) and frontier eval (§4) are built and run**; the methods leg (§5
CSAM) and human eval (Phase 5) are still pending.

Done:

- **Pipeline** in `src/plot_twist/scripts/`: `fetch_pd_stories` (human gold), `run_generate`
  (open-ended "write a story with a plot twist", 3 temps × 10 samples, length-matched,
  durable per-story + per-model subfolders), `run_annotate` (setup/reveal/why), the
  rubric judges (`run_rubric_*`, 3-judge ensemble, median agg, judges disjoint from
  generators), `run_dsi`, `run_realism`, `classify_twists`, `analyze_collapse`,
  `correlate_dsi`, `judge_reliability`/`grm_irt`/`bayes_grm_jrt`, and `make_tc_barplot`.
- **Scoring dimensions (4, equal-weighted z-composite):** surprise, coherence (rubric
  judges); diversity `Div` (mean pairwise reveal-embedding dissimilarity, all-mpnet);
  **realism** (`run_realism`, grounded-vs-fantastical, anti-gaming). Overall = mean of the
  four z-scored facets across the evaluated pool (AGC-style mean-z).
- **Frontier eval:** ≈72 systems scored (frontier + AGC sweep), human gold = STRONG-only
  ceiling. Headline result in `data/plot_twist/tc/` + the scorecard figure: **expert
  humans rank #1 overall by never collapsing on a dimension**; strong models show
  characteristic deficits (DeepSeek low realism; Claude Opus low diversity).
- **Headline figure** `tc_scorecard.png` (Overall ranking + 2×2 per-dimension top-10) in
  the paper (`papers/pt2cb-iclr-2027/`, Overleaf).

Pending: the **thinking/scale probe** (§4 CREATE-style; `llm.py` has the `reasoning`
param), refreshed validity/specificity correlations on the 4-facet overall, the **CSAM
method** (§5, Phase 4) and **blinded human eval** (Phase 5).

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
