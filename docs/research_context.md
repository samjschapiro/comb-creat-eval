# Research Context

## Overall research

Investigation of automatic creativity metrics for large language models.
The recurring question across tracks: when a metric "predicts" creative
ability, is it picking up something creativity-specific, or just
tracking general model capability?

## Active tracks

### dat_eval (primary, ICML 2026 GenAI-Creativity workshop + NeurIPS 2026)

Evaluates three psycholinguistic semantic-distance creativity tests
— DAT, CDAT (and its novelty / appropriateness components), PACE —
on 54 LLMs across three embedding models (GloVe, FastText, SBERT)
and six external benchmarks spanning three target constructs:
creative writing (Arena CW, EQ-Bench CW, Mazur CW), divergent
thinking (Hivemind, NoveltyBench Utility), and scientific ideation
(LiveIdeaBench). Each test is measured on two criteria: *validity*
(raw Pearson r with the benchmark) and *specificity* (semi-partial
r(X, Y − Ŷ_g), Y residualised on a 2-proxy capability stack of
Arena Overall + MMLU-Pro). A covariance-PSD bound (proven in the
appendix) gives a per-benchmark theoretical ceiling on attainable
specificity.

**Status**: full eval run complete; analysis pipeline complete;
draft past second-pass rewrite, near submission. Two parallel
submission variants live in `papers/iccc-2026/`: `main.tex` (ICML
2026 GenAI-Creativity workshop) and `main_neurips.tex` (NeurIPS
2026 main track). Section files (`sections/`) and tables
(`tables/`) feed the ICML build; NeurIPS-only variants live in
`sections_neurips/` and `tables_neurips/`.

**Headline findings**:
- Specificity, not validity, is what separates a creativity test
  from a capability proxy.
- Test effectiveness varies sharply by construct: DAT is the best
  predictor of creative writing; CDAT is the best predictor of
  divergent thinking; PACE has high raw validity on creative
  writing but its specificity collapses under capability control,
  so it is mostly a capability proxy.
- None of the three tests is a valid-and-specific predictor of
  scientific ideation (LiveIdeaBench, n=17): all observed
  specificities are exploratory.
- Across all panels, observed tests sit well below the theoretical
  ceiling — leaving meaningful room for new test designs.

**Sources of truth** (key values updated 2026-05-02):
- MMLU-Pro: TIGER-Lab leaderboard CSV
  (`TIGER-Lab/mmlu_pro_leaderboard_submission`), not AA.
- Mazur CW: `lechmazur/writing` GitHub at commit `80b7f17`.
- EQ-Bench CW: `eqbench.com/creative_writing.js`
  (`leaderboardDataCreativeWritingV3`).
- Specificity computation: true semi-partial `r(X, Y − Ŷ_g)`
  (was full partial pre-2026-05-02; values shifted accordingly).

### new_tests (active, JMLR/ICML preprint in `papers/iccc-2026`)

Designs and evaluates *new* creativity tests on top of the dat_eval
infrastructure, motivated by the dat_eval finding that no existing
semantic-distance test is a valid-and-specific predictor of
LiveIdeaBench (scientific ideation). Headline test:
**DRAT (Divergent Remote Association Test)** — a hybrid of RAT and
DAT. The model produces $\geq 10$ words bridging a $k$-anchor
quadruple ($k = 4$ in the headline configuration); each word is
gated by an embedding-based "near any anchor" utility ($\max$
aggregator over per-anchor cosine), and survivors are scored by
mean pairwise embedding distance — DAT scoring on a RAT-style
constrained vocabulary.

Other benchmark wrappers added under `src/new_tests/` to support
the broader pool: `eqbench_cw`, `hivemind`, `liveideabench`,
`noveltybench`, `rat` (zero-shot strict accuracy on classic 30-item
RAT). Configs in `configs/new_tests/`, runners in
`scripts/new_tests/` and `src/new_tests/scripts/`. Detailed
ongoing notes in `docs/tracks/new_tests/progress.md`.

**Status (2026-05-09)**: DRAT pool n = 20 on LIB after the LIB
facet refresh. Headline $k=4$ scientific-terms cell is significant
on both axes ($v = +0.57^{**}$, $s = +0.50^{*}$). Anchor banks +
RAT prompt now documented in the appendix. Two ablations live —
anchor count $k \in \{2,3,4\}$ and vocabulary corpus
(scientific-terms vs. ConceptNet relation-distant) — both reported
in `04_drat.tex`. Utility-gate aggregator ablation
($\max$ / $\min$ / $\mathrm{avg}$) in `06_appendix.tex`.

### kg_creat (active; venue TBD — reframed 2026-06-04)

**Current (2026-08-22): "Kombine."** Now a three-task combinatorial-creativity benchmark
(association / analogy / blending) over a **flat curated domain-balanced entity pool** (the
CREATE-style seed-BFS graph was dropped — person-biased and unused, since the model connects from
its own knowledge). Each artifact is scored on four criteria — utility, surprise, originality,
emergent creativity. First pilot: factual validity discriminates models cleanly; blending
underperforms (arbitrary anchors rarely admit a 2nd sense) and emergent creativity is not yet a
discriminating signal (judge too lenient). A jsPsych human-generation study is built. See
`docs/tracks/kg_creat/progress.md`. (Older framing below.)

Re-purposes the **comb_eval / Comb-Creat** task setup (constrained labeled-
graph pathfinding with novelty×utility scoring) into a **test-time creativity
task on a real knowledge graph** (Wikidata et al.), administered to frontier
models without retraining — the new_tests survey's "Gap A". **Headline
(reframed 2026-06-04, per Jonah Black):** a **taxonomy of constraints, each a
minimal abstraction of a real-world rule creative generation must obey**, and
the **per-constraint-type novelty–utility tradeoff** — which constraints LLMs
satisfy while keeping novelty high vs which force a tradeoff — as a
**mechanistic diagnostic of the ideation–execution gap** (prior work showed the
*what*, not the *why*). Real entities, OpenRouter only, **no GPU cost**.

**Metric spine:** per constraint type, **ideation** = novelty (DAT embedding
remoteness) of the model's *emitted* path, vs **execution** = exact constraint
satisfaction + judge factuality (failure channels broken out); the
(novelty × satisfaction) 2×2 across types is the core result. **Matched
endpoint-bundle sampling** (fix `(u,v,h)`, toggle only the constraint) makes the
tradeoff causal in constraint *type*. **The LiveIdeaBench validity/specificity
arm is DROPPED** (n≈20, underpowered); the contribution is the diagnostic, not a
correlation.

**Vs CREATE** ([Wadhwa et al. 2026, arXiv 2603.09970](https://arxiv.org/abs/2603.09970),
which already does Wikidata multi-hop paths at test time): the moat is the
**grounded constraint taxonomy + the per-constraint ideation–execution
decomposition** (CREATE = the no-constraint baseline). Verification (open-KG +
LLM judge) and novelty (validated DAT measure) are **shared ground, not
differentiators**. Central risk: *grounding rigor* — each constraint's mapping
to a real-world rule must be defensible/cited.

**Status (2026-07-21)**: **the paper's intended headline result now has data.**
Regime A ran at scale — 8 models (Llama-3.1-8B → Sonnet-4.6) × 30 *fixed* endpoint
bundles × {baseline, exclusion, inclusion, inclusion-rare, ordering, categorical}
= 7,159 judged paths (report: `docs/reports/2026-07-21_kg_creat_regimeA/`).
Constraint types are **not equally hard** (ordering Δsat −0.45 and buys no
novelty; categorical −0.13 and buys the most), constraints **do not degrade
factuality** (a flat ~34–40% tax in every cell including baseline — the entire
cost lands in the constraint channel), and **ordering fails as double-inclusion
rather than as sequencing** (only 11.5% of its failures are real order
violations). Constraints are defined over embedding-derived **relation classes**
with **per-bundle baseline-derived targets**, so each bites by construction rather
than by assumption. Blending was reframed to a **single stimulus** (one anchor,
two parallel structures emanating outward) and smoke-tested; it has not been run
at scale. The earlier analogy result stands (~26% best-model success on 200 random
pairs; `docs/reports/2026-07-20_kg_creat_analogy/`). The blind judge-reliability
human pass is **still owed**, and is now load-bearing since all five Regime-A
cells are judged rather than exactly checked. Detailed state in
`docs/tracks/kg_creat/progress.md`.

### plot_twist — TwistBench (SUBMITTED 2026-06-24, Sci-FM @ COLM 2026)

A **benchmark paper** for *transformational* creativity — the unrepresented third
Boden mode (exploratory = dat_eval/new_tests; combinatorial = comb_eval/kg_creat).
**TwistBench** (renamed from T²C-Bench) has 71 LLMs and 18 expert-human authors write
plot-twist short stories, scored by a 3-judge LLM ensemble on surprise/coherence/realism
plus reveal diversity. The **headline metric is a realism-gated equal-weight z-composite**:
surprise/coherence count only when a story is fully realistic (the "fair-play" gate, not a
4th facet). Result: **humans rank #1/72**; LLMs show two failure modes — *mode collapse*
(low diversity) and *breaking the world model* (unrealistic twists) — and neither
reasoning-effort scaling nor prompting closes the gap; reasoning traces are process-level
homogeneous (twist-first). The earlier CSAM-*method* plan (axiom-modification elicitation,
blinded human study) was **dropped**; the submitted paper is benchmark + analysis only. The
superseded method framing remains below for history.

**Empirical claim**: CSAM beats *compute/token-matched* baselines (free-form
plan-then-twist, thinking mode, self-refine, temperature) at producing
"surprising-yet-inevitable" twists in **blinded human evaluation**, and
`T_mod(a*)` predicts human-rated surprise. Seed prompts = WritingPrompts; a
synthetic controlled leg gives ground-truth `T_mod`.

**One paper — PT²CB** (Plot Twist for Transformational Creativity Benchmark),
benchmark + method in one arc: (§3) the benchmark scores a twist by a
fixed-rubric LLM judge whose surprise/inevitability dimensions come from the SBV
theory, with a judge-free structural score `T_mod × preservation` (+ CREATE-style
diversity) as a check; (§4) frontier models score low and thinking/scale doesn't
help; (§5) CSAM closes the gap, with a short human study validating the win.
**Target: ARR August 2026 cycle (deadline Aug 3 2026; commits to EACL 2027).**
Overleaf paper in `papers/pt2cb-iclr-2027/` (folder name predates the venue
switch; ACL/ARR template). **Status (2026-06-11)**: benchmark (§3) + frontier
eval (§4) built and run — ≈72 systems scored on the 4-facet equal-weight
z-composite (surprise, coherence, diversity, **realism**), with **expert humans
ranking #1 overall by never collapsing on a dimension**; headline scorecard
figure in the paper. Pending: the thinking/scale probe, the CSAM method (§5), and
the blinded human eval.

### comb_eval (background / exploratory)

Earlier track exploring combinatorial-creativity-style evaluation.
Currently dormant; reused only for the Arena-score fetcher and
`benchmarks.json` schema that dat_eval consumes.

## Cross-track conventions

- All API calls go through OpenRouter via the `openai` Python SDK.
- Per-model scores live in `data/<track>/run_v1/<model_key>/`.
- `configs/comb_eval/benchmarks.json` is the shared per-model
  benchmark store, augmented in place by `add_eqbench_scores.py`,
  `add_hivemind_scores.py`, `add_mazur_scores.py`,
  `add_mmlu_pro_scores.py`, `add_noveltybench_scores.py`, and
  `add_arc_agi_scores.py`. **All of these values are transcribed from
  external papers/leaderboards**, so a benchmark's n cannot be raised
  by re-running the benchmark ourselves with a different pipeline (the
  scale won't match — verified for NoveltyBench, 2026-07-23). To raise
  n, fill *our* side instead: run DAT/CDAT/PACE and pull Arena/MMLU for
  models the external source already scored.
- Long-running scripts respect `budget_usd` in their config and abort
  before exceeding the cap.
- Safety scripts in `scripts/safety/` (`status.sh`, `kill_all.sh`,
  `cost_tracker.py`) and `docs/AI_OPERATIONS_PROTOCOL.md` are
  consulted before launching any expensive operation.
