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

### kg_creat (active, ICLR 2027 — Kombine)

**Current (2026-09-03): "Kombine."** A three-task combinatorial-creativity benchmark (association / analogy / blending) over a **flat curated domain-balanced entity pool** (283 Wikidata-grounded anchors). The task formalism is unified under a projection operator — analogy invents `h := M[Φ]`; a blend fuses two inputs through a shared textual **generic space `g`**. Each artifact is scored on utility, surprise, and originality (**pool-relative embedding distance**), and — for analogy/blending only — emergent creativity kept as **separate** dimensions, never aggregated; originality is split into **base** (the scored artifact) and **emergent** (the invention), which behave differently across tasks. **Target venue: ICLR 2027**; paper drafting in `papers/kg_creat-iclr/`; a jsPsych human-generation study is built to the same structure but not yet fielded. See `docs/tracks/kg_creat/progress.md`. (Older framing below.)

**Run and judging (2026-09-07)**: **35 models × 30 items/task × 3 tasks** at temp 0.9. The subjective verdicts come from a **3-judge panel of non-subject frontier models** (Haiku 4.5, GPT-5.4, o3; inter-judge ICC(2,3) 0.48–0.67, fair-to-good), with a single judge on the objective per-triple factuality gate — **`claude-haiku-4.5` since 2026-09-07**, replacing `gpt-oss-120b`, which was returning no parsable verdict on a large share of paths and so had been silently scoring them as factuality failures. A **blind 60-item author re-rating** corroborates the panel: 66% agreement over 150 dimension judgments, 75% where the panel is unanimous, and the leaderboard ordering is robust to restricting it to unanimous items (ρ = 0.945). Blending was **re-elicited for the whole pool** after a format fix — triples now carry a **`uv` tag for a slot both inputs organize**, which the previous format made unrepresentable and so forced concatenation by construction. Cumulative spend **$379.74**, with judging now nearly half of it (o3 $88.96 and GPT-5.4 $58.92 as panel judges, more than any subject model).

**Headline findings** (reports in `docs/reports/2026-08-31_…` → `2026-09-03_…`):
- **The three tasks fail in three different ways**: association and analogy on **factual grounding** (**27.1%** of path triples are hallucinated specific-entity connective facts, measured after the 2026-09-07 judge replacement; the previously reported ~20% was deflated by unjudged paths), blending on **abstraction** (41% of frontier blends die at the generic-space gate), analogy invention on **fidelity** (~19% relabel the target or import an outside concept instead of projecting the mapping).
- **The blend bottleneck is finding a real shared abstraction, not elaborating it.** Every blend claims a shared slot, but only 57% survive verification; 94% of the failures are a **one-sided schema forced onto the other input**. Past the gate, 99% are coherent and 94% fully double-scope. The genuine-fusion rate spans 36–86% across models and does not track raw capability, and every anchor pair yields both a real fusion and a fake, so it measures model skill, not pair difficulty.
- **An artificial hivemind, on a ladder.** Inter-model convergence rises monotonically across each task's characteristic product — association bridge 0.21 → analogy invention 0.24 → blend `c′` 0.48 (excess over a cross-item null +0.12 → +0.14 → +0.34). Blending homogenizes even on its creative leap; analogy stays divergent. Blends carry a provider **house style**; bridges and analogy inventions do not.
- **Inventive multiples, and an operator asymmetry.** Two independent models invent the same entity *by the same abstraction* in 2.3% of co-response pairs; **blending produces ~7× more than analogy** (p = 1.4e-5) and same-provider pairs 3.2× cross-provider (p = 5e-4). More **distant** anchors *raise* the blend convergence rate but leave analogy flat — **blending funnels, analogy fans**, a quantitative signature of Fauconnier–Turner integration vs Gentner projection.
- **Analogy beats blending on utility by 10 points, and item difficulty does not explain it.** Over 1020 matched model×item cells, analogy 54.3% vs blending 44.3% (McNemar exact p = 2.9e-6, OR 1.55 [1.29, 1.88]). **No item is impossible** — every one of the 30 anchor pairs had at least one model produce both a valid analogy and a valid blend. The gap is concentrated on items that are hard *overall* (hardest tercile +27.0 pts; on the easiest third blending is **ahead** by 12.2), and per-item difficulty is essentially **uncorrelated across the two tasks** (r = +0.14) — a pair that is hard to analogise is not the pair that is hard to blend.
- **Thinking effort buys more output, not better output.** Two models × low/medium/high: the overall composite is flat (paired high−low −0.43 and −0.38, both CIs spanning zero) even though mean reasoning tokens rise **7.3×** and **14.6×**. The one apparent decline — association utility under high effort — is a **path-length artifact**: effort lengthens chains and a path counts as factual only if every triple does, while per-triple factuality stays near-flat.
- **The scoring dimensions are not redundant.** Task, not dimension, is the organizing axis; within a task utility trades off against surprise and originality (r ≈ −0.4 to −0.6). Only **emergent** originality transfers across tasks (0.46 vs 0.18 for base), and it is **orthogonal to capability** (r = −0.13 with the leaderboard) — the most inventive models are mid-tier, not the leaders.

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
cells are judged rather than exactly checked. *(Done 2026-09-01 on the Kombine run — see Current above.)* Detailed state in
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
