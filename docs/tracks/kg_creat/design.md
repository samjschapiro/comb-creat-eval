# kg_creat — design

Full task/scoring spec for the test-time, real-KG port of Comb-Creat.
See [progress.md](progress.md) for status and roadmap;
[novelty_vs_create.md](novelty_vs_create.md) for the CREATE comparison.

## Study framing (headline) — 2026-06-04 pivot

**Reframed after discussion with Jonah Black (2026-06-04).** The study is no longer
organized around "does our score correlate with LiveIdeaBench (validity/specificity vs
CREATE)." That correlation arm is **dropped** (it was n≈20 and significance-underpowered;
see [progress.md](progress.md) direction note). The headline is now a **diagnostic**:

> A suite of constraints, each a **minimal abstraction of a real-world rule that creative
> generation must obey**, and the empirical characterization of the **novelty–utility
> tradeoff per constraint *type*** — *which constraints can LLMs satisfy while keeping
> novelty high, and which force them to trade one for the other.*

This gives a mechanistic account of the documented **ideation–execution gap**: prior work
observed the *what* (models generate good ideas but execute them worse), not the *why*.
Our constraint typology decomposes the *why* by constraint type.

**The metric spine — ideation vs execution** (this is what the whole apparatus measures).
For each *emitted* path `P` (before any validity gating):

- **Ideation** = novelty `R(P)` = DAT-style semantic remoteness (embedding distance over
  `P`'s entity set). *Did the model reach for a creative, distant connection?* Measured on
  the emitted path **regardless of whether it is valid or factual.**
- **Execution** = does `P` actually (a) **satisfy the constraint** (exact, deterministic),
  (b) hold up **factually** (LLM judge), and (c) have correct endpoints/hop count? Binary
  per path, and its failures **decompose into channels**: structural / hallucinated-edge
  (factual) / constraint-violation. All three are *execution* failures, none is *ideation*.

Per constraint type `t`, aggregated over prompts and models, the two axes are:

- `sat(t)` = satisfaction (execution) rate under constraint `t`.
- `R_emit(t)` = mean ideation novelty of emitted paths under `t` (what the model *reaches
  for*); `R_valid(t)` = novelty of the *satisfying* paths (what it *realizes*).

The **core result is the 2×2** over constraint types in (novelty, satisfaction) space:

| | high `sat(t)` | low `sat(t)` |
|---|---|---|
| **high `R_emit`** | LLMs handle `t` creatively | **the gap**: creative idea, can't obey `t` |
| **low `R_emit`** | obey `t` by playing safe | `t` just degrades both (hard) |

The **ideation–execution gap for `t`** is the top-right cell: the model proposes a novel
path but it *violates the constraint or hallucinates an edge*. Whether a constraint pushes
a model into "play safe" (low `R_emit`, high `sat`) vs "break the rule" (high `R_emit`,
low `sat`) is the finding we are after.

**Matched/paired sampling is mandatory** (new requirement, §Task): to attribute a novelty
or satisfaction shift to constraint *type* and not to entity difficulty, hold `(u, v, h)`
fixed and toggle only the constraint. The sampler unit is an **endpoint bundle** →
a matched family {no-constraint baseline, one prompt per constraint type at matched count}.
Deltas (`R_emit(t) − R_emit(baseline)`, etc.) are then within-bundle and causal in `t`.

**What stays / what demotes.** The task substrate, constraint typology, open-KG + judge
verification, and the `R`/`D`/`U` scoring primitives below all stand. The per-model
quality×diversity aggregate `C` and any external-benchmark correlation are **secondary**
(the per-constraint decomposition is the product). **Synthetic/pretraining Comb-Creat is
fully abandoned** — real-world entities on a real KG (like CREATE), frontier models via
OpenRouter, **no GPU cost.**

**Analogy and blending are constrained forms of this general formulation** — analogy adds a
metapath (relational-structure) constraint; conceptual blending adds cross-domain
(multi-space) constraints. Our constraint typology is the dial that carries general
combinatorial creativity to these structured special cases; the transformational/emergent
frontier is bracketed (counterfactual variant). Full statement:
[constraints.md](constraints.md) §Analogy and blending as constrained forms.

## Task

Knowledge graph `G = (V, E, Σ)`: `V` = entities, `E` = directed labeled triples
`(e_i, r, e_j)`, `Σ` = relation types (Wikidata properties, e.g. `P31` instance-of,
`P279` subclass-of). A prompt is

```
x = (u, v, K, h)
```

- `u, v ∈ V` — source / target entity, connected by some walk of ≥ `h` hops.
- `K = {c₁, …, c_m}` — a set of **typed constraints**, each a predicate over the path
  (full typology below). Inclusion `I` and exclusion `X` are the two original types;
  we vary the *type* as well as the *count*. (`K` for constraints; `C` is reserved for
  the per-model creativity score in Scoring.)
- `h` — target hop count.

The model returns `k` labeled walks `P = (e_0, r_1, e_1, …, r_h, e_h)` from `u` to `v`.

These mirror the original Comb-Creat failure-mode abstractions: exclusion ≈
"prevent unrealistic assumptions / block expensive plans"; inclusion ≈ "ensure
proper baselines / require detailed steps". The typology generalizes that intuition
to a family of methodological pressures.

### Matched sampling (mandatory — the causal-attribution design)

Because the headline (§Study framing) attributes novelty/satisfaction shifts to constraint
*type*, the sampler's unit is **not** a single prompt but an **endpoint bundle**: a fixed
`(u, v, h)` (and a fixed target count `k`) instantiated as a **matched family of prompts**
— one **no-constraint baseline** plus one prompt per constraint type at a **matched
constraint count** (e.g. a single constraint of each type). Every constraint in a bundle
must be *satisfiable* (∃ a valid path in `G_c`) and *biting* (removes the majority/obvious
routes), checkable on `G_c` before the bundle is used. Because all prompts in a bundle
share endpoints, the within-bundle deltas `R_emit(t) − R_emit(baseline)` and
`sat(t) − sat(baseline)` isolate the *type* effect from entity difficulty. Aggregate these
deltas over bundles (and models) to get the per-type tradeoff. The sampler is
**pre-registered** (§Risks — researcher degrees of freedom).

## Constraint typology (what we vary)

Constraints are the active ingredient, so we treat **constraint type** as a first-class
experimental axis alongside constraint count. **Each type is a minimal abstraction of a
real-world rule creative generation must obey** — the italicized gloss on each entry is
that grounding, and making it *rigorous and defensible per type* (ideally with a citation
to where the rule bites in real scientific/creative practice) is the single most important
open task for the paper, not a decoration (§Risks — grounding rigor). Each constraint is a
predicate over the **emitted path** `P` (once its triples are factuality-verified, see
Per-path validity). The **core set is checked exactly on `P`** — the constraint apparatus
stays rigorous even though *factuality* is judge-based; the "Exact?" tag below means
"enforceable exactly on the verified path," not "requires a held subgraph."

**Relation-level** (edge labels):

- **Inclusion** — `labels(P) ⊇ I`. *Ground in a mechanism type.*
- **Exclusion** — `labels(P) ∩ X = ∅`. *Don't take the trivial shortcut.*
- **Ordering / metapath** — `r₁` precedes `r₂`, or `P` matches a relation-type template
  (Hetionet is built on metapaths). *Methodological order: mechanism before effect.*
- **Budget / cost** — `Σ cost(rᵢ) ≤ B` (generalizes exclusion at `cost = ∞`).
  *Resource-bounded plan.*
- **Polarity / sign** — net up/down-regulation, or must include an inhibitory step
  (signed regulatory edges in Hetionet/PrimeKG). *Mechanistic consistency.*

**Entity-level** (nodes):

- **Categorical inclusion / exclusion** — must pass through / must avoid any entity of
  type `T`. *"Must invoke a gene" / "without any clinical-trial entity."*
- **Waypoint inclusion / exclusion** — must pass through specific entity `w` / must
  avoid hub `H`. *Forces (or forbids) a specific conceptual bridge — a classic analogy
  probe.*

**Structural / set-level:**

- **Cardinality** — exact / range / `≥` / `≤` hop count. *Depth-of-reasoning.*
- **Disjointness across the `k` paths** — node-disjoint or distinct relation sets.
  *Genuinely distinct hypotheses, not rephrasings.*

**Stretch (different verifier — not in the exact-core default):**

- **Novelty floor** — require `R(P) ≥ τ`. Couples constraint to score; keep as a task
  *variant*, not mixed into the default.
- **Counterfactual / missing-edge** — "propose a path that *should* connect `u`–`v` but
  is **not** in the KG." Flips the task to **link-prediction / hypothesis generation** —
  the closest variant to genuine scientific ideation — but **not exactly checkable**
  (answer is non-factual by design; needs a held-out future KG version or a separate
  verifier). Track as the ideation-facing extension thread.

**Headline constraint-type set** (the core, used for the constraint-type ablation):
inclusion, exclusion, categorical, waypoint, ordering/metapath. Budget / polarity are
biomedical-KG extras; counterfactual is the flagged stretch.

## Per-path validity

A path is valid iff all three hold (the third is what CREATE lacks):

1. **structural** — consecutive triples share an entity. *Exact, on the output (same
   check CREATE does mechanically).*
2. **factual** — every triple `(e_i, r_{i+1}, e_{i+1})` is a true KG relation.
   **LLM-as-judge**, over the open KG (CREATE-aligned). *(Reverted from exact
   held-subgraph lookup 2026-06-02: exact checking restricted answers to what we held
   locally and dropped true-but-out-of-coverage paths — too restrictive, and it
   prevented the model from using its full parametric knowledge, which is where
   association-creativity lives.)*
3. **constraint** — `e_0 = u`, `e_h = v`, hop count matches `h`, and **every typed
   constraint `c ∈ K` is satisfied**. *Exact, computed over the judge-verified path:*
   relation-level (incl/excl/ordering/cardinality/budget) and waypoint are syntactic on
   the verified labels/entities; categorical and polarity need a light entity-type / sign
   lookup (KG dump or the same judge).

**Open KG + judge (CREATE-aligned), with exact constraints.** The model answers from
parametric memory over the open KG; factuality is judged. We **adopt and try to improve
on CREATE's judge** (gpt-oss-120b: 85.9% balanced acc, 0.94 recall / 0.52 precision on
incorrect relations) and run our own **judge-reliability analysis** (human spot-check
agreement) — a required component of the paper (see Risks). What stays exact and
judge-free is the **constraint apparatus**: once a path's triples are verified, every
typed constraint is enforced deterministically. So "constraints are rigorously enforced"
holds, but the *factuality* claim is judge-based, not exact.

## Scoring (ideation vs execution, decomposed per constraint type)

**How the primitives roll up (headline).** The three primitives below — novelty `R`,
diversity `D`, utility `U` — are the raw material; the **headline is not a single per-model
number** but the per-constraint-type **ideation vs execution** decomposition (§Study
framing). Concretely, for each constraint type `t`, aggregated over the matched bundles
and models:

- **Ideation** axis `R_emit(t)` = mean `R(P)` over the model's *emitted* paths under `t`
  (validity-agnostic — what it reached for); also `R_valid(t)` over the satisfying subset.
- **Execution** axis `sat(t)` = fraction of emitted paths that pass *all three* validity
  checks (structural ∧ factual ∧ constraint), with the **failure channels** broken out
  (structural / hallucinated-edge / constraint-violation) so we can say *how* execution
  failed. Report `sat` alongside its channels — this is the capability/creativity
  decoupling `comb_eval` already enforces (`solve_rate` reported separately).
- **Tradeoff** = the within-bundle deltas vs the no-constraint baseline
  (`ΔR_emit(t)`, `Δsat(t)`), placed in the 2×2. This is the paper's core figure.

The per-model quality×diversity aggregate `C` (below) is retained as a **secondary**
summary, not the headline.

**Why not label-surprise.** Comb-Creat's relation-frequency surprise does not
transfer to a real KG: relation-*type* frequency is a curation artifact (P31/P279,
Hetionet `associates` dominate by construction, not by novelty), it ignores the
*entities* (where real-KG novelty lives), and a 3-hop path over a ~24-relation
vocabulary carries almost no variance. We replace it with **semantic remoteness** in
entity-embedding space — the DAT lineage `dat_eval` already uses — and keep
relation-surprise only as an ablation baseline.

- **Novelty (per-path) = semantic remoteness** `R(P)` = **DAT-style mean pairwise
  cosine distance over the entity set of `P`** ({`u`, intermediate entities, `v`}).
  Rewards paths that bridge semantically distant concepts rather than staying in a
  tight neighborhood. (Endpoint distance `d(u,v)` is fixed per prompt, so novelty must
  live in the *intermediate* entities — hence the whole-path entity set.) Entity labels
  are embedded with the `dat_eval` infra; **SBERT (all-mpnet-base-v2)** is the default
  since entity names are phrases, with GloVe/FastText (token mean-pool) as the embedding
  ablation. Note: this needs **no graph at all** (just entity embeddings), so it is
  unaffected by the open-KG / judge change and needs no per-KG normalization.
- **Diversity (set-level)** `D` over the valid `k`-path set — a **separate term**, not
  folded into per-path novelty: mean pairwise distance *between* paths (embedding
  distance over path-entity centroids, or `comb_eval`'s intra-response Jaccard / edit
  distance, already implemented). Rewards returning *distinct* routes, not re-emitting
  one. Distinguish from `R`: `R` = within-path spread, `D` = between-path difference.
- **Utility** `U(P; x) = (∏_t (1 + α_t · n_t)) · 1[valid ∧ factual]`, where `n_t` =
  number of constraints of type `t` in `K` and `α_t` its weight, and `factual` is the
  judge verdict. Multiplicative in the constraint load — the **difficulty lever** and
  (via the **constraint apparatus**) the main distinction from CREATE.
  Inclusion/exclusion `(1+α_I|I|)(1+α_X|X|)` is the `t ∈ {I, X}` special case; the
  product generalizes it to the full typology. As the constraint count *and* the mix of
  types grow, a satisfying path is worth more.
- **Per-model creativity (secondary)** `C(model) = E_x[ A({R(P)·U(P;x) : P ∈ valid k-set}; D) ]`,
  a quality×diversity aggregate over the valid set (mirrors CREATE's `s_γ` greedy
  quality×diversity, with embedding `R`/`D` and **constraint-weighted, judge-gated**
  `U`). Greedy-`s_γ` vs simple mean·`D` is a pilot decision. **Demoted to a secondary
  summary** under the pivot — the headline is the per-constraint decomposition above, not
  a single collapsed score (and there is no longer an external benchmark to correlate `C`
  against; see §Evaluation).

**Demoted to baselines/ablations** (report, show the redesign beats them): relation
label-surprise `S(P)`; CREATE-style edge **specificity** `σ(P)` (candidate-class size,
via the judge as CREATE does, or a local construction-subgraph lookup if we keep one);
path **non-obviousness** (excess length over geodesic / hub-avoidance).

Coefficients `α_I, α_X` carry over from Comb-Creat defaults; revisit if the pilot
shows degenerate scoring.

**Judge usage, stated precisely (don't overclaim).** *Factuality* is judge-based
(CREATE-aligned) — we do **not** claim exact/judge-free factuality. What we do keep
deterministic: the **constraint apparatus** (exact on the verified path) and
**novelty/diversity** (embeddings — no LLM judge, same posture as DAT/CDAT and CREATE's
own diversity term). The differentiation from CREATE is the **constraint typology (grounded
as real-world rules) + the per-constraint ideation–execution decomposition**, **not**
judge-freeness and **no longer** an external-benchmark correlation.

## Reuse map (do not rewrite)

| Need | Reuse | Notes |
|------|-------|-------|
| Prompt object + constraint generation | `src/comb_eval/prompts.py` (`EvalPrompt`, BFS) | Already has `start, end, hop_count, include_labels, exclude_labels, difficulty_level`. Swap synthetic graph + a–z labels for KG + relation labels. |
| Per-path validation, set diversity, surprise baseline | `src/comb_eval/scoring.py` (`PathValidation`, Jaccard/edit, surprise) | Port structural + constraint validation + the diversity term `D` + the label-surprise *baseline*; factuality is a separate **judge** step (below). |
| Factuality judge | **NEW** (adopt/improve CREATE's gpt-oss-120b setup) via `src/dat_eval/llm.py` OpenRouter client | Per-triple true/false; run a reliability analysis vs human spot-check. |
| Entity embeddings for remoteness `R` / diversity `D` | `src/dat_eval/cdat.py` (SBERT all-mpnet-base-v2), `src/dat_eval/dat.py` (GloVe), `src/dat_eval/pace.py` (FastText) | SBERT default (entity names are phrases); GloVe/FastText token mean-pool as the embedding ablation. Mean-pairwise-distance scoring is the DAT primitive. |
| Graph backend | **NEW** `src/kg_creat/graph.py` | Replaces `src/comb_eval/graph.py` Erdős–Rényi builder with a common held-subgraph loader over multiple KGs (Wikidata, Hetionet, PrimeKG, …). The only genuinely new component. |
| Model elicitation | `src/dat_eval/llm.py` / `src/new_tests/llm.py` | OpenRouter async client with `budget_usd` cap. |
| ~~Validity / specificity~~ (DROPPED) | ~~`src/dat_eval/scripts/score_evals.py`~~ | LIB correlation arm dropped in the 2026-06-04 pivot; no external-benchmark regression. |
| ~~Criterion store~~ (not needed) | ~~`configs/comb_eval/benchmarks.json`~~ | Only relevant to the dropped LIB arm; the per-constraint decomposition needs no external criterion. |

## Knowledge-graph backends (multi-graph plan)

We run the test on **multiple KGs**. This is a core part of the design, not a single
Phase-1 choice: Wikidata isolates the *method* (clean CREATE contrast), and one or more
*scientific* KGs show **generality across graphs** (the same per-constraint tradeoff
should replicate whichever KG it runs on). The KG is used for **prompt
construction** (sampling endpoints with known connectivity; designing constraints that
are *satisfiable* and *biting*) and as a **factuality reference** the judge can consult
— **not** as a closed answer space (the model answers over the open KG; factuality is
judged). `src/kg_creat/graph.py` exposes a common loader: given a raw KG, materialize a
typed **construction subgraph** `G_c = (V, E, Σ)` around the sampled endpoints with a
relation-frequency table — so adding a KG is a loader, not a new pipeline.

**Hard requirement on any KG: it must be relation-*rich*.** The whole task is
must-include / must-exclude over relation *types* Σ, so a KG with one or two edge
types degenerates to plain pathfinding. This rules out **citation/concept graphs**
(Semantic Scholar, OpenAlex, ACL Anthology, MAG — essentially single-relation
`cites`) and largely hierarchical vocabularies (**MeSH** — broader/narrower). The
earlier `proposals.md` mention of "ACL citation graph / MeSH" is superseded by this.

| Arm | KG | Role | Notes |
|-----|----|------|-------|
| Primary (method) | **Wikidata**, scientific-entity slice | clean head-to-head with CREATE; broad discipline coverage | rich property vocabulary (PIDs); cleaner labels for entity-linking |
| Scientific #1 | **Hetionet** | built *for* typed multi-hop path reasoning (drug-repurposing metapaths); ~24 meaningful relation types; nameable nodes; small enough to hold | biomedicine only |
| Scientific #2 | **PrimeKG** | richer/bigger precision-medicine KG (~30 relation types) → coverage + generality check | more synonyms to entity-link |
| Optional | **DRKG** | very relation-rich (~100 types) for a relation-vocabulary stress test | partly NLP-derived → noisier ground truth; use with care |
| Excluded | SemMedDB | NLP-extracted predications → unreliable as a factuality reference | avoid |

Prefer **curated, expert-integrated** KGs (Hetionet, PrimeKG) — cleaner for prompt
construction and a more reliable factuality reference, and designed around typed paths.

**Access:** local dumps in `resources/` (gitignored) for construction + as the
factuality reference; bounded cost, reproducible. Avoid live SPARQL/REST at runtime.

**Endpoint/constraint sampler** is shared across KGs and should optionally mirror
CREATE's (relation `r`, category `c`) entity sampling on the Wikidata arm so the
head-to-head is clean (Phase 4).

### Cross-KG comparison is itself evidence

Reporting the **per-constraint ideation–execution tradeoff per KG** turns the multi-graph
setup into a robustness result: if the same constraint types show the same tradeoff
signature (e.g. exclusion preserves novelty but tanks satisfaction) on Wikidata *and*
Hetionet *and* PrimeKG, the pattern is a property of the **constraint type**, not of one
graph's idiosyncrasies. Divergence across KGs is also informative — a constraint that
bites only on the biomedical KGs localizes that pressure to domain structure.

### Per-KG note: entity familiarity

With the open-KG + judge setup the model is **not** restricted to held entities, so the
old "coverage collapses → empty test" failure mode is largely gone (this was a main
reason for reverting to a judge). Entity familiarity still matters for *prompt design*:
endpoints must be entities frontier models plausibly know, or they can't form a factual
path at all. Sample endpoints from **well-known** entities per KG (famous drugs/diseases/
major genes for Hetionet). Caveat to state: the score still partly reflects *domain
recall* of the KG's area — appropriate-ish for scientific ideation, but note it.

## Evaluation

**No external-benchmark regression** (the LIB validity/specificity arm is dropped, 2026-06-04).
The evaluation *is* the per-constraint ideation–execution decomposition:

- **Pool**: a spread of frontier models via OpenRouter (no GPU; only elicitation cost).
  Model count is driven by budget, not by which models carry an external benchmark score —
  choose to span a capability range so the tradeoff's model-dependence is visible.
- **Primary analysis** — per constraint type `t` (over matched bundles):
  `R_emit(t)`, `R_valid(t)`, `sat(t)` and its failure channels, and the within-bundle
  deltas vs baseline → the **2×2 novelty × satisfaction figure** (§Study framing). Read
  off *which constraints* force the ideation–execution gap (high `R_emit`, low `sat`) vs
  which induce safe play (low `R_emit`, high `sat`).
- **Model-level view (secondary)** — is the gap universal or does it rank models? Report
  the tradeoff per model and the secondary aggregate `C`. **Comparators**: CREATE
  (unconstrained baseline = the "no-constraint" cell of every bundle) and, if useful, a
  no-novelty-scoring ablation.
- **Robustness** — cross-KG replication of the per-constraint signature (§Cross-KG).

## Risks

- **Grounding rigor (the make-or-break risk under the pivot).** The whole contribution
  rests on each constraint type being a *defensible* minimal abstraction of a real-world
  creative-generation rule. If the mapping reads as arbitrary, the taxonomy — and the
  "which constraints" finding — collapses. Mitigation: a rigorous grounding table, one
  citation per type to where the rule bites in real scientific/creative practice, and
  types chosen for *distinct* real-world pressures (not just formal variety). This is the
  "if done right."
- **Attribution / confound.** A per-type novelty or satisfaction difference must be caused
  by the *type*, not by entity difficulty or hop structure. Mitigation: the **matched
  endpoint-bundle sampling** (§Task) — same `(u,v,h)`, toggle only the constraint — and
  report within-bundle deltas, not raw cross-prompt means.
- **Judge reliability.** Factuality is LLM-judged (CREATE-aligned, 0.52 precision on
  incorrect relations); it gates the execution axis, so a noisy judge inflates/deflates
  `sat(t)`. Mitigation: adopt/improve CREATE's judge + a **human-spot-check reliability
  analysis**. Note the split: constraint-violation failures are *exact* and judge-free;
  only the *factual* failure channel depends on the judge — report the channels separately
  so the judge's noise is isolated to one of them.
- **Researcher degrees of freedom.** Endpoint/constraint sampling has the same
  free-parameter problem DRAT hit with anchor banks (1/3 of pairs had τ > 0.35).
  Pre-register the sampler (and the α weights / embedding choice) before any
  paper-defensible run.
- **Ideation measured only as semantic remoteness.** `R_emit` proxies "creative idea" by
  embedding distance; a model could emit a remote-but-nonsensical path. The factuality
  judge partly guards this on the *valid* subset, but `R_emit` over *all* emitted paths
  is only a proxy for ideation quality — state this, and consider a judged idea-quality
  spot check.
- **Timeline.** ~3 weeks to the June 23 2026 deadline. Scoring primitives + `graph.py`
  exist; the heavy new build is the multi-KG construction + matched sampler + factuality
  judge. Sequence the graph arms (Wikidata first, then Hetionet, then PrimeKG/DRKG) so a
  one-graph result is shippable even if later arms slip. **Reconsider the venue** — the
  ideation–execution-gap framing is broader than LM4Sci/scientific-discovery.
