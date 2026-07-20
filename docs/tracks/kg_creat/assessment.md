# kg_creat — benchmark assessment (pre-registration)

How we actually administer the six-constraint benchmark to models and turn responses into
the headline per-constraint **ideation–execution** 2×2. This is the pre-registration the
design demands (§design.md Risks — researcher degrees of freedom): the task modes, output
formats, satisfaction checks, samplers, and metrics are frozen here *before* any paid run.

Companion docs: [design.md](design.md) (metric spine + formal predicates),
[constraints.md](constraints.md) (taxonomy + grounding), [methods.md](methods.md) (scoring
+ CREATE reuse). Figure: [papers/kg_creat-iclr `content/figures/cc_constraints.png`].

Status: **draft 2026-07-19** — captures the design decisions locked in the 2026-07-19
session (all six constraints; Wikidata common-knowledge entities; interpretability over
strict feasible-fraction matching; V/VI reformulated to the `(u,v)`-only interface;
blending = pivot-path; V/VI on their own cross-domain pools). Nothing here is run yet.

## 1. The unified interface

**Every one of the six modes (plus the baseline) is the same prompt interface:** give the
model a start entity `u` and an end entity `v` from Wikidata; the model does *all* the
structural work. The mode differs only in *what kind of connection* is demanded.

| # | Mode | What the model must return | `sat` check | Regime |
|---|------|----------------------------|-------------|--------|
| 0 | **Baseline** | any factual path `u → v` (`h` hops, `k` distinct) | exact + factual | I–IV bundle |
| I | **Categorical** | factual path through an entity of type `T` | exact + factual | I–IV bundle |
| II | **Ordering** | factual path showing relation `Y` before `Z` | exact + factual | I–IV bundle |
| III | **Inclusion** | factual path including relation `Y` (or entity `X`) | exact + factual | I–IV bundle |
| IV | **Exclusion** | factual path avoiding relation `Y` (or entity `X`) | exact + factual | I–IV bundle |
| V | **Analogy** | two role-aligned parallel structures showing `u` ≈ `v` | **judged** + factual | own pool |
| VI | **Blending** | pivot path `u → … → X → … → v`, `X` double-sensed | **judged** + factual | own pool |

The model always emits CREATE's `<answer>` JSON-triples format (parseable to `EmittedPath`
via `parse.py`); for V it emits two such structures, for VI one pivot path.

## 2. Two measurement regimes

The interface is unified but the *measurement* is not, for two structural reasons that
V/VI's `(u,v)`-only reformulation actually sharpens:

**Regime A — I–IV (+baseline): matched bundles, within-`(u,v)` deltas, exact `sat`.**
Endpoints sampled for *factual connectivity* + *biting constraints*. The unit is an
endpoint bundle (fixed `(u,v,h,k)`) instantiated as {baseline, I, II, III, IV}. The finding
is the **within-bundle delta** `ΔR_emit(t) = R_emit(t) − R_emit(baseline)` and
`Δsat(t)` — causal in constraint *type* because endpoints are held fixed.

**Regime B — V, VI: own cross-domain pools, absolute points, judged `sat`.**
Endpoints sampled for *domain distance* (analogizability / blendability), **not** factual
connectivity — "find an analogy between penicillin and a cell wall" is a non-question;
analogy needs two things structurally similar in *different* domains. V/VI have **no natural
unconstrained baseline** (the analogy/blend *is* the task), so they land on the 2×2 as
**absolute** `(R_emit, sat)` points, read qualitatively against Regime A, and flagged as
judge-scored.

Both regimes share the axes (`R_emit`, `sat`) and both feed the same 2×2 plot; the read is
delta-based for A, absolute for B.

## 3. Per-mode specification

`R_emit` (ideation) is always DAT-style mean pairwise embedding remoteness over the emitted
entity set (SBERT `all-mpnet-base-v2`; validity-agnostic). Common to all modes; below only
the *task* and the `sat` (execution) check differ.

### Regime A (I–IV) — exact `sat`, judge-free except the shared factuality channel
- **Structural** (exact): endpoints `u,v`, hop count `h`, consecutive triples share an
  entity, node-distinct. Reuse CREATE `check_path_validity`.
- **Factual** (judge): every triple a true KG relation. CREATE's gpt-oss-120b.
- **Constraint** (exact, on the verified path): the mode's predicate from
  [constraints.md](constraints.md) §5 — inclusion/exclusion/ordering syntactic on labels;
  categorical via entity-type lookup (KG dump, judge fallback).
- `sat(t) = structural ∧ factual ∧ constraint`; failure **channels** broken out.

### Regime B (V, VI) — judged `sat`, own reliability analysis required
- **V Analogy.** Input `(u,v)`. Output = source structure on `u` ∥ target structure on `v`,
  disjoint nodes. `sat` = *syntactic floor* (both structures share a relation-type template
  — exact) **∧** *semantic* (**judged**: entities play corresponding roles) **∧** factual
  (both structures' triples true). `R_emit` = domain distance between the two structures.
- **VI Blending (pivot-path).** Input `(u,v)`. Output = one path `u → … → X → … → v`.
  `sat` = *syntactic floor* (two isomorphic sub-structures share pivot `X` — exact) **∧**
  *semantic* (**judged**: `X` genuinely carries two colliding/fusing senses) **∧** factual.
  `R_emit` = distance between the two fused domains.

**Cost of the `(u,v)`-only openness:** V/VI `sat` is a fully open quality judgment (no fixed
target), lower-reliability than I–IV's exact checks → V/VI get their **own** human
spot-check calibration, separate from the factuality-judge reliability analysis.

## 4. Sampling (frozen)

**Interpretability over strict matching.** Pick *human-legible* biting parameters — the
excluded relation is the obviously-tempting shortcut, the required waypoint a recognizable
bridge — and **log each prompt's feasible-fraction on `G_c` as a covariate** rather than
matching on it. A reviewer should see at a glance why each constraint is hard.

- **Regime A (I–IV).** Sample `(u,v)` well-known, `≥ h` hops apart on `G_c`. For each
  constraint, choose its parameter by enumerating paths (`graph.py::enumerate_paths`) so it
  is *satisfiable* (∃ valid path) and *biting* (removes the majority/obvious routes).
  Example biting choices: exclusion of the relation on the shortest/taxonomic route;
  inclusion of a relation present on a minority of paths; categorical type on a minority of
  interiors.
- **Regime B — V (analogy).** Pilot uses a **curated common-knowledge seed list** of
  analogy pairs (atom/solar-system, heart/pump, brain/computer, evolution/gradient-descent,
  …) — interpretable and pre-registerable; automated same-type-distant-domain sampling is a
  later extension.
- **Regime B — VI (blending).** **Pivot-first inversion** (the hardest sampler, made
  tractable): enumerate polysemous Wikidata labels whose senses sit in distant domains
  (Crane = bird | machine; Bat; Mercury; Bass; Java; Turkey), pick `u` near sense 1 and `v`
  near sense 2, **hide the pivot from the model.** Guarantees a solution exists; discovering
  it is the creative act. Multiple valid pivots are fine (open-ended) — `sat` is judged on
  whatever pivot the model returns, not the sampler's.

The full sampler (both regimes, α weights, embedding choice) is **pre-registered** before
any paper-defensible run.

## 5. Metrics / output

Per (model, mode) — Regime A aggregated over bundles as within-bundle deltas, Regime B as
absolute means:
- **Ideation** `R_emit`, and `R_valid` over the satisfying subset.
- **Execution** `sat`, with failure channels (structural / hallucinated-edge /
  constraint-violation — and for V/VI, syntactic-floor / semantic-judgment).
- **Diversity** `D` over the valid `k`-set (CREATE's between-path SBERT distance).

**Headline figure:** the six modes as points in (`R_emit`, `sat`) space (Regime A as
deltas vs baseline, Regime B absolute), per model and pooled — the 2×2 of §design.md.
**Robustness:** cross-KG replication of the per-mode signature (Hetionet/PrimeKG, later).
**Required side result:** factuality-judge reliability (human spot-check) + a separate
V/VI semantic-`sat` reliability check.

## 6. Worked examples (common-knowledge Wikidata, illustrative)

**Regime A bundle** — `u = Penicillin`, `v = Bacterial cell wall`, `h = 3`, `k = 5`:
```
0 baseline   : connect, no constraint
I categorical: an intermediate must be an instance of «enzyme»
II ordering  : «has effect» must appear before «part of»
III inclusion: must include «inhibits»  (or entity «transpeptidase»)
IV exclusion : must avoid «subclass of»  (blocks the taxonomic highway)
  ✓ valid IV route: Penicillin —has effect→ transpeptidase inhibition —part of→
                    peptidoglycan cross-linking —regulates→ bacterial cell wall
  ✗ blocked route : Penicillin —subclass of→ β-lactam —subclass of→ … (uses «subclass of»)
```

**V analogy** — `u = Atom`, `v = Solar system`:
```
electron —orbits→ nucleus   ∥   planet —orbits→ Sun
sat: relation templates match (exact) ∧ roles correspond [electron↔planet, nucleus↔Sun]
     (judged) ∧ all four triples factual ;  R_emit: dist(atomic physics, astronomy)
```

**VI blending** — `u = Records`, `v = Squirrels` (sampler pivot: «Boxer» = athlete | dog):
```
Records —chases→ Athlete —is a→ Boxer —is a→ Dog —chases→ Squirrels
sat: two parallel «X chases Y» sides share pivot «Boxer» (exact floor) ∧ «Boxer» carries two
     colliding senses (judged) ∧ triples factual ;  R_emit: dist(sports records, wildlife)
```

## 7. Build order (what this unblocks)

Scorer core (`scoring.py`, `parse.py`) already handles Regime A's exact checks + `R`. New
work, in order:
1. **Factuality judge** + reliability harness (shared by both regimes) — vendored CREATE
   gpt-oss-120b; needs litellm + key.
2. **Regime-B semantic-`sat` checker** (judge-based): analogy role-correspondence + blending
   pivot-sense judgments, each with its own reliability spot-check.
3. **Wikidata builder** (`wikidata.py` → `KnowledgeGraph.from_triples`), cached `G_c`.
4. **Samplers**: Regime-A matched-bundle (connectivity + biting via `enumerate_paths`);
   Regime-B curated analogy list + pivot-first polysemy sampler.
5. **Elicitation runner** (OpenRouter, budget cap, save-raw, resumable) — emits CREATE's
   `path_prediction` schema.
6. **Aggregator** → per-mode `R_emit`/`R_valid`/`sat`/channels/`D` + the 2×2.

**Pilot gate** (before scaling / spend): on a handful of cheap models, the constraint lever
produces variance, the tradeoff appears, exact `sat` is interpretable, and both judge
reliabilities (factuality; V/VI semantic) are acceptable. **Estimate OpenRouter cost and
confirm before any run** (incl. smoke tests) — see memory `estimate-openrouter-cost-first`.

## 8. Open items (not yet frozen)

- Hop count(s) `h`, path count `k`, bundles-per-`(u,v)`, model set — budget-driven; size
  after the elicitation runner exists and a cost estimate is in hand.
- V analogy output format: how many hops per side; whether the two sides must be equal
  length; how the model marks the role alignment in JSON.
- VI: precedence when the model returns a pivot different from the sampler's (accept any
  valid double-sense pivot — confirm the judge rubric).
- Entity linking of emitted names → QIDs for constraint checking + the factuality reference;
  near-miss surface forms.
- α weights (utility), embedding ablation grid — port Comb-Creat defaults, revisit on pilot.
