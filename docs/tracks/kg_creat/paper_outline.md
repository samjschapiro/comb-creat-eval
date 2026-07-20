# Planned submission — kg_creat

> **Reframed 2026-06-04 (Jonah Black).** Headline is no longer "predicts scientific
> ideation (LIB correlation)" — that arm is **dropped**. The paper is now a **diagnostic**
> of the **ideation–execution gap**, decomposed by a taxonomy of constraints (each a
> minimal abstraction of a real-world rule). See [design.md](design.md) §Study framing for
> the authoritative statement; this outline is being brought in line with it (§6, §9
> rewritten; §2/§5 de-LIB'd). **Venue** (was COLM LM4Sci) is under reconsideration — the
> framing is broader than scientific discovery.

Working title (new): **Which Rules Can LLMs Break Creatively? A Constraint Taxonomy for
the Ideation–Execution Gap.** *(alt: "Creative but Non-Compliant: Decomposing the
Ideation–Execution Gap with Knowledge-Graph Constraints")*

This doc is **task-first**: it pins down exactly what the test does, with a worked
example carried through verification and scoring. Prose/section structure follows
[writing_advice.md](../../writing_advice.md) later; the narrative is at the bottom.
Design status in [progress.md](progress.md); CREATE contrast in
[novelty_vs_create.md](novelty_vs_create.md).

---

## 1. The task in one paragraph

Give a model two entities from a real knowledge graph and ask for several distinct
multi-hop paths connecting them — **but require some relation types to appear and
others to be absent** (and, more generally, a *typology* of such constraints). The
constraints are the whole game: they make the task hard in a creativity-relevant way
(the model must *find a different route*, not the obvious one) and give an explicit,
tunable difficulty lever. As in CREATE, the model answers over the **open KG** and each
triple's factuality is checked by an **LLM judge**; what we add is the constraint
apparatus, **enforced exactly** on the verified path. Score each path by how
*semantically remote* the concepts it connects are (novelty, via embedding distance —
the validated DAT measure) times whether it *is factual and satisfies the constraints*
(utility, weighted up by the constraint load), plus a set-level term for how *distinct*
the paths are.

## 2. Substrate: multiple KGs (open, judge-verified)

- We run the test on **several KGs**. The KG is used for **prompt construction**
  (sampling endpoints with known connectivity; designing constraints that are
  *satisfiable* and *biting*) and as a **factuality reference** for the judge — **not**
  as a closed answer space. The model answers over the open KG (CREATE-aligned);
  factuality is judged. The loader materializes a **construction subgraph** `G_c` around
  the endpoints, not a held answer set.
- **Graph arms** (see [design.md](design.md) §Knowledge-graph backends): **Wikidata**
  scientific slice (method arm — clean CREATE contrast, discipline breadth) + scientific
  KGs **Hetionet** (built for typed multi-hop paths), then **PrimeKG**, optionally DRKG.
  **Hard rule: relation-rich only** — citation graphs (ACL/OpenAlex) and MeSH are out
  (too few relation types to constrain over).
- **Cross-KG agreement is evidence**: the same per-constraint tradeoff signature on
  Wikidata *and* Hetionet *and* PrimeKG makes it a property of the constraint *type*, not
  one graph's idiosyncrasies.

## 3. Anatomy of a prompt

A prompt is `x = (u, v, K, h)` (`K` = constraint set; `C` is the per-model creativity
score in §5):

| field | meaning |
|-------|---------|
| `u, v` | source / target entity (QIDs), known to be `≥ h` hops apart in `G_c` |
| `K` | a set of **typed constraints**, each enforced exactly on the verified path |
| `h` | target hop count |

Constraint **types** are a first-class experimental axis (full typology in
[design.md](design.md) §Constraint typology): **inclusion** / **exclusion** of relation
types, **categorical** (route through/avoid an entity *type*), **waypoint** (through/
avoid a specific entity — forces a conceptual bridge), **ordering/metapath** (relation
template), plus budget/polarity (biomedical) and a counterfactual/missing-edge stretch.
Inclusion+exclusion is the original Comb-Creat pair; we vary type as well as count.

The model returns `k` labeled walks, each a sequence
`u —r₁→ e₁ —r₂→ … —r_h→ v`, emitted in a parseable format (entity + property names,
which we entity-link back to QIDs/PIDs against `G_c` / the KG).

### Worked example (illustrative — QIDs/PIDs to be pinned in Phase 1)

```text
u = penicillin            v = bacterial cell wall
I = { "has effect" }      X = { "subclass of" }      h = 3
```

- **Obvious (blocked) route:** penicillin —subclass of→ β-lactam antibiotic —subclass
  of→ antibacterial —…→ cell wall.  ✗ uses an excluded relation (`subclass of`).
- **A valid route:** penicillin —has effect→ inhibition of transpeptidase —part of→
  peptidoglycan cross-linking —regulates→ bacterial cell wall.  ✓ contains `has effect`
  (∈ I), avoids `subclass of` (∈ X), 3 hops, every triple judged factual.

The exclusion forces the model off the taxonomic highway onto a *mechanistic* route —
that redirection is the creative act we are measuring.

## 4. Verification: judged factuality, exact constraints

For each returned path, three checks:

1. **Structural** — consecutive triples share an entity. *Exact, on the output (the
   only check CREATE also does mechanically).*
2. **Factual** — every triple `(eᵢ, rᵢ₊₁, eᵢ₊₁)` is a true KG relation. **LLM-as-judge**
   over the open KG (CREATE-aligned; we adopt/improve their gpt-oss-120b judge).
   *(Reverted from exact held-subgraph lookup 2026-06-02 — exact checking was too
   restrictive: it dropped true-but-out-of-coverage paths and blocked the model's
   parametric knowledge, which is where association creativity lives.)*
3. **Constraint** — `e₀ = u`, `e_h = v`, hop count `= h`, and every typed constraint in
   `K` (per §3). *Exact, on the judge-verified path:* relation-level + waypoint are
   syntactic; categorical/polarity use a light entity-type/sign lookup.

So the **constraint apparatus stays exact and judge-free** even though *factuality* is
judged — "constraints are rigorously enforced," not "verification is judge-free."

**Judge reliability is now a required result** (it replaces the old coverage rate as the
make-or-break number): report human-spot-check agreement with the factuality judge, and
note that a noisy judge corrupts utility `U` (it gates the indicator). CREATE's judge is
0.52-precision on incorrect relations — we must do better or at least quantify it.

Reuse: structural + constraint validation already exists in
`src/comb_eval/scoring.py` (`PathValidation`); factuality is a separate judge step.

## 5. Scoring (semantic-distance novelty × exact constraint utility)

**Why not label-surprise (the original synthetic port).** Relation-*type* frequency on
a real KG is a curation artifact (P31/P279, Hetionet `associates` dominate by
construction), it ignores the *entities* where real-KG novelty lives, and a 3-hop path
over a ~24-relation vocabulary has almost no variance. We score novelty in
**entity-embedding space** instead — the validated DAT measure — and keep
relation-surprise only as an ablation baseline.

Per **valid** path `P` (`|I|`, `|X|` = sizes of the include/exclude sets):

- **Novelty = semantic remoteness** `R(P)` = **DAT-style mean pairwise embedding
  distance over the path's entity set** {`u`, intermediates, `v`}. Paths that bridge
  distant concepts score higher; endpoint distance is fixed per prompt, so novelty lives
  in the intermediates. SBERT (`dat_eval/cdat.py`) default for phrase-named entities;
  GloVe/FastText as the embedding ablation. Needs no graph (just embeddings).
- **Diversity** `D` — *separate* set-level term: mean pairwise distance *between* the
  `k` valid paths (embedding or `comb_eval` Jaccard/edit). `R` = within-path spread,
  `D` = between-path difference.
- **Utility** `U(P; x) = (∏_t (1 + α_t·n_t)) · 1[valid ∧ factual]`, `n_t` = count of
  type-`t` constraints, `factual` = judge verdict. Constraint part **exact**, factuality
  **judged**. The multiplier: **the more constraints imposed, the more a satisfying path
  is worth.** (Inclusion/exclusion `(1+α_I|I|)(1+α_X|X|)` is the special case.)
- **Per-model (secondary)** `C = E_x[ A({R(P)·U(P;x)}; D) ]` — quality×diversity aggregate
  (greedy `s_γ` like CREATE, or mean·`D`; pilot decision). **Demoted** under the pivot: the
  headline is the per-constraint ideation–execution decomposition (§6), not this collapsed
  score, and there is no external benchmark to correlate it against.
- **Baselines/ablations** (show the redesign beats them): label-surprise `S(P)`;
  CREATE-style edge **specificity** `σ` (judge or construction-subgraph lookup);
  path **non-obviousness** (excess over geodesic).

Carrying the example: the valid mechanistic route routes through {penicillin,
transpeptidase-inhibition, peptidoglycan cross-linking, bacterial cell wall} — a
semantically spread set, so `R(P)` is high; `|I|=|X|=1` gives utility multiplier
`(1+α_I)(1+α_X)`. The blocked taxonomic route scores `U=0` (excluded relation)
regardless of its remoteness.

**Judge usage, stated precisely (don't overclaim):** *factuality* is **judge-based**
(CREATE-aligned) — not a differentiator. What stays deterministic: the **constraint
apparatus** (exact on the verified path) and **novelty/diversity** (embeddings, no LLM
judge). The differentiation from CREATE is the **constraint taxonomy (grounded as
real-world rules) + the per-constraint ideation–execution decomposition**, not
judge-freeness and no longer an external-benchmark correlation.

## 6. The headline: per-constraint ideation–execution tradeoff

**This is the paper's core result** (authoritative statement: [design.md](design.md)
§Study framing). For each constraint type `t`, over the matched endpoint bundles, we
measure two axes:

- **Ideation** `R_emit(t)` — novelty (embedding remoteness) of the model's *emitted* path,
  validity-agnostic: *did it reach for a creative connection?*
- **Execution** `sat(t)` — fraction of emitted paths that pass all three validity checks,
  with failure **channels** (structural / hallucinated-edge / constraint-violation).

Plotting constraint types in (`R_emit`, `sat`) space is the finding:

| | high `sat(t)` | low `sat(t)` |
|---|---|---|
| **high `R_emit`** | handled creatively | **the ideation–execution gap for `t`** |
| **low `R_emit`** | obeyed by playing safe | just hard (degrades both) |

The **difficulty lever** (constraint count × type) still exists and is what generates the
tradeoff — as constraint load grows the feasible route set collapses and `R` must drop to
stay valid (Comb-Creat's **novelty–utility trade-off**, expected to recover at frontier
scale, shown only for ≤100M *trained* models before). But the question is no longer "does
the score predict LIB"; it is **which constraint types force the gap vs which force safe
play**, read off the 2×2 and the within-bundle deltas. Reporting `sat` and its channels
separately is the capability/creativity decoupling `comb_eval` already enforces.

This is also the cleanest **construct-validity** evidence: if the task were just "can the
model do graph search," constraints would not trade novelty against satisfaction the way
they do here.

## 7. Why it is hard / creative, not trivial pathfinding

- The **exclusion set removes the obvious route** (taxonomic/`subclass of` chains), so a
  valid answer requires a genuinely different traversal — the redirection in the worked
  example.
- The **inclusion set demands a specific kind of hop** be present, forcing the model to
  route *through* a mechanism/relation it might not otherwise use.
- Producing **`k` distinct** valid paths rewards exploring the constrained region, not
  re-emitting one route — this is where divergent-thinking variance enters.
- Scoring rewards **semantically remote** routes, so degenerate "shortest valid path"
  answers (tight, obvious concept neighbourhoods) score low even when valid.

## 8. Open task-design decisions (resolve in Phase 1–2)

1. **Which KGs + each `G_c` domain & size** — Wikidata slice + Hetionet (+ PrimeKG/DRKG);
   per KG how many seeds, how many hops, which relation whitelist Σ, and how hard to
   restrict to parametrically-known entities. Trades coverage against tractability; gated
   per KG by the Phase-2 coverage rate.
2. **Matched endpoint-bundle sampler** — the unit is a bundle (fixed `(u,v,h,k)`)
   instantiated as {no-constraint baseline + one prompt per constraint type at matched
   count}, so within-bundle deltas isolate the *type* effect (§design.md Task). Each
   constraint must be *satisfiable* (∃ a valid path) and *bite* (remove the majority/obvious
   routes), checkable on `G_c` first. Must be **pre-registered** (DRAT anchor-bank lesson).
2b. **Headline constraint-type set** for the decomposition — recommended core:
   inclusion, exclusion, categorical, waypoint, ordering/metapath (budget/polarity as
   biomedical extras; counterfactual as the flagged stretch).
2c-grounding. **The grounding table** (make-or-break, §design.md Risks) — one defensible
   real-world creative-generation rule per constraint type, ideally with a citation. Draft
   and pressure-test this *before* running, since it is the actual contribution.
2c. **Factuality judge** — adopt CREATE's gpt-oss-120b as-is vs improve it; the
   reliability target; how categorical/polarity entity-type lookups are resolved.
3. **Entity linking of model output** — names → QIDs/PIDs (for constraint checking and
   the factuality reference); how to handle near-miss surface forms.
4. **Path-set aggregator** for `C` (§5): greedy `s_γ` quality×diversity vs simple
   mean·`D`. (Set diversity *is* a separate term — that part is decided.)
5. **Embedding for `R`/`D`** — SBERT default vs GloVe/FastText ablation; **coefficients**
   `α_I, α_X` (utility) and the `R`/`U`/`D` combination weights — port Comb-Creat
   defaults where they exist, revisit if scoring is degenerate on the pilot.

## 9. Narrative & evidence (compressed — full prose later per writing_advice)

**Hook: the ideation–execution gap.** LLMs are widely observed to generate strong ideas
but execute them worse — prior work documents the *what*, not the *why*. We give a
mechanistic decomposition: constrain a creative generation task with a **taxonomy of
constraints, each a minimal abstraction of a real-world rule creative work must obey**, and
measure — per constraint type — **ideation** (did the model reach for a novel connection?)
against **execution** (did it satisfy the rule and stay factual?).

**Two claims.** (1) A **grounded constraint taxonomy** on a CREATE-style open-KG task
turns "connect these entities" into a controllable probe of *rule-following under a
creativity demand* — structure unconstrained association cannot express. (2) The
per-constraint **ideation–execution decomposition reveals which rules LLMs break
creatively vs which make them play safe**: some constraints hold satisfaction high only by
collapsing novelty; others preserve novelty but are frequently violated (the gap). This
*localizes* the ideation–execution gap to specific constraint types — the "why."
**Takeaway:** the gap is not uniform; it is constraint-type-specific, and the taxonomy
predicts where it appears.

**Evidence each claim needs:**
- Claim 1 — the constraint lever produces real variance (count × type sweep, §6); the
  **factuality-judge reliability analysis** (human spot-check; quantify, ideally beat
  CREATE's 0.52 precision), since the judge gates the execution axis.
- Claim 2 — the **per-constraint 2×2** (`R_emit(t)`, `sat(t)` + failure channels) over
  matched bundles, with within-bundle deltas and CIs; the pattern replicated **across KGs**
  (§Cross-KG) and shown **across a capability range of models** (is the gap universal or
  ranked?). CREATE = the no-constraint cell of every bundle (natural baseline).
- Supporting — the **grounding table** (constraint type → real-world rule + citation); the
  secondary per-model aggregate `C`.

**Limitations to state up front:** **grounding rigor** (the taxonomy must be defensible,
not arbitrary — the central risk); **judge reliability** (factuality is LLM-judged,
CREATE-aligned; but note only the *factual* failure channel is judge-dependent —
constraint violations are exact); **ideation proxied by embedding remoteness** (a remote
path can still be nonsensical); sampler degrees of freedom; single graph family per arm
with cross-KG generality as the robustness check. **No LIB / external-benchmark claim** —
dropped by design; the contribution is the diagnostic, not a correlation.
