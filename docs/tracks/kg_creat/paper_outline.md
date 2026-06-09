# Planned submission — kg_creat (COLM 2026 LM4Sci)

Working title: **Constraints Make a Knowledge Graph a Creativity Test:
A Constraint-Tunable, Test-Time Eval that Predicts Scientific Ideation**

Target: COLM 2026 LM4Sci, 8pp main (non-archival), deadline **June 23 2026 AOE**.
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
- **Cross-KG agreement is evidence**: a score that tracks LIB on Wikidata *and* Hetionet
  *and* PrimeKG is a property of the constrained-pathfinding construct, not one graph.

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
- **Per-model** `C = E_x[ A({R(P)·U(P;x)}; D) ]` — quality×diversity aggregate (greedy
  `s_γ` like CREATE, or mean·`D`; pilot decision).
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
judge). The differentiation from CREATE is constraints + utility + LIB validation, not
judge-freeness.

## 6. Difficulty lever & the trade-off we expect to recover

Difficulty is **2-D: constraint count × constraint type.** Sweeping count (`|I|`, `|X|`,
…) is the continuous knob; varying *type* (the constraint-type ablation, §8) asks which
methodological pressure makes the task ideation-predictive. Hypothesis to pre-register:
types map to **LIB facets** (waypoint/categorical "conceptual-bridge" → originality;
ordering/budget → feasibility). Count sweep first:

- Small `|I|`: many routes qualify → models find high-remoteness paths → high `R`, easy `U`.
- Large `|I|`: the feasible route set collapses toward the few constrained paths →
  remoteness must drop to stay valid. This is Comb-Creat's **novelty–utility trade-off**,
  which we expect to **recover at frontier scale** (it was shown only for ≤100M *trained*
  models). Report **solve-rate separately** so a novelty drop is not confounded with the
  model simply failing (the capability/creativity decoupling `comb_eval` already enforces).

This sweep is also the cleanest **construct-validity** evidence: if the score were just
"can the model do graph search," it would not trade novelty against constraints the way
a creativity score should.

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
2. **Endpoint/constraint sampler** — how `(u,v)` and the typed constraints `K` are drawn;
   must be **pre-registered** to avoid the DRAT anchor-bank degrees-of-freedom problem.
   Per constraint: it should be *satisfiable* (∃ a valid path) and *bite* (remove the
   majority/obvious routes), both checkable on `G_c` before the prompt is used.
2b. **Headline constraint-type set** for the type ablation — recommended core:
   inclusion, exclusion, categorical, waypoint, ordering/metapath (budget/polarity as
   biomedical extras; counterfactual as the flagged stretch).
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

**Two claims.** (1) Adding a **typology of constraints** to CREATE-style open-KG
association yields a creativity test with an **explicit, tunable difficulty lever**
(count × type) whose constraints are **exactly enforced** on the judge-verified path —
structure unconstrained association cannot express. (2) Its per-model scores **predict
LiveIdeaBench with specificity** after controlling for capability
`g = (Arena-Overall, MMLU-Pro)`, where semantic-distance tests do not, and `[TBD]` vs
CREATE. **Takeaway:** *constraints*, not a richer graph or a better judge, are the
active ingredient.

**Evidence each claim needs:**
- Claim 1 — the constraint-count sweep produces a real difficulty lever (variance +
  novelty–utility trade-off, §6); the **factuality-judge reliability analysis** (human
  spot-check agreement; must quantify, ideally beat CREATE's 0.52 precision).
- Claim 2 — validity `r(C, LIB)` and specificity `r(C, LIB|g)` with bootstrap CIs,
  on the same ≈31-model pool as DAT/CDAT/PACE/DRAT (and CREATE if Phase 4 runs it).
  Frame as effect-size + frontier comparison, **not** `p<.05` (n is small; dat_eval hit
  this wall on LIB).
- Supporting — the constraint-*type* ablation (which type predicts LIB / its facets).

**Limitations to state up front:** **judge reliability** (factuality is LLM-judged,
CREATE-aligned — we inherit its 0.52-precision risk); **narrower moat vs CREATE**
(verification + novelty are shared ground; the contribution is constraints + utility +
validation); small n; sampler degrees of freedom; LIB itself is a judge-based criterion;
one graph family (Wikidata) with scientific-KG generality only an ablation.
