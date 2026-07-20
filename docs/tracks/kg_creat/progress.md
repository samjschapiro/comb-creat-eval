# kg_creat — progress

## Goal (reframed 2026-06-04 — see Direction pivot below)

Re-purpose the Comb-Creat task setup
([Schapiro et al. 2025, arXiv 2509.21043](https://arxiv.org/abs/2509.21043))
into a **test-time, real-KG creativity task administered to frontier models**, and use a
**taxonomy of constraints — each a minimal abstraction of a real-world rule creative
generation must obey** — to produce a **per-constraint-type ideation–execution
decomposition**: which constraints can LLMs satisfy while keeping novelty high, and which
force a tradeoff. This is a **mechanistic diagnostic of the ideation–execution gap** (prior
work showed the *what*, not the *why*). CREATE
([Wadhwa et al. 2026, arXiv 2603.09970](https://arxiv.org/abs/2603.09970)) is the closest
prior work (unconstrained = our no-constraint baseline). Authoritative spec:
[design.md](design.md) §Study framing.

Original Comb-Creat ran constrained labeled-graph pathfinding on a *synthetic*
graph against models ≤100M params *trained* on that graph. This track ports the
same constraint + novelty/utility apparatus onto a **real KG (Wikidata et al.)** and
administers it to frontier models **without retraining** — the survey's "Gap A"
([survey §5](../new_tests/survey.md)). **No GPU cost** — real entities, OpenRouter only.

**Target venue:** was COLM 2026 LM4Sci (8pp, non-archival, deadline June 23 2026 AOE);
**under reconsideration** since the ideation–execution-gap framing is broader than
scientific discovery.

## Direction pivot — 2026-06-04 (Jonah Black discussion)

Headline moved from "our score correlates with LiveIdeaBench (validity/specificity vs
CREATE)" to the **per-constraint novelty–utility tradeoff / ideation–execution gap**
diagnostic above. Confirmed with the user:

- **Metric spine:** per constraint type, **ideation** = novelty (DAT embedding remoteness)
  of the *emitted* path, vs **execution** = exact constraint satisfaction + judge
  factuality (failure channels broken out). The (novelty × satisfaction) 2×2 across types
  is the core result. Requires **matched endpoint-bundle sampling** (fix `(u,v,h)`, toggle
  only the constraint) so the tradeoff is causal in constraint *type*.
- **DROPPED:** the entire LIB validity/specificity arm (capability controls, external
  correlation) — it was n≈20 and significance-underpowered (dat_eval hit this wall).
- **DROPPED:** synthetic/pretraining substrate — real entities on a real KG, OpenRouter only.
- **Demoted:** the per-model quality×diversity aggregate `C` (secondary summary).
- **Central open risk:** *grounding rigor* — each constraint's mapping to a real-world
  creative rule must be defensible/cited, or the taxonomy reads as arbitrary.

Docs brought in line: [design.md](design.md), [paper_outline.md](paper_outline.md),
[novelty_vs_create.md](novelty_vs_create.md). Memory: `kg-creat-constraint-tradeoff-pivot`.

## Why a separate track (not new_tests / comb_eval)

Per [repo_usage.md](../../repo_usage.md), tracks separate *fundamentally
different approaches*:
- `comb_eval` — synthetic Erdős–Rényi graph, models *trained* on the graph.
- `new_tests` — Cre-DPO and preference-optimization *training* methods (+ DRAT).
- `kg_creat` (this track) — a *real-KG test-time eval* of frontier models, no
  training. Reuses `comb_eval`'s scoring primitives and `dat_eval`'s embedding-novelty
  infra, but the task substrate and research question are distinct.

## Headline methodological novelty vs CREATE (updated 2026-06-04)

CREATE already does Wikidata multi-hop paths between two endpoints with
quality×diversity scoring at test time. Our distinguishing contribution is:

1. **A taxonomy of constraints, each a minimal abstraction of a real-world rule creative
   generation must obey** — inclusion / exclusion / categorical / waypoint /
   ordering-metapath / budget / polarity (full list in [design.md](design.md)
   §Constraint typology). CREATE has none. Difficulty is **2-D: constraint count × type**.
2. **The per-constraint ideation–execution decomposition** these constraints enable —
   ideation (emitted-path novelty) vs execution (constraint satisfaction + factuality) per
   type, a mechanistic diagnostic of the ideation–execution gap. Backed by
   **constraint-load-weighted utility** `U = ∏_t(1+α_t·n_t)·1[valid∧factual]`, enforced
   **exactly on the judge-verified path**.

**Verification is shared ground with CREATE, not a differentiator.** We **reverted to
CREATE's open-KG + LLM-judge factuality on 2026-06-02** (exact held-subgraph checking
was too restrictive — it dropped true-but-out-of-coverage paths and blocked the model's
parametric knowledge). We adopt/improve CREATE's gpt-oss-120b judge + a reliability
analysis. Novelty also uses the **validated DAT semantic-distance measure** (not a new
formula), so it too is **not** a differentiator. **The moat is 1–2 above** (grounded
constraints + the diagnostic) — *not* an external-benchmark correlation (LIB arm dropped).
(Scoring history: label-surprise → DAT remoteness 2026-06-02; exact → judge 2026-06-02.)

The per-constraint ideation–execution decomposition is the *empirical demonstration* and
the headline. Full comparison table: [novelty_vs_create.md](novelty_vs_create.md).
Planned submission narrative/outline: [paper_outline.md](paper_outline.md).

## Status — 2026-06-04 (Phase 1 started: KG abstraction landed)

The KG-agnostic core of the eval engine is in. `src/kg_creat/graph.py` defines
the **common `KnowledgeGraph` abstraction** — the construction subgraph
`G_c = (V, E, Σ)` from [design.md](design.md), as an `nx.MultiDiGraph` wrapper
(directed + multi-relational, vs `comb_eval`'s undirected single-label
`nx.Graph`). Implemented and validated (38/38 checks on a synthetic biomedical
`G_c`, network-free):

- `from_triples(...)` — the common builder entry: any KG (Wikidata, Hetionet,
  PrimeKG) becomes a `KnowledgeGraph` via a flat triple list + label/type maps,
  so adding a KG is a *builder*, not a new pipeline. Fails loud on any
  unlabeled entity/relation.
- Accessors (`label`, `types`, `relation_label`, `has_triple`), a
  **relation-frequency table** (`relation_frequency()`), and `stats()`.
- **Typed, direction-agnostic path enumeration** (`enumerate_paths`,
  `count_paths_up_to_k`): node-distinct labeled walks `P=(e₀,r₁,…,r_h,e_h)` of
  exact hop count, traversing the *undirected projection* (a meaningful path may
  use an edge against its stored direction) while recording each step's true
  directed triple (`LabeledWalk.forward`). Supports inclusion/exclusion relation
  gating — the satisfiable-and-biting sampler check. Ports
  `comb_eval.prompts.bfs_paths`/`count_valid_paths_up_to_k`, typed.
- `subgraph_around(seeds, radius)`, `adjacency_text(...)` (factuality
  reference / debugging view), and versioned JSON `save`/`load` (`GC_SCHEMA_VERSION`,
  fails loud on schema drift / missing cache).

**Decision (2026-06-04):** Wikidata *sourcing* (SPARQL-BFS vs REST-BFS vs full
dump) deferred — built the source-agnostic abstraction first so the sourcing
choice is now just a builder that emits `from_triples`. Convention: the cached
`G_c` is a derived artifact → `data/kg_creat/` (gitignored); raw dumps
(Hetionet/PrimeKG) → `resources/`.

**Next:** the Wikidata builder (`src/kg_creat/wikidata.py` → `KnowledgeGraph`),
plus a config-driven build orchestration script writing `G_c` to `data/kg_creat/`.

## Status — 2026-06-01 (track created)

Phase 0 scaffold only. Directory layout, design spec
([design.md](design.md)), and the CREATE comparison doc written. No eval code
yet. Design decisions locked with the user:

- Novelty lead = constraints + novelty/utility scoring (above).
- CREATE head-to-head baseline **deferred**. *(Superseded 2026-06-04: CREATE is now the
  no-constraint baseline cell; the r-with-LIB goal this line referenced is dropped.)*
- **Multi-graph by design**: run on **Wikidata** (method/CREATE contrast) **and one
  or more scientific KGs** (validity payoff + cross-graph generality). Scientific
  arm = **Hetionet** (primary, built for typed multi-hop paths), then **PrimeKG**,
  optionally **DRKG**. Hard rule: KG must be *relation-rich* — citation graphs
  (ACL/OpenAlex) and MeSH are excluded (too few relation types). See
  [design.md](design.md) §Knowledge-graph backends.

## Phased roadmap (reframed 2026-06-04 — per-constraint diagnostic, no LIB arm)

1. **Phase 1 — eval engine.** Common KG loader (`src/kg_creat/graph.py`, **done**) →
   **Wikidata** builder; the **grounding table** (constraint type → real-world rule +
   citation); the **matched endpoint-bundle sampler** (fix `(u,v,h)`, toggle only the
   constraint; each constraint satisfiable + biting on `G_c`); port structural/constraint
   scoring (`src/comb_eval/scoring.py`) + embedding novelty `R` (`dat_eval`); **factuality
   judge** (adopt/improve CREATE's gpt-oss-120b). Smoke test on 1–2 cheap models.
2. **Phase 2 — pilot (Wikidata).** A handful of cheap models spanning capability. Gates:
   the matched-bundle design yields interpretable per-constraint `R_emit`/`sat`; the
   constraint lever produces variance; the novelty–utility trade-off appears; and the
   **judge reliability** (vs human spot-check) is acceptable.
3. **Phase 3 — full run, multi-graph.** A capability-spanning model set (OpenRouter,
   budget-driven) on **Wikidata + Hetionet** (add **PrimeKG**/DRKG as they land). Per KG,
   compute the **per-constraint ideation–execution 2×2** (`R_emit`, `R_valid`, `sat` +
   failure channels, within-bundle deltas). **Cross-KG replication of the per-constraint
   signature** is the robustness result.
4. **Phase 4 — comparisons/ablations.** CREATE = the no-constraint cell (baseline);
   **constraint-type analysis** = the headline (which type forces the gap vs safe play, and
   whether the pattern is universal or model-ranked); constraint on/off; additional
   scientific KGs; aggregator and embedding ablations; label-surprise baseline. Stretch:
   the counterfactual / missing-edge variant as the ideation-facing thread.
5. **Phase 5 — write-up.** 8pp (venue TBD — LM4Sci or a broader LLM-eval/creativity venue).
   One-graph result is shippable; extra KGs strengthen generality.

## Next steps

Full scoring + CREATE-extension methodology is now in [methods.md](methods.md) (scoring
operationalization, the key scientific question, the CREATE reuse/adapt/replace/add plan).
CREATE is cloned to `resources/repos/CREATE`. **Build order is scorer-first** (network-free,
testable before spending on elicitation):

1. ~~Common KG loader `src/kg_creat/graph.py`~~ **done 2026-06-04**.
2. ~~Grounding table~~ **done 2026-07-02** → [constraints.md](constraints.md). Still to do:
   verify the Si et al. quotes against the PDF; firm up the two loose anchors (ordering,
   hub-avoidance).
3. ~~Scoring + CREATE-extension methodology~~ **documented 2026-07-03** → [methods.md](methods.md).
   Still to do: confirm CREATE's *code* license before vendoring.
4. ~~**Scorer core (scorer-first):**~~ **done 2026-07-05.** CREATE vendored (author-cleared)
   → `src/kg_creat/vendor/create/` + `NOTICE.md`; `src/kg_creat/parse.py` bridges their parser
   → `EmittedPath`; `src/kg_creat/scoring.py` = well-formedness + all syntactic (I–IV)
   constraint predicates + novelty `R` (concept/triple ablation), 40/40 toy checks + parser
   bridge tested. **Remaining scorer work:** wire the vendored factuality judge (needs litellm
   + API key) + reliability harness; wire `s(U)` (needs sentence-transformers); the **semantic
   (V–VI) analogy/blending checker** (judge-based); the **aggregator** (per-constraint
   `R`/`D`/`sat` + failure channels + structural-reference modulation profile).
5. **Wikidata builder** (`src/kg_creat/wikidata.py` → `KnowledgeGraph.from_triples`) — the
   deferred sourcing decision (SPARQL-BFS / REST-BFS / dump), cached `G_c` → `data/kg_creat/`.
6. **Matched endpoint-bundle sampler** — pre-registered; fix `(u,v,h)`, toggle only the
   constraint; each constraint satisfiable + biting on `G_c`; difficulty-matched feasible
   fraction. Reuses `graph.py::enumerate_paths` for the structural feasible-set reference.
7. **Elicitation runner** — OpenRouter, budget cap, save-raw-immediately, resumable;
   emits CREATE's `path_prediction` schema (drop-in to the scorer).
8. Acquire scientific-KG dumps to `resources/` (gitignored): **Hetionet** then **PrimeKG**.

## Open decisions (tracked)

- **Grounding table (highest priority)** — the exact real-world rule + citation per
  constraint type; must be defensible or the taxonomy reads as arbitrary (§design.md Risks).
- **Wikidata sourcing** — SPARQL-BFS vs REST-BFS vs dump (deferred 2026-06-04; abstraction
  built source-agnostic so this is now just a builder).
- **Matched-bundle sampler parameters** — hop count(s), `k`, constraint count per bundle,
  endpoint familiarity (well-known entities so models can form factual paths); pre-register.
- **Verification = open KG + LLM judge** (reverted from exact 2026-06-02). Open: adopt
  CREATE's gpt-oss-120b judge as-is vs improve it; how to fold categorical/polarity
  entity-type lookups in (KG dump vs judge).
- Novelty = DAT-style semantic remoteness `R` (per-path); relation-surprise demoted to
  baseline (2026-06-02). Embedding: SBERT default vs GloVe/FastText ablation.
- Path-set aggregator for the **secondary** `C`: greedy `s_γ` vs mean·`D` (pilot decision;
  no longer headline-critical since `C` is demoted).
- Venue: LM4Sci COLM vs a broader LLM-eval/creativity venue (framing is now broader).
