# kg_creat — progress

## Goal

Re-purpose the Comb-Creat task setup
([Schapiro et al. 2025, arXiv 2509.21043](https://arxiv.org/abs/2509.21043))
into a **test-time creativity eval administered to frontier models on a real
knowledge graph**, and show its per-model scores **correlate with LiveIdeaBench
(LIB)** — ideally better than CREATE
([Wadhwa et al. 2026, arXiv 2603.09970](https://arxiv.org/abs/2603.09970)) —
under the [dat_eval](../dat_eval/progress.md) validity/specificity framework.

Original Comb-Creat ran constrained labeled-graph pathfinding on a *synthetic*
graph against models ≤100M params *trained* on that graph. This track ports the
same constraint + novelty/utility apparatus onto a **real KG (Wikidata)** and
administers it to frontier models **without retraining** — the survey's "Gap A"
([survey §5](../new_tests/survey.md)).

**Target venue:** COLM 2026 LM4Sci workshop (Language Models for Scientific
Discovery). 8pp main, non-archival. **Deadline: June 23 2026 AOE.**

## Why a separate track (not new_tests / comb_eval)

Per [repo_usage.md](../../repo_usage.md), tracks separate *fundamentally
different approaches*:
- `comb_eval` — synthetic Erdős–Rényi graph, models *trained* on the graph.
- `new_tests` — Cre-DPO and preference-optimization *training* methods (+ DRAT).
- `kg_creat` (this track) — a *real-KG test-time eval* of frontier models, no
  training. Reuses `comb_eval`'s scoring primitives and `dat_eval`'s
  validity/specificity pipeline, but the task substrate and research question
  are distinct.

## Headline methodological novelty vs CREATE (locked)

CREATE already does Wikidata multi-hop paths between two endpoints with
quality×diversity scoring at test time. Our distinguishing contribution is:

1. **A typology of methodological constraints** — inclusion / exclusion / categorical /
   waypoint / ordering-metapath / budget / polarity (full list in
   [design.md](design.md) §Constraint typology). CREATE has none. Difficulty is now
   **2-D: constraint count × type**; recovers Comb-Creat's *novelty–utility trade-off*
   and adds a *constraint-type* diagnostic (which pressure predicts LIB / its facets).
2. **Constraint-load-weighted utility** `U = ∏_t(1+α_t·n_t)·1[valid∧factual]` —
   structure CREATE's unconstrained association task cannot express. Constraints are
   enforced **exactly on the judge-verified path** (so the apparatus stays rigorous).

**Verification is shared ground with CREATE, not a differentiator.** We **reverted to
CREATE's open-KG + LLM-judge factuality on 2026-06-02** (exact held-subgraph checking
was too restrictive — it dropped true-but-out-of-coverage paths and blocked the model's
parametric knowledge). We adopt/improve CREATE's gpt-oss-120b judge + a reliability
analysis. Novelty also uses the **validated DAT semantic-distance measure** (not a new
formula), so it too is **not** a differentiator. The moat is 1–2 above + LIB validation.
(Scoring history: label-surprise → DAT remoteness 2026-06-02; exact → judge 2026-06-02.)

The LIB correlation is the *empirical demonstration* of the test, not the
headline novelty. Full comparison table: [novelty_vs_create.md](novelty_vs_create.md).
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
- CREATE head-to-head baseline **deferred** — establish our own r-with-LIB first.
- **Multi-graph by design**: run on **Wikidata** (method/CREATE contrast) **and one
  or more scientific KGs** (validity payoff + cross-graph generality). Scientific
  arm = **Hetionet** (primary, built for typed multi-hop paths), then **PrimeKG**,
  optionally **DRKG**. Hard rule: KG must be *relation-rich* — citation graphs
  (ACL/OpenAlex) and MeSH are excluded (too few relation types). See
  [design.md](design.md) §Knowledge-graph backends.

## Phased roadmap

1. **Phase 1 — eval engine.** Common KG loader (`src/kg_creat/graph.py`) with a
   **Wikidata** loader first — builds a **construction subgraph** for endpoint/constraint
   sampling (not a closed answer space); port `EvalPrompt` (`src/comb_eval/prompts.py`) +
   structural/constraint scoring (`src/comb_eval/scoring.py`); **factuality judge**
   (adopt/improve CREATE's gpt-oss-120b). Smoke test on 1–2 cheap models (mirror
   `src/new_tests/scripts/run_drat_smoke.py`).
2. **Phase 2 — pilot (Wikidata).** 6 cheap LIB-pool models. Verify the score is
   discriminating, the constraint-count lever produces variance, the novelty–utility
   trade-off is recovered, and — gate — the **judge reliability** (vs human spot-check)
   is acceptable.
3. **Phase 3 — full run + validation, multi-graph.** Full ~31-model LIB pool on
   **Wikidata + Hetionet** (add **PrimeKG**/DRKG as they land). Per KG, compute
   validity `r(C, LIB)` and specificity `r(C, LIB | g)`, `g = (arena_overall,
   mmlu_pro)`; place `C` alongside DAT/CDAT/PACE (dat_eval) and DRAT (new_tests).
   Cross-KG agreement is itself a robustness result.
4. **Phase 4 — comparisons/ablations.** CREATE head-to-head (run vs reuse
   leaderboard — decide here); **constraint-type ablation** (inclusion / exclusion /
   categorical / waypoint / ordering-metapath — which methodological pressure best
   predicts LIB, and whether types map to LIB facets); constraint on/off; additional
   scientific KGs; aggregator and embedding ablations. Stretch: the counterfactual /
   missing-edge variant as the ideation-facing thread.
5. **Phase 5 — write-up.** LM4Sci COLM 2026, 8pp. One-graph result is shippable;
   extra KGs strengthen generality.

## Next steps

1. Build the **common KG loader** (`src/kg_creat/graph.py`) with a local **Wikidata**
   dump first — materializes a **construction subgraph** `G_c` around sampled endpoints
   for prompt design + as a factuality reference. The loader interface (raw KG → typed
   `G_c` + relation-frequency table) makes adding Hetionet/PrimeKG a loader, not a new
   pipeline. Structural/constraint scoring ports from `comb_eval`.
2. Stand up the **factuality judge** (adopt/improve CREATE's gpt-oss-120b via
   `src/dat_eval/llm.py`) and a human-spot-check reliability harness.
3. Pre-register the endpoint/constraint sampler (shared across KGs); require each
   constraint *satisfiable* (∃ valid path) and *biting* (removes the obvious routes),
   checkable on `G_c` before a prompt is used.
4. Acquire scientific-KG dumps to `resources/` (gitignored): **Hetionet** then **PrimeKG**.

## Open decisions (tracked)

- **Verification = open KG + LLM judge** (reverted from exact 2026-06-02). Open: adopt
  CREATE's gpt-oss-120b judge as-is vs improve it; how to fold categorical/polarity
  entity-type lookups in (KG dump vs judge).
- Novelty = DAT-style semantic remoteness `R` (per-path) + separate set diversity `D`;
  relation-surprise demoted to baseline (decided 2026-06-02, see [design.md](design.md)).
- Path-set aggregator for `C`: greedy `s_γ` quality×diversity vs mean·`D` (the *separate
  diversity term* is decided; only the aggregator form is open). Pilot decision.
- Embedding for `R`/`D`: SBERT default vs GloVe/FastText ablation (dat_eval infra).
- Per-KG endpoint familiarity (sample well-known entities so models can form factual
  paths); the score partly reflects domain recall — note it.
- Whether to reuse CREATE's exact (relation, category) entity-sampling recipe on the
  Wikidata arm for a cleaner head-to-head (Phase 4).
