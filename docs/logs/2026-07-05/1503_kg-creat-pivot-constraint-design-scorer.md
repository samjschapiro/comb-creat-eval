# kg_creat: pivot to constraint-tradeoff diagnostic, constraint design, and scorer build

**Date:** 2026-07-05 (session spanned 2026-06-04 → 2026-07-05) · **Track:** kg_creat ·
**Paper:** `papers/kg_creat-iclr/` (target **ICLR 2027**, Sep; possible NeurIPS workshop Aug)

## Summary

Long multi-part session that took kg_creat from a Phase-0 scaffold to (a) a **reframed
research question**, (b) a **cognitively-grounded constraint taxonomy** with formal
predicates and failure-mode grounding, (c) a **CREATE-extension methodology** (their code
reviewed and vendored), and (d) the **exact core of the scorer built and tested**. The
headline pivoted from "our score correlates with LiveIdeaBench" to a **diagnostic of the
ideation–execution gap**: how within-path novelty/diversity *modulates* as a function of
constraint type. LiveIdeaBench arm **dropped**; synthetic/pretraining substrate **abandoned**
(real-KG only, OpenRouter, no GPU).

## Phase 1 — KG abstraction (`src/kg_creat/graph.py`)

- Built the KG-agnostic `KnowledgeGraph` (construction subgraph `G_c`), an `nx.MultiDiGraph`
  wrapper (directed + multi-relational, vs `comb_eval`'s undirected single-label graph):
  `from_triples` builder, accessors, relation-frequency table, **typed direction-agnostic
  path enumeration** (`enumerate_paths`/`count_paths_up_to_k`, `LabeledWalk` with orientation),
  `subgraph_around`, `adjacency_text`, versioned JSON save/load.
- Validated 38/38 on a synthetic biomedical `G_c`. **Wikidata sourcing deferred** (built
  source-agnostic first). Convention: cached `G_c` → `data/kg_creat/`; raw dumps → `resources/`.

## The pivot (2026-06-04, per Jonah Black)

- **Headline** → per-constraint-type **novelty–utility tradeoff** as a mechanistic account of
  the **ideation–execution gap** (prior work showed the "what," not the "why").
- **Metric spine (confirmed):** per constraint type, **ideation** = novelty (embedding
  remoteness) of the *emitted* path, vs **execution** = exact constraint satisfaction + judge
  factuality (failure channels broken out). The (novelty × satisfaction) 2×2 is the core result.
  Requires **matched endpoint-bundle sampling** (fix `(u,v,h)`, toggle only the constraint) +
  **difficulty-matching** (hold feasible-fraction constant) so effects are causal in type.
- **Dropped:** the entire LIB validity/specificity arm; synthetic/pretraining substrate.
  **Demoted:** per-model aggregate `C`.
- Reframed all track docs (`design.md`, `paper_outline.md`, `novelty_vs_create.md`,
  `progress.md`) + `research_context.md`; saved memory `kg-creat-constraint-tradeoff-pivot`.

## Constraint taxonomy + failure-mode grounding (`constraints.md`)

- **Organizing axis:** novelty–utility duality → 4 cognitive operations (defeat dominant
  response / bridge remote concepts / respect the conceptual space / elaborate in feasible
  order) + set-level divergence, each anchored to a cog-sci construct and a documented LLM
  failure mode.
- **Failure-mode grounding table** — each constraint abstracts a *quoted* failure mode from
  **Si, Yang & Hashimoto 2024 (arXiv:2409.04109)**: exclusion↔unrealistic assumptions,
  budget↔infeasible compute ("fine-tuning BLOOM 176B"), inclusion↔missing baseline,
  depth↔hand-wavey, categorical↔dataset misuse ("StereoSet is not a QA dataset"),
  waypoint↔ungrounded, ordering↔incoherent pipeline, hub-avoidance↔generic-hook clustering,
  disjointness↔lack of diversity ("only 200 non-duplicate of 4,000"). *Caveat: quotes were
  WebFetch-summarized — verify against the PDF before camera-ready.*
- Also: plain-English cores, an experimental-checkability table (exact vs judge), condensed
  formal predicates. Full formal predicates worked out (well-formedness + all constraint
  indicators + reductions: exclusion⊂budget, hub⊂categorical-by-degree, metapath⊃incl+ord+card).

## Analogy & blending as constrained forms of general CC

- **Most-defensible framing (user-directed):** analogy and conceptual blending are **constrained
  special cases** of our general CC formulation, obtained by *adding* constraints — analogy =
  CC + a metapath (relational-structure) constraint; blending = CC + cross-domain (multi-space)
  constraints. Our constraint typology is the dial from general CC to these structured forms.
- **Bisociation dropped** entirely (subsumed by blending); scrubbed from docs.
- **Boundary:** we subsume the *combinatorial* core, not blending's *emergent structure* (concept
  invention = Boden transformational → the counterfactual/missing-edge frontier).
- Figure V/VI: **V Analogy** = isomorphic structures, disjoint nodes; **VI Blending** (user
  renamed from "Antanaclasis") = colliding/fused isomorphism sharing a pivot node. Both are the
  **semantic tier** (satisfaction is judged, not syntactic) — the yellow-star = human/judge eval.
  I–IV are syntactic (exact, judge-free execution).

## CREATE integration

- Cloned to `resources/repos/CREATE`; read the full codebase. Key structural fact: **their eval
  is a post-hoc scorer on a `path_prediction` JSONL**, decoupled from elicitation → our outputs
  are drop-in and our scoring is a superset that imports theirs. CREATE runs as the
  **no-constraint baseline cell** for free.
- **Extension plan** (methods.md): reuse parser + factuality judge (gpt-oss-120b) + `s(U)`
  verbatim; adapt the prompt (constraint block + controlled relation vocab); replace novelty σ
  (class-size specificity) with embedding remoteness `R`; add the constraint checker + matched
  sampler + aggregator; keep our OpenRouter runner for elicitation.
- **Critique — where CREATE is weak for CC:** their "novelty" is *specificity* not *remoteness*
  (`A —spouse→ B` scores maximally strong = most obvious); `min`-over-triples bottleneck; no
  surprise term; endpoint sampling has obvious-connection bias; utility can't express the
  novelty–utility tradeoff (no constraint lever); diversity over strings not concepts; creativity
  entangled with capability. Keep: open-KG+judge, greedy quality×diversity, parser+judge, σ as a
  secondary signal.

## Scoring design (methods.md)

- **Novelty `R`** = within-path mean pairwise embedding distance (DAT primitive). **Embedding
  unit `(c,c')` vs `(c,r,c')` is a post-hoc ablation** (lean `(c,r,c')`; report both as a
  robustness check) — a scoring choice, not a collection-time commitment, so **log full triples**.
  Axes need not be orthogonal.
- **Diversity `D`** = between-path (reuse CREATE's path embeddings; matched `k`).
- **Utility `U`** = leveled: L0 coherence → L1 factuality → L2+ constraints (weighted); maps onto
  execution floor + `sat` axis.
- **Key scientific question:** how novelty/diversity *modulates* per constraint type, measured
  against a **model-free structural reference** (feasible set via `enumerate_paths`) — the gap
  `ΔR_struct − ΔR_model` = the model's *avoidable* creativity loss. No profile predicted (empirical).
- Surprise-as-separate-dimension considered and **dropped for v1** (within-path novelty only).

## Overleaf cleanup (`papers/kg_creat-iclr/`)

- Cloned the Overleaf repo (git-bridge was down for ~1 day, an external outage — diagnosed and
  waited). It is the **original synthetic Comb-Creat paper** being repurposed. Decision:
  **"kg_creat replaces the empirics"** — keep `04_background`, the ideation–execution intro
  framing, and `media/07_table_of_utility_constraints` (the grounding-table seed); swap synthetic
  small-model experiments for the real-KG constraint study.
- **Archived** (moved to `archive/`, not deleted) all obsolete/orphaned synthetic content + old
  drafts/slides + orphaned figures + unused `jmlr.sty`; fixed a dangling `\Cref{sec:discussion}`;
  `main.tex` untouched and still compiles. Committed + pushed (`db7ccf9`). The two synthetic
  sections still in the build (`old/05_theory_cc_paths`, `old/06_experiments_scaling`) left for a
  later main.tex trim.

## Code built + tested (this session's implementation)

- **`src/kg_creat/scoring.py`** — `EmittedPath` (from CREATE-format triples) + well-formedness
  (L0, failure channels) + all **syntactic constraint predicates** (exclusion/inclusion/ordering/
  metapath/budget/waypoint/hub-avoidance/categorical/depth/disjointness) + **novelty `R`**
  (concept/triple ablation). **40/40 toy checks**, incl. hand-verified pairwise-cosine math.
- **`src/kg_creat/vendor/create/`** — CREATE **vendored** (`path_evaluator`, `prompt_bank`,
  `creative_utility`, `inference`, `prompt`) + `NOTICE.md`. Upstream has **no code license**;
  **author gave permission** (user knows her) → copied, only imports relativized, logic untouched
  for comparability.
- **`src/kg_creat/parse.py`** — bridges CREATE's `Path.parse_path_from_text` → `EmittedPath`.
  9/9 (real `<answer>` JSON, `</think>`+markdown messiness, garbage→`[]`).
- End-to-end front-of-scorer verified: **raw output → parse → constraint checks + `R`**.

## Example outputs per constraint (pilot grounding)

Worked out plausible good/bad outputs (CREATE triple format, biomedical running example
aspirin→colorectal cancer) for all six constraint sets, each bad case labeled with its failure
channel (constraint-violation / hallucination / generic-low-novelty / incoherent). V/VI examples
(insulin:glucose::thermostat:temperature analogy; Mercury planet/element blend) confirmed the
semantic tier — their "bad" cases (metapath-matches-but-not-analogy; sense-conflation) are
invisible to the exact checker and need a judge.

## Open decisions / next steps

- **Verify Si et al. quotes** against the PDF; firm up the two loose cog-sci anchors (ordering,
  hub-avoidance); confirm `04_background` cites **Gentner** (now load-bearing for analogy).
- **Remaining scorer:** wire vendored **factuality judge** (litellm + API key) + reliability
  harness; wire **`s(U)`** (sentence-transformers); the **semantic (V–VI) checker** (judge-based);
  the **aggregator** (per-constraint `R`/`D`/`sat` + failure channels + structural-reference
  modulation profile). Also: **Wikidata builder** + **matched-bundle sampler** + **elicitation
  runner** (OpenRouter, budget, resumable).
- Offered but not yet done: turn the example outputs into `data/kg_creat/fixtures/` end-to-end
  scorer regression tests.
- Later: trim `main.tex` to drop the two remaining in-build synthetic sections.
