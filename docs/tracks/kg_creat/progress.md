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

## Status — 2026-07-26 (benchmark redesign: arbitrary endpoints, antanaclasis, diversity, CREATE-parity)

Major methodology pivot toward the paper's real benchmark. Full session record:
[docs/logs/2026-07-26/1625_kg_creat_diversity_pivot.md](../../logs/2026-07-26/1625_kg_creat_diversity_pivot.md).

- **Arbitrary-entity endpoints.** Dropped the connectivity/biting sampler filters (they selected
  hub entities and defeat combinatorial creativity; biting is already handled post-hoc by
  baseline-derived targets). `sample_random_bundles`, `sampler.strategy=random`. Verified on an
  obscure-pair probe that models attempt real paths and the judge adjudicates the tail well.
- **Blending = true antanaclasis.** Fixed anchor; the task is to find a *valid polysemy* (same word,
  two senses — the C6 'Boxer' figure). Prompt + judge reworked; smoke-tested (Turkey passes; judge
  rejects forced/fabricated senses).
- **Set-level diversity + systematic decoding.** M=10 resamples per prompt × temperature sweep
  {0.7, 0.9, 1.0}; `diversity.py` computes D over all + valid items. Diversity is free (embeddings);
  only utility judging scales with M.
- **CREATE-parity size.** ~931 instances (120 bundles×5 + 165 analogy + 166 blending). Grew G_c
  (`gc_domains_v2`, 51 seeds/16 domains): prominent pool 424 → **1,066**. Domain balance uneven
  (politics/geography heavy).
- **Explicit "be creative" in every task** (Regime A previously asked only for strong+diverse, not
  novelty). Domains tagged as reference metadata, never shown to the model.
- **Infra hardened:** M/temperature knobs in `run_elicit`; crash-proof path parsing (a set-valued
  triple once killed a run at json.dumps).
- **Budget reality:** this OpenRouter key has ~$217 left (not the account's $1.3k). Measured output
  ~250 tok, so runs are cheaper than the conservative estimate; the cheap pilot is ~$15.

**Running now:** Stage-1 diversity pilot (3 cheap models × 451 instances × 3 temps × M=10).
**Next:** report temp×diversity surface + per-task rates + categorical derivability → derive targets
→ Stage 2 constrained cells → judge slice. Owed: human judge-reliability pass.

## Status — 2026-07-22 (ordering dropped; constraint set is now four types)

**Ordering removed from the constraint set.** Round-1 flagged it as the most damaging constraint
(−86 % creativity), but that number is a construction artifact of deriving the target as the
*reverse* of the natural class order: (1) it is really a conjunction — only ~12 % of unconstrained
paths contain both target classes; (2) the demanded direction is anti-natural — of paths where
both classes co-occur, 89 % are in the reverse (natural) order; (3) it is sometimes infeasible —
8/30 bundles had zero satisfying paths across all 8 models. Decomposing its 495 failures, only
11.5 % are true order inversions. Dropped from figures, reports, and future generation
(`make_pass2.py` no longer emits the cell; raw scored data kept on disk for the record).

**Headline finding — a constraint is a novelty lever and an adherence tax at once.** Creativity
factorises as `E[R·U] = R_valid × adherence`, and a constraint moves the two factors in opposite
directions: it *raises* the novelty of successful paths (+9–11 %, except common-inclusion +1 %) and
*lowers* the adherence rate (−27 to −50 %). The novelty gain is causal, not survivorship — it shows
up in `R_emit` over **all** emitted paths (+0.019 to +0.053, no success filtering) and within fixed
endpoints (paired ΔR_valid +0.028 to +0.072, 56–76 % of bundles). Net creativity falls because the
adherence tax dominates: categorical −16 % (n.s.), exclusion −27 % (**), rare-inclusion −37 % (**),
common-inclusion −48 % (**). The novelty lever is specifically "go somewhere you usually wouldn't" —
the one constraint requiring a *common* relation buys no novelty and is the worst cell; categorical,
which redirects the waypoint without restricting the vocabulary, buys the most and is the only cell
that can net positive (2/8 models). Headline figure `fig_creativity_mechanism.pdf`. Primary writeup:
[`docs/reports/2026-07-22_kg_creat_creativity/`](../../reports/2026-07-22_kg_creat_creativity/report.md);
the 2026-07-21 report is superseded (ordering rationale preserved there). Camera-ready creativity
figure (Nature MI spec) at `papers/kg_creat-iclr/media/fig_creativity_by_constraint.pdf`.

## Status — 2026-07-21 (Regime A run at scale — the headline per-constraint result exists)

**The core deliverable of the track now has data.** 8 models × 30 fixed endpoint bundles ×
{baseline, exclusion, inclusion, inclusion_rare, ordering, categorical} = 1,440 prompts /
7,159 judged paths. Report:
[docs/reports/2026-07-21_kg_creat_regimeA/](../../reports/2026-07-21_kg_creat_regimeA/report.md).

**Framing (per the user, 2026-07-21):** creativity = novelty **and** utility; the constraints
*are* the utility operationalization. So the analysis is one dependent variable (creativity,
`E[R·U]` per design.md §Scoring) across levels of one independent variable (constraint type),
not a novelty-vs-satisfaction tradeoff between two goods. Reported at α = 0; since every cell
carries exactly one constraint, a uniform α cannot reorder the constraint types.

**Findings.**
- **Creativity by constraint type** (pooled, baseline 0.209): categorical 0.176 (−16 %),
  exclusion 0.155 (−26 %), inclusion-rare 0.127 (−40 %), inclusion 0.116 (−45 %),
  ordering 0.029 (**−86 %**).
- **Not explained by difficulty.** Against the models' own default behaviour, rare-inclusion
  rules out 99.2 % of baseline paths and ordering 98.6 % — matched — yet ordering yields
  **4.4× less creativity**. Constraint *type* carries information beyond restrictiveness; it is
  the **conjunction** of two class requirements that is destructive.
- **Constraints raise novelty but never enough.** ΔR_valid > 0 for all five cells, and ordering
  has the *highest* realized novelty in the study (0.496 vs 0.420) — but utility falls 0.13–0.45
  while novelty rises only 0.02–0.08. A novelty-only reading would wrongly report that
  constraints increase creativity: they increase **ideation** and decrease **creativity**.
- **Categorical is the only constraint that ever beats the baseline** — 2/8 models
  (Sonnet 4.6 +0.072 → 0.328, the highest cell in the study; GPT-4.1-mini +0.042).
- Constraints **don't degrade factuality** — the factual channel is a flat ~34–40 % tax in
  every cell *including baseline* (34.3 %). The entire cost lands in the constraint channel.
- **Ordering fails as double-inclusion, not as sequencing**: only 11.5 % of its failures are
  genuine order violations. Next run should add a "both classes, any order" cell.

**Design changes made to get here** (recorded as amendments in
[assessment.md §7b](assessment.md)): constraints are over **relation CLASSES** (k-means over
embeddings of the top-150 relations models actually emitted) rather than single labels, since
an open vocabulary makes label-level constraints a wording lottery; targets are **derived per
bundle from that bundle's own baseline behaviour**, so each constraint bites by construction;
added an `inclusion_rare` cell; and **blending was reframed to a single stimulus** (one anchor,
two structures emanating outward into different domains, sharing the anchor and nothing else).

**Two measurement defects found and fixed** (both would have corrupted the headline):
- `max_tokens=1200` truncated long answers, and truncated JSON parses to **zero** paths.
  GPT-4o-mini lost 104/180 prompts — it would have read as a 60 % structural failure rate that
  was really a token cap meeting a verbose model. Fixed with truncation salvage in `parse.py`
  (keeps only paths whose array closed, so a half-emitted path isn't scored as "never reached
  the target"). All models now ~5.0 paths/prompt.
- The categorical judge ran at `max_tokens=400`; a reasoning judge spends a small budget
  thinking and never emits JSON, silently turning satisfaction into `unjudged` (123 paths).
  Raised to 800; added `scripts/rejudge.py` to repair a cell without re-scoring the corpus.
  Unjudged fell **196 → 9** paths (0.13 %).
- Also: one malformed provider response used to propagate out of `asyncio.gather` and kill an
  entire model's scoring mid-run (~25 min of paid judging lost). Judge calls now retry and
  degrade to one unjudged record instead of taking the run down.

**New this session:** `src/kg_creat/relation_classes.py`, `src/kg_creat/regime_b.py` (shared
structure-mapping predicates for analogy + blending, so scorer and figures use one definition),
`scripts/make_pass2.py`, `scripts/rejudge.py`, `scripts/plot_regime_a.py`.

**Still owed:** the human blind judge-reliability pass (owed since the analogy round, and now
load-bearing since all five Regime-A cells are judged rather than exactly checked); running the
reframed blending task at scale.

Cost to date this round: ~$6.6 (elicitation $4.32, judging ~$2.2, re-judge $0.09).

## Status — 2026-07-20 (full pipeline built end-to-end; first analogy result)

The entire eval pipeline now runs end-to-end, and the **analogy tier (constraint family V)
became the session's empirical focus** — a first real result exists. Report:
[docs/reports/2026-07-20_kg_creat_analogy/](../../reports/2026-07-20_kg_creat_analogy/report.md).

**Environment (Mac).** The repo's pinned `torch==2.6.0+cu124` is unusable on macOS/py3.14, so
`uv sync` fails here. Round-1 is **torch-free**: the main `.venv` (3.14) runs graph/sample/
elicit with `uv pip install`'d deps; a second **`.venv_mlx` (3.12)** runs MLX for local
model serving, local embeddings, and the scorer. (Lambda/CUDA hosts unaffected.)

**Built this session** (all in `src/kg_creat/`, orchestration in `scripts/`, configs in
`configs/kg_creat/`, outputs to gitignored `data/kg_creat/`):
- `wikidata.py` — REST-BFS builder over a **frequency-derived** relation vocabulary (top-N
  most-used properties, minus a documented admin/attribute stoplist — *not* hand-picked);
  **domain-tagged seeds** (each entity inherits its reaching seed's domain, so domain is a
  study variable); cached typed `G_c` (domain build: 3,442 entities / 13 domains / 24 relations).
- `sample.py` — matched-bundle sampler (Regime A: biting exclusion/inclusion/categorical/
  ordering) + **random** Regime-B endpoint sampler (analogy/blending pairs drawn at random
  from `G_c`, seeded — deliberately hard, no curation).
- `prompts.py` — CREATE-aligned prompts (its scaffolding + our constraint block); **open
  vocabulary** for the model (controlled-vocab was tried then reverted).
- `run_elicit.py` (OpenRouter + local-MLX via `LLM_BASE_URL`, budget cap, per-model resume),
  `judge.py` (CREATE's gpt-oss-120b factuality prompt + semantic analogy/blending + relation-
  constraint judges), `embed.py` (local MLX MiniLM for novelty `R`), `aggregate.py`, `score.py`,
  four plotters, and the blind judge-reliability review harness (`sample_review.py`,
  `review_server.py` web UI with auto-logging, `score_review.py`).

**Design decisions locked this session** (several reverse earlier assumptions):
- **No exact hop count** — variable-length paths, matching CREATE (`h` used only for sampling).
- **Open-vocabulary relations** for the model; the derived vocabulary is a graph/BFS reference only.
- **Constraint checking is judge-based** under open vocab (the "exact/judge-free constraints"
  claim is dropped; the contribution remains the typology + the ideation–execution decomposition).
- **Analogy validity is strict structure-mapping**: exact relation-sequence match ∧ disjoint
  structures ∧ node-distinct ∧ factual ∧ judged role-correspondence. (Getting this right took
  several iterations — paraphrased relations, loop-backs, and shared entities each inflated the rate.)

**First analogy result (n=200 random pairs × 8 models, gpt-oss-120b judge, ≈$5.7).**
Even the best models (Sonnet-4.6 26.0%, Haiku-4.5 25.5%) find a valid analogy between two
arbitrary entities only ~1-in-4 times; field spans 1–26% (Llama-3.1-8B → Sonnet). Complementary
per-pair analysis: **anchor embedding distance is a weak predictor** (Pearson −0.14, Spearman ≈0)
— analogical difficulty is *structural, not distributional*.

**Regime-A (the constraint typology, still the paper's intended headline)** was only *piloted*
(≈10 matched bundles, weak local Qwen-3B/7B): the `plot_novelty_utility` 2×2 shows exclusion
"handled" vs inclusion/categorical/ordering pushing toward the gap, but capability failures
(structural/factual) dominate at that model scale — **not yet scaled to frontier models**.

**Next:** (1) run the blind judge-reliability review (harness ready) → the CREATE-comparable
number; (2) scale Regime A to frontier models (the constraint 2×2 is the paper's core, and is
under-explored vs analogy this session); (3) `score.py` analogy-success is computed in the
plotters, not `score.py` itself — fold it in; (4) broader, less academia-heavy seeds.

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
