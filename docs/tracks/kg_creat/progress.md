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

**Target venue: ICLR 2027** (confirmed 2026-09-04; Overleaf repo `papers/kg_creat-iclr/`).
Earlier candidates: COLM 2026 LM4Sci (8pp, non-archival, deadline June 23 2026 AOE), dropped when
the framing outgrew scientific discovery.

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

**Stage-1 pilot done** (3 cheap models × 451 baseline/analogy/blending instances × 3 temps × M=10,
~$6): diversity rises monotonically with temperature; weaker models are *more* diverse (Llama-8B
high, Llama-70B low); tasks ladder baseline > analogy > blending; structural rates discriminate
models sharply (analogy 3/19/81%, antanaclasis blending 1/3/58%). Sets up a **diversity↔validity
trade-off across models**. Raw outputs: `data/kg_creat/responses_rand_v2_stage1/`.

**Stage-2 built, not run.** `make_pass2` extended to derive categorical on arbitrary endpoints
(type baseline interior entities via G_c, drop generic types, most-contrastive → biting). 480 specs
generated, fired + killed on user request.

**Constraint-taxonomy correction (DEFERRED).** The set should be a clean **2×2: inclusion/exclusion ×
relation/entity** (categorical = *inclusion of entity*). Current Stage-2 covers 3/4 quadrants —
**exclusion of entity is missing**; `inclusion_rare` is a redundant inclusion-of-relation. Fix
deferred: drop `inclusion_rare`, rename categorical → inclusion-of-entity, add exclusion-of-entity.

**Next:** analyze Stage-1 analogy/blending results (immediate); then resolve the 2×2 + rebuild/run
Stage 2. Owed: human judge-reliability pass. This OpenRouter key has ~$217 left.

Full session detail: [docs/logs/2026-07-26/1625_kg_creat_diversity_pivot.md](../../logs/2026-07-26/1625_kg_creat_diversity_pivot.md).

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

## 2026-08-22 — Kombine pivot: flat pool, four criteria, first pilot

**State.** The track is now framed as **Kombine**: three tasks (association / analogy / blending),
each scored on four criteria — **utility (factual + judge)**, **surprise** (cosine distance),
**originality** (inverse frequency; *paper-defined, not yet coded*), **emergent creativity**
(count of true inferences licensed by the whole but not any part). Prompts are open-ended
(array-of-objects, each item carries an `inferences` field) and motivate the four criteria
(TRUE/REMOTE/UNCOMMON/GENERATIVE).

**Big change: dropped the seed-BFS knowledge graph.** It was person-biased (biographical/family
relations dominate the top-28 frequency vocab → ~60% humans) and unnecessary (the model connects
from its own knowledge; the graph is never traversed). Replaced by a flat curated domain-balanced
pool: `data/kg_creat/entities_curated.json` + `src/kg_creat/scripts/sample_flat.py` (cross-domain
stratified). No graph / min_degree / relation vocabulary.

**Pilot (curated pool, ~$2.1).** Elicit 4 cheap models × 90 prompts × 3 temps; score 3 (dropped
llama-3.1-8b, 54% parse), batched factuality.
- Factual validity discriminates cleanly (llama-70b > gpt-4o-mini > gemini-lite).
- Combinatorial **scarcity** confirmed (verified-genuine/prompt): association 6.0, analogy 1.5,
  **blending 0.5 (78% of anchors yield zero)**.
- **Emergent creativity flat (~0.5)** across strong models → emergent judge too lenient; not
  discriminating.
- Surprise is item-driven (fixed cross-domain pairs), not a model signal.

**Open / next (incl. user's 2026-08-22 revision prompts):**
1. **Blending underperforms** post-pivot (arbitrary anchors rarely admit a 2nd sense) — rethink how
   to assess it (curated polysemy anchors, or a different blending operationalization).
2. **Models are weak at emergent creativity**, and the metric doesn't discriminate — beef up how we
   *prompt for* and *assess* it (interesting real cases: discovering a **common cause** between two
   things, a shared hidden mechanism, a transferred prediction). Likely: stricter emergent judge +
   prompt that asks for a specific *kind* of emergent inference.
3. Implement the **originality (inverse-frequency)** scorer.
4. Deploy the **human generation** experiment (jsPsych, in `llm_creativity_mech_interp`).

Details: `docs/logs/2026-08-22/1913_kombine_pilot_flat_pool_and_human_study.md`.

### 2026-08-22 (late) — emergent creativity is task-specific; blending → fusion (decisions locked, redesign pending)

Per-task emergent-creativity mechanism, grounded in the Combinatorial Creativity paragraph of the
paper (`04_background.tex`):
- **Association has NO emergent creativity** — it only *reveals a connection*. Score it on utility,
  surprise, originality only (no emergent field, no GENERATIVE elicitation).
- **Analogy: emergent = transferred inferences** — novel true claims about one/both domains via the
  mapping (Gentner's *systematicity principle* → candidate inferences; NOT comp-gen systematicity).
- **Blending: emergent = the emergent structure of the blend** — properties/behaviors the fused
  concept has that neither input has (may be false of each), per the F&T quote.
- **Blending also reframed from lexical polysemy → two-concept FUSION**: given two concepts, fuse into
  a blend (name + `structure` triples from both) with emergent structure. (computer virus = biology +
  software.) Sampler draws blending as a *pair*, like analogy.

**Done in the paper (pushed):** Table 1 (`tab:cg-vs-cc`) split into association/analogy/blending with
A4 = ✗/✓/✓; `tab:scoring` association emergent cell → em-dash.

**Pending redesign (next session):**
- Paper prose: Preliminaries + `tab:scoring` caption (emergent = analogy/blending only); association
  motivation (drop "elicits novel inferences"); analogy motivation (emergent = transferred inference);
  blending prose reframe to two-concept fusion; §2.2 A4 discussion.
- Code (`src/kg_creat`): association prompt drops emergent field; analogy emergent = transferred
  inferences; blending → two-concept fusion (blend + structure) + emergent structure; `sample_flat.py`
  blending → pairs; `parse.py`/`run_elicit.py` new blending format; **task-specific emergent judge**
  (analogy: transferred inference; blending: emergent structure; none for association); `score.py`
  scores emergent only for analogy/blending. (prompts.py currently at the committed uniform-`inferences`
  version — realign to the above.)

### 2026-08-22 (late) — fusion blending implemented + first run

Implemented the fusion reframe end-to-end (prompts, parse_blend, sampler reuses analogy pairs,
fusion + blend-emergent judges, score/aggregate) — see docs/tracks/kg_creat/blending_fusion.md.
First run (temp 0.9, 30 pairs, 3 scored models; data/kg_creat/kombine_blend_v2/):

- **Parse 100%** (vs polysemy's 54–83%); single-object format is robust.
- **Utility (fusion pass) discriminates**: llama-3.3-70b 33% > gemini-flash-lite 23% > gpt-4o-mini 13%.
  The generic-space HARD GATE does all the rejecting, and its reasoning is sharp: ~80% of blends use a
  *conjunction pseudo-schema* ("harnesses energy AND has therapeutic effects" = one property from each
  input stapled together), not a real shared abstraction. Passing blends have genuine shared schemas
  ("a communication system that unifies distant regions through a central authority" = telephone+Rome).
  **Finding: models default to conjunction-mashup blending; true generic-space fusion is rare.**
- **Emergent creativity ~2.2 verified statements/blend** (vs polysemy ~0.5), rich and meaningful; only
  weakly model-discriminating (level, not spread, is the signal). Models reproduced hand-predicted
  emergent structure (llama-70b: antibiotic-resistance for photosynthesis+penicillin).

Two fixes landed while validating:
1. **Blending utility must NOT gate on factuality** — a blend's structure is intentionally false of the
   real world (F&T), so CREATE's factuality judge was killing 88/90 on the `factual` channel. Fixed:
   blending sat = well-formed ∧ fusion-judge (judge-only), factuality skipped for blends.
2. **Per-draw keying bug (pre-existing, all modes)**: judges keyed responses by `prompt_id`, but every
   temperature shares one `prompt_id`, so `{prompt_id: r}` collapsed the draws — emergent ran on only
   1/3 of draws and analogy/blending judged the wrong draw's structure. Fixed with a `_draw_key`
   (prompt_id, temperature, sample_idx) throughout score.py. Needed for the temp ablations.

First benchmark run pinned to **temp 0.9 only** (temp ablations deferred).

### 2026-08-30 — task formalism finalized, scoring pipeline realigned, cost ledger, first multi-model run

Big multi-day session (08-28 → 30). Full log: `docs/logs/2026-08-30/1002_kombine_scoring_pipeline_ledger_and_test30.md`.

**Task formalism (paper, pushed).** Unified analogy/blending under a projection operator `M[·]`:
analogy invents `h := M[Φ]`; blend `c' := M_u[Φ_u] ∪ M_v[Φ_v] ∪ Δ` with two independent selective
projections. **Generic space `g` reintroduced as a textual schema** (not a triple set). Blend base
dimensions all score `g`: utility `U_bl = J^gen` (both inputs instantiate `g`), surprise
`S_bl = ½(d(u,g)+d(v,g))`, originality `O_bl = ρ_g(g)`. Double-scope quality `Q_bl ∈ {1,2,3}`.
**Emergent creativity is kept as SEPARATE dimensions** (originality/coherence/validity-or-scope), never
aggregated. **Originality = pool-relative embedding distance ρ (kNN)** everywhere, replacing
inverse-frequency. Restored `tab:scoring`; minimalized `tab:examples`; judge-prompt appendix (F +
analogy-invention + blend `J^gen`/`J^coh`/`Q_bl`), all user-approved.

**Prompts (`prompts.py`).** Analogy asks for ONE analogy; blend example → cyborg; no mid-sentence line
breaks; reward bullets map 1:1 to scoring dims (blend surprise instruction dropped); brevity rule (short
recognizable entities, no CamelCase/dash coinages); blend structure capped at 4–6 triples.

**Scoring realigned (`judge.py`/`parse.py`/`score.py`/`run_elicit.py`/`dat_eval/llm.py`).** `parse_blend`
reads `{triple, from}` + keeps u/v/emergent tags; `parse_items` keeps analogy invention/projection.
Blend judge → `generic_ok`/`coherent`/`scope` (**must** get the tagged structure); new analogy-invention
judge. Surprise made paper-exact per task; analogy utility = structural (relation-identity) ∧ factual
(dropped the old semantic judge). **Judge explanations + verdicts now persisted** (`blend_judges`,
`invention_judges`). Bugs fixed: `_majority` bool-coerced ordinal `Q_bl` (→ `_majority_val`); reasoning
judge (`gpt-oss-120b`) truncation at low `max_tokens` (→ 3000).

**Cost ledger (`src/kg_creat/cost_ledger.py`, NEW).** Persistent `data/kg_creat/cost_ledger.jsonl`
(gitignored), actual token usage → USD, by phase/model; `python -m src.kg_creat.cost_ledger`. Wired into
elicit + score.

**Run + findings.** 6 models × 30/task (gpt-5-mini, sonnet-4.5, deepseek-chat, gemini-3.7-flash,
llama-3.3-70b, grok-4.6): **100% parse**, blend triples in 4–6, concise entities; scores discriminate
(grok-4.6 tops utility). **Cost $10.53 total** (elicit $10.08 + score $0.45); **grok-4.6 alone $5.43**
(870k reasoning tokens) — reasoning models dominate cost and the flat budget-cap misses it.

**Human study** (sibling repo `llm_creativity_mech_interp/.../kombine_generation/`) redesigned to the new
structure: ID page; per-task intros; association = path of full triples; analogy = a full triple per
domain per row (path_a | path_b) + invention triple below a divider; blend example → cyborg;
"I can't think of one" opt-out.

**Next:** persist factuality-judge explanations; swap single judge → 3-panel; add an actual-cost stop to
`run_elicit` + per-model effort caps before a full ~30-model × 70-item run.

## 2026-08-31 → 09-03 — pool scaled to 21 models, 3-judge panel + human corroboration, the blend `uv` fix, four analyses

Two of the three 08-30 "next" items are done (3-judge frontier panel, actual-cost stop in `run_elicit`) and the run scaled from 6 models to the full pool; factuality-judge explanations are still not persisted. Paper drafting moved into its own Overleaf repo, `papers/kg_creat-iclr/`, targeting **ICLR 2027**. Everything below is uncommitted in the outer repo as of 09-04.

**Pool (`kombine_test30_frontier.yaml`, `kombine_test30_anthropic3.yaml`).** 13 frontier flagships appended to the original 6 (resume-safe, cheap → expensive ordering, actual-cost stop against `budget_usd`), plus the three legacy-priced Anthropic whales, giving **21 models × 30 items/task × 3 tasks at temp 0.9**. mistral-large is elicited but dropped from the homogeneity analysis (sparse). Leaderboard top: gpt-5.6-sol, grok-4.6, opus-4.6, grok-4.5, gemini-3.1-pro.

**Judging (`kombine_test30_panel_score.yaml`).** The subjective verdicts (blend `generic_ok`/`coherent`/`scope`, analogy invention, emergent counts) are now a **3-judge majority panel — Claude Haiku 4.5, GPT-5.4, o3 — none of which is a subject in the pool** (no self-scoring bias); `panel_open_ended: true` panels the analogy invention too. The objective per-triple factuality gate stays a single cheap judge (gpt-oss-120b) to keep bulk spend down. Panel ICC(2,3) is **fair-to-good, 0.48–0.65** (`content/14_human_judge_reliability.tex`, `tab:icc`).

**The blind human pass is no longer owed.** `build_blind_review.py` + `blind_review_server.py` serve a 60-item blind subsample (30 analogy, 30 blend) with model identity and panel verdicts hidden; an author rated all 60 (`data/kg_creat/kombine_test30/human_review/ratings.jsonl`). Human vs panel majority agree **66%** over 150 dimension judgments, rising to **75%** on unanimous-panel items (n=96). Restricting the leaderboard to unanimous items barely moves it (Spearman ρ = 0.945, Pearson r = 0.941, n = 21; ρ = 0.79 within the top eight) — `leaderboard_unanimous.py`, `leaderboard_single_judge.py`.

**Blending `uv` fix + v3 re-elicitation (the big methodological change).** The old blend format made shared-property fusion *unrepresentable*: triples could only be tagged `u`, `v`, or `emergent`, so a property both inputs organize had nowhere to go and the format forced concatenation by construction. The prompt now carries a **`uv` tag — one slot both inputs organize** — and the blend judge scores **scope 1/2/3** (1 = no genuine shared slot, 2 = a real shared slot, 3 = shared slot + emergent structure) with the named `shared_properties` returned. Blending was **re-elicited for all 21 models on the same 30 anchor pairs** (`kombine_test30_blendv3.yaml`, separate `blends_v3/` output, merged back), and the paper's Blending + Overall columns refreshed; association and analogy are unchanged.

**Other scoring changes.** Originality is **split into base (the scored artifact) and emergent (the invention)** for analogy and blending — they correlate only 0.63–0.74 within a task, and collapsing them hid the cross-task result below (`rescore_split_originality.py`). Originality was **rescored pool-relative over all 21 models** when the pool grew (`rescore_originality.py`), as it must be. `rejudge_factuality.py` fixes a real scoring bug: the batched factuality judge (10 paths/call) truncated on models emitting many long association paths, returning None, and those paths were marked `unjudged` and wrongly counted as utility failures — re-running them in batches of 3 repairs the affected models' `path_scores.json`/`summary.json`. Per-judge explanations and raw verdicts are persisted for the blend and invention panels (`blend_judges`, `invention_judges`), not for factuality. `repair_elicit.py` backfills failed draws.

**Four analyses written up (all in `docs/reports/`):**

1. **`2026-08-31_kg_creat_invention_homogeneity` — the artificial hivemind.** Inter-model convergence of each artifact against a cross-item null (18 models × 30 pairs, MiniLM). The analogy *mapping* is the most convergent artifact (0.54 vs 0.16 chance) — it is just facts about u and v. The **convergence ladder across each task's characteristic product**: association bridge 0.21 → analogy invention 0.24 → blend `c'` 0.48 (excess +0.12 → +0.14 → +0.34). Blending homogenizes even on its creative leap; analogy stays divergent. Blends carry a provider **house style** (same-provider 0.52 vs cross 0.47); bridges and analogy inventions do not.
2. **`2026-09-01_kg_creat_inventive_multiples` — models reinvent the same thing.** 21 models, 1,248 inventions, 12,363 co-response pairs. A **structural multiple** (same invention *and* same underlying abstraction: invention cosine ≥ 0.69 ∧ abstraction cosine ≥ 0.50, both thresholds calibrated from lexically-identical pairs) occurs in **2.3%** of pairs; 5.9% at topic level, so most topical convergence is not the same invention. **Blending produces ~7× more structural multiples than analogy** (3.9% vs 0.6%, paired Wilcoxon p = 1.4e-5), and **same-provider pairs 3.2× cross-provider** (5.7% vs 1.8%, permutation p = 5e-4). Clusters reach 17 of 21 models ("Crystalline Imperium"). **Operator asymmetry:** more distant anchors *raise* the blend multiple rate (ρ = +0.41, p = 0.025; 2.5% → 5.4% by distance tercile) but leave analogy flat at a 0.6% floor — blending funnels, analogy fans, a quantitative signature of Fauconnier–Turner integration vs Gentner projection.
3. **`2026-09-02_kg_creat_facet_correlations` — what the dimensions measure.** n = 21 models. **Task, not dimension, is the organizing axis** (analogy is the hub at r ≈ 0.6 with both others; association–blending only 0.41; no cross-task "originality skill"). **Utility trades off against surprise and originality within a task** (r ≈ −0.4 to −0.6) — the novelty–appropriateness tension as a measured property. Only **emergent-to-emergent originality transfers across tasks** (0.46, the one cell reaching p < 0.05; base–base 0.18), and that inventiveness is **orthogonal to capability** (r = −0.13 with the leaderboard) — the most inventive models are mid-tier (gemini-2.5-pro, gemini-3-flash, qwen3-max, glm-4.6). This is the justification for the multidimensional scoring and for the base/emergent split.
4. **`2026-09-03_kg_creat_frontier_failures` + `2026-09-03_kg_creat_blend_integration` — how they fail.** Three tasks, three failure modes: association/analogy break on **factual grounding** (~20% of path triples are hallucinated specific-entity connective facts — an upper bound, single judge), blending on **abstraction** (41% of frontier blends die at the generic-space gate), analogy invention on **fidelity** (~19% relabel the target or import an outside concept instead of projecting the mapping — opus 20–27%, grok 0%). The blend bottleneck is the *whole* story: past the gate, 99% are coherent and 94% are full double-scope. Corpus-wide (630 blends, 21 × 30), **every blend claims a shared slot but only 57% survive verification**; of the 276 scope-1 failures, **94% are a one-sided/unbalanced schema** ("faking the slot") and 11% categorical absurdity. **All 30 anchor pairs contain both a genuine fusion and a fake**, so scope measures the model's fusion skill, not the pair's difficulty; the genuine-fusion rate spans 36–86% (gpt-5.6-sol 86%, qwen3-max 36%) and does not track raw capability.

**Cost.** The ledger now totals **$379.74** — elicit $191.68, elicit-repair $19.72, score $167.99. **Judging is now nearly half the spend**: o3 $88.96 and gpt-5.4 $58.92 as panel judges, ahead of any subject model. The actual-cost stop and per-model reasoning caps (`reasoning.max_tokens: 8000` in blendv3) are in place.

**Paper (`papers/kg_creat-iclr/`).** Full 21-model leaderboard with provider logos; per-task per-dimension appendix tables (% of max); entity-pool appendix (all 283 anchors + Wikidata grounding); judge-reliability + human-corroboration appendix; radar figure replaced by the invention landscape + worked example; "utility gate" → "utility-conditioned" terminology.

**Open / next.** The blend-v3 review items are built but **not yet rated** (`human_review_blendv3/` has `items.json` + `key.json`, no `ratings.jsonl`) — the human corroboration currently covers the pre-`uv` blends. Factuality is still a single judge, so the 20% hallucination rate is an upper bound; spot-checks found false positives. The homogeneity nulls are single-draw (no bootstrap CI) and the house-style gap is untested. The human generation study (`scripts/kg_creat/deploy_study.sh`) has not been fielded against the current structure.

## 2026-09-05 — pool scaled to 30 models, a dimension-dropping bug caught, every downstream artifact regenerated

**Pool (`kombine_test30_spread9.yaml`).** Nine cheaper models elicited and scored — phi-4, gpt-4o-mini, gpt-4.1, glm-4.5-air, llama-4-maverick, deepseek-chat-v3-0324, qwen-2.5-72b, kimi-k2, gemini-2.5-flash — taking the pool to **30 models × 30 items/task × 3 tasks at temp 0.9**. Scoring cost **$41.49** against a ~$72 estimate (o3 $23.05, gpt-5.4 $14.73, haiku-4.5 $3.47, factuality $0.24). The frontier subset is unchanged at 15, so every frontier-only number in the failure reports is untouched. Leaderboard top five unchanged: gpt-5.6-sol, grok-4.6, opus-4.6, grok-4.5, gemini-3.1-pro; the new models land 21st–30th except gemini-2.5-flash (18th).

**Two pool-relative scoring bugs, both fixed.** Originality is pool-relative, but `score.py`'s resume builds the element pool over *all* models and then skips any model that already has a `summary.json` — so the 21 old models kept originality measured against a 21-model pool while the 9 new ones got the 30-model pool, and the two were not comparable. `rescore_originality.py` (new, judge-free, no API cost) recomputes it for every model at once; the dry run confirmed the diagnosis exactly, with **every new model shifting 0.0000 and every old model dropping 0.016–0.033**. Second and worse: `em_originality` is written by a *separate* script (`rescore_split_originality.py`) that had never been run on the new models, so `compute_composite.py` silently **dropped `analogy.em_originality` and `blending.em_originality`** from the composite — a leaderboard on 4 dimensions where the previous one had 6. Running the split script pool-wide restored them (`skipped_dims` is now empty). **After both fixes the relative order of all 21 previously-scored models is identical — max rank move 0.**

**Provider-map drift, fixed at the root.** `microsoft` and `moonshotai` were absent from the provider/brand/display maps, which were **duplicated across five scripts**. The copies had diverged, so new providers were drawn grey and unlabelled in some figures and crashed `analyze_inventive_multiples` outright (`sorted()` over a provider set containing `None`) — which meant that analysis had been *silently failing* and the JSON on disk was stale from the 21-model pool. `plot_radar` is now the single source for `BRAND`/`DISPLAY`/`LOGO_SLUG`/`_provider`, `plot_profiles` imports rather than copies, `_provider` falls back to the model key's own prefix so a missing logo no longer greys a provider, and `plot_invention_landscape` raises if a provider in the data has no legend entry. No logo asset exists for Microsoft or Moonshot; they get a brand colour and no mark.

**What changed in the numbers (n = 21 → 30).**

- **Inventive multiples.** 1,773 inventions over 25,321 co-response pairs. Structural rate **1.6%**; **24%** of inventions are reinvented by at least one other model (44% of blends, 5% of analogy inventions); 109 clusters, largest **28 of 30 models** (Opera + Documentary film). Blending **12×** analogy (3.0% vs 0.2%, p = 1.7e-6); same-provider **2.6×** cross-provider (3.5% vs 1.4%, p = 5e-4). The `--prepost` comparison is now **restricted to the 21 models that have a pre-v3 backup**, so the format change is no longer confounded with the pool change: structural 6.3% → 3.1%, clusters 52 → 65.
- **Facet correlations.** The associative-hypothesis gap **strengthened**: association↔analogy +0.74 [0.52, 0.87] vs association↔blending +0.44 [0.10, 0.69], Williams' t = 3.17, p = .004 (was t = 2.24, p = .038). Association↔blending utility became significant (+0.21 n.s. → +0.40, p = .028), so the utility chain's endpoints are weakly linked rather than independent.
- **One claim reversed.** Emergent originality vs the overall composite went from **+0.13 (n.s.) to +0.47 (p = .008)** — a restricted-range effect, not a data error: the nine weaker models widened the spread of capability. The facet report now carries this as a worked caution that its coefficients are pool-conditional.
- **Homogeneity.** Convergence ladder holds and steepens slightly: bridge 0.22 → analogy invention 0.25 → blend `c'` 0.46 (excess +0.14 → +0.16 → +0.32).
- **Blend integration.** 885 scored blends, **52% survive scope verification** (was 57% on 630). The failure-theme split is now **62% one-sided / 45% categorical absurdity**, which is *not* comparable to the previously reported 94% / 11% — that came from an unrecorded one-off scan, and the patterns are now written down in the new `analyze_blend_integration.py`.
- **Frontier failures: unchanged**, as expected — same 15 models. Generic-space gate 46.7%, invention incoherence 20.2%, mapping-not-applied 8.5%.

**New: analogy/blending dissociate at the item level (`analyze_task_dissociation.py`).** Analogy and blending run on the same 30 pairs, so every model gives a matched attempt on every item — 873 cells. **45% disagree**: 235 cells where a model built a working invention on a pair it then failed to abstract over, 161 the other way. φ = +0.086, and **+0.136 after disattenuating** for panel reliability (estimated from the individual judge votes: mean pairwise judge r = 0.30 and 0.44, panel reliabilities 0.57 and 0.70). Yet *aggregated over the 30 pairs* the two rates correlate at **r = +0.64** across models. The competence is real at the model level and does not transfer to any particular pair — the jaggedness is at the item level, and model-level scores average it away.

**Regenerated downstream.** Paper: `02_leaderboard.tex`, `03_per_task_full.tex`, `inventive_multiples.png` (now stacked reproducibly by the new `make_paper_multiples_figure.py` instead of by hand), `profiles_grid`, `radar_profiles`, `bar_profiles`, and Findings #2 + #3 in `06_results.tex` — #3 also finally moved off *conditional* means onto the **ungated** means the analysis has used since 09-05, and off n = 21. Reports: all five kg_creat reports updated; the inventive-multiples report was **rewritten**, since it still documented the pre-redefinition cosine criterion rather than the name-free property criterion in force.

**Open / next.** `analyze_facet_correlations.py` no longer hardcodes its significance threshold, but several reports still quote coefficients that are pool-conditional — worth a standing note. The dropped-dimension bug argues for a check in `compute_composite.py` that refuses to silently skip a dimension present in a previous run. Blend-v3 human review items are still unrated. The nine new models are not in the human-corroboration subsample.

## 2026-09-07 — pool 35, the factuality judge replaced pool-wide, and a thinking-effort study

**Pool 30 → 35.** Four Anthropic models plus `gpt-6-astra-flex` as its own entry. Elicitation moved off OpenRouter onto the user's own LiteLLM and Anthropic keys via a new `src/kg_creat/providers.py`, which deliberately bypasses `LLM_BASE_URL` so the OpenRouter budget guards stay intact; OpenRouter was used for **scoring only**. The gateway needs a top-level `reasoning_effort` (not OpenRouter's `extra_body.reasoning`) and `allowed_openai_params` for models it has not registered as reasoning-capable; the Anthropic route must stream, cannot take `temperature` on SDK 1.4, and rejects `thinking.type.enabled` on the newest models. `run_elicit.py` now raises rather than writing an empty result set when every draw fails at the API.

**The factuality judge was silently failing, and it is now `claude-haiku-4.5`.** `gpt-oss-120b` returned no parsable verdict on a large fraction of paths; `score.py` recorded these as `channel="unjudged"` and the aggregate counted them as utility failures, deflating association and analogy. `rejudge_factuality_haiku.py` (new) re-judges every path and preserves the original verdict as `factual_gptoss`/`channel_gptoss` through `setdefault`, so re-running is idempotent and never destroys the original. Main pool: **2,039 unjudged → 0**, and the hallucination rate rose from the reported 21% to **27.1%** — the old figure was deflated by unscored paths, not by better models. User decision: haiku everywhere, "for factuality bigger is better."

**The same bug was in the effort study, and there it scaled with the independent variable** — unjudged 27.3/41.0/47.0% for sol at low/medium/high and 76.9/89.1/86.5% for astra. Uncorrected it would have produced a clean, entirely spurious "effort hurts association" result. The re-judge cost **$6.36** (2,433 calls, 382 s at concurrency 32) and left ~1% unjudged.

**(#1a) rigorously retested.** Macro means over 35 models: analogy `U_an` **54.0%**, blending `U_bl` **43.8%**. Over the 1020 matched model×item cells the gap is **+10.0 pts** (54.3 vs 44.3; McNemar exact **p = 2.9e-6**, OR **1.55** [1.29, 1.88]) — down from the 14 points reported at n = 30 with the old judge. **Item difficulty is controlled**: all 30 pairs had at least one model produce both a valid analogy and a valid blend, so nothing is dropped as impossible. The advantage is concentrated on hard items — hardest tercile **+27.0** (p = 3.9e-13), middle **+15.5**, easiest **−12.2**, where blending is *ahead* — and per-item difficulty is essentially **uncorrelated across the two tasks** (r = +0.14, n = 30). `test_utility_analogy_vs_blending.py` (new) scripts all of it. **(#1b)** generic space found **43.8%** of the time, coherent blend **97.1%** conditional on it.

**Thinking-effort study: more thinking buys more output, not better output.** `gpt-5.6-sol` and `gpt-6-astra-flex` × {low, medium, high}, scored in **one pooled pass** so pool-relative originality stays comparable across levels. Overall composite is flat — paired high−low **−0.43** [−6.2, +5.4] for sol and **−0.38** [−8.1, +7.0] for astra. Of 38 effort contrasts only three exclude zero, all on sol's association, and they are *one* effect rather than three: surprise and originality are utility-gated, so a utility drop drags both down mechanically. That drop is itself a **path-length artifact** — effort lengthens chains (4.81 → 5.52 hops) and a path is factual only if every triple is, while per-triple factuality stays near-flat (93.9 → 92.2%); `p_triple ^ mean_hops` predicts observed path utility within 1–3 pts in all six configs. The **manipulation check passes**: mean reasoning tokens rise 7.3× for sol (1,786 → 13,076) and 14.6× for astra (1,011 → 14,779), so the models genuinely thought an order of magnitude longer and scored the same. Reasoning trace **text is unavailable** — the LiteLLM gateway strips it (`merge_reasoning_content_in_choices: false`) and returns counts only. New: `analyze_effort_study.py`, `plot_effort_composite.py` (three camera-ready figures + `effort_composite.json`).

**Bugs fixed.** `compute_composite.py` gained the dropped-dimension guard argued for on 09-05; it fired for real on the effort study, where `rescore_split_originality` had never been run, so the composite was quietly running on 5 of 6 dimensions. The effort bootstrap was resampling the union of anchor pairs, but **association is posed over a different 30 pairs than analogy/blending** (union 60), letting each task's item count drift — now stratified within task and paired across effort levels. Per-panel autoscaling in the effort figures was exaggerating 2-point ranges into apparent collapses; all composite panels now share one y-scale. `model_names.py` (the 09-05 consolidation, now its own module) had `BRAND` defined twice and `DISPLAY` three times — byte-identical, so nothing misbehaved, but an edit to the first block would have been silently discarded; deduped.

**Open / next.** The paper's method section justified `gpt-oss-120b` by citing reliability evidence from `wadhwa2026create`; that citation does not transfer to haiku, so it was removed rather than re-attributed, which **leaves the judge choice unjustified** and needs a sentence pointing at our own Table 9. `content/15_embedding_robustness.tex` remains in `stash@{0}` with n = 30-era rank-stability numbers. The `#1a`/`#1b` paper edits are staged locally and unpushed. 81 effort-study paths remain unparsed, and a few high-effort API calls failed outright (sol 88/90, astra 83/90).

## 2026-09-09 — the Kombine tasks load on divergent thinking; τ-inventive multiples rebuilt

**Headline: DAT is the strongest external correlate of Kombine performance.** Correlating each Kombine task and facet against four reference tests over the 35-model pool (n = 31–35 per cell after a validity filter): overall composite vs **DAT +0.66**, RAT +0.39, CDAT +0.31, DRAT −0.21. Per task, blending loads on DAT most heavily (**+0.63**), analogy +0.60, association +0.49 — the gradient the task design predicts, with association closest to remote-associates retrieval and blending furthest from it. **Twelve facet cells survive Benjamini-Hochberg** at q=0.05 over 60 tests, where at n=21 none did; nine of the twelve are DAT. New `analyze_cognitive_facets.py`.

**Utility and originality dissociate on the other two tests.** Utility facets load on RAT (+0.38 to +0.48) and not CDAT; originality facets load on CDAT (+0.39 to +0.59) and go *negative* on RAT (`blend.originality` −0.34). Convergent thinking predicts whether a model clears the task gate; constrained divergent thinking predicts whether what it produces is novel.

**Three measurement bugs, each of which had inverted a result.** (1) **Utility gating destroyed the facet structure** — `compute_composite` zeroes surprise/originality on artifacts failing utility, making within-task facets near-identical (mean pairwise r **+0.95 to +0.97**), so every facet merely re-expressed the utility rate; computing them ungated dropped that to −0.05/+0.35/+0.25 and the dissociation appeared. (2) **Temperature was not fixed** — DAT/CDAT pooled 1.0/1.5/2.0, RAT is 0.0, Kombine 0.9, and stored CDAT averaged only the temperatures each model *passed the gate at* (49 models passed all three, 6 only 1.0), so models were not compared at a common operating point. Fixing both at **temp 1.0** erased a "clean CDAT dissociation" that was entirely this artifact, raised CDAT coverage 18→28, and cut DAT–RAT collinearity from **+0.73 to +0.40**. (3) **The 256-token cap truncated reasoning models** in DAT, CDAT, DRAT *and* RAT: `claude-opus-5` CDAT went **2% → 100%** valid at 4000, `claude-sonnet-4-6` 0% → 100%. Both models the CDAT gate had "failed" were truncation artifacts.

**DRAT does not measure what its name says, and should probably be dropped.** `score_drat` computes each word's utility as **`max`** cosine over the anchors, not min — a word survives by being near **any one** anchor — and the score is then mean pairwise distance among survivors. So it is a relevance-gated DAT, not a joint divergent-convergent test. Its variance is dominated by the item: τ (the 90th percentile of a random-noun null under the same max rule) rises with how semantically *broad* the anchor set is, spans **0.246–0.486** over 194 groups, and correlates **−0.73** with survivor count (p=9e-34). With `n_min: 5` the per-trial score is near-binary (0 or ~76) and **76% of all cells score 0**. That explains the nano models topping the pool, the −0.64 correlation with CDAT, and the negative loadings on every Kombine facet.

**Backfill across three routes, $27.87 on OpenRouter.** Coverage DAT 21→33, CDAT 18→31, RAT 29→35, DRAT 27→33. Anthropic's OpenAI-compat endpoint serves all 10 Claude models and the user's LiteLLM gateway serves the two GPT ones, so OpenRouter was needed for only 13. `strip_unsupported()` in `dat_eval/llm.py` keys off the *endpoint*: Anthropic rejects `temperature` on newer models; `extra_body` is an OpenRouter extension; newer OpenAI models need `max_completion_tokens` and reject temperature ≠ 1.0; and the gateway's `gpt-6-astra-flex` deployment **leaks its own config keys** (`mode`, `max_input_tokens`, `max_output_tokens`) into the upstream request, which is suppressed by nulling them client-side (worth fixing at the source).

**Provider data quality is a real constraint.** `glm-4.5-air` and `glm-4.6` return **null content on half or more of calls**, and with `max_retries: 4` each failure cost 5× the work; `qwen-2.5-72b` kept its older `run_v1` data because the new collection was worse (69% vs 41%). `grok-4.5` cost ~$18 and 5 hours for 4 of 6 files then failed the validity filter anyway. `analyze_cognitive_facets` now excludes any model below **70%** of trials parsed. Missing from DAT/CDAT: grok-4.5/4.6, glm-4.5-air/4.6, qwen3-max — a skew toward US labs that matters for generality more than power.

**τ-inventive multiples rebuilt (09-08).** τ is now the reported axis rather than a hidden constant (`tau_curve` gives the rate at every τ); the abstraction clause is **removed**, matching the paper's Definition — property overlap and nothing else; θ is **calibrated against a cross-item null** (`calibrate_theta.py`) at **0.545**, the null's 99.75th percentile (α = 0.25%), replacing an underived 0.58. Three independent routes — size-normalised overlap, null-corrected similarity, exact-match per opportunity — all put blending's advantage over analogy at **~2×**, against the paper's 11×; blends carry 5.01 properties to analogy's 2.60, so a fixed τ is a 40% bar for blending and 77% for analogy. **Anchor echo** is a live confound: 67.5% of analogy object-collisions are the anchor entities themselves, and unfiltered this *inverts* #3b. Exploratory: within-item invention similarity has **no cluster structure** (first eigenvector explains 0.08–0.09 of positive variance), so multiples are not attractors — property support is the better unit (12.0% of blending property clusters asserted by ≥2 models vs 1.3% by chance).

**Open.** RAT scores for `gpt-5.6-sol` (43.3%) and `gpt-6-astra-flex` (96.7%) are **artifacts of RAT's 32-token cap** — both were correct on every question they actually answered (13/13 and 29/29), with the rest empty; needs a ~60-call re-run. DRAT needs dropping or rescoring at fixed τ. DAT has little spread at the top (30 of 35 models between 82 and 92), so the correlation may rest on a few low outliers. Paper `#3a/#3b/#3c` still carry pre-τ numbers (27.4%, 11×, 3.5%).

## 2026-09-09 (pm) — inventive multiples finished: anchor echo, fair operator comparison, paper #3 rewritten

**Headline: the operator effect is 2–6×, not 11× and not 36×.** The 09-08 rebuild left three things in a scratch session that never reached the code — the anchor-echo confound, the property-count asymmetry, and the "three routes" comparison. All three are now in `analyze_inventive_multiples.py` and every number in the paper, the report, and the showcase comes from its JSON.

**Anchor echo is now part of the criterion.** A property whose object *is* an anchor (`(X, resembles, Pi)` on *(Don Quixote, Pi)*) is dropped before matching — exact match on the normalised object, not word overlap. It is **8.7%** of analogy properties (234/2,692) and 0.3% of blend properties; removing it cuts the analogy τ=2 rate **0.82% → 0.32%** and leaves blending at 11.4%. So the fixed-τ blend/analogy ratio *rose* from 14× to **36×** — the opposite of the ~2× the 09-08 log expected, because that log's routes were computed before the filter. θ re-calibrated after the filter stays at **0.545** (α = 0.24%).

**A fixed τ is not a fair operator comparison, and the JSON now says so.** Blends carry 5.0 properties and analogy inventions 2.4 after the filter (21% of analogy inventions carry one and can never qualify). `task_routes` reports blend/analogy by five routes: fixed τ **36.2×**; eligible pairs 22.7×; per-property re-use (shared / min(k,k′)) **3.2×** (paired Wilcoxon p = 3.7e-9); exact-match per property, encoder-free, **2.0×** (p = 1.5e-3); size-matched strata at min(k,k′) = 3 and 4, 6.2× and 4.5×. A cross-item null (30,000 same-task pairs on different anchors) is 0.02%/0.00% at τ=2, so null correction changes nothing. Paper `#3b` now states 36× at fixed τ and 2–6× property-count-free, with a new routes table (`media/07_task_routes.tex`).

**Clusters are chains.** At τ=2 eight blending items pull ≥32 of 35 models into one component and *(Photosynthesis, Bread)* pulls all 35 — but within-cluster pair density is **0.13–0.35** for those and 0.57 on average, so a component is pairwise overlap chained, not 35 models asserting the same thing. `clusters_have_outsiders` is now computed (False) and each cluster carries `density`; `#3a` and the figure caption say "component".

**Other numbers, all regenerated.** 46.1% of inventions in ≥1 τ=2 multiple (85% of blends, 7% of analogy), 16.6% at τ=3; same-name pairs 21% multiples; provider RR **2.0×** (p = 5e-4), holding within each task and per property (1.34×, p = 5e-4); pre/post `uv` under the current criterion 23.4% → 12.0% with names unchanged. Abstract updated (27% → 46%, 3.5× → 2×). Report rewritten; showcase fixed (it read `k_shared`/`tau_slot` keys that no longer existed and was crashing).

**Open.** The paper cannot be compiled locally (`algorithm.sty` missing, `tlmgr` needs a self-update); the #3 section and both tables were checked in a standalone build with the ICLR style. The residual anchor-echo question — paraphrases of the anchor that do not name it — is unmeasured and would only inflate analogy. The paper's `#3` prose is machine-authored and sits inside `\ai{}` for review.

## 2026-09-09 (later) — θ raised to 0.674 after inspecting what a match means

**Sampling matched property pairs by cosine band showed that θ = 0.545 admitted shared-noun matches.** Above ~0.65 a match is a paraphrase (`transforms atomic nuclei` / `splits atomic nuclei` 0.70; `cushions sleeper` / `conforms to sleepers` 0.674). In 0.50–0.60 the encoder is anchored on one noun (`includes innings` / `rotates formation with innings` 0.57; `has a grammar` / `legislates by revising grammar rules` 0.55) and still misses real paraphrases (`cushions with blubber padding` / `insulated by blubber` 0.53). Decision (user): **θ = 0.674**, the bottom of the paraphrase band. θ is now a documented judgment, not a null-derived number; the null reports the implied α = **0.03%** (one property pair in ~3,000).

**Every number moved down; every direction held.** At τ = 2: 1.13% of pairs (393); **20%** of inventions in ≥1 multiple (37% of blends, 2% of analogy); τ = 3 leaves 30 pairs. Blend/analogy **32×** at fixed τ (2.21% vs 0.07%, p = 1.7e-6), **3.4×** per property (p = 3.5e-8), **2.0×** exact-match (p = 1.5e-3); size-matched strata are now too thin to read (3 analogy hits at min(k,k′) = 4). Provider RR **2.5×** (p = 5e-4), 1.6× per property. Same-name pairs: only **6%** are multiples. Clusters: 111, largest 19 (Opera + Documentary film, density 0.42), mean density 0.78. Pre/post `uv`: 6.3% → 2.7%, names unchanged. Abstract now says one in five, 2.5×.

**Known softness in the primitive, recorded in the report.** σ is a greedy one-to-one matching, a lower bound on the Definition's "there exist S, injective μ"; with k ≤ 7 an exact maximum matching is cheap and should replace it. Relation and object are embedded jointly as one string, so there is no separate relation-agreement bar.

**Names and properties dissociate both ways (09-09, user-flagged).** Crossing the held-out name with the criterion over 34,687 pairs: 68% of the 753 same-name pairs share *no* property (30% have no property pair above cosine 0.5), and only 5.6% are multiples vs 1.0% of other pairs — the name is a weak predictor. In the other direction 89% of the 393 multiples carry different names (26 of 30 at τ = 3), and the 111 components' 409 members carry 346 distinct names. Now `name_property_dissociation` in the JSON (with examples), paper `#3d`, and a report section. The same-name-different-invention cases are mostly analogy portmanteaus the anchors nearly dictate (`fission tomography`, `Ethical adjuvant`).

**σ is now an exact maximum matching; figures redone for the paper's switchable set (09-09, later).** `shared_properties` uses `scipy.sparse.csgraph.maximum_bipartite_matching` on the θ-threshold graph, matching the paper's Definition ("admit a pairing of size ≥ τ"); the greedy assignment undercounted 2 of 34,687 pairs. τ=2 multiples 393 → **395** (analogy 12 → 14), blend/analogy fixed-τ 32× → **27×**, per-property 3.3×, 112 clusters. Overleaf now switches figure sets (`\paperfigure`, `figureversion`); `plot_invention_landscape.py` also emits each panel alone and `make_paper_multiples_figure.py` writes `media/figures/old/inventive_multiples.png` plus the four `media/figures/new/` PDFs (matrix, mds_a, mds_b, three-page combined), all on the θ = 0.674 data and the (Opera, Documentary film) item. Definition rewritten in plain words (a *pairing* in which no property is used twice) and pushed to Overleaf with Table 5; the figures and #3 prose are local.

**Anchor distance redone (09-09, later): still a null.** New `analyze_anchor_distance.py` reads the analysis JSON's new `per_item` block and tests four distance measures (label cosine, label+Wikidata-description cosine, min/mean log sitelinks as the prominence confound) against four outcomes per task (τ=2 rate, τ=1 rate, per-property re-use, largest component), n = 30 anchor pairs, permutation p, LOO, partial ρ given prominence. The KG geodesic was tried and dropped (archived graph connects 6/30 pairs). Headline rate: blending ρ = +0.34 (p = 0.066) on label cosine, +0.13 on description cosine; analogy flat. One cell is nominally strong — blending largest component vs label cosine, ρ = +0.53, p = 0.003, LOO 30/30 — but it is 1 of 32 screened, fails on the description distance (+0.09), and the two distances agree only at +0.31, so it is treated as a property of the label embedding. Not a finding; report section replaced with the full table and a figure.

## 2026-09-11 — human study wording finalized to the paper's task definitions

The Kombine human generation study (source `llm_creativity_mech_interp/src/experiments/kombine_generation/`, served at https://schapiro.ai/kombine/ via `scripts/kg_creat/deploy_study.sh`) had task intros and per-item prompts written before the analogy and blending redesigns: the analogy intro still described "name the relationship and what each maps to", and the blending prompt never named the generic space or asked for a property both inputs feed. Rewritten to the paper's §3 definitions: analogy = two paths of triples sharing the relation at each step, entities corresponding row by row, then an invention row (a triple true on one side carried across with the corresponding entities); blending = name the new concept, name the generic space stated specifically (the vacuous-schema exclusion is in the prompt), project triples from each input into the new concept with at least one property both feed, then emergent triples true of neither input. The blending worked example now mirrors `tab:examples` exactly (four projections including the two `allocates` rows that land on the same property, plus the emergent triple). "The blend" became "the new concept" throughout the blending UI. Association wording (user-authored) unchanged. Rendered headlessly and deployed.

**Open.** The blending UI has no explicit `uv` tag: double scope is expressed as two rows (one from each input) landing on the same blend triple, so the human→model converter must merge those into `uv` before the integration-quality judge sees them. The consent form still says the study is about how people *evaluate* creative ideas. Data sink is still unwired (`DATA_SUBMISSION_URL=''`, debug mode).

## 2026-09-10 → 09-11 — paper floats rebuilt around concrete multiples; retest30 item set drawn

**Paper (Overleaf, pushed as scoped patches).** The τ table is now a float-less fragment with the author's caption, columns renamed (Multiples: Count / % of pairs / % of inventions; % of pairs by task; % of pairs by model family; the eligible column dropped, still in `07_task_routes`). Under it sit **two side-by-side LaTeX tables** of concrete multiples (`make_multiples_examples_table.py`, from the JSON's `multiples` block): (a) Opera + Documentary film, Fable 5.1 × Gemini 3.7 Flash, τ = 3; (b) Democracy :: Banking, Fable 5 × Fable 5.1, τ = 2 — properties indexed p_i / p'_j in each invention's own order, matched rows shaded green by cosine. The inventive-multiples figure is gone; in its place under the leaderboard is the **generic-space failures grid transposed** (`plot_abstraction_failure.py --by-item`): the 19 hardest anchor pairs as rows, models as columns with logos, red-filled (#EA8485) cells. Two Rice + Radio cells are blank (Opus 4.7/4.8 returned null content). The author block is left-aligned with 15pt rows. **Rules adopted after two incidents:** generators emit no captions or labels (captions are the author's), and Overleaf pushes are hunk patches on the fresh head, never whole-file copies.

**Overleaf repo consolidated (09-11).** `main.tex` only inputs `setup/{imports,configurations}` and `sections/00_title` … `18_app_judge_prompts` (the reproducibility statement is now `08_`, acknowledgements `09_`, contributions `10_`); every figure is a fragment in `media/fig_NN_*.tex` with its image in `media/figures/NN_*` numbered in reading order (01 overview, 02 cc_overview, 03 generalization_taxonomy, 04 abstraction_failure, 05 facet_corr); tables stay `media/tab_*.tex`. The `\paperfigure`/`figureversion` switch, `setup/figures.tex`, `FIGURES.md`, `archive/`, `media/figures/{old,new}` and unused packages are gone. Style files are now `setup/iclr2027_conference.{sty,bst}` from the ICLR Master-Template (only the year differs from 2026; the left-aligned 15pt author block carried over). Full `latexmk` compile: 0 errors, 0 undefined, 29 pages. Report figures that feed the paper: `fig_abstraction_failure_by_item.pdf` → `04_abstraction_failure.pdf`, `fig_facet_corr_reduced.png` → `05_facet_corr.png` (copied by hand; the generators keep writing to their report folders).


**Findings #3, worked through in chat.** Convergence is uncorrelated with capability (per-model blend convergence vs blending composite ρ = +0.14, p = 0.41; most convergent Fable 5, Opus 4.8/4.7, DeepSeek-R1, Grok 4.5; least GPT-6 Astra, GPT-4o-mini, Gemini 2.5 Pro). The same-family effect is **provider-uneven**: within-family τ=2 rates DeepSeek 14%, xAI 13%, Meta 10%, Anthropic 5.5% vs OpenAI 1.1% and Google 2.0% against 1.8% cross-family; Anthropic supplies 45 of 83 same-family pairs. Invention-level versions of the table's (b)/(c): by task 37.3% vs 2.6% of inventions (14×); by family the naive invention share inverts (9.3% vs 15.1%) because models have ~5 same-family vs ~29 cross-family partners, and per partner it is 3.6× — the pair rate is the opportunity-corrected quantity. `model_pair_matrix` heatmap added (`plot_pair_matrix.py`).

**Retest30 (prepared, NOT run).** `sample_flat.py` gained `shared_pairs`, `unique_entities`, `exclude_prompts`; `configs/kg_creat/kombine_retest30_sample.yaml` drew 30 cross-domain pairs disjoint from test30 at the entity level, every anchor once, the same 30 for association/analogy/blending (`data/kg_creat/kombine_retest30/prompts/`). Plan: 7 Anthropic models direct (≈$80), gpt-5.6-sol + gpt-6-astra-flex via the gateway, gpt-5-mini / gpt-4.1 / gpt-4o-mini via OpenRouter (≈$3), same judge panel (≈$100); test–retest r and ICC per model/task/facet over the two sets, rescoring test30 originality over the same 12 models for comparability; the multiples rates replicated on new items. Same items can serve the human study (35 responses per item ⇒ 210 usable humans, ~250–280 recruits).

**Other.** `twistbench-code` (private GitHub repo) holds the plot_twist track for Sumuk Shashidhar. Anchor-distance study is a null on four measures (report section + `analyze_anchor_distance.py`).

## 2026-09-12 — human study platform finalized with Tom and Akshay's edits; live at schapiro.ai/kombine

Source `llm_creativity_mech_interp/src/experiments/kombine_generation/` (commits 5d163b8..8a3e4cf); pushed to the site by the user. Wording now follows the Google doc's "Study Items (REVISED)" tab plus the user's later edits; "generic space" is never shown to participants (it is "the abstract structure that both concepts share"; data key unchanged). Worked examples: analogy is a real high-originality LLM invention (claude-fable-5, The blue whale :: The mattress, vacuum cleaner → whale groomer drone, unanimous judges, emergent originality 0.58; "epidermis" rendered as "skin", "ticking" as "sheets"); blending is a trimmed Liquid Franchise (one Both row, one Banking row, one new link). Invented concept shaded gold everywhere.

**Form mechanics.** Progressive reveal (analogy invention after the mapping rows; blend "what is new" after name, structure and link rows); no invention step when every analogy row is skipped; opt-outs for analogy rows, invention, whole blend, emergent. Relation typed on one side mirrors to the other in real time (analogy rows, blend rows). Analogy invention rows carry a required direction arrow; the invented side is shaded and `projection[i].direction` is recorded with `source` = true side. Blend rows carry required source pills Democracy / Banking / Both; a Both row stacks one source link per concept into one link of the new concept (`from: "uv"`, `source_u`/`source_v`), which is the paper's double-scope property. Bonus banner on each overview.

**Items.** 5 association, 4 analogy + Life :: A journey first, 4 blending + Breakfast + Lunch first (15 total). Dropped Antibiotic resistance :: A spam filter and Democracy + Banking (the worked example). The easy items are flagged `is_control`; random placement was built and then replaced by the warm-up ordering at the user's request, so position is fixed.

**LLM-use detection.** Paste/drop blocked on answer fields (`paste_attempts` recorded). Two visually hidden traps per item, phrased as plain task requirements ("One of the entities in your answer must be the word {word}"), a different word per item, recorded as `trap_word`; the honeypot field exported as `honeypot`. Lesson: Chrome's Ask Gemini ignored traps addressed to "AI assistants" and traps hidden with aria-hidden, and followed them once they were in innerText and phrased as task text. Per-trial typing summary (keys, backspaces, chars, first-key latency, active time, median inter-key interval, long pauses) exported for screening retyped answers. Debug bar (skip page / skip rest of task) when no PROLIFIC_PID.

**Open.** Data sink still `DATA_SUBMISSION_URL=''`. Consent still says the study is about how people *evaluate* ideas. Human→model converter must map `uv` rows and the analogy `direction` field. Analysis-time screen against consumer-model answers on the same items not yet built. Deploy pushes need the user's terminal: this shell has no GitHub credential, gh, or SSH key.

## 2026-09-12 — Overleaf build fixed and figures numbered; ICLR 2027 style; showcase candidates

**Paper repo.** Build fixes pushed (`(#3b)` heading closed, `sec:cg_vs_cc` reference, reproducibility statement input as `08_`, sections 08–10 renumbered); figure assets and fragments numbered in reading order (`media/figures/01_overview.png` … `05_facet_corr.png`, `media/fig_01_*.tex` … `fig_05_*.tex`). Style files replaced by `setup/iclr2027_conference.{sty,bst}` (upstream differs from 2026 only in the header year; the left-aligned author block carried over). The running header had been empty on every page: fancyhdr ≥ 4 makes `\lhead` local and the stock ICLR sty sets it inside `\@maketitle`'s `\vbox`; the sty now sets it `\AtBeginDocument`. New `\iclrpreprintcopy` mode (author names shown, header "Preprint") is active in `setup/imports.tex`. Full local compile: 0 errors, 0 undefined, 29 pages.

**Showcase candidates.** `compile_showcase_inventions.py` collects the inventions every panel judge passed (blend: generic space valid, coherent, scope 3; analogy: valid, coherent, path fully factual): 105/1,033 blends (31 models, 24 pairs) and 286/1,037 analogy inventions (35 models, 30 pairs); listing in `scratch/showcase_inventions/candidates.md` ranked by originality (one per pair, ≤ 2 per model), mocks of a card gallery, a chip table and an originality × surprise scatter in the same folder. The unanimous set is **not** more original or surprising than the rest (medians equal), so a scatter would undercut the showcase; cards or a table are the layouts to choose between. A plainness ranking (wordfreq Zipf) was tried and reverted at the user's request.

**Open.** Showcase picks and layout; results-section organisation (author's); retest30 go-ahead; the two blank Rice + Radio cells.
