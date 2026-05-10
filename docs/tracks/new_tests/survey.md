# Survey: new creativity tests for LLMs (beyond semantic distance)

A comprehensive survey of recent creativity tests / benchmarks for
LLMs that go beyond classical semantic-distance instruments
(DAT, CDAT, PACE, forward-flow). Compiled to inform the `new_tests`
track design space, with particular emphasis on **scientific
ideation** — the construct on which `dat_eval` showed every existing
semantic-distance test is non-predictive.

Throughout, "validity" and "specificity" carry the [dat_eval](../dat_eval/progress.md)
sense: validity = raw Pearson r with an external benchmark;
specificity = semi-partial Pearson r | g with g = (Arena Overall,
MMLU-Pro). The [validity-specificity frontier](../../../papers/iccc-2026/sections/04_method.tex)
gives the per-benchmark theoretical ceiling on attainable specificity
as a function of validity and the benchmark's R^2 on g.

---

## 1. Why we need tests beyond semantic distance

The `dat_eval` paper makes three points that motivate this track.

1. **Construct-bound effectiveness.** DAT is the best predictor of
   creative writing; CDAT is the best predictor of divergent
   thinking; *no* semantic-distance test reliably predicts scientific
   ideation (all r ∈ [−0.11, +0.20] on LiveIdeaBench, no specificity
   cell reaches p < 0.05).
2. **Capability bleed.** PACE, despite r ≈ 0.74 on creative writing
   benchmarks, has near-zero specificity — it tracks general
   capability, not creativity-specific variance.
3. **Headroom.** Even where validity is decent, observed tests sit
   well below the validity-specificity frontier on every panel of
   the headline figure. There is design room for new tests.

Cognitively, semantic distance only probes one primitive — distant
association — out of many that creativity research implicates:
constrained search, look-ahead planning, conceptual combination,
analogy formation, axiom modification, exploration/exploitation
trade-off. A test that targets one of these directly is more likely
to recover variance that capability proxies do not already capture.

---

## 2. The three named primary tests

### 2.1 Roll the Dice & Look Before You Leap (Nagarajan et al. 2025)

**Reference.** Nagarajan, Wu, Ding, Raghunathan. ICML 2025 oral.
[arXiv 2504.15266](https://arxiv.org/abs/2504.15266).

**Construct.** "Algorithmic creativity": the computational primitives
(combinational and exploratory in Boden's taxonomy) that creativity
plausibly requires, abstracted into minimal graph tasks. Explicitly
*not* sufficient for creativity, but argued to be necessary.

**Tasks (all defined over a knowledge-graph G stored in model
weights via training):**

| Task | Setup | Output | Cognitive analog |
|------|-------|--------|------------------|
| Sibling Discovery | Bipartite G with parents and children. | Triplets (γ, γ', Γ) where γ, γ' ∈ nbr(Γ). | Wordplay; analogy; "punchline" connections. |
| Triangle Discovery | Graph with many triangles. | (v₁, v₂, v₃) forming a complete triangle in G. | Higher-order multi-constraint coordination. |
| Circle Construction | Implicit cycle structure via adjacency. | Edge pairs that under some permutation form a cycle. | Designing word problems / proteins under constraints. |
| Line Construction | Same as Circle but a path. | Edge pairs forming a path under some permutation. | Same, simpler. |

**Scoring.** A single creativity score per task:
cr̂_n(T) = unique{s ∈ T | ¬mem_s(s) ∧ coh(s)} / |T| — the fraction of
generated outputs that are (a) coherent w.r.t. the graph,
(b) non-memorised from training, and (c) unique within the sample.
Permutations of the same underlying object collapse to one unit.

**Method contribution: hash-conditioning ("seed-conditioning").**
Prepend a per-example random hash string during training and
inference. With even greedy decoding, hash-conditioning produces
diverse outputs and beats standard temperature sampling; longer
hashes monotonically increase creativity. Authors interpret it as
"fixing a random seed upfront so the model fleshes out one thought
per sample, rather than maintaining a running set of multiple
thoughts."

**Headline findings.**
- Multi-token approaches (teacherless training, diffusion models)
  achieve ~5× the algorithmic creativity of standard NTP.
- NTP memorises training data; multi-token approaches don't.
- Hash-conditioning + multi-token is the strongest combination;
  benefit is robust across Gemma 2B and larger models.
- Limited transfer to real summarisation tasks (XSUM,
  CNN-DailyMail): "slight" diversity gains for large GPT-2,
  inconsistent for small models.

**Limitations the authors flag.**
- Tasks are minimal; success here is *necessary not sufficient*.
- Coverage is restricted to combinational and exploratory creativity
  in Boden's taxonomy — transformational creativity is excluded.
- Real-world creativity involves much larger context, social and
  cultural value, and subjective judgment — none captured here.

**Why it matters for `new_tests`.** The seed-conditioning insight
suggests our tests should *measure how much per-sample diversity is
unlocked by mechanisms beyond temperature*. Concretely: a test could
prompt the same model with different random "seeds" injected into
context and score the diversity-vs-quality trade-off.

### 2.2 Combinatorial Creativity (Schapiro et al. 2025)

**Reference.** Schapiro, Shashidhar, Gladstone, Black, Moon,
Hakkani-Tur, Varshney. [arXiv 2509.21043](https://arxiv.org/abs/2509.21043).

**Construct.** Combinatorial creativity (Boden) operationalised as
constrained labeled-graph pathfinding. Designed to model the
**ideation–execution gap** observed in LLM-for-science work
([Si et al. 2024](https://arxiv.org/abs/2409.04109)): LLMs propose
novel ideas but generate infeasible plans.

**Setup.** Conceptual space = labeled undirected graph G = (V, E, Σ).
Empirically, V = all 3-letter strings AAA–ZZZ (|V| = 17,576), Erdős–
Rényi edges with average degree 6 and labels from the alphabet. A
prompt x specifies an endpoint pair (u, v), an inclusion constraint
set I (labels that *must* appear), and an exclusion constraint set X
(labels that *must not* appear). Output = a labeled walk
P = (v₀, ℓ₁, v₁, …, ℓ_h, v_h) of length h.

**Constraints model real LLM-for-science failure modes:**
- Exclusion constraints abstract "prevent unrealistic assumptions /
  block prohibitively expensive plans."
- Inclusion constraints abstract "ensure proper baselines / require
  detailed (not vague) implementation steps."

**Scoring.**
- **Novelty:** N(P) = α_h · h + α_r · S(P), where S(P) is mean label
  surprise (negative log-likelihood under empirical label
  frequencies).
- **Utility:** U(P; x) = (1 + α_I |I|)(1 + α_X |X|) · 1[v₀ = u, v_h = v,
  labels(P) ⊇ I, labels(P) ∩ X = ∅] — multiplicative, weighted by
  how many constraints the prompt imposes.
- **Creativity:** C(θ) = E_x [U(G_θ(x); x) · N(G_θ(x))].

**Headline findings.**
- Scaling: predictable creativity gains over 1M–100M parameters.
- Architectural sweet spot at fixed compute: ~8 layers at 100M;
  optimal embedding-to-depth ratio E/L ∈ [200, 300]. Wider/shallower
  beats narrower/deeper.
- **Novelty–utility trade-off** recovered empirically and shown to
  be *scale-invariant* across 1M–10M–100M: as |I| grows, novelty
  drops monotonically.
- Error-mode shift with scale: at 1M–10M most failures are
  hallucinations (edges that don't exist); at 100M, hallucinations
  drop sharply and "invalid path" errors (well-formed but failing
  utility) rise to roughly equal frequency.

**Limitations the authors flag.**
- Synthetic graph; doesn't carry real-world semantic complexity.
- Restricted to combinatorial creativity (no exploratory or
  transformational coverage).
- Scale capped at 100M; frontier models are billions.

**Why it matters for `new_tests`.** This is the only published test
that explicitly *tunes* difficulty via constraint counts and recovers
a quantitative novelty–utility trade-off. A natural extension is to
port the constraint paradigm onto a *real* knowledge graph
(Wikidata / a domain ontology) so the construct retains semantic
content and the test becomes administerable to frontier models
without retraining.

### 2.3 CREATE (Wadhwa et al. 2026)

**Reference.** Wadhwa, Roy, Lederman, Li, Durrett.
[arXiv 2603.09970](https://arxiv.org/abs/2603.09970).

**Construct.** Associative creativity: "the ability to draw novel
yet meaningful connections between concepts." Targeted at hypothesis-
generation-style tasks.

**Setup.** Wikidata as the underlying knowledge graph. For each
manually selected (relation r, category c) pair, sample two entities
x, y from the class C_{r,c} = {x : (x, r, c) ∈ G}. Source paths are
~3 triples long. Models generate sets U of multi-hop paths
connecting the two endpoints. A path is structurally valid if
consecutive triples share entities; factuality requires each triple
to be a true Wikidata relation. 931 queries across diverse relation
types.

**Scoring.**
- **Specificity** of a triple (e_i, r_i, e_{i+1}): the rarer the
  relation among possible completions, the more specific.
  σ = g(max(|C_A|, |C_B|)) where C_A, C_B are the predicate-induced
  candidate classes and g maps class size to {1, 2, 3, 4, 5} via
  logarithmic bucketing.
- **Path quality:** f(u) = 1[factual] · min_{triple in u} σ.
- **Distance** between paths: a cosine-annealed function of string
  cosine distance (saturates above 0.7).
- **Creative utility (the headline metric):** s_γ(U) = max over
  permutations τ of Σ_i γ^{i−1} · f(u_τ(i)) · min_{j<i} d(u_τ(i),
  u_τ(j)). Greedy ordering of paths to maximise marginal
  utility; γ ∈ {0.7, 0.9} controls patience for diverse-but-low-
  quality additions.
- **Distinctiveness** ν: minimum distance to *any* path produced by
  *any* other model in the population — a cross-model diversity
  signal.

**Headline findings.**
- GPT-5 (medium reasoning) achieves the top creative utility
  s_{0.9} = 12.03; Gemini-3-pro 10.41; Claude and open-source models
  noticeably lower.
- "Thinking" models are *not* always better; high token budget alone
  doesn't unlock creative utility.
- Prompt-based interventions ("be creative", iterative refinement)
  give only weak and inconsistent gains.
- Cross-model homogeneity: average distinctiveness 0.02–0.04, max
  0.08–0.12 — frontier models converge on similar associations.
- Saturation is hard to reach due to answer multiplicity.

**Limitations the authors flag.**
- LLM-as-judge for specificity (Pearson 0.67 to humans); factuality
  evaluator has 0.94 recall but only 0.52 precision on incorrect
  relations.
- Long-tail entity coverage is weak — factuality grading degrades.
- No causal claim that high CREATE utility = genuine associative
  creativity vs. better Wikidata recall.
- Ensemble homogeneity may impose a population-level ceiling.

**Why it matters for `new_tests`.** CREATE is the closest existing
test to "a real knowledge-graph version of comb-creat that is
administered at test time." It gives us a working scoring template
(quality × diversity, greedy permutation, cosine-annealed distance)
that can be ported to other domains. Whether it has *specificity* on
LiveIdeaBench is exactly the question we'd ask of any candidate
new test for scientific ideation.

---

## 3. Adjacent benchmarks, by category

The three primary tests above are not the entire design space.
Below are ~30 adjacent benchmarks, grouped by what construct or
cognitive primitive they probe. For each: a 3–5-sentence summary
focused on what would inform a new-test design.

### 3.1 LLM-for-science / research-ideation benchmarks

These are the closest *external* benchmarks to the
"scientific ideation" construct from `dat_eval`. They are all
candidates for the criterion side of a new test (i.e., what we
would correlate against).

- **Si, Yang, Hashimoto 2024** —
  [arXiv 2409.04109](https://arxiv.org/abs/2409.04109).
  100+ NLP PhDs blind-rate ideas from human experts vs. an LLM
  ideation agent. LLM ideas judged *more* novel (5.64 vs 4.84 / 10)
  but slightly less feasible. The most carefully controlled
  scientific-ideation eval to date; expensive to run repeatedly.

- **Ruan et al. 2024 — LiveIdeaBench** —
  [arXiv 2412.17596](https://arxiv.org/abs/2412.17596). 1,180
  single-keyword prompts across 22 disciplines; LLM-judge panel
  scores originality / feasibility / fluency / flexibility / clarity.
  Demonstrates dissociation from general capability — QwQ-32B-preview
  competes with frontier on Originality despite weaker reasoning.
  This is the headline criterion benchmark from `dat_eval`.

- **Guo et al. 2024 — IdeaBench** —
  [arXiv 2411.02429](https://arxiv.org/abs/2411.02429). Title +
  abstract + reference list as context; LLM proposes an idea; GPT-4o
  scores novelty / feasibility into an "Insight Score." LLMs strong
  on novelty, weak on feasibility — same gap as Si et al.

- **Liu et al. 2025 — AI Idea Bench 2025** —
  [arXiv 2504.14191](https://arxiv.org/abs/2504.14191). Updated AI-
  research ideation benchmark with explicit contamination control.

- **Lu et al. 2024 — The AI Scientist** /
  **AI Scientist-v2** —
  [arXiv 2408.06292](https://arxiv.org/abs/2408.06292) /
  [arXiv 2504.08066](https://arxiv.org/abs/2504.08066). Full agentic
  ideation → experimentation → write-up pipeline. v2 produced the
  first AI-authored workshop-accepted paper. Independent audit
  (Beel et al. 2025, [arXiv 2502.14297](https://arxiv.org/abs/2502.14297))
  finds weak novelty filtering and 42% experiment-failure rate.
  Useful as upper bound but not a clean test (system, not
  population, evaluation).

- **Baek et al. 2024 — ResearchAgent** —
  [arXiv 2404.07738](https://arxiv.org/abs/2404.07738). Iterative
  ideation over a citation/concept graph with multi-LLM reviewers.
  A natural baseline whenever a new test claims combinatorial /
  graph-grounded ideation.

- **Jansen et al. 2024 — DiscoveryWorld** —
  [arXiv 2406.06769](https://arxiv.org/abs/2406.06769). 120
  parametric tasks across 8 toy science domains in a 2D virtual
  environment requiring full hypothesise/experiment/conclude loops.
  Best baselines below 20% on Normal/Challenge. Process-level
  scientific creativity in a controllable sandbox.

- **Chen et al. 2024 — ScienceAgentBench** —
  [arXiv 2410.05080](https://arxiv.org/abs/2410.05080). 102 data-
  driven scientific tasks from 44 peer-reviewed papers; best agent
  solves 32–42%. Measures scientific *execution* rather than
  ideation; useful as a complement (high CREATE utility but low
  ScienceAgentBench → ideation ≠ execution).

- **Liu et al. 2025 — HypoBench** —
  [arXiv 2504.11524](https://arxiv.org/abs/2504.11524). 7 real and 5
  synthetic tasks, 194 datasets. Hypotheses scored on explanatory
  power, utility, generalisability. Best methods recover only 38.8%
  of ground-truth hypotheses on hard synthetic tasks.

- **Zhou et al. 2024 — HypoGeniC** —
  [arXiv 2404.04326](https://arxiv.org/abs/2404.04326). UCB-style
  data-driven hypothesis generation; 88.6% on deceptive-review
  classification. Process-level: search algorithm wrapped around an
  LLM, in the spirit of Nagarajan's "look before you leap."

### 3.2 Process-level creativity (planning, exploration,
look-ahead)

These tests probe *how* a model arrives at outputs — closer in
spirit to Nagarajan than to DAT.

- **Lee et al. 2025 — Mind Evolution** —
  [arXiv 2501.09891](https://arxiv.org/abs/2501.09891). Evolutionary
  search over LLM-generated candidates; >98% on TravelPlanner and
  Natural Plan. Decomposes creative cognition into explicit
  divergent-then-convergent search.

- **Sudoku-Bench** —
  [arXiv 2505.16135](https://arxiv.org/abs/2505.16135). Variant-
  Sudoku puzzles with "break-in" insight moves. Probes lateral
  search rather than rote constraint propagation.

- **LatEval** —
  [arXiv 2308.10855](https://arxiv.org/abs/2308.10855). Interactive
  lateral-thinking puzzles; the model must *ask questions* to
  uncover hidden information. Process-level, interactive — a
  primitive distinct from search depth.

### 3.3 Combinatorial / compositional creativity (beyond comb-creat)

- **Tian et al. 2023 — MacGyver** —
  [arXiv 2311.09682](https://arxiv.org/abs/2311.09682). 1,600
  problems requiring unconventional use of household objects
  (functional fixedness). Closest published combinatorial test in a
  semantically rich, non-scientific domain.

- **Lu et al. 2025 — DeepMath-Creative** —
  [arXiv 2505.08744](https://arxiv.org/abs/2505.08744). Constructive
  math problems (build an object satisfying a property; build a
  counterexample). O3-Mini at 70% on undergrad tasks, fails on
  harder ones. Authors argue successes come from pattern
  recombination, not novel synthesis.

- **COGS / CFQ** (Kim & Linzen 2020; Keysers et al. 2020). Canonical
  compositional-generalisation suites. Not creativity per se but
  the foundation of "novel combinations of known components" that
  combinatorial-creativity claims build on.

### 3.4 Analogy and conceptual leaps

- **Webb, Holyoak, Lu 2023 — "Emergent Analogical Reasoning"** —
  [arXiv 2212.09196](https://arxiv.org/abs/2212.09196). GPT-3
  matches/exceeds humans on Raven-style matrix and verbal-analogy
  tasks. **Hodel & West 2023** rebuttal —
  [arXiv 2308.16118](https://arxiv.org/abs/2308.16118) — shows
  failure on counterfactual variants. The cautionary tale on
  construct validity for LLM analogy tests.

- **Mitchell et al. — Counterfactual Concept-ARC** (related to ARC
  family). Stricter conceptual-leap probe under perturbation.

- **SemEval-2024 Task 9 — BRAINTEASER** —
  [arXiv 2404.16068](https://arxiv.org/abs/2404.16068). Sentence and
  word puzzles defying commonsense. Lateral-thinking benchmark with
  modest direct relevance to scientific ideation but useful for
  conceptual-leap construct validity.

- **NYT-Connections benchmarks** — Samadarshi et al.
  ([arXiv 2406.11012](https://arxiv.org/abs/2406.11012)) and Todd
  et al. ([arXiv 2404.11730](https://arxiv.org/abs/2404.11730)).
  Lateral semantic grouping. GPT-4o solves only ~8% of puzzles
  fully. Public, automatically scorable analogue of "Only Connect."

### 3.5 Long-horizon / open-world creativity

The Kapoor 2026 line: "long-horizon, messy, real-world tasks
assessed through small-sample qualitative analysis."

- **Chan et al. 2024 — MLE-Bench** —
  [arXiv 2410.07095](https://arxiv.org/abs/2410.07095). 75 Kaggle
  competitions; o1-preview + AIDE earns Kaggle bronze on 16.9%
  (pass@1).

- **Wei et al. 2025 — BrowseComp** —
  [arXiv 2504.12516](https://arxiv.org/abs/2504.12516). 1,266 hard
  fact-finding browsing questions; Deep Research solves ~50%.

- **Tian et al. 2024 — SciCode** —
  [arXiv 2407.13168](https://arxiv.org/abs/2407.13168). 80 research
  problems, 338 sub-problems, 16 natural-science subfields,
  scientist-curated. Claude-3.5-Sonnet solves 4.6% of main
  problems. The closest existing benchmark for "scientific
  implementation creativity."

- **Glazer et al. 2024 — FrontierMath** —
  [arXiv 2411.04872](https://arxiv.org/abs/2411.04872). Hundreds of
  unpublished research-level math problems. SOTA <2% at release.

- **Tsoukalas et al. 2024 — PutnamBench** —
  [arXiv 2407.11214](https://arxiv.org/abs/2407.11214). 1,724 Putnam
  problems formalised in Lean / Isabelle / Coq.

- **Petrov et al. 2025 — "Proof or Bluff?" USAMO 2025** —
  [arXiv 2503.21934](https://arxiv.org/abs/2503.21934). Expert-
  graded natural-language proofs on the 6 USAMO problems within
  hours of release. Gemini-2.5-Pro 25%, all others <5%. Surface
  accuracy hides creative-proof failure.

- **Romera-Paredes et al. 2024 — FunSearch** (Nature). LLM +
  evolutionary code search discovers new cap-set lower bounds and
  improved bin-packing heuristics. Proof-of-concept that creative
  leaps in math are achievable when wrapped in a search procedure
  (echoes Nagarajan).

- **ARC-AGI-1 / ARC-AGI-2** (Chollet 2019;
  [arXiv 2412.04604](https://arxiv.org/abs/2412.04604);
  [arXiv 2505.11831](https://arxiv.org/abs/2505.11831)). Few-shot
  abstract grid puzzles; v2 explicitly designed to defeat the
  brute-force regime. Boundary case for whether creativity is
  reducible to abstraction.

### 3.6 Output diversity (beyond Hivemind)

- **Zhang et al. 2025 — NoveltyBench** —
  [arXiv 2504.05228](https://arxiv.org/abs/2504.05228). Diversity-
  eliciting prompts plus filtered real user queries; 20 SOTA models.
  Larger models within a family are *less* diverse than smaller
  ones — the empirical hook for the "mode collapse hurts ideation"
  thread.

- **Jiang et al. 2025 — Artificial Hivemind / Infinity-Chat** —
  [arXiv 2510.22954](https://arxiv.org/abs/2510.22954) (NeurIPS 2025
  Best Paper). 26K open-ended prompts, 31K human ratings, 70+
  models. Documents intra-model repetition *and* cross-family inter-
  model homogeneity.

- **CreativityPrism** —
  [arXiv 2510.20091](https://arxiv.org/abs/2510.20091). Holistic
  creativity evaluation across multiple subtasks; usable as a meta-
  benchmark / aggregator.

### 3.7 Originality / TTCT-style for LLMs

- **Chakrabarty et al. 2024 — TTCW** —
  [arXiv 2309.14556](https://arxiv.org/abs/2309.14556). Torrance-
  Test-of-Creative-Writing adaptation: 14 binary expert-judged
  criteria across fluency / flexibility / originality / elaboration.
  LLM stories pass 3–10× fewer criteria than New Yorker writers.

- **Zhao et al. 2024 — "Assessing and Understanding Creativity
  in LLMs"** —
  [arXiv 2401.12491](https://arxiv.org/abs/2401.12491). Adapts the
  classical TTCT to LLMs; GPT-4 in human top 1% on fluency /
  originality. The "LLM TTCT" paper.

- **LitBench** —
  [arXiv 2507.00769](https://arxiv.org/abs/2507.00769). 2,480
  pairwise creative-writing comparisons + 43,827 training pairs from
  r/WritingPrompts; designed to train and evaluate LLM-judges of
  creative writing. Methodology for building a scorable creative
  writing target.

### 3.8 Math / scientific reasoning targeting transformative leaps

Already partially covered above (FrontierMath, PutnamBench, USAMO,
FunSearch, DeepMath-Creative). Add:

- **OlympiadBench** —
  [arXiv 2402.14008](https://arxiv.org/abs/2402.14008). Olympiad-
  level bilingual multimodal problems.

- **MathArena** —
  [arXiv 2505.23281](https://arxiv.org/abs/2505.23281). Uncontaminated
  competition tracking.

- **Sun et al. 2025 — OMEGA** —
  [arXiv 2506.18880](https://arxiv.org/abs/2506.18880). Maps Boden's
  exploratory / compositional / transformative typology onto
  mathematical generalisation across geometry, number theory,
  algebra, combinatorics, logic, and puzzles. Verifiers are
  symbolic, numerical, or graphical. Headline qualitative finding:
  fine-tuning helps exploratory generalisation; compositional
  remains limited; **transformative reasoning shows little to no
  improvement**. Most directly speaks to the construct that
  semantic-distance tests miss.

### 3.9 Theoretical foundations

- **Boden 2004** — combinatorial / exploratory / transformational
  trichotomy. Cited by every test above. Transformational creativity
  alters "enabling constraints" of a conceptual space.

- **Schapiro, Black, Varshney 2025 — "Transformational Creativity
  in Science: A Graphical Theory"** —
  [arXiv 2504.18687](https://arxiv.org/abs/2504.18687). Formalises
  transformational creativity as graph-theoretic operations on
  conceptual spaces; argues that modifications to *axioms* yield the
  most transformative impact. Historical examples: Copernicus,
  Einstein, Darwin. Suggests evaluation directions: whether LLMs
  can generate axiom modifications that violate current scientific
  assumptions while preserving internal consistency, and whether
  they can identify which modifications have historically proven
  transformative.

- **Kapoor et al. 2026 — "Open-world evaluations"** —
  [cruxevals.com](https://cruxevals.com/open-world-evaluations.pdf).
  Argues for "long-horizon, messy, real-world tasks assessed
  through small-sample qualitative analysis rather than
  benchmark-scale automation." Counterweight to anything in §2 and
  §3 of this survey: even an excellent minimal test cannot capture
  long-horizon real-world creativity on its own.

---

## 4. Synthesis: design principles distilled

Across the surveyed work, a few recurring design principles emerge.

1. **Decouple the test from the criterion.** The reason DAT, CDAT,
   PACE all failed on scientific ideation is not that semantic
   distance is wrong, but that the criterion (LiveIdeaBench)
   measures something the test does not probe. Any new test should
   pre-register the construct it claims to measure and the
   criterion it expects to correlate with under the validity-
   specificity frontier.

2. **Constraints are the lever.** Both comb-creat and CREATE recover
   meaningful per-model variance only when the prompt imposes hard
   constraints (inclusion / exclusion labels, endpoint pairs,
   factuality). DAT-style "be diverse" prompts have no constraint
   surface to gradient over.

3. **Score the trade-off, not the value.** comb-creat's novelty ×
   utility, CREATE's quality × diversity, OMEGA's per-regime
   verifier — the pattern is product or curve over two axes. Single-
   number diversity scores are easier to game and harder to
   triangulate.

4. **Target a cognitive primitive that semantic distance misses.**
   Candidates from the survey:
   - far-sighted planning / look-ahead (Nagarajan, Mind Evolution);
   - constrained graph traversal (comb-creat, CREATE);
   - axiom modification (Schapiro 2025 transformational theory,
     OMEGA's transformative regime);
   - counterfactual analogy (Hodel & West rebuttal);
   - exploration vs exploitation under a budget (HypoGeniC,
     FunSearch);
   - cross-domain transfer of an unfamiliar primitive (BRAINTEASER,
     ARC-AGI-2).

5. **Knowledge graphs > synthetic graphs for population-administered
   tests at the frontier scale.** Synthetic graphs (Nagarajan, comb-
   creat) require training; frontier models cannot be retrained for
   a test. CREATE's Wikidata grounding is the existence proof that
   a real KG works.

6. **LLM-as-judge has known failure modes.** CREATE's specificity
   judge correlates with humans only at 0.67; its factuality judge
   has 0.94 recall but 0.52 precision on incorrect relations. Any
   new test that uses an LLM judge needs an analogous reliability
   analysis.

7. **Ensemble homogeneity caps tests with cross-model distance
   components.** CREATE's max-distinctiveness ν tops out at 0.12
   across frontier models. Tests scored by *cross-model* diversity
   inherit this ceiling.

---

## 5. Where the gap is widest

The construct on which `dat_eval` showed the cleanest non-coverage
is **scientific ideation**. Inspecting the categories above:

- The semantic-distance family (§ historical, not above) probes
  associative distance only, which LiveIdeaBench does not reward
  much (correlations near zero).
- The LLM-for-science benchmarks (§3.1) directly *measure* the
  construct, but none of them serve as a *test* — they require
  agentic scaffolding, retrieval, or human raters.
- CREATE (§2.3) and comb-creat (§2.2) probe the right cognitive
  primitive (constrained creative path/association) but neither has
  been validated on LiveIdeaBench. CREATE in particular is a
  candidate to score against LiveIdeaBench in this track.
- OMEGA's transformative regime (§3.8) — and the Schapiro 2025
  graphical theory of transformational creativity (§3.9) — point at
  axiom-modification as the most distinctive primitive of scientific
  creativity, and yet no minimal, automatically-scorable LLM test
  yet exists for it.

The two cleanest gaps are therefore:

- **Gap A.** Port comb-creat-style constraint + novelty/utility
  scoring to a real knowledge graph (e.g., a domain ontology of a
  scientific field), administer to frontier models *at test time*,
  validate against LiveIdeaBench. This is "real-world comb-creat,"
  partially overlapping CREATE but with the inclusion/exclusion
  constraint apparatus.
- **Gap B.** Operationalise *transformational* creativity in the
  Schapiro 2025 sense as a minimal test: present a small axiom set
  + observed phenomena that the axioms cannot explain, score
  whether the model proposes a *minimal axiom modification* that
  resolves the contradiction. Verifier: a downstream symbolic /
  numerical check (à la OMEGA), or a structured judge with a
  rubric like CREATE's specificity buckets.

Initial sketches for both gaps are in
[proposals.md](proposals.md).
