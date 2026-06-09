# kg_creat vs CREATE — methodological novelty

The single most important thing to articulate for the LM4Sci resubmission: how a
real-KG port of Comb-Creat differs from CREATE
([Wadhwa et al. 2026, arXiv 2603.09970](https://arxiv.org/abs/2603.09970)), which
*already* runs Wikidata multi-hop paths between two endpoints with quality×diversity
scoring at test time.

**Lead with the constraints** (locked): the **typology of methodological constraints**
(inclusion/exclusion + categorical/waypoint/ordering/…), **constraint-weighted utility**,
and **validation against LiveIdeaBench**. We share CREATE's open-KG + LLM-judge
verification (reverted from exact checking 2026-06-02 — see below), so **verification is
*not* a differentiator**, and novelty uses the *validated* DAT measure, so **novelty
scoring is *not* a differentiator** either. The moat is constraints + utility + LIB
validation. State this honestly.

## Side-by-side

| Dimension | CREATE (Wadhwa et al. 2026) | kg_creat (this track) |
|-----------|-----------------------------|------------------------|
| Substrate | Wikidata | Wikidata (head-to-head); scientific KG as ablation |
| KG role | query construction (seed sampling); answers from parametric memory | **same** — query construction + factuality reference; answers from parametric memory |
| Answer space | open (all of Wikidata) | **same** — open KG |
| Factuality check | **LLM-as-judge** (0.94 recall, 0.52 precision on bad relations) | **same** — LLM-as-judge (we adopt/improve CREATE's, + a reliability analysis) |
| Prompt | two endpoints `(x, y)` sampled from a `(relation, category)` class | endpoints `(u, v)` **+ a set `K` of typed constraints + hop count `h`** |
| **Constraints** | **none** | **a typology**: inclusion / exclusion / categorical / waypoint / ordering-metapath / budget / polarity — enforced exactly on the verified path |
| **Difficulty tuning** | not tunable (fixed by sampled class) | **2-D: constraint count × type** (type ablation = which pressure predicts LIB / its facets) |
| Novelty | judge-graded specificity buckets `σ ∈ {1..5}` | **semantic remoteness** `R(P)` = DAT mean pairwise embedding distance over path entities — *deliberately the validated DAT measure, not a novel formula* |
| **Utility** | `f(u) = 1[factual] · min σ` | `(∏_t (1+α_t·n_t)) · 1[valid ∧ factual]` — **constraint-load weighted** |
| Diversity / set scoring | greedy `s_γ` with cosine-annealed string distance | separate set-level term: mean pairwise distance over the `k` valid paths (embedding / Jaccard·edit) |
| Trade-off recovered | — | **novelty–utility trade-off** (Comb-Creat's headline curve), as constraint load grows |
| **External-criterion validation** | none (no correlation to an ideation benchmark) | **validity + specificity vs LiveIdeaBench** under the dat_eval frontier |

## The three sentences for the paper

1. CREATE measures associative creativity as endpoint-to-endpoint multi-hop path
   generation scored by judge-graded specificity and cross-path diversity, with no
   way to *control task difficulty* and no validation against an external creativity
   criterion.
2. On the **same open-KG, judge-verified setup** as CREATE, we add a **typology of
   methodological constraints** (inclusion/exclusion/categorical/waypoint/ordering) and
   a **constraint-load-weighted utility**, which makes difficulty an explicit, 2-D
   (count × type) lever and recovers the novelty–utility trade-off — structure CREATE's
   unconstrained association task has no way to express. Novelty is the validated DAT
   semantic-distance measure (not a new formula).
3. We then validate the test where CREATE does not: its per-model scores **correlate
   with LiveIdeaBench** under the validity/specificity framework — the first KG-based
   creativity test shown to predict scientific ideation.

## Why we (like CREATE) use a judge, not exact checking

We initially designed exact factuality checking against a held subgraph as the headline
differentiator, then **reverted to an LLM judge (2026-06-02)** because exact checking is
too restrictive. The reasons are exactly why CREATE judges in the first place — once you
let the model answer from **parametric memory over the open KG** (which is the right
setup for measuring association creativity), exact verification fails for four reasons:

1. **Open-world incompleteness** — the KG is incomplete; absence of a triple ≠ false.
   A true-but-unrecorded triple fails an exact lookup but passes a judge.
2. **Unbounded answer set** — the model freely picks intermediate entities/relations
   across ~100M entities; there is no small pre-chosen region to hold and query, and
   restricting to one cripples the model's use of its own knowledge.
3. **Entity/relation linking** — models emit natural-language names, not QIDs/PIDs;
   exact lookup needs an error-prone linking step the judge sidesteps.
4. **Specificity is a count, not a membership test** — class-size `COUNT`s per triple;
   judge-approximated.

So **verification is shared ground with CREATE** (open KG, LLM-judged factuality; we
adopt/improve CREATE's gpt-oss-120b judge and add a reliability analysis). The
distinction is *what we ask and how we score it*: **constraints** restructure the task,
and they are enforced **exactly on the judge-verified path** (so the constraint
apparatus stays rigorous even though factuality is judged).

## What is *not* the novelty (avoid overclaiming)

- Not "first to use a real KG for creativity" — CREATE did that.
- Not "first test-time creativity eval on frontier models" — CREATE did that too.
- Not exact / judge-free verification — we **reverted to CREATE's open-KG + judge**;
  verification is shared ground, not a differentiator.
- Not a new **novelty measure** — we deliberately use the *validated* DAT semantic-
  distance measure (both we and CREATE score in embedding-distance space).
- The contribution is concentrated in **(1) the constraint *typology* (inclusion /
  exclusion / categorical / waypoint / ordering / …) as a 2-D difficulty + diagnostic
  axis, (2) constraint-load-weighted utility (exactly enforced on the verified path),**
  and (3) *validating* the test against LiveIdeaBench — not in verification or novelty.
