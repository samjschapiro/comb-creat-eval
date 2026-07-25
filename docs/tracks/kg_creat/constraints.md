# kg_creat — constraint taxonomy & failure-mode grounding

> **Ordering dropped (2026-07-22).** *Ordering* appears throughout this doc as a Regime-A
> constraint, but the round-1 results showed that as derived (target = reverse of the natural
> class order) it measured an anti-natural double-inclusion, not sequencing, and it has been
> removed from the reported constraint set. The mentions below are retained as taxonomy history;
> see [assessment.md](assessment.md) §7c and
> [`reports/2026-07-22_kg_creat_creativity/`](../../reports/2026-07-22_kg_creat_creativity/report.md)
> Appendix A. It was flagged here (§6) as one of the "two loose fits" — that instinct was right.

The constraint set is the study's contribution, so this doc is its single reference:
the **cognitively-grounded taxonomy**, the **failure-mode grounding** (each constraint
abstracts a documented LLM ideation failure), the plain-English cores, how each is
checked experimentally, and the formal predicates in condensed form.

See [design.md](design.md) §Study framing for the metric (ideation vs execution);
[paper_outline.md](paper_outline.md) for the narrative; and the CREATE contrast in
[novelty_vs_create.md](novelty_vs_create.md). The full formal predicates live in the
paper's methods (and are condensed in §5 below).

## 1. Organizing principle

The taxonomy's top-level split is the field-standard definition of creativity —
**novel *and* useful** (Boden; Simonton BVSR; `limit_theorems`) — which maps 1:1 onto the
paper's **ideation–execution decomposition**:

- **Novelty pole = ideation constraints** — do they preserve the novelty the model reaches
  for (`R_emit`)?
- **Utility pole = execution constraints** — do they get satisfied (`sat`)?

Within each pole, a small set of **cognitive operations** creativity requires, each
anchored by (a) an established cog-sci construct and (b) a documented LLM failure mode, and
mechanized by a graph constraint:

| Pole | Cognitive operation (cog-sci construct) | Documented LLM failure | Constraints |
|------|------------------------------------------|------------------------|-------------|
| Novelty | **Defeat the dominant response** — associative hierarchy (Mednick); path of least resistance (Ward) | clustering / duplication; reinvents the obvious | exclusion, hub-avoidance |
| Novelty | **Bridge remote concepts** — remote associates / RAT (Mednick); analogy (Gentner); conceptual blending (Fauconnier–Turner) | shallow / local associations | waypoint-through |
| Utility | **Respect the conceptual space** — exploratory creativity (Boden); Geneplore; constraint satisfaction | unrealistic assumptions; ungrounded / wrong-domain; missing mechanism or baseline | inclusion, categorical |
| Utility | **Elaborate in feasible order** — means–ends analysis (Newell & Simon); elaboration (Guilford); verification (Wallas) | vague / underspecified; out-of-order logic; infeasible plans | ordering/metapath, depth, budget |
| (set-level) | **Genuine divergence** across the answer set | lack of diversity / duplication | disjointness |

> **Grounding-anchor confidence.** Rock-solid: Mednick associative hierarchy / RAT, Gentner
> structure-mapping, Fauconnier–Turner blending, Boden exploratory-within-a-space, the
> novelty–utility backbone (mostly already in `04_background.tex`). To pressure-test before
> camera-ready: whether *path of least
> resistance* (Ward) vs *functional fixedness* (Duncker) is the right anchor for "defeat the
> dominant response," and whether the means–ends / elaboration row is the tightest fit for
> ordering/budget (the weakest link).

### Analogy and blending as constrained forms of combinatorial creativity

The most defensible way to relate our framework to the two canonical theories of *structured*
combinatorial creativity — **analogy** (Gentner, structure-mapping) and **conceptual blending**
(Fauconnier & Turner) — is not to treat them as separate phenomena, but as **constrained special
cases of our general formulation**. Our task (novel-and-useful multi-hop paths connecting concepts
in a KG) is the general case; each structured theory is recovered by *adding* a constraint regime.

- **Analogy = structurally-constrained CC.** Fix the *relational structure*: require the path to
  instantiate a given metapath `M` (relation-type template — our `c^meta_M`). Analogy is
  combinatorial creativity with the relational *form* held constant while the content
  (entities/domains) varies: `heart —pumps→ blood —through→ vessels` ≈ `pump —moves→ water —through→
  pipes`. Structure-mapping's *systematicity* (prefer deep relational systems) = our **depth** lever.
- **Conceptual blending = cross-space-constrained CC.** Require the path to integrate ≥2 input
  spaces: endpoints in distinct domains `D1, D2` and/or categorical constraints (`c^cat`) demanding
  the path touch each. Blending is combinatorial creativity where the connection must span multiple
  conceptual spaces; the novelty `R` (space distance) *is* the blend's conceptual leap.

Both are **proper restrictions** — obtained by *adding* constraints — which is exactly why a
constraint-based formulation is the natural general theory: **our constraint typology is the dial
that carries general CC to these specific structured forms.** Analogy and blending are points in our
constraint space, not phenomena outside it. Experimental payoff: the metapath condition *is* the
analogy regime and the cross-domain condition *is* the blending regime, so the per-constraint
novelty-modulation results speak directly to how well LLMs perform analogy vs blending.

**Boundary (state honestly).** We subsume the *combinatorial* core of both theories — connecting
distant structure coherently (Boden's combinatorial creativity, our remit). We do **not** capture
blending's genuine *emergent structure* (inventing concepts in neither input space), which is
Boden's *transformational* creativity. The counterfactual / missing-edge variant (propose a path
that *should* exist but is not in the KG) is the frontier where that emergence lives — the
transformational edge of the framework, explicitly bracketed. We do **not** use Koestler's
"bisociation" — it is subsumed by conceptual blending; the associative-distance intuition it named
is carried by Mednick's remote-association and by our novelty `R`.

### Operationalizing analogy (V) and blending (VI): the semantic tier

The figure's two starred panels are the **semantic** constraints — satisfaction is a *judgment*,
not a syntactic check on the KG path — as opposed to the syntactic constraints I–IV.

- **V. Analogy = structure-preserving generation.** Prompt supplies a **source structure**
  `S = a₀ —m₁→ … —mₖ→ aₖ` (a real path sampled from the KG) + a **target anchor** `u'`; the model
  produces the analog `P = u' —r₁→ … —rₖ→ v'` (`Wheel —part_of→ Bike` ⟹ `Tire —part_of→ Car`).
  It is the metapath constraint (IV) with the template derived from a meaningful source and the
  target left to the model. **Satisfaction:** *syntactic floor* = relation sequence matches `M`
  (exact); *semantic* (**judged**) = entities play corresponding roles (same metapath ≠ good
  analogy). **Novelty `R`** = source↔target domain distance. Two structures, **disjoint** nodes.
- **VI. Blending = colliding/fused isomorphism.** Read as a single **pivot path**
  `Records —chases→ Athlete —is→ Boxer —is→ Dog —chases→ Squirrels`: a pivot node `X` (`Boxer`)
  invoked in two senses, with structurally parallel sides (`X chases Y` on both). Blending =
  *connect two distant frames through a shared node whose two roles collide/fuse*.
  **Satisfaction:** *syntactic floor* = two isomorphic sub-structures sharing pivot `X`; *semantic*
  (**judged**) = `X` genuinely carries two colliding/fusing senses. **Novelty `R`** = distance
  between the two fused domains. Two structures, **shared** node.

**Methodological consequence of the star.** I–IV are *exactly* checkable (controlled relation
vocab + type map) → execution axis judge-free (only factuality judged). V–VI's `sat` **is** a
judgment → they need a reliability analysis like factuality does, and their execution axis is not
judge-free. **Study fit:** I–IV slot into matched bundles directly; V needs a source structure
appended, VI needs cross-domain endpoints — so V–VI run on the subset of bundles that support them,
compared via the same within-bundle novelty-modulation deltas, flagged as the semantic tier.

## 2. Failure-mode grounding (the central table)

Each constraint turns a **documented, quoted way LLM research ideas fail**
([Si, Yang & Hashimoto 2024, arXiv:2409.04109](https://arxiv.org/abs/2409.04109)) into a
checkable rule on a path. This is the ICLR-scale expansion of the paper's existing
`media/07_table_of_utility_constraints.tex` (which had only 4 inclusion/exclusion rows).
Examples use ML entities the audience knows (several are Si et al.'s own: BLOOM, StereoSet,
CLIP, self-flagging hallucinations).

| Constraint | Documented failure mode (Si et al. 2024) | ✓ What it ensures | ✗ How it's violated |
|------------|-------------------------------------------|-------------------|---------------------|
| **Exclusion** | *Unrealistic assumptions* — reviewers flagged an idea assuming "the model can accurately flag its own hallucinations," an assumption "unlikely to be true in practice" | bans the wishful link, forcing an externally-grounded mechanism | path routes `LLM —self-detects→ hallucination —removes→ clean output` (uses the banned assumption) |
| **Budget** | *Infeasible compute* — "the project calls for fine-tuning BLOOM (176B params)… quite a lot of GPUs" | total operation-cost ≤ B; full fine-tunes priced out, pushing toward prompting/LoRA | path goes `task —full-finetune→ BLOOM-176B —eval→ gain` (over budget) |
| **Inclusion** | *Missing baseline* — "needs to be compared to simply asking the model to think" | path must contain a `compared-to-baseline` hop | `new-method —claims→ SOTA` with no baseline edge |
| **Depth** | *Hand-wavey* — "ad-hoc + hand-wavey suggestion"; "unclear how CLIP is connected to the LM" | path ≥ geodesic + δ hops: a real mechanism chain | one-hop leap `CLIP —improves→ language-model` |
| **Categorical** | *Dataset/method misuse* — "StereoSet is not a QA dataset; it simply contains statements" | path must pass through an entity of the required *type* (a QA dataset) | evaluation routed through StereoSet (a bias-statement set) when a QA dataset is required |
| **Waypoint-through** | *Ungrounded* — "does not mention some methods which seem to do quite well with LLMs" | path must pass through the specified prior method (e.g. retrieval-augmentation) | problem→solution path that never touches the required prior work |
| **Ordering** | *Incoherent pipeline* — steps whose connection is "unclear"/ad-hoc | relation order = mechanism→effect (`encode` before `condition` before `generate`) | `—evaluate→` applied before the method is `—trained→` |
| **Hub-avoidance** | *Clustering on generic hooks* — ideas gravitate to a few dominant concepts | path can't route through an overused hub (`prompting`, `attention`) | two concepts bridged only via `—uses→ prompting —improves→` |
| **Disjointness** | *Lack of diversity* — "only 200 non-duplicate unique ideas" out of 4,000 generated | the k returned paths share no interior concepts / relation sets | 5 paths all `metformin→AMPK→cancer` with relabeling |

## 3. Plain-English cores

The sentence a skimming reviewer should remember per constraint.

| Constraint | Core |
|------------|------|
| Exclusion | No shortcuts — the obvious/wishful move is banned |
| Hub-avoidance | Don't route through the obvious middleman |
| Waypoint-through | Bridge through *this* specific concept |
| Inclusion | Use the required kind of link |
| Categorical | Involve (or avoid) something of this kind |
| Ordering / metapath | Right steps, right order |
| Depth | Show the full chain, not the shortcut |
| Budget | Stay within budget |
| Disjointness | Genuinely different routes, not rephrasings |

Unifying one-liner: **novelty-pole constraints all say "don't take the easy path";
utility-pole constraints all say "respect the rules of the space."**

## 4. Experimental checkability (see design.md methods / CREATE-extension)

The model emits free-text JSON paths (CREATE-style), so checkability hinges on specifying
constraint parameters as **named** entities/relations the model echoes, plus a **controlled
relation vocabulary**. Then most constraints are exact string checks — only categorical
typing and factuality touch the judge.

| Constraint | Check method | Exact or judge? |
|------------|--------------|-----------------|
| Well-formed (endpoints/hops/shared entities) | string equality (CREATE's structural filter) | exact |
| Exclusion / Inclusion / Ordering / Budget | typed relation-label logic | **exact** |
| Waypoint / Hub-avoidance | named-entity string/alias match | **exact-ish** |
| Depth | hop count vs geodesic on `G_c` | **exact** |
| Disjointness | interior-entity / relation-set overlap across paths | **exact** |
| Categorical | is each entity of type `T`? | **judge** (KG type-lookup as robustness check) |
| Factuality (all triples) | CREATE's gpt-oss-120b judge | **judge** |

So the **execution axis is judge-free except the shared factuality channel** (plus
categorical typing). Novelty `R_emit` needs neither linking nor judge — embed emitted entity
strings (SBERT), mean pairwise distance.

## 5. Formal predicates (condensed)

Path `P = (e₀, r₁, e₁, …, r_h, e_h)`. Derived: `Rel(P)` = relation set, `Int(P)` =
`{e₁,…,e_{h-1}}` interior entities, `τ` = entity-type map, `deg_G` = degree, `cost` =
per-relation cost, `d_G(u,v)` = geodesic. Full definitions + reductions in the paper methods.

| Constraint | Predicate (satisfied iff) | Parameters |
|------------|---------------------------|------------|
| Exclusion | `Rel(P) ∩ X = ∅` | forbidden relations `X ⊆ Σ` |
| Inclusion | `I ⊆ Rel(P)` | required relations `I ⊆ Σ` |
| Hub-avoidance | `Int(P) ∩ H = ∅` | hub set `H` (or degree threshold θ) |
| Waypoint-through | `W ⊆ Int(P)` (all) / `W ∩ Int(P) ≠ ∅` (any) | required waypoints `W ⊆ V` |
| Categorical | `∃/∀ e ∈ Int(P): T ∈/∉ τ(e)` | type `T`, sign (incl/excl) |
| Ordering | `fst_a(P) < fst_b(P) < ∞` | ordered pair `(a,b)` |
| Metapath | `ρ(P)` matches template `M ∈ (Σ∪{∗})^h` | template `M` |
| Depth | `h(P) ≥ d_G(u,v) + δ` | margin `δ` (set at sampling) |
| Budget | `Σ cost(rᵢ) ≤ B` | budget `B`, cost table |
| Disjointness | `∀ j≠l: Int(P^{(j)}) ∩ Int(P^{(l)}) = ∅` | set-level, over the k paths |

Composition: `valid(P;x) = wf(P) · ∏_{c∈K} c(P)`; utility
`U(P;x) = (∏_t (1+α_t n_t)) · valid(P;x)`; novelty
`R(P) = mean pairwise (1 − cos φ(eᵢ),φ(eⱼ))`. Per constraint type `t`: ideation
`R_emit(t) = E[R(P)]` over emitted paths; execution `sat(t) = E[valid(P;x)]`.

Useful reductions to state: exclusion = budget with `cost=∞` on `X`; hub-avoidance =
degree-thresholded categorical; metapath subsumes inclusion + ordering + cardinality.

## 6. Caveats to resolve before camera-ready

- **Verify the Si et al. quotes.** They were surfaced via WebFetch's summarizer, not the PDF
  directly. Pull exact sentences from arXiv:2409.04109 and wrap any unconfirmed ones with the
  `% --- AI-GENERATED / VERIFY ---` bib convention. Double-check the "200 of 4,000" phrasing
  verbatim.
- **Two loose fits.** *Ordering* and *Hub-avoidance* lack a single crisp quote (ordering
  leans on the "unclear how X connects" critiques; hub-avoidance on the duplication/clustering
  finding, which disjointness also claims). Soft spots if a reviewer pushes.
- **Controlled-relation-vocabulary deviation from CREATE.** Required to make relation-level
  constraints exact; ablate free-form vs controlled relations to bound the side-effect.
- **Precedence convention (ordering).** "both present, a first" vs the weaker "no b before any
  a" — pick deliberately.
