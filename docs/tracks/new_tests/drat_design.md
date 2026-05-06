# Divergent Remote Association Test (DRAT) — design

A test-time, vocabulary-space creativity test targeted at scientific
ideation. The stimulus structure is the Remote Associates Test
(Mednick 1962): two distant anchors that need bridging. The response
format and scoring inherit from the Divergent Association Task
(Olson et al. 2021): produce many items, score by embedding
geometry. Neither parent on its own probes what we want — RAT is
convergent and accuracy-scored, DAT has no anchors at all.

## Motivation

DRAT is designed to bridge convergent and divergent thinking measures
into a single unified test. The cognitive psychology literature
since Guilford (1967) has split creativity along this axis:
convergent thinking — the search for a single correct answer to a
constrained problem (Mednick 1962) — and divergent thinking — the
generation of many candidate responses to an open-ended one
(Guilford 1967, Torrance 1974). Real creative cognition uses both:
divergent generation of candidates followed by convergent selection,
or convergent retrieval of analogues followed by divergent
elaboration. RAT measures the convergent face. DAT measures the
divergent face. No vocabulary-space test currently combines them.

The dat_eval results show that DAT, CDAT, and PACE all fail to
predict scientific ideation on LiveIdeaBench: raw validity
$r \in [-0.11, +0.20]$, no specificity cell reaching $p < 0.05$.
CDAT and DAT show non-zero but underpowered specificity
($r \mid g \approx +0.21$ and $+0.24$). The diagnosis is that these
three tests are operationalizations of one face of Mednick's
associative-hierarchy theory — divergent fluency — and miss the
convergent constraint that real creative cognition layers on top.

### The shared cognitive theory

DAT, CDAT, and PACE all descend from Mednick's *associative theory
of creativity* (Mednick 1962, *Psychological Review*). The theory's
central claim is that creative individuals have a flatter
*associative hierarchy*. Given a stimulus, a non-creative person
produces a steep distribution of associates: a few primary responses
dominate, the rest decay rapidly. A creative person produces a flat
distribution: many associates of comparable strength, including
remote ones. The flatness is what predicts creative ability, because
it provides more candidate connections when a problem demands
non-obvious bridging. Every creativity test in this lineage is some
operationalization of measuring that flatness.

### What RAT measures

The Remote Associates Test (Mednick 1962, refined by Bowden &
Jung-Beeman 2003) instantiates the theory through *convergence*. It
presents three remote stimulus words and asks for the single word
that connects all three. The classic example: *broken / clear / eye*
$\to$ *glass*. The cognitive demand is to activate remote associates
of three stimuli simultaneously and locate their intersection. RAT
has been the standard psychometric instrument for remote-associative
ability for sixty years; it correlates positively with creative
achievement in human populations at $r \approx 0.3$ in meta-analyses
(Lee & Therriault 2013). Its main weakness, well-documented, is that
the single-answer format makes performance partly a function of
whether the bridge word is in the respondent's vocabulary, not just
whether their associative hierarchy is flat enough to find it.

### What DAT measures

The Divergent Association Task (Olson, Nahas, Chmielewski, Branch,
Cousineau, Webb, & Beaty 2021, *PNAS*) instantiates the same theory
through *divergence*. There are no stimulus words. The instruction
is to produce 10 words as different from each other as possible. The
score is the mean pairwise distance in semantic embedding space. The
intuition: if a person's associative hierarchy is flat, they can
travel further between produced words than someone with a steep
hierarchy. DAT correlates with creative achievement at levels
comparable to RAT ($r \approx 0.2$–$0.3$), takes one minute to
administer, and removes RAT's vocabulary-knowledge confound by
removing anchors entirely.

### Why neither parent works for the scientific-ideation problem

For LLMs, RAT in its canonical form has the wrong response format. A
single correct answer makes performance mostly a function of
retrieval, collapses response variation, and tells us nothing about
whether the model has *many* candidate bridges available — only
whether it can find one. DAT goes the opposite way: by removing
anchors, it cannot probe bridging at all. It measures unconditioned
divergence, which is the right primitive for some creative tasks
(free generation, brainstorming) but not for the scientific-ideation
problem we care about.

The cognitive primitive most implicated in scientific creativity is
bridging between remote concepts, not pure divergence. Three
independent literatures converge on this:

- *Structure-mapping theory* (Gentner 1983, *Cognitive Science*)
  characterizes analogy as alignment between the relational
  structures of two distant domains. The cognitive operation is
  finding the mapping; the vocabulary-space shadow is finding words
  that occupy both domains' neighborhoods.
- *Far-analogy* work (Holyoak & Thagard 1995, *Mental Leaps*)
  distinguishes within-domain mappings from cross-domain ones, with
  the latter the engine of scientific innovation. The empirical
  finding most relevant here: human subjects asked for cross-domain
  analogies produce many candidates of varying quality, not a single
  answer. The cognitive process is divergent, not convergent.
- *In-vivo* studies of working scientists (Dunbar 1995, on
  molecular-biology labs) found that breakthrough analogies in real
  research were systematically long-distance — between fields, not
  within them. Productive scientists generate many candidate
  analogies during problem-finding; the breakthrough analogies are
  the survivors of an internal divergent process.

The vocabulary-space form of remote bridging is producing words
that simultaneously occupy the neighborhood of two distant anchors.
RAT probes this for a single bridge. DAT does not probe it at all.
The natural test is the one that probes it for *many* bridges per
pair, in DAT-shaped form.

### What DRAT inherits and what it changes

From RAT, DRAT keeps the anchor structure but reduces three anchors
to two and converts the task from convergent (find *the* bridge) to
divergent (find many bridges). The reduction from three anchors to
two is partly practical: with three anchors and a divergent
instruction, the intersection region of the embedding space becomes
too sparse to support ten items per response. With two anchors, the
intersection is sparse enough to require remote association but
populated enough to be answerable. The convergent-to-divergent shift
is the substantive change: it exposes the flatness of the model's
associative hierarchy in the bridging region, rather than only
checking whether the hierarchy reaches at least one bridge. This is
the move that makes the test cohere with the analogy literature
above, where the relevant cognitive process is divergent generation
of candidate mappings, not retrieval of a single one.

From DAT, DRAT keeps the embedding-based scoring (because
measurement at population scale needs an automatic, replicable
score) but conditions it on bridge-finding instead of free
generation. The conditioning is what targets scientific ideation
specifically: free divergence is the right measure for free-generation
creativity (closer to creative writing), while conditioned divergence
is the right measure for analogy-driven creativity (closer to
scientific ideation).

Shorthand: divergent two-anchor RAT, scored with DAT machinery. The
cognitive primitive being measured is whether a model has many
remote associates available between two distant fields — Mednick's
flatness hypothesis applied to the kind of bridging that analogy
and scientific creativity actually demand.

## Formal specification

### Stimulus

A pre-registered bank of $K$ ordered concept pairs
$\mathcal{B} = \{(A_k, B_k)\}_{k=1}^K$. Each pair's two anchors are
drawn from distinct top-level scientific divisions
(life / physical / social / formal / engineering). The embedding
distance $d(A_k, B_k)$ falls in a fixed quantile range of the
pairwise-distance distribution over a reference vocabulary of
scientific lemmas — default $[Q_{0.60}, Q_{0.80}]$, distant enough
that bridging is non-trivial, close enough that non-empty bridges
exist.

### Elicitation

For each $(A, B) \in \mathcal{B}$, prompt the model with a fixed
template:

> *Please give 10 words that are as different from each other as
> possible, in all meanings and uses of the words, and each of which
> connects "A" and "B". Only use single nouns. Do not use proper
> nouns. Do not use the anchor words themselves or variations of
> them. Respond with ONLY a JSON array of exactly 10 words.*

The instruction explicitly asks for both divergence (DAT-style: "as
different from each other as possible") and bridging (RAT-style:
"each of which connects A and B"). This makes the elicitation
faithful to what the score actually rewards, rather than letting
the model guess at the implicit task.

Parse the response into a list $W = (w_1, \ldots, w_n)$ with
$n \leq 10$. Discard items not in the embedding vocabulary.

### Score

Fix an embedding $\phi : \text{word} \to \mathbb{R}^d$ (run
separately for GloVe-840B, FastText-2M, and SBERT-mpnet, matching
the dat_eval appendix protocol). Let $d(\cdot, \cdot)$ denote
embedding distance (cosine distance in the default).

**Per-word utility.** A word's utility is its similarity to the
closer of the two anchors:
$$
\mathrm{Utility}(w \mid A, B) \;=\; \max\!\big(1 - d(\phi(w), \phi(A)),\;
                                              1 - d(\phi(w), \phi(B))\big).
$$

Using $\max$ rather than $\min$ is the substantive design choice
that makes the test a unification of convergent and divergent
thinking. A word qualifies as utility-passing if it is anchored in
*either* $A$'s or $B$'s neighborhood, not necessarily both. This is
what allows the score to integrate three response strategies in one
number: divergence within $A$'s neighborhood, divergence within
$B$'s, and bridging between them.

**Per-pair calibration.** For each pair $(A, B)$, sample $N$ random
nouns from a fixed pool (the same pool CDAT uses), compute
$\mathrm{Utility}(w \mid A, B)$ for each, and set the threshold
$\tau_{A, B}$ as the $90$th percentile of the null distribution.

**Per-response gate.** Define the survivor set
$S = \{w \in W \mid \mathrm{Utility}(w \mid A, B) > \tau_{A, B}\}$.
Require $|S| \geq n_{\min}$ with default $n_{\min} = 5$. If the gate
fails, the response scores zero.

**Per-pair score.** With $k = |S|$ survivors and $w_1, \ldots, w_k$
the survivors enumerated in any order:
$$
\mathrm{DRAT}(W \mid A, B) \;=\; \frac{100}{k(k-1)}
  \sum_{i \neq j}^{k} d(\phi(w_i), \phi(w_j)).
$$

This is $100 \times$ mean pairwise distance over the survivor set,
matching the DAT/CDAT convention. The sum runs over all ordered
pairs $(i, j)$ with $i \neq j$, giving $k(k-1)$ terms (each
unordered pair counted twice). The prefactor $100/(k(k-1))$ then
converts the sum into a mean and rescales by $100$.

**Per-model score.**
$$
\mathrm{DRAT}_\theta \;=\; \frac{1}{K} \sum_{k=1}^K
  \mathrm{DRAT}(W^{(k)}_\theta \mid A_k, B_k).
$$

The score rewards conceptual breadth among words that are anchored
in the $A \cup B$ universe. A model that produces only synonyms of
$A$ passes the gate but its survivors cluster, giving low DRAT. A
model that produces only random words fails the gate. A model that
spans both anchor neighborhoods — possibly with bridges in between —
scores highest, because cross-cluster pairs sit further apart in
embedding space than within-cluster pairs.

### Validity and specificity

Same protocol as dat_eval §4. Validity is Pearson
$r(\mathrm{DRAT}_\theta, Y_\theta)$ where $Y_\theta$ is each
benchmark; specificity is the semi-partial
$r(\mathrm{DRAT}_\theta, Y_\theta - \hat Y_\theta^g)$ with
$\hat Y^g$ the OLS fit of $Y$ on the capability stack
$g = (\text{Arena Overall}, \text{MMLU-Pro})$.

## Why DRAT might pick up specificity that CDAT did not

CDAT measures the flatness of a model's associative hierarchy
conditioned on a single concept. A high-capability model with deep
vocabulary near the anchor can score well even with a steep
hierarchy, because steepness within a sufficiently rich neighborhood
still produces enough distinct words to look flat under the metric.
This is the mechanism by which CDAT-A reduces to capability.

DRAT measures the flatness of the associative hierarchy *across two
distant anchor neighborhoods simultaneously*. The score is dominated
by cross-cluster pairwise distances, which are much larger than
within-cluster ones in embedding space. So a model can no longer
score well by being deep in one neighborhood; it has to populate
both, and produce conceptually distinct items within each. The
predictor changes from "vocabulary depth near a concept" to
"vocabulary breadth across two distant concepts". The latter is
plausibly less correlated with general capability, because vocabulary
breadth across distant fields is a different property from raw
vocabulary size.

The construct-validity argument runs through Mednick's theory
applied at the cross-domain scale. A flat associative hierarchy at
the within-concept scale predicts free-generation creativity (DAT,
CDAT, creative writing). A flat hierarchy at the cross-concept scale
predicts the kind of creativity that requires drawing concepts from
multiple fields — exactly what the analogy and scientific-creativity
literatures (Gentner, Holyoak, Dunbar) characterize as the cognitive
basis of scientific innovation. If DRAT measures cross-concept
flatness, its empirical claim is that LiveIdeaBench responds to
this property where the within-concept tests do not.

## Stress-test

Seven failure modes, in roughly decreasing order of how much they
worry me.

### 1. Anchor-bank construction is the test

Whatever 30 pairs we pick *is* the test. A bank weighted toward
biology pairs is a different test than one weighted toward
physics pairs. There is no clean fix; only commitments:

1. Each pair's two anchors drawn from two of the five top-level
   divisions above.
2. $d(A_k, B_k) \in [Q_{0.60}, Q_{0.80}]$ in a reference vocabulary
   of scientific lemmas.
3. $K \geq 30$ pairs for adequate per-model variance under the
   $n = 17$ LiveIdeaBench coverage.
4. Bank pre-registered before any model is scored.

Even then, bank choice carries researcher degrees of freedom. Worth
keeping in scope from the start.

### 2. Capability leakage via vocabulary breadth

Models with larger effective vocabulary produce rarer, more
peripheral words; rarer words tend to be farther apart in embedding
space, raising $\nu$. CDAT has the same issue; we partial it out
via $g$, but the partialing is only as good as the proxy stack.

Empirical check: regress $\mathrm{DRAT}_\theta$ on
$(\text{mean IDF}_\theta, \text{mean word length}_\theta, n_\theta)$.
If $R^2$ is large, DRAT is mostly measuring lexical-rarity capability
and the structural-skill claim is wrong.

### 3. The vocab-space ceiling may be at CDAT, not above it

DRAT extends CDAT to two anchors with a per-response gate.
Conceptually it should improve on CDAT for sci-ideation.
Empirically, it might not. Two reasons:

- $n = 17$ on LiveIdeaBench makes a $+0.10$–$+0.15$ specificity gap
  hard to detect at $p < 0.05$.
- The bottleneck for scientific ideation may be mechanism articulation
  and falsifiable commitment, neither reachable from vocab space at
  all. In that case DRAT pushes specificity from $+0.21$ (CDAT) to
  perhaps $+0.35$ and plateaus, and the right pivot is structured
  outputs.

### 4. Generic process vocabulary on scientific anchors

A model can produce abstract process words ("system", "process",
"transition", "regime", "dynamics") that are weakly close to almost
any scientific concept pair. On word-level anchors these cluster
tightly, and $\nu$ catches them. On scientific-concept anchors they
*are* the legitimate vocabulary of dynamical systems, so they sit
close to both anchors *and* are more dispersed in embedding space.
The per-response gate doesn't help here — these words pass the gate.
The discrimination signal that remains is $\nu$, and on scientific
anchors it does less work.

### 5. Embedding artifacts and shared-corpus confound

GloVe overweights co-occurrence; SBERT is closer to semantic
similarity; FastText sits between. Running all three (the dat_eval
protocol) gets us free triangulation. The harder confound: the LLM
under test and the embedding were both trained on overlapping text,
so high DRAT may partly reflect shared co-occurrence statistics
rather than bridge-finding skill. This applies equally to
DAT/CDAT/PACE.

### 6. Prompt sensitivity

"Connect $A$ and $B$" is underspecified. Different models may default
to topical overlap, causal chain, analogical mapping, or
shared-property. Worth ablating across two framings:

- *Topical:* "Give 10 words that come up in discussions of both $A$
  and $B$."
- *Analogical:* "Give 10 words that could be metaphorically applied
  to both $A$ and $B$."

The analogical framing is closer to scientific creativity per the
Gentner literature; topical is more conservative.

### 7. Zero-bridge anchor pairs

Some $(A, B)$ pairs may have no genuine intersection — the anchors
live in disjoint concept islands. RAT items came pre-validated by
human responses; DRAT items don't. Detect by computing bridge scores
for a reference bank produced by a strong model and discarding items
where the top score is below threshold. This pre-screening step
should also be pre-registered.

## Worked examples

Cosines below are illustrative priors over what SBERT would return,
not measured. Real values from a pilot would shift absolute scores
by $\pm 0.1$ in places without changing the ranking. The point is
to make the score concrete and to show how the unified design treats
each response strategy.

For the worked examples I assume $\tau_{A, B} \approx 0.20$ in
similarity space (the 90th percentile of random-noun max-similarity
against typical scientific anchors), and $n_{\min} = 5$. DRAT scores
are reported as $100 \times$ mean pairwise distance per the formula
above, matching DAT/CDAT convention.

### Word-level anchors: $A=$ "immune system", $B=$ "supply chain"

Six response strategies, listed roughly cheap-to-best:

| strategy | example words | survivors | $\bar{d}(S)$ | DRAT |
|---|---|---|---|---|
| DAT-mode (ignores anchors)  | volcano, jazz, harpoon, calculus, broccoli… | 0/10 | — | 0 |
| Generic-hypernyms cheat     | system, process, function, mechanism, factor… | 10/10 | 0.30 | 30 |
| CDAT-mode for $A$           | antibody, lymphocyte, pathogen, infection, immunity… | 10/10 | 0.40 | 40 |
| CDAT-mode for $B$           | logistics, inventory, warehouse, shipping, depot… | 10/10 | 0.40 | 40 |
| Pure-bridges (RAT-mode)     | bottleneck, surveillance, redundancy, network, threat… | 10/10 | 0.55 | 55 |
| Spanning ($A \cup B$)       | 5 from $A$-cluster + 5 from $B$-cluster | 10/10 | 0.65 | 65 |

The ranking under the new design is intentionally non-trivial.

The DAT-mode failure scores zero — the max-utility gate filters out
words anchored in neither neighborhood. This is the only catastrophic
failure mode preserved from the previous design.

The CDAT-modes pass the gate (their words are highly anchored in one
side) but score moderately. Within-cluster diversity is bounded by
the conceptual radius of one anchor neighborhood, which is small
compared to the cross-anchor distance.

The pure-bridges strategy — what the previous min-utility design
specifically rewarded — does well but loses to the spanning strategy.
Bridges live in the intersection region, which is denser than either
anchor cluster on its own; the spanning strategy puts half its mass
in $A$'s cluster and half in $B$'s, and pairs across clusters are
further apart in embedding space than pairs within bridges.

This is the design intent. The unified test rewards either pure
divergent fluency *within* one anchor (the CDAT face), pure
convergent retrieval *between* both anchors (the RAT face), or the
integrated strategy that spans both. Each of these is a face of the
underlying associative-hierarchy flatness; the spanning strategy is
the one that exposes flatness at the cross-domain scale, which is
the construct DRAT is most directly designed to probe.

### Scientific-concept anchors: $A=$ "phase transition", $B=$ "neural network training dynamics"

| strategy | example words | survivors | $\bar{d}(S)$ | DRAT |
|---|---|---|---|---|
| DAT-mode                    | volcano, jazz, harpoon, calculus, broccoli… | 0/10 | — | 0 |
| Generic-process cheat       | transition, change, dynamics, regime, state… | 10/10 | 0.40 | 40 |
| CDAT-mode for $A$           | criticality, order-parameter, susceptibility, fluctuation, universality… | 10/10 | 0.45 | 45 |
| CDAT-mode for $B$           | gradient, loss, optimizer, batch, learning-rate… | 10/10 | 0.40 | 40 |
| Pure-bridges                | critical-point, energy-landscape, basin, fluctuation, scaling… | 10/10 | 0.50 | 50 |
| Spanning ($A \cup B$)       | 5 from $A$-cluster + 5 from $B$-cluster | 10/10 | 0.55 | 55 |

The compression on scientific anchors persists in a different form:
the generic-process cheat now scores within 30% of the spanning
strategy ($40$ vs $55$). This is the same failure mode as before —
generic process vocabulary lives close enough to both anchors to
pass the gate, and is moderately dispersed in embedding space so
its $\bar{d}$ isn't tiny. The discriminating signal is still
$\bar{d}$, and on scientific anchors it does less work than on
word-level anchors.

### Three readings of the worked examples

The first is that the unified design produces a smooth ranking
across response strategies, with no catastrophic-failure zone except
for true DAT-mode (no anchoring). This is the unification working as
designed: CDAT-mode and RAT-mode are no longer treated as failures.

The second is that the *spanning* strategy strictly dominates on
both pair types. This is the design rewarding the integrated
convergent-divergent strategy over either pure component, which is
the central claim of the motivation.

The third is that the genuine-vs-cheat compression on scientific
anchors persists. The new design doesn't fix this; the structural
issue is embedding geometry, not the scoring rule. Running both
bank styles (word-level and concept-level) and reporting them
separately is the honest response.

## Open design choices

1. **Per-word threshold.** Quantile of the random-noun null. $90$th
   percentile is a guess; $75$th and $95$th are worth ablating.
2. **Minimum survivors $k_{\min}$.** Default $3$; $5$ is more
   conservative but bites on responses with sparse bridges.
3. **Aggregator.** $\min$ vs $\mathrm{mean}$ for $b(w; A, B)$.
   $\min$ is principled; $\mathrm{mean}$ is less noisy.
4. **Output cardinality.** $10$ (matching DAT) vs $7$.
5. **Token granularity.** Single words (works for GloVe/FastText)
   vs short phrases (works for SBERT, breaks single-token embeddings).
6. **Anchor bank construction.** Hand-curated, auto-mined from a
   scientific concept graph, or auto-generated and validated by a
   held-out judge. Auto-mined with a fixed seed is most reproducible.
7. **Difficulty quantile.** $[Q_{0.60}, Q_{0.80}]$ is a guess; a
   small prior pilot would tell us the right setting.

## Decision rule

Two stages.

**Pre-pilot.** Hand-craft 4–5 scripted responses per anchor pair on
a small bank ($K \approx 5$). Score under all three embeddings.
Verify that the rank ordering — genuine $>$ partial $>$ generic-cheat
$>$ CDAT-mode $>$ DAT-mode — holds in every embedding on both
word-level and concept-level pairs. If the ordering breaks anywhere,
the metric needs revising before any model is queried. No API cost.

**Pilot.** $K = 30$ pairs, $10$ models from the existing dat_eval
pool, all three embeddings. Roughly $300$ API calls,
under \$5 at typical OpenRouter rates. Resolution criteria:

- If $r(\mathrm{DRAT}, \text{LIB}) \mid g$ exceeds $+0.30$ on at
  least two embeddings, scale to the full $54$-model pool.
- If specificity is below $+0.20$ on all three embeddings, the
  vocab-space hypothesis is wrong for scientific ideation. Move to
  structured outputs (mechanism + falsifier) and give up the pure
  embedding score.
- Intermediate result ($+0.20$ to $+0.30$): try the analogical
  prompt framing before scaling.

**Sanity check.** Administer classic RAT items to the pilot models
as a control. If RAT accuracy correlates with $\mathrm{DRAT}_\theta$
across the pool, the cognitive lineage from RAT is empirically
grounded. If it doesn't, the framing has drifted further from RAT
than the design claims, and we should weaken the construct-validity
argument accordingly.
