# plot_twist — design

Full method/scoring/eval spec for **conceptual-space axiom modification (CSAM)** —
an inference-time method that elicits *transformational* creativity (plot twists)
from LLMs by making them externalize a story's conceptual-space DAG and perform a
controlled axiom flip. See [progress.md](progress.md) for status and roadmap.

Working method name: **CSAM** (provisional). Working twist-quality metric: **PTC**
(plot-twist creativity), used here as the *analysis instrument*, not a standalone paper.

## Motivation / the construct

The lab's portfolio measures two of Boden's three creativity modes — **exploratory**
(DAT/CDAT/PACE semantic distance, DRAT) and **combinatorial** (comb_eval, kg_creat
constrained pathfinding). The third, **transformational** creativity — restructuring
the conceptual space itself — is unrepresented and is the hardest to measure and to
elicit. A **plot twist** is the cleanest everyday instance: the reveal forces the
reader to *reinterpret prior events under new rules*, i.e. it modifies an axiom of the
reader's world-model. The classical quality criterion, **"surprising yet coherent"**
(Aristotle's *peripeteia* + *anagnorisis*), is exactly **novelty × appropriateness** —
the same two-factor structure used across the lab's other tracks.

Documented deficiency this method targets: LLM stories are low plot-diversity and their
twists are poorly foreshadowed
([Echoes in AI, arXiv 2501.00273](https://arxiv.org/pdf/2501.00273);
[Human-Level Narratives?, arXiv 2407.13248](https://arxiv.org/html/2407.13248v1)).

## Theoretical grounding — SBV graphical theory → story DAG

Built directly on the lab's own
[Transformational Creativity in Science: A Graphical Theory
(Schapiro, Black, Varshney, ICCC 2025 best short paper, arXiv 2504.18687)](https://arxiv.org/abs/2504.18687).

SBV: a conceptual space is a finite DAG `S = (V, E)`; vertices are constraints (subsets
of a formal language), `(u,v) ∈ E iff u ⊆ v` (`u` further-constrains `v`); **axioms**
`𝒜` are sink nodes (self-justifying); **rules** depend on them; **artifacts**
`a = (h, σ, w)` are generated from a support set `σ ⊆ V`. Key quantities:
`depends(v) = {u : path u→v}` (everything downstream relying on `v`) and
`T_mod(v) = |depends(v)|` (transformative potential), with Thm 4:
`argmax_v T_mod(v) ∈ 𝒜` — modifying an axiom is maximally transformative.

The story DAG instantiates this on the **reader's world-model** at the pre-twist point:

| SBV | Story |
|---|---|
| Conceptual space `S` | Reader's working model of the story world right before the reveal |
| Axioms `𝒜` (sinks, self-justifying) | Load-bearing reader assumptions ("narrator reliable", "X is dead", "present day", "A is the hero") |
| Rules (depend on axioms) | Derived inferences ("A will save B because A is the hero") |
| Artifacts `(h, σ, w)` | Narrated scenes/events, each with a support set `σ` of assumptions making it coherent |
| Edges (logical dependency) | Inferential dependency among assumptions/events |

**A plot twist = an axiom modification** `a → a'` (SBV's Thm-4 operation applied to the
reader's space). This yields two *structural* twist metrics:

- **Surprise / magnitude** `= T_mod(a*)` — how much prior story must be re-evaluated when
  the flipped axiom propagates downstream. Memorization-proof (structural, not surprisal).
- **Coherence / retro-fit** `= preservation(a')` — fraction of prior artifacts whose
  support set `σ` remains satisfiable (non-contradicted, ideally *better* explained) after
  the flip. Random twist → support sets break (low preservation); predictable ending →
  `T_mod ≈ 0`; genuine twist → high `T_mod` × high preservation.

Unifying payoff: all three creativity modes are operations on the **same graphical
substrate** — comb_eval/kg_creat = *pathfinding* in the space, DAT/DRAT = *semantic
distance* in the space, CSAM = *axiom modification* of the space.

## The method — CSAM (the paper's contribution)

Inference-time, no training. Given a seed prompt `x`:

1. **Build `G`.** Model emits a DAG of the story's conceptual space with *rich axioms* —
   an explicit set `𝒜` of load-bearing reader assumptions plus dependent rules, in a
   parseable format (typed nodes + dependency edges).
2. **Narrate `≤ t`.** Model writes the story consistent with `G` up to a cut point `t`
   (the pre-twist setup), planting artifacts whose support sets reference `𝒜`.
3. **Axiom flip `G → G'`.** Model modifies a single axiom (or a small set) `a → a'` —
   the controlled transformational move. Choice of which axiom is itself a lever
   (model-chosen vs forced high-`T_mod` vs random — see ablations).
4. **Narrate `t' > t`.** Model continues/reveals under `G'`, re-contextualizing the
   planted artifacts so the prior text remains valid (preservation) while the reveal is
   surprising (high `T_mod`).

The explicit `G`/`G'` makes the twist **inspectable**: we can verify which axiom flipped,
compute `T_mod(a*)`, and check artifact preservation — turning a fuzzy "creativity"
intervention into a measurable, ablatable mechanism.

## Scoring

**Primary: blinded human evaluation** (CSAM vs each baseline) — see protocol below.

**Secondary / analysis instrument (PTC):** structural metrics from the emitted DAG —
`T_mod(a*)`, `preservation(a')`, and their combiner (product / geom-mean / min, ablated).
The key analysis claim: `T_mod(a*)` predicts human-rated **surprise** and
`preservation(a')` predicts human-rated **coherence** — i.e. the method
works *because* it flips high-`T_mod` axioms while preserving artifacts. This absorbs the
earlier "PTC metric" idea as the explanatory mechanism, not a separate paper.

## Baselines (compute/token-matched — the make-or-break confound)

All baselines matched to CSAM's total token/inference budget, else "it's just more
test-time compute" sinks the paper.

- **Free-form plan → write → twist** (prose plan, *no graph*) — isolates whether DAG
  structure helps over a text plan.
- **Reasoning/thinking mode**, equal thinking-token budget.
- **Direct prompt** "write a story with a surprising twist" + **self-refine**
  ("make the twist more surprising yet coherent").
- **Temperature sweep** — included but weak (buys surprise by destroying coherence);
  beating it alone is not the headline.

## Ablations (isolate the active ingredient)

- **Graph vs prose plan** — does the DAG do anything a text plan doesn't?
- **Explicit axiom-flip vs "just continue with a twist"** — is the *controlled
  modification* the key?
- **Model-chosen vs forced high-`T_mod` vs random axiom flip** — does flipping a deep
  axiom matter? (Predicts the `T_mod` → human-surprise link.)
- **Single vs multiple axiom modification** — dose-response on `T_mod`.
- **Combiner** for PTC (product / geom-mean / min) — DRAT max/min/avg pattern.
- **Extractor/model** invariance — DAT GloVe/FastText/SBERT-style robustness check.

## Datasets

- **WritingPrompts** ([Fan et al. 2018](https://arxiv.org/pdf/2212.04634), 303k
  prompt→story) — **primary seed prompts** (the credible default). Contamination caveat:
  in pretraining, so judge *generation* quality by humans, not surprisal of existing text.
- **Synthetic controlled leg** (TinyStories-style, "plot twist" feature) — ground-truth
  axiom/`T_mod`; the pre-registered, hard-to-deny evidence.
- **Flawed Fictions** (controllably induces plot holes) — the incoherent **negative
  control** ("random ending") and a preservation probe.
- **STORIUM** ([Akoury et al. 2020, EMNLP](https://aclanthology.org/2020.emnlp-main.525.pdf)) —
  structured story cards; cite as precedent for structured story representation, optionally
  align `G` to its scene structure.
- **WHODUNIT** ([2502.07747](https://arxiv.org/pdf/2502.07747)) — twist-bearing human
  narratives with ground-truth reveal structure (validation, not generation).

## Human evaluation protocol

- **Blinded pairwise forced-choice** (CSAM vs each baseline), randomized order,
  **length/fluency-matched** to kill the "longer = better" confound.
- Dimensions: **surprise**, **coherence**, joint **"surprising-yet-
  coherent"**, **overall preference**, + a **fluency control**.
- Annotators: crowd **+ a few expert writers**; report **inter-annotator agreement**
  (Krippendorff's α); analyze via **Bradley–Terry win-rates** with bootstrap CIs.
- **Pre-register** dimensions + analysis; hold the headline to **p < .001**
  (per [writing_advice.md](../../writing_advice.md)).
- Corroborate with PTC (secondary automatic signal).

## Faithfulness / verification

Reuse the comb_eval/kg_creat verification ethos: check (a) the story realizes `G`,
(b) the continuation respects `G'` and the flipped axiom, (c) prior artifacts stay
non-contradicted. Without this a reviewer asks "did the model even use the graph?"

## Risks (red-team)

- **Compute confound** — neutralized only by token-matched baselines (above). Highest risk.
- **DAG-extraction reliability** — report inter-extractor agreement; lean on the synthetic
  controlled leg for the load-bearing claim.
- **"Is a twist transformational or exploratory?"** — answer via reader-reinterpretation
  (the flip modifies a reader axiom with large `depends`), not author intent.
- **Contamination** — famous twists in pretraining deflate surprisal; judge generation by
  humans + use fresh/synthetic stimuli.
- **Graph may hurt fluency** — measure via the fluency control; if so, that's a finding.
- **Scope** — frame as eliciting *transformational creativity / plot twists*, not general
  creativity (the only thing the theory/metric speak to).

## Reuse map

- `dat_eval` validity/specificity pipeline + embedding infra (for any PTC↔benchmark leg).
- `comb_eval`/`kg_creat` verification + graph-handling ethos.
- OpenRouter client (`src/*/llm.py`), budget-capped runners, `scripts/safety/`.
- SBV graphical theory (`T_mod`, `depends`, axiom = sink) as the structural metric.
