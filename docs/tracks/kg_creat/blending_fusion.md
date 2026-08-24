# Blending as conceptual fusion (design)

Reframed 2026-08-22. Supersedes the earlier **polysemy/antanaclasis** blending
(one anchor, two senses of the same word), which failed: on the pilot it degenerated
into a dictionary exercise (list word-senses of "Trade") with trivial "both involve
exchange of value" inferences — no new concept, no emergent structure, ~78% zero yield.

## The task

Given **two concepts** `u`, `v` (a cross-domain pair — the *same* pairs analogy draws),
fuse them into a **single new blended concept** that is both at once, and read off the
structure the fusion generates. Grounded in Fauconnier & Turner conceptual integration:
two input spaces → a shared generic space → a blend with its own **emergent structure**,
irreducible to either input (and often false/impossible of each).

**One blend per pair** (convergent), unlike association/analogy which diverge ("as many
as you can"). Two specific concepts fuse into essentially one coherent blend; generativity
lives not in the *count of blends* but in how much the single blend generates — its
emergent structure. This is F&T's "running the blend."

## Artifact (a single JSON object)

```json
{
  "concept": "jazz pantheon",
  "generic_space": "an oral, performed tradition transmitted by improvised retelling",
  "structure": [
    ["jazz pantheon", "improvises on", "standards"],
    ["jazz pantheon", "venerates", "archetypal figures"]
  ],
  "emergent": [
    "recording a standard freezes it the way writing down an oral epic kills it — so no version is definitive"
  ]
}
```

- `concept` — name of the fused concept.
- `generic_space` — one phrase naming the shared schema (what makes u,v fusable). Must be
  specific; "both exist / involve change" is vacuous and fails the utility judge.
- `structure` — triples describing the blend, drawing on **both** inputs (a star around the
  blend, not a chain — so blending well-formedness does NOT require a continuous path).
- `emergent` — statements each true of the blend but true of **neither input alone**. The
  prompt states this operational test verbatim so the model self-filters before the judge.

Structurally the blend item = an **association-shaped item** (one path = the `structure`
star) plus two string fields, with `emergent` mapping onto the `inferences` slot.

## Scoring (four criteria)

| criterion | blending |
|---|---|
| **utility** | **judge-only**: a genuine, coherent fusion — real (non-vacuous) generic space, selective projection from *both* inputs — vs. a forced mashup. A blend is a novel concept, so literal factuality of the blend is NOT the gate (F&T: elements "false or impossible in both inputs" are a feature). Factuality applies only to the input-grounded structural claims. Hard gate: vacuous `generic_space` → utility fails. |
| **surprise** | mean of cos(`u`, `g`) and cos(`v`, `g`), the semantic distance from each anchor to the blend's `generic_space` `g` — a proxy for how far the shared schema abstracts away from the concrete inputs. |
| **originality** | inverse item-frequency of the bridging concepts introduced (pending scorer). |
| **emergent creativity** | count of `emergent` statements the judge confirms are (a) coherent / follow from the blend, (b) present in neither input alone, (c) non-trivial. **Not** "true in the real world" — blends may be fictional. Padding doesn't pay: only verified statements count. |

## Sampling

Blending reuses the **analogy** cross-domain pairs verbatim (same `(u,v)`), so analogy and
blending run on identical inputs — isolating the map-between vs. fuse-into distinction and
directly supporting Table 1's claim that the two license *different* emergent creativity.

## Resampling

One blend per *prompt*, but N samples per pair across temperatures — so we can measure
whether a model produces the *same* fusion every time (blending mode-collapse), the same
signal TwistBench found for twists.
