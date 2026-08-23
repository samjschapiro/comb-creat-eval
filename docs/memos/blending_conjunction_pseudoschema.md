# Failure mode: conjunction pseudo-schemas in conceptual blending

**Found 2026-08-22**, first fusion-blending run (`data/kg_creat/kombine_blend_v2/`, temp 0.9,
30 cross-domain pairs = the analogy pairs, 3 scored models: gpt-4o-mini, gemini-flash-lite,
llama-3.3-70b). Fusion utility judge = `openai/gpt-oss-120b` with a hard gate on the generic space.

## The finding

When asked to fuse two concepts into one blend (Fauconnier–Turner), **LLMs overwhelmingly fake the
generic space by *concatenation* rather than *abstraction*.** They produce a "shared schema" that is
one salient property of input A stapled to one salient property of input B with an "and" — a schema
that fits *neither* input cleanly, rather than an abstraction both genuinely instantiate.

~80% of blends fail the generic-space gate, and this is the dominant reason. Utility (fusion) pass
rates: llama-3.3-70b 33%, gemini 23%, gpt-4o-mini 13%.

## Examples (the conjunction move)

| pair | pseudo-schema emitted | why it's a conjunction |
|---|---|---|
| Photosynthesis + Penicillin | "a biological agent that **harnesses energy and exhibits therapeutic effects**" | energy←photosynthesis, therapeutic←penicillin; photosynthesis isn't an "agent", penicillin doesn't "harness energy" |
| Northern Lights + honey bee | "illuminates **and** influences its environment through collective behavior" | illuminates←aurora, collective←bees |
| Moon + Diamonds | "a **luminous** celestial body with **precious** qualities" | luminous←Moon, precious←diamond |
| Microscope + Mirror | "enhances perception through **reflection and magnification**" | reflection←mirror, magnification←microscope |

A second, smaller failure texture is the opposite pole — **vacuous over-generalization**: a schema
true of both but empty ("a system with energetic transformations" for solar system + radioactivity;
"a process that amplifies and transforms patterns" for golden ratio + fission).

## Contrast: what a genuine shared schema looks like (the 20% that pass)

- Telephone + Roman Empire → "a system for **rapid, centralized communication and control across a
  vast territory**" (both literally are this).
- Orchestra + Chaos theory → "a dynamic system where components interact to create emergent patterns".
- The Western + Mushrooms → "a **territorial, decentralized network of independent nodes**" (→ "Mycelial West").

The tell: a real generic space is a single abstraction *both* inputs instantiate; a pseudo-schema is a
list of per-input properties joined by "and".

## Why it matters

- It is a **specific, mechanistic** account of *how* LLM conceptual blending fails — not "models are
  bad at blending" but "models substitute concatenation for abstraction." This is the analogue, for
  blending, of the mode-collapse / world-model-break failure modes TwistBench found for plot twists.
- It is **model-discriminating** (utility gate separates the three models) and grounded in a real
  cognitive-science construct (F&T generic space).
- Even a pair a human fuses cleanly (Photosynthesis + Penicillin → "a molecular mechanism mediated by
  a reactive agent") defeated **all three** models — evidence the default is structural, not incidental.

## Two-axis payoff (related observation)

Utility (is it a real fusion?) and emergent creativity (does the fusion generate new structure?) come
apart: some fusions pass utility but score **0** emergent (immune+F1 "Adaptive Racing Protocol" —
coherent but inert), while others are rich (Imperial Telegraphic Network: "its failure creates regional
information vacuums that can foster secessionist movements"). Utility rank ≠ emergent rank: **llama
passes the most, gemini generates the deepest.** Keep both axes.

## Caveats before this is a headline

1. **Judge boundary noise (~10–20%).** Telephone + Roman Empire: gpt-4o-mini's schema failed while
   gemini's/llama's near-identical schemas passed. A 3-judge fusion vote would quantify/tighten this.
2. **Emergent judge is somewhat lenient** — it credited "combines visual expression with meditation"
   (a restatement of the combination) as emergent. The operational test needs tightening against
   restatements.

## Next steps to harden the finding

- 3-judge fusion vote on the current 90 blends to measure boundary noise.
- Tighten the emergent judge against restatements.
- A small human-label pass on conjunction-vs-genuine to validate the judge's schema calls.
- Then scale (more models, more pairs) — the conjunction rate per model is the headline number.
