# 2026-08-22 — Kombine: four-criteria redesign, flat entity pool, pilot, human study

## Summary
Large session on the **kg_creat / Kombine** benchmark: reframed all three tasks (association,
analogy, blending) around four scoring criteria; rebuilt the elicitation format (open-ended,
array-of-objects, emergent "inferences"); **dropped the CREATE-style seed-BFS knowledge graph
entirely** for a flat curated domain-balanced entity pool; ran a paid pilot (4 models elicit, 3
models score) and diagnosed real findings; and built a jsPsych **human generation** experiment.
Paper (Overleaf `kg_creat-iclr`) was updated throughout.

## Paper (papers/kg_creat-iclr — pushed to Overleaf)
- **Scoring table** `tab:scoring`: consolidated per-task scoring into one table; defined the LLM judge
  `J`; made **originality item-specific** (`p̂_a`, convention stated in Preliminaries); font/resizebox.
- **Renamed "emergent novelty" → "emergent creativity"** globally (kept F&T's "emergent structure").
- **Added the originality dimension** (inverse frequency of non-anchor concepts/relations vs the pool).
- **Rewrote the three task motivations** from primary sources, each motivating the four criteria:
  Mednick remote-association + Uzzi(2013)/Shi–Evans scientometrics; Gentner structure-mapping +
  Kepler + Dunbar in-vivo; Koestler bisociation + Fauconnier–Turner emergent structure + Thagard.
- **Consolidated** the §3 intro paragraph + Preliminaries (removed duplicated scoring description).
- **Background** analogy-vs-blending distinction (structure-mediated vs concept-mediated; virus example).
- **Appendix: full task prompts, one per page** in TwistBench `promptbox` style (added `inconsolata`
  + T1 fontenc; reflowed prose so lines don't break mid-sentence).
- Bib added (VERIFY-wrapped): `gentner_kepler, uzzi2013, dunbar1995, hope2017, thagard1984,
  runco_jaeger2012, holyoak_thagard1995`.

## Benchmark redesign (src/kg_creat)
- **prompts.py**: all three tasks now **open-ended** ("as many as you can"), **array-of-objects**
  output; every item carries an emergent **`inferences`** field; **criteria-aligned framing**
  (TRUE→utility, REMOTE→surprise, UNCOMMON→originality, GENERATIVE→emergent creativity); analogy
  fields renamed `source/target → path_a/path_b` (neutral, non-directional); symmetric generative
  inferences (both u and v).
- **parse.py**: `parse_pairs`, `parse_items` (array + inferences), truncation salvage.
- **run_elicit.py**: unified item parsing; stores `items` (paths+inferences), flat `paths`, `pairs`.
- **judge.py**: **`judge_emergent` + `count_emergent`** — the previously-missing emergent-creativity
  scorer (an inference counts iff **true AND licensed by the whole but not any single part**).
- **score.py**: judges **every** pair (was only the first), added the emergent-judge phase, **batched
  factuality** (`FACT_BATCH=10`), `exclude_models`, and a printed **verified-genuine per-prompt
  distribution** (scarcity check).
- **aggregate.py**: `verified_genuine`, `n_items`, `emergent_mean` per mode.

## Entity pool — dropped BFS entirely (the big pivot)
- Diagnosed the CREATE-style **seed-BFS graph** as (1) **person-biased** — its top-28
  frequency-derived relation vocabulary is dominated by biographical/family relations, so
  degree-filtering yields ~60% humans and their obscure relatives (Bach's 20 children, etc.) — and
  (2) **unnecessary**: the model connects entities from its own knowledge, the graph is *never
  traversed at inference*, and we explicitly reject CREATE's connectivity pre-filtering.
- Built **`sample_flat.py`** + **`data/kg_creat/entities_curated.json`**: a flat, domain-balanced,
  concept/object/idea-rich pool (~283 entities, 20 domains) with **cross-domain stratified**
  sampling. **No graph, no `min_degree`, no relation vocabulary.**
- (Interim before abandoning BFS: expanded `build_gc.yaml` 51→139 seeds as `gc_domains_v3`.)

## Pilot (paid, ~$2.1 total)
Configs: `configs/kg_creat/kombine_pilot_{sample,run,score}.yaml`.
- **Elicitation** (~$0.5): 4 cheap models × 90 prompts (30/task) × 3 temps on the curated pool.
- **Scoring** (~$1.6): 3 models (dropped llama-3.1-8b — only 54% parsed), batched factuality.
- **Findings:**
  - Format works end-to-end; **quality scales with model** (llama-70b > gpt-4o-mini > gemini-lite).
  - **Factual validity is the clean discriminator** (association 52/38/28%, blending 20/5/2%).
  - **Scarcity confirmed & quantified** (verified-genuine per prompt): association **6.0** (not
    scarce), analogy **1.5** (a third yield 0), blending **0.5** (**78% yield 0**).
  - **Emergent creativity is flat** (~0.5) across the two strong models → the **emergent judge is too
    lenient** and doesn't discriminate; needs tightening.
  - **Surprise is nearly constant** — it's item-driven (fixed cross-domain pairs), not a model signal.
  - **Blending needs polysemy-friendly anchors** — arbitrary entities almost never admit a 2nd sense.

## Human generation experiment (llm_creativity_mech_interp/src/experiments/kombine_generation/)
- jsPsych 7.3.4 **generation** study: 5 association + 5 analogy + 5 blending (blend anchors are
  genuinely **polysemous words**: virus/spring/current/crane/bank).
- Follows the template repo conventions (consent → instructions → 15 trials → completion, Prolific
  params + data POST). **AUT consent form** used (verbatim except the task-description lines, which
  must accurately describe the typed-generation task). **AUT font/UI palette.** **No self-report
  ratings** (surprise/factuality are measured directly). Files: `index.html`, `js/experiment.js`,
  `js/stimuli-data.js`, `README.md`.

## Other
- `docs/tracks/kg_creat/primary_sources_motivations.md` — verified primary-source lit review for the
  three mechanisms.
- Memory saved: `paper-terms-not-code-jargon` (never use `sat`/`R_emit`; use surprise/factual/
  originality/emergent creativity).
- Earlier in-session (pre-compaction): a Claude artifact concept-pairs mockup + `human_study/`
  (jsPsych 8, standalone) — superseded by `kombine_generation`.

## Open questions / next steps
1. **Tighten the emergent judge** (too lenient; doesn't discriminate) — or reconsider self-emitted
   inferences vs a post-hoc diff.
2. Decide **"as many as possible" vs a bounded ask** — the scarcity confound hits analogy/blending.
3. **Implement the originality (inverse-frequency) scorer** — defined in the paper, not yet in code.
4. **Curated polysemy anchor set for blending** in the LLM pipeline (as done for the human study).
5. **Deploy** the human experiment (Vercel + backend + Prolific).
