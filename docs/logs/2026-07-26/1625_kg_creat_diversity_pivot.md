# 2026-07-26 · kg_creat — creativity reframe → arbitrary-entity + diversity redesign

**Summary.** Two arcs. (1) Finished the Regime-A creativity writeup: reframed the result around a
two-mechanism decomposition, dropped the ordering constraint as confounded, rebuilt figures to
Nature spec, and made "report the sample frame" a standing convention with a reproducible datasheet.
(2) A larger methodology pivot for the paper's real benchmark: arbitrary-entity endpoints (drop the
connectivity filters), blending reworked to true antanaclasis, set-level diversity via M=10
resampling with a systematic temperature sweep, CREATE-parity dataset size (~931 instances), a grown
G_c, and an explicit "be creative" instruction in every task. Ended with a cheap 3-model diversity
pilot (Stage 1) firing.

---

## Arc 1 — the creativity writeup (finished)

**Two-mechanism decomposition is the headline.** Creativity factorises exactly:
`E[R·U] = R_valid × adherence`. A constraint moves the two factors in opposite directions — it
**raises** the novelty of successful paths (+9–11 %) and **lowers** the adherence rate (−27 to
−50 %); net creativity falls because adherence dominates. The novelty gain is **causal, not
survivorship** — it shows up in `R_emit` over *all* emitted paths (no success filtering) and within
fixed endpoints (paired ΔR_valid). The novelty lever is specifically "go somewhere you usually
wouldn't": the one constraint requiring a *common* relation buys ~0 novelty and is the worst cell;
categorical redirects the waypoint without restricting vocabulary, buys the most, and is the only
cell that can net positive (2/8 models). Report:
[docs/reports/2026-07-22_kg_creat_creativity/](../../reports/2026-07-22_kg_creat_creativity/report.md).

**Dropped ordering.** Its −86 % creativity drop was a construction artifact, not a sequencing
result: target derived as the *reverse* of the natural class order made it (a) a conjunction (~12 %
of unconstrained paths contain both target classes), (b) anti-natural (89 % of co-occurrences are in
the reverse order), (c) sometimes infeasible (8/30 bundles unsolved by all 8 models). Only 11.5 % of
its failures are true order inversions. Removed from figures, reports (kept as Appendix A rationale),
`make_pass2`, and the taxonomy docs; the reported set is 4 constraints.

**Fixed an over-lenient scorer.** `_entity_matches` used bidirectional substring matching, so a path
ending at `australia group export controls` counted as reaching `Australia Group`. 6.4 % of
well-formed paths had inexact endpoints, ~81 % genuinely wrong. Tightened to equality up to a
trailing parenthetical; re-derived offline (no re-judging) — 313 paths moved, **no conclusion
changed**.

**Figures to Nature Machine Intelligence spec.** Read the artwork guide from source: 88 mm
single-column, Arial 5–7 pt, vector PDF, RGB. Built `fig_creativity_mechanism` (the two-mechanism
bars, no false additivity), `fig_creativity_by_constraint` (boxplot over the 8 per-model paired
effects + Holm-corrected significance stars). Verified 87.9 mm / ArialMT from the emitted PDF.

**Standing convention: report the sample frame.** Added `src/kg_creat/scripts/datasheet.py` — emits
dataset dimensions, sampling frame, and per-finding n's from the scored data (never hand-typed).
Surfaced the main generality limit of the *old* run: "30 bundles" was only 21 distinct entities
(UN in 11/30). Codified in `docs/repo_usage.md` (Reports section) + memory
`report-findings-with-sample-frame`.

---

## Arc 2 — the benchmark redesign (in progress)

### Arbitrary-entity endpoints (drop connectivity filters)
The connectivity/biting filters (`min_degree≥4`, `min_routes≥6`, `require_constraints≥3`) selected
hub entities and caused the geopolitical skew. Verifying a path exists between `u` and `v` selects
for the *unsurprising* pairs and defeats combinatorial creativity; and biting is already guaranteed
post-hoc by `make_pass2`'s baseline-derived targets. New `sample_random_bundles` draws random
arbitrary pairs (like the analogy task), keeping only a light per-node prominence floor.
`sampler.strategy = random | matched`. Validated with an obscure-pair probe: models attempt real
paths and the judge adjudicates the tail well (precise catches on Bavarian orders, Golden-Fleece
membership, KBE-vs-Knight-Bachelor). Baseline success becomes a *measured* quantity.

### Blending reworked to true antanaclasis
The prior "two structures from one anchor" was just parallel facts about one sense — **not a blend**.
Per the C6 'Boxer' figure (Boxer = athlete vs dog), a true blend hinges on the anchor word carrying
two genuinely distinct **senses**. Rewrote the prompt (fixed anchor; find a *valid polysemy*; Boxer
example embedded; strict antanaclasis, no homophones) and the judge (verify two distinct senses of
the same word, not two facts). Smoke test: Turkey (country/bird) passes; judge correctly rejected
Denmark (fictional-setting ≠ distinct sense), Internet ("fishing net" pun), Amazon rainforest
(fabricated sense). Models fabricate rather than abstain on anchors with no polysemy — the judge
catches it; that's the intended difficulty.

### Set-level diversity + systematic decoding
Diversity is central to the paper and needs a *set*, so we resample each prompt **M=10** times
(Regime B returns only one structure per call, so it needs resampling; k-paths alone don't suffice).
Key cost insight: **diversity is free** (embedding measure); only the judge (utility) scales with M.
- `run_elicit` gained `eval.n_samples` (M) and `eval.temperatures` (list); every response tagged
  with `(temperature, sample_idx)`.
- `src/kg_creat/diversity.py` + `scripts/compute_diversity.py`: set-level `D` = mean pairwise
  embedding distance over a prompt's M samples per temperature, over **all** items and
  **valid-only** items (item = a path for Regime A, a whole analogy/blend structure for Regime B).
- Temperature sweep **{0.7, 0.9, 1.0}** (systematic decoding). Validated on a tiny M=3 run:
  diversity rises with temperature (D_all 0.54→0.60), the expected sanity check.

### CREATE-parity size + grown G_c
Target **~931 instances** (CREATE has 931 queries; instances = distinct prompts, M is depth not
size). Split: **120 bundles × 5 cells + 165 analogy + 166 blending ≈ 931**. The old G_c had only 424
prominent entities → grew it (`gc_domains_v2`): 22→51 seeds across 16 domains, radius 2,
max_neighbors 40→50. Result: **4,891 nodes, 1,066 prominent (deg≥3)** — enough for ~931 pairs without
heavy reuse. Caveat: domain balance is uneven (politics/geography dominate; biology/tech/food thin),
because political/geographic entities are densely interlinked in Wikidata.

### Domain tagging
Every spec carries domains as reference metadata, **never shown to the model** (verified no leak into
prompt text): Regime A / analogy → `domain_u`/`domain_v`/`cross_domain`; blending → `domain_u` = the
pivot's domain.

### Explicit "be creative" everywhere
Caught that Regime A (CREATE's prompt) asked for strong + diverse paths but **never for novelty** —
so we'd have measured emergent novelty of default behaviour, not creative capacity. Added an explicit
**BE CREATIVE** directive (novel, surprising, non-obvious; favour remote/unexpected entities; every
triple still factual) to all three tasks, aligning Regime A with Regime B.

### Crash-proofing
A malformed output made CREATE's parser emit a `set` as a triple element; `json.dumps` threw
mid-write and discarded a whole model's 13,530 completed draws, killing the run (~$0.93 wasted).
Fixed at two layers: `parse_paths` coerces every triple element to `str`; `run_elicit` writes with
`default=str`. Would have crashed the full run identically.

---

## Cost / budget facts (established this session)
- OpenRouter account: ~$1,279 remaining, **but this key has a $500 limit with ~$217 left** — plan
  against ~$217, not the account balance.
- Measured actual output: baseline ~400 tok, analogy ~224, blending ~162 (mean ~250) — the runner's
  900-token estimate was 3.4× conservative; corrected `EST_OUTPUT_TOKENS` 900→400. `max_tokens` does
  **not** affect cost (billed on actual tokens); lowering it only risks truncation.
- Real full 3-cheap-model pilot ≈ $12 elicit + $3 judge ≈ **$15** (not the $30–36 the conservative
  estimate implied). Opus is now $5/$25 (was $15/$75), so a frontier suite is affordable later.
- Memory rule reaffirmed: estimate + confirm before any paid run.

## Current run (as of 16:25)
Stage-1 diversity pilot **firing**: 3 cheap models (llama-3.1-8b, gemini-2.5-flash-lite,
llama-3.3-70b) × 451 instances (baseline + analogy + blending) × 3 temps × M=10 = 40,590 draws,
concurrency 16, ~$6, $15 cap. Output → `data/kg_creat/responses_rand_v2_stage1`. Watcher armed.

## Key files
Created: `src/kg_creat/diversity.py`, `scripts/compute_diversity.py`, `scripts/datasheet.py`,
`scripts/show_failures.py`, `scripts/fig_creativity_mechanism.py`, `configs/kg_creat/build_gc.yaml`
(v2), `configs/kg_creat/run_elicit_rand_stage1.yaml`, `data/kg_creat/gc_domains_v2/`,
`data/kg_creat/prompts_rand_v2/`, `docs/reports/2026-07-22_kg_creat_creativity/`.
Modified: `sample.py` (random sampler), `prompts.py` (antanaclasis blend + creativity directives),
`judge.py` (polysemy blend judge), `run_elicit.py` (M/temperature/serialization), `parse.py`
(str coercion), `scoring.py` (strict entity match), `aggregate.py` (creativity term),
`docs/repo_usage.md`.

## Next steps
1. Stage-1 lands → report temperature×diversity surface, per-task rates on v2 pool, categorical
   derivability (→ 4 or 5 cells).
2. Derive constraint targets (`make_pass2` on the M-sampled baseline) → Stage 2 (constrained cells,
   same M=10 × 3 temps).
3. Judge a slice for utility; compute diversity (free) across everything.
4. Human blind judge-reliability pass — **still owed** (load-bearing: all cells are judged).
5. Then scale to the frontier suite (mind the ~$217 key limit; may need it raised or tranched).

## Open decisions still parked
- Categorical derivation on arbitrary endpoints (needs interior-entity typing) — decide after
  Stage-1 baseline.
- Frontier suite membership + whether to keep Opus-5 given the key limit.
- Whether to add biology/tech/food seeds to close the domain-balance gap in G_c.

---

## Continuation — 2026-07-27

**Paper: forms-of-creativity table.** Added Table `tab:forms` to `content/05_benchmark.tex` mapping
the three tasks to their cognitive-science traditions (remote association / Mednick; analogy &
metaphor / Gentner + Lakoff; blending / Fauconnier-Turner + Koestler) with one short example each
(*Einstein→violin→Mozart*; *atom : solar system*; *Boxer: athlete/dog*) and a concise caption.
Added Gentner 1983 + Lakoff 1980 to the bib. **Committed + pushed to Overleaf** (paper is its own
git repo). Also discussed (and scoped OUT) conceptual combination as a 4th form — it's the one
literature gap, but scoring a generative hybrid needs a softer, judge-heavy utility target
(ground the parts, judge the synthesis + emergence); noted for related-work, not built.

**Stage-1 pilot results (3 cheap models, baseline + analogy + blending, M=10 × 3 temps).**
All three models ~96–98% parse, 1–5 API fails; ~$6.
- **Diversity rises monotonically with temperature** (the sanity check) — e.g. Llama-8B baseline
  D_all 0.54→0.59 across T 0.7→1.0.
- **Diversity captures model + task structure:** weaker models are *more* diverse (Llama-8B highest,
  Llama-70B lowest/flattest — the strong model is consistent); tasks ladder baseline > analogy >
  blending.
- **Structural rates:** baseline well-formed 77–96% (arbitrary endpoints not floored); analogy valid
  3/19/81% and antanaclasis blending 1/3/58% (Llama-8B / Flash-Lite / Llama-70B) — the tasks
  discriminate models sharply, blending hardest.
- **Finding it sets up:** a **diversity↔validity trade-off across models** (Llama-70B most valid,
  least diverse; Llama-8B the reverse), plus the monotone temperature effect.

**Stage-2 built (not yet run).** `make_pass2` extended to derive **categorical on arbitrary
endpoints** — type the interior entities models actually used (via G_c), drop over-generic types
(human/country/sovereign-state, which don't bite), pick the most-contrastive specific type. Yields
biting targets for all 120 bundles across 25 specific types (island country, music genre, academy of
sciences...). Generated 480 Stage-2 specs (120 bundles × 4 cells). **Fired and immediately killed on
user request** — nothing spent.

**Constraint-taxonomy correction (DEFERRED).** User clarified the constraint set is a clean **2×2:
inclusion/exclusion × relation/entity** (categorical is just *inclusion of entity*; no separate
"categorical" type). Current Stage-2 cells cover 3 of 4 quadrants — **exclusion of entity is
missing**, and `inclusion_rare` is a redundant second inclusion-of-relation, not a distinct type.
Fix (deferred): drop `inclusion_rare`, rename categorical → inclusion-of-entity, add
exclusion-of-entity (same contrastive-type machinery as an *avoid* constraint + new prompt clause +
judge branch). Open confirmations: entity constraints by TYPE (symmetric with relation-by-class) vs
specific entity; whether to keep `inclusion_rare` as an appendix ablation.

**Budget:** this OpenRouter key has ~$217 left (checked). Stage-1 ~$6; Stage-2 ~$8 when run.

**Next session:** (1) resolve the 2×2 taxonomy + rebuild Stage-2 spec generation; (2) run Stage 2;
(3) **the immediate ask: analyze the Stage-1 analogy/blending results in depth.** Raw outputs saved
at `data/kg_creat/responses_rand_v2_stage1/` (local-disk only, not backed up).
