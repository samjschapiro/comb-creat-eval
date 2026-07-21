# 2026-07-21 · kg_creat — Regime A run at scale + blending reframe

**Summary.** Ran the track's headline experiment: 8 models × 30 fixed endpoint bundles ×
6 cells (baseline + 5 constraint types) = 1,440 prompts / 7,159 judged paths, producing the
per-constraint ideation–execution decomposition. Also reframed blending to a single-stimulus
task, and caught two measurement defects that would have corrupted the result.

## Tasks completed

**Pass-2 design (constrained cells).**
- Derived relation **classes** from the Pass-1 baseline corpus: k-means over embeddings of the
  top-150 relations models actually emitted → 8 classes, LLM-named with collision
  disambiguation (membership 31.9 %, location/origin 16.7 %, affiliation/collaboration 16.4 %,
  hosted/participation 15.8 %, international relations 7.3 %, agent 6.7 %, position 3.1 %,
  affiliation-specialized-agency 2.2 %).
- **Per-bundle, baseline-derived targets** so each constraint bites by construction: exclusion =
  the class that bundle used most when unconstrained; inclusion = least-used usable class;
  inclusion_rare = least-used class with share < 8 %; ordering = the REVERSE of that bundle's
  most frequent class co-occurrence order.
- `make_pass2.py` → 150 specs (30 bundles × 5 cells) on the **same fixed endpoints** as Pass 1.

**Elicitation + judging.**
- Pass 2 elicited on the 8-model suite ($3.60), then judged with `gpt-oss-120b` (~$2.2).
- Re-judge pass to repair judge holes ($0.09). Total round cost **~$6.6**.

**Two measurement defects found and fixed.**
1. *Truncation.* `max_tokens=1200` cut long answers mid-JSON, and truncated JSON parses to
   **zero** paths. GPT-4o-mini lost 104/180 prompts — this would have been reported as a 60 %
   structural failure rate when it was really a token cap meeting a verbose model. Added
   truncation salvage to `parse.py`, keeping only paths whose array actually closed (so a
   half-emitted path isn't scored as "never reached the target"), and re-fired the 12 responses
   cut before any path closed. All 8 models now sit at ~5.0 paths/prompt.
2. *Judge hole.* `judge_categorical` ran at `max_tokens=400`; a reasoning judge spends a small
   budget thinking and never emits JSON, silently turning satisfaction into `unjudged`
   (123 categorical paths). Raised to 800, built `rejudge.py` to repair a cell without
   re-scoring the corpus: unjudged **196 → 9** paths (0.13 %).

**Robustness.** One malformed provider response body propagated out of `asyncio.gather` and
killed a whole model's scoring mid-run, losing ~25 min of paid judging (crashed at 6/8 models).
Judge calls now route through `_ask()` — retried, and degrading to a single unjudged record
rather than taking the run down. Resumed and completed the remaining two models.

**Blending reframed to a single stimulus** (per the user's framing: blending is analogy with one
stimulus, where the model must find two directions of relational structure emanating outward).
- New `src/kg_creat/regime_b.py` holds the structure-mapping predicates for BOTH semantic tasks,
  so scorer and figures share one definition of validity. Analogy pins two endpoints and requires
  fully disjoint structures; blending pins one anchor and requires branches sharing the anchor
  **and nothing else**, with an identical relation sequence. Failures are attributable
  (`branch_not_anchored` / `branches_overlap` / `relations_differ` / `revisits_node`).
- Rewrote the prompt, the judge (now anchor + two branches, asking whether the branches reach
  genuinely different domains), the sampler (single anchors — blending's second domain becomes an
  *outcome* rather than a design variable), and folded pair-level Regime-B success into
  `score.py` (it previously lived only in the plotters). Novelty for blending = branch-tip
  distance.
- The shared output-format example showed both paths reconverging on one entity, which
  demonstrates exactly the overlap blending forbids — added a divergent variant.
- Smoke-tested on 8 anchors × 2 models (~$0.02): **Sonnet 4.6 8/8 structurally valid, Gemini
  2.5 Flash-Lite 0/8.**

**Analysis + figures.** New `plot_regime_a.py` (ideation–execution 2×2 + failure-channel
decomposition). De-duplicated the structure-mapping predicates out of the three analogy plotters
onto `regime_b.py`. Added `scoring.cosine_distance`.

## Key findings

1. **Constraints are not equally hard.** Ordering Δsat **−0.448** (buys +0.002 novelty);
   categorical **−0.131** (buys **+0.055**). Ordering is pure cost; categorical is the efficient
   lever. Ordering is < 0.10 for **every** model tested, strongest included.
2. **Constraints don't degrade factuality.** The factual channel is a flat ~34–40 % tax in every
   cell *including baseline* (34.3 %). The entire cost of a constraint lands in the constraint
   channel — under ordering, 41.6 % of all paths vs 8–16 % elsewhere.
3. **Ordering fails as double-inclusion, not as sequencing.** Decomposing its 495
   constraint-failures: only **11.5 %** are genuine order violations; 88 % never get both required
   classes into the path at all. This is a lower bound (exact string matching understates class
   presence), but it reframes finding #1 — ordering may be hard because it's the only cell
   demanding two classes at once, not because sequencing is hard.
4. **Rarity isn't what makes inclusion hard.** Pooled, rare and common inclusion are nearly
   identical (−0.228 vs −0.234), and individual models move in opposite directions.

## Files created / modified

Created: `src/kg_creat/regime_b.py`, `src/kg_creat/scripts/rejudge.py`,
`src/kg_creat/scripts/plot_regime_a.py`, `configs/kg_creat/score_regimeA.yaml`,
`docs/reports/2026-07-21_kg_creat_regimeA/` (report + 2 figures).
Modified: `parse.py` (salvage), `judge.py` (`_ask` retry wrapper, token budgets, blending judge),
`prompts.py` (blending prompt + divergent output example), `sample.py` (single-anchor blending),
`scoring.py` (`cosine_distance`), `scripts/score.py` (`finalize_regime_b`, blending dispatch),
the three analogy plotters, `configs/kg_creat/run_elicit.yaml`,
`docs/tracks/kg_creat/assessment.md` (§7b amendments), `progress.md`, `research_context.md`.

## Open questions / next steps

- **Add a "both classes, any order" cell** — the single most informative next run; it separates
  ordering-as-sequencing from ordering-as-double-inclusion (finding #3).
- **Human blind judge-reliability pass** — owed since the analogy round, and now load-bearing:
  with class-level constraints, all five Regime-A cells are judged rather than exactly checked.
- **Run blending at scale** (harness ready, smoke-tested, not yet run).
- Re-run the finding-#3 decomposition with semantic rather than exact class matching.
- The `categorical` target for some bundles is a very generic type (e.g. `'human'`), which may be
  near-trivially satisfiable; worth auditing whether that inflates the categorical cell.
- Figure 2 in the ICLR paper may need updating against the current constraint definitions.
