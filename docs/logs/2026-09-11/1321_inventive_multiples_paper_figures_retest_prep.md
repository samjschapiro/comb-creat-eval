# 2026-09-09 → 09-11 — inventive multiples finished, paper figures/tables rebuilt, retest set prepared

One long session (09-09 morning through 09-11). The first day's analysis work is logged in detail in
[`../2026-09-09/1030_inventive_multiples_finished.md`](../2026-09-09/1030_inventive_multiples_finished.md)
(four addenda); this log summarises the whole session proportionally and covers 09-10/11 in full.

## Summary

The inventive-multiples study (Findings #3) is finished on a documented criterion (θ = 0.674 set by
inspection, τ = 2, exact maximum matching, name and anchor-echo properties excluded), every number in
the paper comes from `inventive_multiples.json`, and the paper's Findings #3 / Benchmark floats were
rebuilt around concrete examples (two side-by-side tables) and a transposed generic-space-failures
grid. A second, disjoint 30-item set for a test–retest reliability study is sampled and reviewed but
**not run** (cost ≈ $190; awaiting go-ahead). The plot_twist track was extracted into a standalone
GitHub repo for a collaborator.

## Tasks completed

**Analysis (09-09; details in the 09-09 log)**
- Anchor-echo filter (properties whose object is an anchor: 8.7% of analogy properties) made part of
  the criterion; θ moved 0.545 → **0.674** after sampling matched pairs by cosine band (0.50–0.60 is
  shared-noun matching); per-task percentile rule considered and rejected; θ = 0.65 examined and
  rejected. Cross-item null reports the implied α (0.03%).
- Operator comparison by five routes (fixed τ 27×, eligible 17×, per-property 3.3×, exact-match 2.0×,
  size-matched); `shared_properties` switched to an exact maximum bipartite matching (differs from
  greedy on 2 of 34,687 pairs); cluster density recorded (largest component 19 models, density 0.42).
- Name/property dissociation (Finding #3d): 68% of same-name pairs share no property; 89% of
  multiples carry different names.
- Anchor distance vs multiples redone (4 distances × 4 outcomes × 2 tasks): null.
- Analysis JSON gained `per_item`, `model_pair_matrix`, `multiples` (every τ-multiple with its
  matching and cosines), `name_property_dissociation`, `task_routes`, `anchor_echo`, `null`.

**Figures and tables (09-09/10)**
- `plot_pair_matrix.py`: model × model shared-property heatmap averaged over items (blend / analogy).
- `plot_multiples_examples.py`: side-by-side inventions with matched properties joined (report 2×2;
  paper one-row); then replaced in the paper by `make_multiples_examples_table.py`, two LaTeX tables
  (indexed properties p_i / p'_j, rows shaded green by cosine).
- `plot_abstraction_failure.py` gained the by-item orientation for the paper: 19 hardest anchor pairs
  as rows (bold, black), models as columns with logos under the grid, cells filled #EA8485, no marginal
  rates.
- `make_tau_multiples_table.py`: float-less fragment; eligible column dropped; headers renamed
  (Multiples: Count / % of pairs / % of inventions; % of pairs by task; % of pairs by model family).
- All three table generators now emit **no captions or labels** (captions belong to the author).
- `plot_invention_landscape.py` emits single panels; `make_paper_multiples_figure.py` writes the
  paper's old/new figure sets.

**Paper (Overleaf, all pushed as scoped patches)**
- Plain-words τ-inventive-multiples definition; τ table; examples tables; by-item failures grid
  replacing the multiples figure; author block left-aligned with tighter rows; τ-table caption
  rewritten on request. The Findings #3 prose drafts (#3a–#3d) exist only locally — the author is
  writing #3 on Overleaf.
- Two incidents, both fixed and both now guarded by memory rules: a whole-file push overwrote a float
  the author had moved (restored; pushes are now hunk patches), and a regenerated fragment overwrote
  the author's caption (restored; generators no longer emit captions).

**Other**
- `twistbench-code`: private GitHub repo (`samjschapiro/twistbench-code`) holding the plot_twist
  track (src, scripts, configs, docs, site source; no data), collaborator `sumukshashidhar` invited.
- Human-study sizing table (recruits = ceil(6K/(1−N))); with K = 35 to match the LLM pool: 210 usable,
  250–280 recruits at 15–25% exclusion.
- Findings #3 discussion: convergence uncorrelated with capability (ρ = +0.14, p = 0.41); same-family
  effect is Anthropic/DeepSeek/xAI/Meta, not OpenAI or Google (1.1% and 2.0% vs 1.8% cross-family).
- Retest set: `sample_flat.py` gained `shared_pairs`, `unique_entities`, `exclude_prompts`;
  `configs/kg_creat/kombine_retest30_sample.yaml` drew 30 cross-domain pairs, disjoint from test30 at
  the entity level, unique anchors, the same 30 for all three tasks
  (`data/kg_creat/kombine_retest30/prompts/`).

## Files modified / created

- `src/kg_creat/scripts/`: `analyze_inventive_multiples.py` (major), `calibrate_theta.py`,
  `plot_tau_theta_grid.py`, `make_tau_multiples_table.py`, `make_multiples_examples_table.py` (new),
  `plot_multiples_examples.py` (new), `plot_pair_matrix.py` (new), `analyze_anchor_distance.py` (new),
  `plot_abstraction_failure.py`, `plot_invention_landscape.py`, `plot_multiples_matrix.py`,
  `make_paper_multiples_figure.py`, `make_multiples_showcase.py`, `sample_flat.py`.
- `configs/kg_creat/kombine_retest30_sample.yaml` (new).
- `docs/reports/2026-09-01_kg_creat_inventive_multiples/` (report rewritten; figures), `docs/reports/2026-09-03_kg_creat_frontier_failures/figures/`.
- `docs/tracks/kg_creat/progress.md`, `docs/tracks/plot_twist/progress.md`, `docs/research_context.md`,
  `docs/structure.md`, `README.md`.
- Paper repo: `content/05_benchmark.tex`, `content/06_results.tex`, `content/01_authors.tex`,
  `setup/iclr2026_conference.sty`, `media/04_tau_multiples.tex`, `media/08_multiples_examples.tex`,
  `media/figures/{old,new}/…`.

## Key decisions

- θ is a **judgment, documented** (paraphrase band), not a null-derived number; the null gives α.
- The fixed-τ blend/analogy ratio is mostly property count; the paper should quote the per-property
  routes beside it.
- Captions are the author's; generated fragments carry only tabular bodies.
- Overleaf pushes are hunk patches onto the fresh remote head, never whole-file copies.

## Open / next steps

1. **Retest30 run** — awaiting go-ahead: 7 Anthropic models (direct API, ≈$80), gpt-5.6-sol +
   gpt-6-astra-flex (gateway), gpt-5-mini / gpt-4.1 / gpt-4o-mini (OpenRouter, ≈$3); same judge panel
   (≈$100 on OpenRouter). Then per-model test–retest correlations and ICC across the two item sets,
   plus a replication of the multiples rates. The same 30 items can serve the human study.
2. Two Rice + Radio cells are blank in the failures grid (Opus 4.7 / 4.8 returned null content) —
   re-elicit (two calls) or caption it.
3. Paper: Findings #3 prose (#3b should state the per-property ratio next to the fixed-τ one; #3c's
   family effect is provider-uneven; #3d not yet written on Overleaf); "elig." no longer appears.
4. A separate relation-agreement bar (so a shared noun cannot carry a match) remains the obvious
   refinement of the primitive.
