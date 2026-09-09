# 2026-09-09 — inventive multiples finished

Session goal: finish the τ-inventive multiples study. The 09-08 rebuild had left the anchor-echo
confound, the property-count asymmetry and the "three routes" only in a scratch session, the showcase
script broken on renamed keys, and the paper/report on pre-τ numbers.

## What was done

- `analyze_inventive_multiples.py`: anchor-echo filter in `slot_texts` (object == anchor, exact after
  normalisation), `slot_objects`/`exact_shared` for the encoder-free route, `cross_item_null` (30k
  same-task cross-item pairs per task), `task_routes` (fixed τ, eligible, null-corrected, per-property,
  exact per-property, size-matched), tau_curve rows now carry eligible/null/excess/inventions-touched,
  provider effect also per property and within task, cluster `density`, `clusters_have_outsiders`
  computed. Stale docstring/comments (abstraction clause, "COS_SLOT is not calibrated") corrected.
- `calibrate_theta.py`, `plot_tau_theta_grid.py`: pass anchors to `slot_texts`; calibration re-run after
  the filter: θ = 0.545 still α = 0.24%.
- `make_tau_multiples_table.py`: emits `04_tau_multiples.tex` (Inv. column, eligible ratio) and a new
  `07_task_routes.tex`; both checked in a standalone ICLR-style build (no overfull boxes).
- `make_multiples_showcase.py`: `k_shared`/`tau_slot`/`tau_con` → `tau`/`cos_slot`/`cos_con`; abstraction
  wording removed from the criterion text; outsider rows show shared count + schema cosine for reading.
- Regenerated: JSON, showcase HTML + examples, landscape, matrix, stacked paper figure, τ–θ density.
- Paper: Definition paragraph (anchor exclusion), #3a/#3b/#3c prose, routes table input, figure caption,
  abstract clause. All machine-authored text inside `\ai{}`. Not pushed.
- Report rewritten to the current criterion; progress.md updated.

## Numbers (35 models, 30 anchor pairs, 2,070 inventions, 34,687 pairs, θ = 0.545, τ = 2)

- pairs 5.9%; inventions in ≥1 multiple 46.1% (blend 85.0%, analogy 7.4%); τ=3 → 0.95% / 16.6%.
- anchor echo: analogy 8.7% of properties, rate 0.82% → 0.32%; blending 0.3%, 11.54% → 11.44%.
- blend/analogy: fixed τ 36.2×; eligible 22.7×; per-property 3.2× (p = 3.7e-9); exact 2.0× (p = 1.5e-3);
  size-matched 6.2× (kmin 3), 4.5× (kmin 4). Null at τ=2: 0.02% / 0.00%.
- provider: 10.3% vs 5.1% (RR 2.0, p = 5e-4); per property 1.34× (p = 5e-4).
- clusters: 69; eight ≥ 32 models, density 0.13–0.35; mean density 0.57.
- pre/post uv (21 models, blending): 23.4% → 12.0%, nominal 4.6% → 4.7%.

## Surprises

- The 09-08 log said all routes gave ~2×. After the anchor filter the fixed-τ ratio is 36× and the
  property-count-free routes 2–6×. The log's routes had been computed on unfiltered analogy.
- The largest cluster is the whole pool. Density shows it is a chain.

## Open

- Local paper build fails on `algorithm.sty` (tlmgr needs `update --self`); only the #3 section was
  compiled standalone.
- Paraphrased anchor echo is unmeasured.

## Addendum — θ = 0.674

Sampled matched property pairs by cosine band (scratch `theta_examples.py`): 0.65+ paraphrases;
0.50–0.60 shared-noun matches and missed paraphrases. User set θ = 0.674. Re-ran calibrate (implied
α = 0.03%), analysis, prepost, density, showcase, landscape, matrix, stacked figure, tables; rewrote
paper #3a–c, Definition θ paragraph, abstract clause, report, progress.

Numbers at θ = 0.674, τ = 2: pairs 1.13%; inventions 19.8% (blend 37.3%, analogy 2.3%); τ=3 30 pairs;
blend/analogy 32× fixed τ, 20× eligible, 3.4× per property (p = 3.5e-8), 2.0× exact (p = 1.5e-3);
provider 2.3% vs 0.9% (RR 2.5, p = 5e-4), 1.6× per property; same-name 6% multiples; 111 clusters,
max 19, mean density 0.78; prepost 6.3% → 2.7%.

Open: σ is greedy, not maximum matching (lower bound on the Definition); relation/object embedded
jointly. Size-matched strata too thin at this θ.
