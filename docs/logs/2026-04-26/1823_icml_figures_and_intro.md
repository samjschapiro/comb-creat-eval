# 2026-04-26 — ICML draft: combined headline figure, palette overhaul, intro rewrite

## Summary

Rebuilt Figure 4 of the ICML draft as a single two-row figure that
absorbs the previous fig_specificity_ceilings; switched the
per-test categorical palette to Okabe-Ito (propagated to the
`C_DAT/C_CDAT/C_CNOV/C_CAPP/C_PACE` globals); recoloured the
benchmark-correlation heatmap to `cmc.broc` and masked the upper
triangle in-data to remove anti-aliased edges; and rewrote §1.1
("Problems" → "Three Untested Premises") with three sentence-style
problem statements and a moved-to-appendix greedy-DAT failure-mode
demo. Also pushed the ARC-AGI v2 correlation block authored 2026-04-25
(per-metric correlations + per-benchmark matrix, partials gated at
n≥7/8).

## Tasks completed

### Combined headline figure with construct-level ceiling
- Added `_benchmark_signed_R(bench_key, BMARKS)` and
  `_panel_avg_ceiling(R_list, v_grid)` helpers in
  `src/dat_eval/scripts/make_figures.py`. Construct-level ceiling is
  the unweighted mean across the panel's benchmarks of the
  per-benchmark bound `v·√(1−R²) + |R|·√(1−v²)`.
- Per-panel R values used (computed on the n-subset with Y, Arena
  Overall, MMLU-Pro all present): CW = +0.98 / +0.83 / +0.80;
  DT = -0.68 / -0.33; SI = +0.62.
- Wrote `fig_headline_combined()` (top row = 3 construct-level scatter
  panels with the panel-averaged ceiling overlay; bottom row = 6
  per-benchmark feasibility-lens panels; single shared legend at
  bottom). `main()` now calls only this; the old `fig_headline()` and
  `fig_specificity_ceilings()` remain defined but are no longer
  invoked.
- Tightened layout: row titles "(a) Prediction by construct" and
  "(b) Prediction by benchmark"; bumped legend / x-axis / panel-title
  font sizes; added gap between the bottom-row x-axis and the legend.

### Heatmap colormap iteration
- Cycled `RdBu_r` → `cmc.cork` → `cmc.bam` → `cmc.broc` for the
  `fig_benchmark_correlations` heatmap; settled on `cmc.broc`
  (Crameri perceptually-uniform, blue/cream/brown, colorblind-safe).
- Switched the upper-triangle masking strategy from "white-rectangle
  overlay" to "masked imshow array" so the upper-right region simply
  isn't drawn — eliminates the faint anti-aliased outline at the top
  and right of each heatmap. Spines explicitly hidden.

### Test-color palette overhaul
- Replaced the batlow gradient samples with a 5-entry Okabe-Ito
  categorical palette (DAT blue `#0072B2`, CDAT vermillion `#D55E00`,
  CDAT-N green `#009E73`, CDAT-A pink `#CC79A7`, PACE orange
  `#E69F00`) so DAT and CDAT are clearly distinguishable rather than
  adjacent samples of a sequential map.
- Propagated to the `C_DAT/C_CDAT/C_CNOV/C_CAPP/C_PACE` globals;
  `C_CDAT` is a new constant for gated CDAT.
- Fixed two semantic-mis-coloring callsites where gated CDAT had been
  drawn with `C_CAPP` (appropriateness) or `C_CNOV` (novelty):
  `fig2_combined_grid:357` and `fig_validity_specificity:1237` now
  use `C_CDAT`.

### ARC-AGI v2 correlation block
- (Authored 2026-04-25, pushed today.) Added an ARC-AGI v2 block to
  `src/dat_eval/scripts/score_evals.py`: per-creativity-metric
  correlation against ARC-AGI plus an ARC-AGI vs every-other-benchmark
  matrix. Inclusion gate is independent of the existing arena_cw gate
  to capture the full n=10 ARC-AGI cohort.
- Partials gated at n≥7 (single control) / n≥8 (two controls) — at
  the n=6 intersection with Arena Overall ∩ MMLU-Pro the partials
  return degenerate |ρ|=1 / NaN, so they're skipped rather than
  reported.
- Headline (Pearson, exploratory): no creativity metric correlates
  meaningfully with ARC-AGI at n=9–10 (all |r|<0.5, all p>0.26);
  ARC-AGI tightly tracks MMLU-Pro (r=+0.89, n=6) and EQ-Bench CW
  (r=+0.98, n=5) but is essentially decorrelated from Arena CW
  (r=+0.44, n=6). Directional only — n=5–10 is too thin to support
  inferential weight; appendix-grade only.

### §1.1 rewrite + greedy appendix
- Renamed `\subsection{Problems}` → `\subsection{Three Untested
  Premises}` (paper-side, in
  `papers/iccc-2026/sections/01_introduction.tex`) — note the user
  later removed the subsection wrapper entirely on Overleaf and
  promoted the three `\textbf{Problem \#k:}` blocks to inline §1
  paragraphs.
- Three sentence-style problem statements (parallel structure
  anchored on "validity"):
  1. "Their validity as measures of *machine* creativity has not been
     established."
  2. "Their validity as measures of *human* creativity is itself only
     modestly established."
  3. "Tests should measure something outside of what general capability
     already predicts."
- Activated the previously commented-out greedy-DAT appendix
  (`papers/iccc-2026/sections/07_appendix.tex`, label `app:greedy`):
  Algorithm 1 (greedy argmax-distance pseudocode) plus the
  `fig:greedy-baseline` figure environment. Added a leading
  paragraph explaining why this matters (transformers implement
  algorithmic circuits per Olsson induction heads / Nanda
  modular-arithmetic, plus DAT admits a trivial solver — together
  they make "high LLM DAT score" compatible with "algorithmic
  shortcut").
- Added `\usepackage{algorithm}` and `\usepackage{algorithmic}` to
  `main.tex` for the algorithm box.

### Draft audit against `docs/writing_advice.md`
- Provided a structured audit (good points / bad points) of the
  current draft against the writing-advice guidelines: flagged the
  unfinished holes (abstract sentence 3, contributions item 3,
  background §2.4 TODO, empty Results, empty Conclusion, discussion
  placeholder), the absent headline number in the abstract, the
  Pearson/Spearman inconsistency in §2 PACE description, the
  Methods/Results boundary blur, the missing related-work and
  reproducibility statement, and the AI-generated bib entries.
- Drilled in on §1.1 specifically: original three problems were at
  three different abstraction levels and only Problem #3 directly
  motivated the paper's contribution. Recommended renaming them as
  three diagnoses anchored on "validity," which the user subsequently
  implemented (with their own further edits on Overleaf).

## Files modified

### Main repo (`comb-creat-eval`)
- `src/dat_eval/scripts/make_figures.py` — combined headline figure;
  Okabe-Ito globals; `cmc.broc` heatmap; masked-array upper triangle.
- `src/dat_eval/scripts/score_evals.py` — ARC-AGI v2 correlation
  block (per-metric + per-benchmark matrix; partials gated at n≥7/8).
- `docs/tracks/dat_eval/progress.md` — added 2026-04-25 (ARC-AGI)
  and 2026-04-26 (combined figure) entries.
- `docs/reports/2026-04-12_preliminary_correlations/figures/*.pdf,*.png`
  — regenerated for new palette + heatmap colormap.

### Papers repo (`papers/iccc-2026`, Overleaf-synced)
- `figures/fig_headline.pdf, .png` — new combined two-row figure.
- `figures/fig_benchmark_correlations.pdf, .png` — `cmc.broc` heatmap,
  masked upper triangle.
- `sections/01_introduction.tex` — Problems subsection rewrite (then
  user follow-up edits on Overleaf).
- `sections/03_method.tex` — Figure 4 caption rewritten to describe
  both rows.
- `sections/05_discussion.tex` — removed standalone
  `fig:spec-ceilings` figure environment; refs redirected to
  `fig:headline`.
- `sections/07_appendix.tex` — activated greedy-DAT appendix; added
  motivating paragraph; `app:spec-bound` reference text updated.
- `main.tex` — added `algorithm` and `algorithmic` packages.

### Memory
- `feedback_pearson_only.md` (already created 2026-04-25) — captures
  the user preference: report Pearson r only in chat/doc summaries,
  never Spearman; the underlying scoring code can keep storing both.

## Decisions / insights

- **Construct-level ceiling = mean of per-benchmark ceiling curves**
  (mean-of-ceilings, not ceiling-of-mean-R). Justified because the
  "Overall" composite point in the headline is also an unweighted
  mean across the panel's benchmarks, so the visual interpretation
  is consistent: a hypothetical test that hit each benchmark's
  individual ceiling would land on this curve.
- **Partial correlations need a higher n floor than raw correlations.**
  At n=5 with one control, the rank-residual partial routinely
  returns |ρ|=1 or NaN as a small-sample artifact. Gating at n≥7
  (single control) / n≥8 (two controls) avoids reporting these
  spurious extremes for the ARC-AGI block. The same caveat applies
  in principle to the existing partials in the main loop, but the
  Arena-CW-anchored cohort is large enough (n≥34) that this hasn't
  surfaced.
- **`multi_embed_appendix.py` writes its output inside the
  `--overwrite`-able `output_dir`.** Re-running
  `score_evals.py --overwrite` deletes `multi_embed_scores.json`
  silently, breaking the bottom row of the headline figure on the
  next render. Fix candidate (not done): move the file one level up,
  next to `cdat_gated_scores.json`. Tracked in the progress.md
  heads-up at 2026-04-26.
- **ARC-AGI v2 is a clean reasoning-capability proxy at the small n
  available**: r=+0.89 with MMLU-Pro and r=+0.98 with EQ-Bench CW,
  but only r=+0.44 with Arena CW. Directionally consistent with the
  paper's "creativity ≠ general capability" framing — but n=5–6 is
  appendix-grade only. The 5 leaderboard models we don't have
  (GPT-5.5, GPT-5.2 Pro, Muse Spark, Gemini 3 Pro non-image, Claude
  Opus 4) are either not on OpenRouter or not in our pool; running
  the ones we could add (~3 models) would lift the cohort to ~13.

## Open questions / next steps

- §1.1 still has an orphaned/unfinished transition sentence on
  Overleaf (line 16): "We explain why these comparisons are
  problematic, and more broadly, why naive applications of human
  psychometrics to LLMs". User flagged this as their own TODO and
  rejected my draft completion as "clunky and unconvincing"; offered
  three alternatives, no replacement chosen yet.
- Title-vs-body repetition in Problem #1 and Problem #3 (both have
  bodies that paraphrase their `\textbf{...}` titles); not yet fixed.
- Footnote in Problem #2 sits between "forward flow" and its citation
  rather than after the citation; not yet fixed.
- Two `figures/fig_qualitative_*.pdf` files are untracked in the
  papers repo (not from this session).
- Two stashes exist on the papers repo (`stash@{0}` from this
  session's drafted §1.1 polish, superseded by user edits;
  `stash@{1}` predates the session). Neither dropped without user
  confirmation.
- Bigger writing-advice gaps still open: empty Results section, empty
  Conclusion, abstract sentence 3 unfinished, contributions item 3
  unfinished, background §2.4 placeholder, no related-work section,
  no reproducibility statement, AI-generated bib entries unverified.
