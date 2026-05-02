# 2026-05-02 — NeurIPS-2026 setup, MMLU-Pro switch to TIGER-Lab, semi-partial fix, paper polish

## Summary

Long working session. The substantive findings: (1) caught and fixed a
silent bug where our specificity computation was computing the *full*
partial correlation `r(X − X̂_g, Y − Ŷ_g)` instead of the paper-defined
*semi-partial* `r(X, Y − Ŷ_g)`; the full partial systematically inflated
specificity, in one case driving NoveltyBench × CDAT to a frontier-violating
+0.96 (n=8) that disappears under the correct definition (+0.60). (2)
Switched the MMLU-Pro source from artificialanalysis.ai to the
TIGER-Lab leaderboard CSV after a co-author flagged that AA's snapshot
mixes evaluation methodologies; cut model-overlap by ~7 cells but
restores methodological consistency with the original MMLU-Pro paper.
The rest of the session was paper polish: rewriting §5.1 around the
four main findings, weaving a "specificity not validity" thesis through
§5.1/§5.2/§6/§7, adding a NeurIPS 2026 submission variant in parallel
to the ICML version, autogenerating Appendix Table 4 from benchmarks.json,
and many figure/typography passes.

## Tasks completed

### Bug: full-partial vs semi-partial correlation
- The paper defines specificity as the semi-partial
  `r(X, Y − Ŷ_g)` (Y residualised on g, X kept raw). The code in
  `partial_pearson_multi` (and `partial_pearson`, and
  `multi_embed_appendix.joint_partial_r`) was residualising **both**
  X and Y, computing the full partial. Co-author asked
  "are you sure you did a semi-partial correlation?" — diff confirmed
  the bug.
- Fixed all three call sites in `src/dat_eval/scripts/score_evals.py`
  and `src/dat_eval/scripts/multi_embed_appendix.py`. Reran the full
  correlation pipeline + multi-embed regen.
- Concrete cell impact (Overall block):
  - Mazur DAT spec: +0.56* → +0.49 (lost a star)
  - EQ-Bench DAT spec: +0.50** → +0.41* (downgraded star)
  - **NovBench × CDAT spec: +0.96** (n=8) → +0.60** (no longer
    frontier-violating; the +0.96 was a spurious sample artifact of
    the full-partial computation)
  - NovBench × CDAT-N: +0.80* → +0.45 (lost star)
  - NovBench × CDAT-A: −0.73* → −0.40 (lost star)
- Updated `tables/main_result.tex` programmatically, regenerated
  fig_headline hardcoded data, refreshed §5.1 narrative numbers.
- Commit: `f096688`.

### MMLU-Pro source switch: AA → TIGER-Lab
- Verification campaign: re-fetched AA model pages, cross-checked
  against TIGER-Lab CSV
  (`TIGER-Lab/mmlu_pro_leaderboard_submission/results.csv`).
  Concluded TIGER-Lab is methodologically tied to the original
  MMLU-Pro paper while AA is opaque about evaluation conditions
  (CoT, prompt style, etc.). Co-author preferred TIGER-Lab.
- Rewrote `src/comb_eval/scripts/add_mmlu_pro_scores.py` to map
  OpenRouter keys to TIGER-Lab model-name strings; replaced
  `data/dat_eval/mmlu_pro_raw.json` with the full TIGER-Lab CSV
  (260 entries) keyed by TIGER-Lab name.
- Coverage: 47 of our 64 model entries map to TIGER-Lab; 13
  previously had MMLU-Pro values that had to be cleared.
  Specificity sample sizes shrunk slightly (e.g., Arena CW val/spec
  was 52/49, now 54/40) but values are now defensible.
- Documented the source switch in §07 appendix ("MMLU-Pro source"
  paragraph) with HF dataset link and rationale.
- Commit: `40ec98d`.

### Significance methodology
- Added `\textbf{Significance}` paragraph to §3 Evaluation Metrics
  naming the test (two-sided Pearson t with df = n−2−k) and star
  thresholds. Co-author flagged a Mann-Whitney suggestion; clarified
  that for correlations the relevant non-parametric alternative is
  Spearman (not MWU), and the user accepted the parametric test as
  sufficient given paper scope.
- Fixed a follow-on bug: partial p-values were using df = n−2 instead
  of n−2−k. Now use the correct df throughout (commit `3fb22a5`).
  Only one star flipped (FastText DAT × Mazur CW spec ** → *).

### TIGER-Lab Mazur snapshot provenance
- Verified all 50 hardcoded Mazur CW V2 scores against
  `lechmazur/writing` GitHub at commit `80b7f17`. Documented the
  source in §07 appendix ("Mazur CW snapshot"); explained why we kept
  this snapshot rather than the more recent re-grades (largest
  intersection with our pool — n=20 vs ≤10 for any later snapshot).
- Renamed `Mazur CW v2` → `Mazur CW` throughout the paper (the "v2"
  was an internal label that didn't match Mazur's own versioning).

### Paper structural rewrite around main findings (§5.1)
- Replaced the three construct-organised subsubsections with four
  finding-organised paragraphs: "No single test predicts all
  constructs well", "DAT is the best predictor of creative writing",
  "CDAT is the best predictor of divergent thinking", "None of the
  tests is a good predictor of scientific ideation".
- Tightened each to ~3 sentences after writing_advice.md feedback.
- Added a closing "Why each test wins where it does" synthesis
  paragraph (test design → construct fit), removing prior unverified
  claims about scientific ideation needing "knowledge-grounded
  recombination".

### Cohesive thesis: "specificity, not validity"
- Weaved the central premise through five touchpoints per
  writing_advice's "repeat the key idea in varied ways" guidance:
  - §5.1 opener: leads with specificity as the criterion that
    separates a creativity test from a capability proxy.
  - §5.1 end: structural explanation of which test fits which
    construct.
  - §5.2 post-theorem: ties Eq.~\eqref{eq:spec-ceiling} back to
    the criterion.
  - §6 opener: new "Specificity, not validity, decides what a test
    measures" synthesis paragraph.
  - §7 conclusion: appended single-sentence methodological
    prescription.

### NeurIPS 2026 submission variant
- Added `papers/iccc-2026/main_neurips.tex` parallel to `main.tex`,
  using the official NeurIPS 2025 style file
  (`neurips_2025.sty`, fetched from media.neurips.cc; user later
  renamed to `neurips_2026.sty`).
- Duplicated `sections/` → `sections_neurips/` and
  `tables/` → `tables_neurips/` so NeurIPS-only edits don't affect
  the ICML build. Per-section table inputs rewired.
- Bibliography: switched to plain bib style with `[nonatbib]`
  option. Replaced all `\citet{}` and `\citep{}` calls with `\cite{}`
  in the NeurIPS sections (28 occurrences swept via sed).
- Table 1: wrapped the 13-column tabular in
  `\resizebox{\textwidth}{!}{...}` so it no longer overflows the
  NeurIPS single-column right margin.
- Figures 1 and 2: converted from full-width `figure*` to
  `wrapfigure[N]{r}{0.55\textwidth}` for vertical-space efficiency.
  Required iterating on the [N] line-count parameter to prevent
  overlap with text below.

### Auto-generated appendix table 4 (per-model benchmark scores)
- Co-author caught a stale Claude-Sonnet-4.6 EQ-Bench Elo (1938 →
  1991) in the hand-edited `tables/per_benchmark_scores.tex`. Wrote
  `src/comb_eval/scripts/build_per_benchmark_table.py` to regenerate
  the table from `benchmarks.json` so it stays in sync. 64 model
  rows, supports re-running after any benchmark refresh. Commit:
  `6d03a33`.

### Many figure/typography polish passes
- `fig_headline`: bigger fonts, Times font, `(v*, s*)` optima
  marked on each ceiling with black diamonds + coordinate labels.
  Bottom-row diamonds explained in §5.2.
- `fig_benchmark_correlations`: side-by-side layout (re-recombined
  after a brief split into two figures); panel (b) anchored to
  bottom of allocated slot to align bottoms; titles aligned at top;
  RdBu_r colormap; subplot-title fontsize bumped to 19; panel (b)
  square cells preserved while panel (a) auto-aspects.
- `fig_overview` (NeurIPS only): half-width `wrapfigure[N]` on the
  right, attached to paragraph 2, with iteration on N to land
  cleanly in page-2 wrap zone.
- `fig:examples` (sd_test_examples; NeurIPS only): converted from
  full-width to `wrapfigure[22]{r}{0.55\textwidth}` with
  footnote-size text inside, attached to first paragraph of §2.

### Stylistic / consistency sweeps
- Standardised the model-pool count to **54** throughout (was a
  mix of "52", "55", "50+"). Updated abstract, §3 Method, §02a
  preliminaries, §07 appendix sanity-check ranges, and the
  greedy-DAT figure caption (n=2078 trials, mean=83.75, std=5.13
  recomputed over the 54-model subset).
- Removed hyphens from "creative writing", "divergent thinking",
  "semantic distance", etc. across all section files.
- Reduced abbreviations to {DAT, CDAT, PACE, LLM} + benchmark
  proper names; expanded SD, CW, DT, SCI, OLS, SBERT, FDR, PSD,
  AUT, M, SD-stat throughout.
- Various typo fixes flagged in audit pass: "$r - 0.59$ →
  $r = -0.59$", "Arena All → Arena Overall", missing close paren
  in §5.1, "their novelty → its novelty", and several others.

## Files modified / created (main repo)

```
src/dat_eval/scripts/score_evals.py            (semi-partial fix, df=n-2-k)
src/dat_eval/scripts/multi_embed_appendix.py   (semi-partial fix in joint_partial_r)
src/dat_eval/scripts/make_figures.py           (many figure tweaks)
src/comb_eval/scripts/add_mmlu_pro_scores.py   (rewritten for TIGER-Lab)
src/comb_eval/scripts/build_per_benchmark_table.py  (NEW)
configs/comb_eval/benchmarks.json              (TIGER-Lab MMLU-Pro values)
data/dat_eval/mmlu_pro_raw.json                (TIGER-Lab CSV snapshot)
docs/reports/2026-04-12_preliminary_correlations/figures/*.{pdf,png}
papers/iccc-2026/figures/*.{pdf,png}           (regenerated)
papers/iccc-2026/figures/archive/              (old fig_benchmark_correlations parked)
```

## Files modified / created (Overleaf paper repo)

```
sections/{00_abstract,01_introduction,02_background,02a_preliminaries,
          03_method,04_results,05_discussion,06_conclusion,07_appendix}.tex
tables/{main_result,per_benchmark_scores}.tex

main_neurips.tex                  (NEW)
neurips_2025.sty / neurips_2026.sty  (NEW; user renamed)
sections_neurips/*.tex            (NEW; \citep/\citet → \cite)
tables_neurips/*.tex              (NEW; resizebox on main_result)
```

## Key decisions / insights

- **Verify the formula, not just the numbers.** The full-partial bug
  produced consistently plausible-looking specificity values for over
  a year (the cells that flipped were small-n cells that we'd already
  flagged as marginal). One frontier violation surfaced it. Lesson:
  if a result sits unusually close to the theoretical ceiling, that
  ceiling is itself a useful audit lens.
- **TIGER-Lab over AA for any benchmark with a published-paper
  methodology.** AA is a moving target; if a benchmark has a
  reference implementation (like MMLU-Pro), the academic source is
  more defensible.
- **Auto-generate everything that depends on `benchmarks.json`.**
  The per-model appendix table had drifted across multiple columns
  (EQ-Bench Elo, MMLU-Pro accuracies) because it was hand-maintained.
  `build_per_benchmark_table.py` makes it a one-command regen.

## Open / next

- **NeurIPS prose pass**: the `\citet → \cite` mechanical replacement
  produces awkward sentences like "introduced by [1], the DAT asks..."
  Some sentences want manual rewriting to read smoothly with numbered
  bracket citations.
- **NeurIPS Paper Checklist**: required by the call for papers; not
  yet filled in (commented stub in `main_neurips.tex` points to the
  scaffold in `Styles/neurips_2025.tex`).
- **`\bibliographystyle{plain}` confirms only `\cite` works**; if the
  user wants author-year style for NeurIPS, they'd swap to
  `unsrtnat`/`plainnat` and re-add natbib.
- The greedy-DAT figure caption uses n=2078 trials over 54 models;
  if the pool grows again, recompute via the snippet in this log.
