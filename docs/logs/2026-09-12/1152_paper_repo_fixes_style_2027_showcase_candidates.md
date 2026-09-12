# 2026-09-11 → 09-12 — Overleaf build fixed and numbered, ICLR 2027 style, preprint header, showcase candidates

Continuation of the 09-09 → 09-11 session (see
[`../2026-09-11/1321_inventive_multiples_paper_figures_retest_prep.md`](../2026-09-11/1321_inventive_multiples_paper_figures_retest_prep.md)
for the analysis and the repo consolidation proposal). This log covers the work after that log was written.

## Summary

The consolidated Overleaf repo now builds cleanly with figure assets numbered in reading order, the
ICLR 2027 style files, a working running header (a fancyhdr ≥ 4 incompatibility in the stock ICLR
template) and a new preprint mode. A candidate set of "unanimously good" inventions was compiled for a
creativity showcase figure, with three layout mocks; a plainness-based ranking was tried and reverted.

## Tasks completed

**Paper repo (Overleaf, four scoped pushes; local clone merged after each)**
- Build fixes: the unbalanced `\textbf{(\}` in `sections/06_results.tex` closed as an empty `(#3b)`
  heading (no prose added); `\Cref{subsec:distinguishing_cc_from_cg}` → `sec:cg_vs_cc`; the
  reproducibility statement is now `\input` after the conclusion, with sections renumbered
  (08 reproducibility, 09 acknowledgements, 10 author contributions).
- Figures numbered in reading order: `media/figures/01_overview.png`, `02_cc_overview.png`,
  `03_generalization_taxonomy.png`, `04_abstraction_failure.pdf`, `05_facet_corr.png`; fragments
  `media/fig_01_overview.tex` … `fig_05_facet_corr.tex`. The aux file confirms Figures 1–5 in that order.
- ICLR 2027 style: `setup/iclr2027_conference.{sty,bst}` from the ICLR Master-Template replace the 2026
  files. The upstream 2027 sty differs from 2026 only in the header year; the left-aligned 15 pt author
  block was carried over. The bst is unused (`abbrvnat` is set in `configurations.tex`).
- Running header restored: the stock ICLR sty sets `\lhead` inside the `\vbox` of `\@maketitle`, and
  fancyhdr ≥ 4 makes that assignment local (global only under the deprecated `compatV3`), so the header
  was empty on every page — before the style swap too. The sty now sets it once `\AtBeginDocument`.
- Preprint mode: `\iclrpreprintcopy` (final-copy layout, author names, no ruler, header "Preprint");
  `setup/imports.tex` uses it, with the two alternatives noted on the same line.
- Every push was verified by a full local `latexmk` (0 errors, 0 undefined references, 29 pages), with
  page text checked via pypdf (installed in the venv; `pdftotext` is not on this machine).

**Showcase candidates (main repo)**
- `src/kg_creat/scripts/compile_showcase_inventions.py`: collects every invention all three panel
  judges passed (blend: generic space valid, coherent, scope 3; analogy: valid, coherent, and every path
  triple factual) into `analysis/showcase_inventions.json`, with the tagged blend structure or the
  source→image projection pulled from the response files, and writes a ranked listing
  (`scratch/showcase_inventions/candidates.md`; top by originality, one per anchor pair, ≤ 2 per model).
  105 of 1,033 blends qualify (31 models, 24 anchor pairs); 286 of 1,037 analogy inventions (35 models,
  all 30 pairs). Top producers: GPT-5.6 Sol (11 blends, 19 analogies), Opus 4.5, Gemini 3.1 Pro, Grok 4.5.
- Three layout mocks in `scratch/showcase_inventions/` (`mocks.py`): (A) card gallery — coined name in
  provider colour, generic space, properties with provenance chips (input 1 / input 2 / fused /
  emergent), analogy cards as source → image rows; (B) the same as a four-column booktabs table with the
  chips; (C) originality × surprise scatter with the unanimous set highlighted and callouts.
- Tried and **reverted** (commit `fd82ced`, revert `dc1d7f9`): ranking candidates by vocabulary
  plainness (wordfreq Zipf; rare-word count then mean frequency) with a ≥ 2-projection floor for
  analogies. It surfaced plain but thin inventions from small open models; the user asked to go back.

## Files modified / created

- Paper repo: `main.tex`, `setup/iclr2027_conference.{sty,bst}` (new; 2026 files removed),
  `setup/imports.tex`, `sections/06_results.tex`, `sections/14_app_related_work.tex`, renamed
  `sections/08–10`, renamed `media/fig_0N_*.tex` and `media/figures/0N_*`.
- Main repo: `src/kg_creat/scripts/compile_showcase_inventions.py` (new),
  `data/kg_creat/kombine_test30/analysis/showcase_inventions.json` (new, force-added),
  `scratch/showcase_inventions/{candidates.md, mocks.py, mock_A_cards.png, mock_B_table.{tex,pdf,png},
  mock_C_scatter.png}`, `scratch/paper_previews/kombine_restructured.pdf`, docs (this log, progress,
  research context, structure).

## Key decisions / insights

- The "3/3 judges pass" set is **not** more original or surprising than the rest (median originality
  0.43 vs 0.43 for blends, 0.48 vs 0.48 for analogies): judge-approved quality and the novelty measures
  are independent. Good for a finding, weak for a scatter-based showcase; the cards or table are the
  better layouts.
- Plainness and provider are correlated: the plainest analogies come from smaller open models and are
  also the flattest. Ranking stays by originality.
- Header bug lesson: modern fancyhdr breaks any template that sets headers inside a box; the fix is a
  begin-document hook, not the `compatV3` option.

## Open / next steps

1. Pick the showcase examples and layout (cards vs table); whether "3/3 judges" is on the figure or in
   the caption; whether to cap small-model or single-provider picks.
2. Results section organisation (discussed in chat; the section is the author's to write).
3. Retest30 run still awaits go-ahead (≈$190); two blank Rice + Radio cells in the failures grid.
