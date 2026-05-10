# 2026-05-09 19:57 — Sync substantive ICML/NeurIPS paper edits

## Summary
Ported substantive main-body content edits made on the NeurIPS branch
(`papers/iccc-2026/sections_neurips/`) over to the ICML branch
(`papers/iccc-2026/sections/`). Layout-driven divergences (`\cite` vs
`\citep`/`\citet`, wrapfigure conversions, `\resizebox`/scriptsize
table tweaks, the figure-and-table relocation from §4 → §5, equation
alignment compaction, appendix `\sigma`/`\cdot` cleanups, and the
NeurIPS Paper Checklist file) were intentionally NOT ported — those
are layout choices specific to NeurIPS single-column.

## Tasks completed
- Diffed `sections/` vs `sections_neurips/` and `tables/` vs
  `tables_neurips/` to separate substantive content edits from
  NeurIPS-only layout changes.
- Pulled latest from Overleaf master before editing (caught a second
  commit `ba4d137` that had additional substantive NeurIPS edits the
  prior diff missed).
- Ported the following content edits to the ICML branch, using
  `\citep` to match ICML's natbib convention:
  - **00_abstract.tex** — full rewrite: dropped `(LLMs)` redefinition
    and "as it stands"; "two psychometric criteria" → "two criteria";
    "far from theoretical upper bounds" → "far below the theoretical
    limits"; final sentence shortened.
  - **01_introduction.tex** — "first systematic and large-scale" →
    "first systematic"; rewrote the Our Contributions intro paragraph
    around the $r=0.98$ benchmark/capability finding; added new
    bullet on PACE-as-capability-proxy; added "in contrast to general
    beliefs" emphasis on the scientific-ideation bullet; trimmed the
    frontier-bound bullet.
  - **02_background.tex** — added training-data-leakage footnote
    citing Stevenson2022; "Finally, **creativity tests…**" → "Finally,
    we argue that **creativity tests…**"; "confounded by" → "correlate
    with" general intelligence (from earlier in this session).
  - **03_preliminaries.tex** — added "cosine" to DAT distance; "Equation
    \eqref{eq:cdat-novelty}" → "the CDAT-N"; removed redundant
    "(see \Cref{sec:method})".
  - **05_results.tex** — "CDAT the best predictor" → "CDAT as the best
    predictor"; "none … a good predictor" → "none … as good predictors".
  - **06_discussion.tex** — appended "as post-training often reduces
    diversity~\citep{yue2025does}" to limitations (cite already added
    to `main.bib` via Overleaf-side commit).

## Files modified
- `papers/iccc-2026/sections/00_abstract.tex`
- `papers/iccc-2026/sections/01_introduction.tex`
- `papers/iccc-2026/sections/02_background.tex`
- `papers/iccc-2026/sections/03_preliminaries.tex`
- `papers/iccc-2026/sections/05_results.tex`
- `papers/iccc-2026/sections/06_discussion.tex`

Pushed as Overleaf commit `db9a680` to
`https://git.overleaf.com/69dc0aaf7d0d6aa7082a44af`.

## Key decisions
- **Pull-before-edit on Overleaf-synced repo.** First batch of edits
  was done without pulling; user flagged this and we pulled, finding
  one additional Overleaf-side commit (`ba4d137`) with the bulk of
  the substantive content edits to port. Lesson: always pull
  `papers/iccc-2026/` before any edit because the user co-edits in
  Overleaf.
- **Layout edits stay branch-local.** The two paper variants
  intentionally diverge on bib style, figure placement, table
  formatting, and the checklist; only main-body wording is kept in
  sync.

## Open questions / next steps
None for this sync. The two variants are now content-aligned through
Overleaf master `db9a680`. Code repo (`comb-creat-eval`) had no
changes this session — main is clean.
