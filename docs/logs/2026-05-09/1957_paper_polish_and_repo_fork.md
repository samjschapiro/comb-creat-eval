# 2026-05-09 — Paper polish, gate-ablation rerun on new LIB pool, repo fork to samjschapiro

## Summary

Late-evening session focused on the JMLR/ICML preprint draft
(`papers/iccc-2026/main_preprint.tex`). Re-ran the DRAT utility-gate
ablation on the refreshed LIB pool (now n=20 with the new facet rows
in `benchmarks.json`); LIB now hits significance on both axes for the
$\max$ gate ($v=+0.57^{**}$, $s=+0.50^{*}$) while $\min$ and $\mathrm{avg}$
lose validity significance at the larger sample. Cleaned up several
loose ends in the appendix: empty TOC, `(R = ±#)` subplot annotations,
"expert" anchor-bank label, missing anchor-bank tables, missing RAT
prompt. Also forked the outer `comb-creat-eval` repo from
`jean-technologies` to `samjschapiro` (private) and pushed the entire
in-progress `src/new_tests/` track + `comb_eval/dat_eval` refresh as
two clean commits to the new origin.

## Tasks completed

### Paper (papers/iccc-2026, Overleaf submodule)

- **Gate ablation rerun.** `scratch/drat_paper_comparison/compute_gate_ablation.py`
  on the refreshed `configs/comb_eval/benchmarks.json` (LIB n=20
  with the new per-facet rows). Updated `tab:drat_gate_ablation` in
  `06_appendix.tex`. Headline LIB row now $v=+0.57^{**}$, $s=+0.50^{*}$
  for $\max$; $\min$ drops to $v=+0.32, s=+0.21$ (no longer sig);
  $\mathrm{avg}$ drops to $v=+0.44, s=+0.34$ (validity now p<.10
  rather than p<.05). Caption rewritten — the old "validity is
  roughly invariant to the gate" sentence is no longer true at the
  new n; replaced with "validity is also highest under $\max$".
- **Removed `(R = ±#)` subplot titles** from `figures/si_headline.pdf`
  (Average + 5 facet panels) and verified `figures/fig_headline.pdf`
  has none. Re-rendered both PDFs and copied into the paper figures
  directory.
- **Fixed empty appendix TOC.** Root cause:
  `\localtableofcontents` from etoc was placed at top level (no
  enclosing sectional unit) and produced an empty TOC. Switched to
  the etoc tag-filter pattern: tag `mainmatter` in
  `main_preprint.tex` and `main_jmlr.tex` (before
  `\input{...01_introduction}`); in `06_appendix.tex` set
  `\etocsettagdepth{mainmatter}{none}`,
  `\etocsettagdepth{appendix}{section}`, then call
  `\tableofcontents`. Compile twice to populate.
- **Rename "expert" → "scientific terms"** throughout the DRAT
  vocabulary-corpus ablation (body, caption, both ablation
  paragraphs in `04_drat.tex`; gate-ablation body and caption in
  `06_appendix.tex`). Plot legend in
  `scratch/drat_paper_comparison/plot_ablation_lines.py` updated
  ("Expert" → "Scientific terms"); `figures/ablation_lines.pdf`
  re-rendered.
- **Added `\subsection{DRAT Anchor Banks}` (`app:anchor_banks`)**
  inside the new `\section{Divergent Remote Association Test}`
  appendix umbrella. Lists all 30 quadruples for both banks
  (scientific terms + ConceptNet relation-distant) in two
  side-by-side tabulars. Resolves the previously dangling
  `\Cref{app:conceptnet_bank}` reference (now `app:anchor_banks`).
- **Added RAT prompt** to the Prompts appendix as
  `\subsection{RAT (stems: cracker, fly, fighter)}` using the exact
  template from `src/new_tests/rat.py:rat_prompt`.
- **Glued anchor-bank headings to their tables.** Wrapped each
  `\paragraph{...bank.}` + tabular pair in a
  `\begin{minipage}{\linewidth}...\end{minipage}` so LaTeX can't
  break a page between heading and table; `\bigskip` separates the
  two minipages.
- **Rebase reconciliation.** During the second push, Overleaf had
  restructured the appendix in the meantime (pulled "Connection of
  DRAT to Existing Tests" and the gate ablation under a single new
  `\section{Divergent Remote Association Test}`). Rebased manually:
  kept the upstream structure, applied my "expert" rename to
  upstream's gate-ablation subsection, and placed the new anchor
  banks subsection between Connection and Gate Ablation.

### Outer repo (samjschapiro/comb-creat-eval)

- Forked to `https://github.com/samjschapiro/comb-creat-eval`
  (private, empty repo created on github.com first).
- Changed `origin` URL via `git remote set-url`. Old jean-technologies
  remote dropped from local config (the GitHub repo still exists
  upstream; not deleted).
- Pushed `main` (74 commits) — first push hit HTTP 408 on the 46 MB
  pack; succeeded with `-c http.postBuffer=524288000`.
- Then committed and pushed all accumulated working-tree changes (52
  files) as two reviewable commits:
  - **`dbe350a` — new_tests: scaffold DRAT pilots/ablations and new
    benchmark wrappers.** `src/new_tests/` modules
    (drat/distinctness/hivemind/llm/noveltybench/rat + scripts), all
    `configs/new_tests/*.yaml` (DRAT k×{expert,conceptnet} ablation
    grid + DRAT pilot phases + eqbench/hivemind/lib/noveltybench/rat),
    `scripts/new_tests/*.sh`, `docs/HYPOTHESES.md`,
    `docs/memos/preference_optimization_for_novelty.md`,
    `docs/tracks/new_tests/{proposals,survey}.md`, `pyproject.toml`.
  - **`950d71d` — comb_eval/dat_eval: refresh benchmarks with LIB
    facets; regen figures.** `configs/comb_eval/benchmarks.json`
    (per-facet LIB rows added, overall LIB averages refreshed),
    `src/dat_eval/scripts/make_figures.py` (fig_headline_combined
    updates, R-annotation removal), `src/comb_eval/scripts/{build,
    augment}_per_test_table.py`, regenerated
    `docs/reports/2026-04-12_preliminary_correlations/figures/`
    pdfs+pngs.

## Files modified / created

### papers/iccc-2026 (separate Overleaf repo, not the outer repo)
- `main_preprint.tex`, `main_jmlr.tex` — added `\etocdepthtag.toc{mainmatter}` before body inputs
- `sections_jmlr/04_drat.tex` — expert → scientific terms; ref to `app:anchor_banks`
- `sections_jmlr/06_appendix.tex` — TOC fix, gate-ablation table+caption update,
  rename, new Anchor Banks subsection, RAT prompt subsection, minipage glue
- `figures/{si_headline,fig_headline,ablation_lines}.{pdf,png}` — regenerated

### Outer repo (comb-creat-eval)
- 52 working-tree files committed in `dbe350a` and `950d71d` (see above).
- `scratch/drat_paper_comparison/{plot_ablation_lines,plot_si_headline_combined,
  compute_gate_ablation}.py` invoked / edited; outputs at
  `scratch/drat_paper_comparison/{si_headline,ablation_lines}.{pdf,png}`,
  `gate_ablation.json`, `ablation_results.json`. (`scratch/` is gitignored,
  so these don't enter version control.)

## Key decisions / insights

- The LIB pool growing from n=17 → n=20 changed the gate-ablation
  story qualitatively. Previously $\min/\mathrm{avg}$ both reached
  validity significance and the gate looked roughly invariant;
  now only $\max$ retains significance on both axes, which is a
  cleaner story for the appendix. Worth checking if the same
  enlargement also strengthens the headline (k, vocab) ablation
  in `04_drat.tex` — those numbers are still cited at the old
  n=19 / smaller-pool values (see open questions).
- The empty appendix TOC was an `etoc` semantics issue, not a
  package conflict. `\localtableofcontents` requires a parent
  sectional unit; at top level it produces nothing. The tag-based
  alternative is the documented idiom for an "appendix-only TOC"
  in a single-document setup.
- The HTTP 408 on the first push to GitHub for the 46 MB pack is a
  recurring symptom for repos this size on default git settings.
  `http.postBuffer=524288000` is the standard fix; could be set
  globally if it recurs.
- `papers/*` is gitignored in the outer repo (treated as separate
  Overleaf-synced sub-repos); the paper work is invisible to
  outer-repo `git status` and pushes. This is by design.

## Open questions / next steps

- The `04_drat.tex` (k, vocab) ablation paragraphs still cite
  pre-refresh LIB values (`v=+0.56*, s=+0.35*` for k=4 scientific
  terms; monotonic series `+0.33→+0.50→+0.56`) and `n=19` in the
  figure caption. The latest `ablation_results.json` shows
  k=4 scientific-terms at $v=+0.57^{**}, s=+0.50^{*}$ at n=20 —
  worth aligning with the gate-ablation table.
- `04_drat.tex` line 74 (the `fig:si-headline` text) still cites
  `n=19` for the headline DRAT vs. existing-tests comparison.
  Same enlargement question.
- The outer repo's `docs/structure.md` and `docs/research_context.md`
  predate the `new_tests/` track and `papers/iccc-2026/sections_jmlr/`
  (preprint/JMLR variants); brought them in line in this session
  but worth a second look next time around.
- The `papers/drat-icml-2026/` directory exists but is mostly a
  stub; status TBD.
