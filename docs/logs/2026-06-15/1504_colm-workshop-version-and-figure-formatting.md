# COLM workshop version + percentile radar/table + figure formatting

**Date:** 2026-06-15 · **Track:** plot_twist · **Paper:** `papers/pt2cb-iclr-2027/` (T²C-Bench)

## Summary

Long formatting/figure session on the plot-twist paper, split across two deliverables:
(1) a **percentile-based Figure 4 radar + Table 1** overhaul of the results presentation, and
(2) a **separate COLM 2026 workshop version** (`colm_main.tex`) for the *Scientific Understanding of
Foundation Models* (Sci-FM) workshop (deadline **June 23 AoE**), de-anonymized with HF/GitHub badges
and single-column wrapped figures. The paper lives in its own Overleaf git repo
(`papers/pt2cb-iclr-2027`, branch `master`) — **fetch + rebase before every push** (many concurrent
Overleaf edits landed mid-session; several pushes required rebase/conflict resolution).

## Figure 4 → radar small-multiples (`make_tc_radar.py`, `fig_radar.tex`)

Replaced the old 5-panel `tc_scorecard` with a **3×3 grid of radar profiles**, one system per panel,
the expert-human profile overlaid on every panel as a dashed reference.

- **Rows = the three failure modes**, headed **(a) Expert humans**, **(b) Mode collapse**,
  **(c) Breaking the world model**, **(d) Incoherent**. Row 0 is split: (a) over the single human
  panel, (b) over the two model panels.
- **Percentile scale** (chosen over z despite the caveat that percentile is non-linear and makes the
  human→Claude *coherence* gap look bigger than *diversity*, which drives the z-composite): rings =
  per-facet **percentile rank within the 72-model pool** (25/50/75, median ring dashed);
  "Overall pctl" = mean of the four facet percentiles.
- **Curated exemplars** (user-picked, data-driven): (b) under-diversification = `claude-opus-4.6` +
  `claude-opus-4.5` (diversity craters, everything else high); (c) = `gemma-4-31b`,
  `deepseek-v3.2-exp`, `qwen3-next-80b` (high surprise+coherence, realism collapses; Gemma realism
  p2). Dropped `gpt-5.4` (mediocre surprise, not a clean example).
- **Row subtitles**: single-line, prominent bold "(x) Title" + same-size unbold parenthetical for
  (c)/(d); (b) = "Mode collapse (high quality, low diversity)". Uniform auto-fit font so every header
  fits one line; wider title↔paren gap; raised above panel titles.
- Iterations: removed the legend; big camera-ready fonts; radial range tightened; facet-label offsets
  (only Realistic/Coherence pushed out); vertical row-gap tuning.

## Table 1 = percentile top-systems (`make_tc_pctl_table.py`, `tab_pctl_top.tex`)

Brought back the leaderboard the scorecard used to carry, as a **booktabs table of the top-8 systems
per facet + Overall, all as pool percentiles**, humans bolded. Added two extra fully-populated rows
at **rank 9 (Diversity)** and **rank 14 (Surprise)** — the two facets where humans fall outside the
top eight. Full (non-abbreviated) column titles; model names abbreviated (`-aNb`/date suffixes
stripped) so the `\resizebox` scales the table **up** instead of shrinking it. (A full 72-row
appendix longtable was briefly added, then dropped — "just the abbreviated version in main body".)

## Other figure/text changes

- **Removed the "Key Failure Modes of LLMs" subsection** (prose + the three per-mode tables
  `tab_mode_{diversity,realism,coherence}.tex`, now deleted) — the radar + caption carry that story.
- **Figure 1** (`tc_over_time.py`): y-axis → **mean facet percentile** (was mean z); removed the
  "Frontier (best so far)" legend label (kept the step line); enlarged the ceiling label + 3 model
  callouts (8/7.5 → 11pt); nudged `claude-sonnet-4.5` between the y-axis and the staircase.
- **Removed the `05_method` section** (deleted file, removed input, rephrased the dangling
  `\S\ref{sec:method}`).
- **Long-URL wrapping**: `\PassOptionsToPackage{hyphens}{url}` before `acl`/`colm` so bibliography
  URLs (Short Story Guide) break at hyphens instead of overflowing.

## COLM 2026 workshop version (`colm_main.tex`)

Downloaded the COLM template (`colm2026_conference.sty`/`.bst` from `COLM-org/Template`). Created a
**separate main file** reusing the same `sections/`, `figures/`, `tables/`, `refs.bib` as the ARR
`main.tex`. Single column, Palatino (tgpagella/mathpazo), natbib.

- **De-anonymized**: option `[submission]` → **`[preprint]`** (shows authors, drops review line
  numbers; comment notes how to flip back for the double-blind submission).
- **HF + GitHub badges** beneath the author block: downloaded the official 🤗 logo
  (`figures/hf_logo.png`) and GitHub mark (`figures/github_logo.png`); two centered `\href` lines to
  `huggingface.co/datasets/PLACEHOLDER/T2C-Bench` and `github.com/PLACEHOLDER/T2C-Bench`.
- **Running-header fix**: the COLM style sets `\lhead{Preprint. Under review.}` *inside*
  `\@maketitle`'s box (local, lost after the title) — re-assert the option-appropriate header right
  after `\maketitle` so the banner shows on every page.
- Local-build font deps (Overleaf already has them): `tlmgr --usermode install tex-gyre psnfss
  urw-base35 palatino fpl mathpazo wrapfig`.

## COLM figure formatting via `\ifcolm` (gated so ARR is untouched)

Added `\newif\ifcolm` (`\colmtrue` in colm_main, `\colmfalse` in main.tex) to gate single-column
choices in the **shared** figure files:

- Figs **1/2/5 → half-width `wrapfigure`** (text flows around the figure+caption). Fig 2 is the
  gen-prompt promptbox; fixed its wrap indentation (`breakautoindent=false, breakindent=0pt`).
- Starred floats `figure*`/`table*` → `figure`/`table` for single-column reliability.
- Numbered `\section{Limitations}` (ARR keeps unnumbered `\section*`).
- Intro headline figure: skip `\suppressfloats` in COLM so it sits below the title.
- Replaced `\shortcite` (unsupported by COLM natbib) with `\citeyearpar`.

## Mis-step + recovery (important)

At the user's request ("remove the `\ifcolm` from figures, ignore ACL for now") I **stripped the
conditionals** and exposed a wrapfig `[N]` line-count param. This **broke the two-column `main.tex`**
(figures lost their `figure*`/column formatting → cramped). Per the user, **reverted** the commit
(`git revert`) back to the `\ifcolm` split (plain wrapfigures, no `[N]`). **Lesson:** the `[N]`
spacing knob (and any COLM-only change) must live **inside** the `\ifcolm` COLM branch, never by
deleting the conditional.

- **Wrapfigure spacing**: excess space above/below the half-width figures was `wrapfig`'s
  `\intextsep` (default ~12pt) → set to **4pt** in `colm_main.tex`.

## State / conventions

- Both builds compile clean: `main.tex` (ARR, two-column, ~14pp) and `colm_main.tex` (COLM,
  single-column, ~15pp). COLM is **~15pp and must be trimmed to the 8-page limit** before submission.
- On Overleaf, set `colm_main.tex` as the main document to build the workshop version.
- Generator scripts (`make_tc_radar.py`, `make_tc_pctl_table.py`, `tc_over_time.py`) live in the
  **main repo** (`src/plot_twist/scripts/`) and are currently **uncommitted there** — only the paper
  repo (Overleaf) has been pushed this session.
- Local build: user-mode `tlmgr` installs above; `.venv/bin/python` for figure generation.

## Open follow-ups

- Trim the COLM body to 8 pages (preliminary cut).
- Align the radar caption wording ("under-diversification"→"mode collapse" etc.) with the new
  subtitles.
- Optionally re-add the wrapfig `[N]` spacing knob **inside** the `\ifcolm` COLM branch.
- Commit the `src/plot_twist/scripts/` generators to the main repo.
