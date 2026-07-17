# TwistBench: realism-gated metric, figure/table overhaul, and paper submission

**Date:** 2026-06-24 · **Track:** plot_twist · **Paper:** `papers/pt2cb-iclr-2027/` (TwistBench, Sci-FM @ COLM 2026)

## Summary

Finalized and **submitted** the plot-twist benchmark paper to the *Scientific Understanding of
Foundation Models (Sci-FM)* workshop at COLM 2026. The session's two big pieces: (1) a redefinition
of the headline metric to a **realism-gated z-composite**, propagated through every figure/table; and
(2) a large round of paper writing/figure work culminating in submission, including renaming the
benchmark **T²C-Bench → TwistBench**.

## Headline metric change: realism gate ("V1 hard gate")

- Surprise and coherence now count toward the composite **only when a story is fully realistic
  (realism == 5)**; realism is the **gate**, not a separate 4th facet (avoids double-counting).
  Composite = equal-weight z of `{gated surprise, gated coherence, diversity}`.
- Prototyped 6 gating variants (`explore_realism_gate.py`); user chose V1 (literal "counts only if
  fully realistic", widest human margin, demotes the sci-fi "gamers" hardest).
- Effect: human `overall_eq` +1.21 → **+2.00, still #1/72**; GLM-5.1 (sci-fi twists) and DeepSeek-V3.2
  (low realism) demoted out of the top tier.
- Centralized in `src/plot_twist/join.py`: `REALISM_GATE=5.0`, `gated_means()`,
  `EQ_FACETS=(mean_surprise_g, mean_coherence_g, div)`.

## Verification (the payoff)

- `verify_gate_ablations.py`: under the gate, **no reasoning-effort cell and no prompting cell
  exceeds the human best-8 line**, except Claude-Sonnet-4.5 in-context-regen which **ties** it
  (+2.30 vs +2.29 — a diversity-by-construction effect, reported as a tie). GLM-5.1 medium effort
  (the old "winner") drops to +1.90 < +2.29.

## Propagation + figures/tables

- Gate flowed into: `make_tc_barplot` (tc.json + scorecard), `run_thinking_analysis`,
  `run_prompt_analysis` (cells carry `*_g`), `make_effort_temp_boxplots`, `tc_over_time`,
  `make_over_time_appendix`, `make_tc_vs_temp`, `make_tc_pctl_table`, `make_tc_radar`.
- New `make_tc_leaderboard_table.py` → full appendix leaderboard with **raw** facet values.
- **Percentile ↔ z**: flipped the whole paper percentile→z, then settled per-figure — radar spokes z
  (0-ring = pool mean), Table 2 + leaderboard z, over-time z; `facets_over_time` shows the **gated
  composite built up** so its Overall matches the headline.
- **Effort/strategy boxplot** evolution: dropped the temperature panel → 2-panel (effort, prompting),
  stacked vertically, z y-axis (de-saturates the top vs percentile's ceiling pile-up), annotated the
  human-tying cells (Sonnet-4.6 low, Sonnet-4.5 in-context), rendered as a half-width `wrapfigure`.
- **Figure renaming**: images + wrappers renamed to `fig{N}_*` matching compiled figure numbers
  (1–6 body); generators (`make_tc_radar`, `make_effort_temp_boxplots`) emit the numbered names;
  unused figures moved to `figures/archive/`. (Caveat logged: the §3 task-prompt promptbox is itself
  Figure 3, so the image files are off-by-one — cosmetic only, PDF numbering is correct.)
- Appendix prompt boxes all given captions (now numbered Figs 8–13) and **extracted into
  `prompts/*.tex`**, referenced via `\input`.

## No-spend hygiene

- Moved reveal-annotation caches **outside** the `--overwrite`-wiped `output_dir` (now under
  `downstream/annotate_cache`) so re-runs of the analyses don't trigger paid re-annotation. All
  re-runs this session were instant cache hits; no meaningful new API spend.

## Paper writing (Overleaf repo `papers/pt2cb-iclr-2027`, branch master)

- §3 metric definition rewritten to the gated form; §3 motivating-example realism condition clarified
  (addressed a co-author comment on why rule-rewriting is a *distinct* operation + admits a gaming
  shortcut).
- §4 failure modes: concrete model-attributed examples (Opus-4.5 dead-spouse cluster c9 mode-collapse;
  Gemini-2.5-Pro / DeepSeek-V3.2 "synthetic being"/duplicate for world-breaking), with cluster colors
  matched to Fig. 4 (`clusterAI` #C82828 = c6, `clusterDS` #8C5050 = c9).
- §4 ablations: prompting-strategy + reasoning-effort paragraphs written, referencing the appendix
  prompt figures; footnote listing the 9 effort-controllable models.
- New appendix sections: `app:moves` (step-coding prompt) and `app:strategies` (be-creative,
  twist-summarizer, in-context-regen prompts).
- **Benchmark renamed T²C-Bench → TwistBench** throughout (title, abstract, intro, §3, `\bench` macro);
  metric `\text{TC}` left untouched.
- Reveal-points longtable (Table 4) forced flush-left (`\LTleft=0pt`).
- Limitations brainstorm reframed for the Sci-FM "evaluation science / reliability" audience
  (construct validity of the LLM-judge metric, researcher-degrees-of-freedom in the metric,
  confounds in the human comparison, reproducibility/CIs, statistical fragility at the top).

## Files (this repo)

- **New:** `src/plot_twist/scripts/{explore_realism_gate,verify_gate_ablations,make_tc_leaderboard_table}.py`
- **Modified:** `src/plot_twist/join.py` (gate helpers); `scripts/{make_tc_barplot,run_thinking_analysis,
  run_prompt_analysis,make_effort_temp_boxplots,tc_over_time,make_over_time_appendix,make_tc_vs_temp,
  make_tc_pctl_table,make_tc_radar}.py`
- Paper edits live in the separate Overleaf repo (`papers/pt2cb-iclr-2027`, already pushed/submitted).

## Key decisions

- Realism as a **hard gate at R=5** (not a soft discount, not a 4th facet) — literal "fair play",
  no double-count, demotes gamers hardest.
- Report the Sonnet-4.5 in-context case as a **tie**, not a win (honest framing; the gap is +0.006).
- Headline display in **z** (not percentile) for the de-saturation of the frontier cluster.

## Full task inventory (chronological, exhaustive)

Roughly the order things happened across the session (compaction-spanning; the summary IS this
session's work):

1. **GLM-5.1 story pull/summary** — dumped and summarized the 7 GLM-5.1 medium-effort stories (the
   n=7 cell that exceeded the old human line); found perfect 5/5 surprise+coherence (judge ceiling)
   but 2/7 sci-fi (realism 1–2) dragging realism to ~4.0 → motivated the realism gate.
2. **Realism-gate prototyping** (`explore_realism_gate.py`) — 6 variants (hard R=5, soft R/5,
   threshold R≥4, gated product, ±realism-facet); user picked **V1 hard gate**.
3. **Gate implementation** in `join.py` (`REALISM_GATE`, `gated_means`, `EQ_FACETS`).
4. **Verification** (`verify_gate_ablations.py`) before regenerating — confirmed effort cells all
   below human, one prompting tie (Sonnet-4.5 in-context); decided to report as a tie.
5. **Regenerated** `tc.json` + all main figures; propagated gate to thinking/prompt analyses.
6. **Annotate-cache relocation** to avoid `--overwrite` wiping paid reveal annotations.
7. **`make_tc_leaderboard_table.py`** (new) → appendix raw-facet leaderboard.
8. Effort/strategy boxplot: **drop temperature panel** → 2-panel; then several iterations —
   percentile y-axis → back to **z**; **stack vertically**; **half-width wrapfigure**; `\intextsep`
   plus an explicit `[N]` line count; **annotate** Sonnet-4.6 (low) / Sonnet-4.5 (in-context), font size 10,
   shifted clear of the "(a)" title.
9. **Ablation-verification pushback** — user asked to confirm interventions don't beat humans before
   any regen; confirmed and reported.
10. **Percentile→z flip** across the whole paper; then per-figure reconciliation (radar spokes z with
    0-ring = pool mean; over-time gated z; `facets_over_time` gated build-up so Overall = headline).
11. **Figure renaming** to `fig{N}_*` + wrappers + `\includegraphics`/`\input` + generators; unused
    figures → `figures/archive/`. Discovered/logged the task-prompt-is-Fig-3 off-by-one.
12. **Appendix reorg** — Full-leaderboard table moved to the top; commented out over-time companions
    and creativity-taxonomy sections; captioned all prompt boxes (Figs 8–13); **extracted prompts to
    `prompts/*.tex`** via `\input`.
13. **`app:moves`** (step-coding prompt) and **`app:strategies`** (be-creative, summarizer,
    in-context-regen prompts) appendix sections added.
14. **§4 failure-mode prose** — concrete model examples folded in (Opus-4.5 c9 mode-collapse;
    Gemini-2.5-Pro/DeepSeek-V3.2 world-breaking), cluster color-coding (c6 red `#C82828`, c9 maroon
    `#8C5050`) matched to Fig. 4.
15. **§4 ablation paragraphs** (prompting + reasoning-effort) written w/ appendix-figure refs and the
    9-effort-model footnote.
16. **§3 realism clarification** (Alexi comment: rule-rewriting is a *distinct* operation + gaming
    shortcut) and metric-definition + backronym cleanups.
17. **Benchmark rename T²C-Bench → TwistBench** everywhere (title/abstract/intro/§3/`\bench`), metric
    `\text{TC}` untouched; resolved several live-merge conflicts (co-author proofreading in parallel).
18. **Table 4 (reveal points) flush-left fix** (`\LTleft=0pt`; dropped `center` wrapper).
19. **Limitations brainstorm** — general + reframed for the Sci-FM "evaluation science / reliability"
    audience (construct validity of the LLM-judge metric, researcher DoF, human-comparison confounds,
    reproducibility/CIs, top-of-leaderboard statistical fragility).
20. **Copyediting help** — smoother phrasings for a couple of §3 sentences (advisory).
21. **Paper SUBMITTED**; closing tasks + this log.

Note: many paper edits were pushed to the Overleaf repo throughout via a tight fetch→merge(-X theirs)
→push loop because a co-author was proofreading live; every push ended in-sync.

## Open / next

- The off-by-one in image filenames vs. compiled figure numbers (task-prompt = Fig 3) is unresolved —
  cosmetic; re-sync the four body image names if desired post-deadline.
- Intro/abstract still say "Claude Opus 4.5" for the in-context match while results say
  "Claude-Sonnet-4.5" — reconcile (flagged to user).
- Memory `plot-twist-headline-metric` updated to the gated definition.
