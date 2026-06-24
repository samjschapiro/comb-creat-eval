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

## Open / next

- The off-by-one in image filenames vs. compiled figure numbers (task-prompt = Fig 3) is unresolved —
  cosmetic; re-sync the four body image names if desired post-deadline.
- Intro/abstract still say "Claude Opus 4.5" for the in-context match while results say
  "Claude-Sonnet-4.5" — reconcile (flagged to user).
- Memory `plot-twist-headline-metric` updated to the gated definition.
