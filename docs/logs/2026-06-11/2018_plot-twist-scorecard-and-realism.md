# 2026-06-11 — plot_twist: realism dimension + headline scorecard figure

## Summary

Added a **realism** scoring dimension (4th equal-weighted facet) to the PT²CB
transformational-creativity benchmark and built the **headline scorecard figure**
(`tc_scorecard.png`) summarizing all ~72 evaluated systems vs the expert-human gold set,
pushed to the Overleaf paper.

## Tasks completed

**Realism dimension (anti-gaming facet).**
- New `src/plot_twist/scripts/run_realism.py` + `configs/plot_twist/realism.yaml`: scores
  every story 1–5 on whether genuinely-impossible elements actually occur in the story's
  world (grounded vs sci-fi/fantastical "escape hatch"). Single cheap judge
  (`claude-sonnet-4`), durable per-story cache, scores both LLM generations and the human
  gold set. Output: `data/plot_twist/realism/realism_scores.json`.
- Prompt anchors tightened so dreams/hallucinations/hoaxes/figurative prose/unreliable
  narrators do **not** count as unreal (only real ghosts/magic/time-travel/AI/aliens/
  simulation/monsters/psychic powers do). Validated: all 18 STRONG human stories → 5.0.
- Effect: confirms the hypothesis that the prior #1 (DeepSeek) was a sci-fi artifact —
  with realism in the composite, **expert humans reclaim #1**; DeepSeek drops on realism
  (~3.4). Claude Opus 4.8 scores low realism (~2.5): it mode-collapsed onto a
  supernatural "lighthouse/drowning/dead-narrator" template.

**Metric.** Headline = equal-weight **z-composite** of the four facets
(surprise, coherence, diversity, realism), each z-standardized across the evaluated pool
(population-relative, like AGC-Bench's mean-z). Wired `realism_scores` into
`configs/plot_twist/tc.yaml`.

**Scorecard figure** (`src/plot_twist/scripts/make_tc_barplot.py`, heavy redesign):
- Layout: **(a) Overall ($z$)** as a tall vertical panel ranking all systems on the LHS;
  **(b–e)** Surprise / Coherence / Diversity / Realistic as a 2×2 grid of per-dimension
  **top-10** horizontal bars on the RHS (each panel re-ranked by its own facet).
- Font: registered and switched to **Inter** (TTFs in `resources/fonts/inter`).
- Provider colour-coding via `batlowS` categorical palette; expert humans in black;
  long-tail providers (minimax, morph, z-ai, deepcogito, nousresearch, tencent, ai21,
  baidu) collapsed into a single grey **"Other"** bucket (bars + legend).
- Fixed near-square canvas (no `bbox_inches="tight"` blow-up) so fonts stay large at
  `\textwidth`; explicit gap-columns give independent control of the (a)→(b) vs
  (b)→(c) spacing; legend wraps to 2 rows beneath all panels with no clipping.
- Facet value-axes **zoomed to each dimension's data range** (not 0) so small
  differences are visible; 2-decimal x-ticks for (b)/(d); tick count capped to avoid
  crowding; exact values labelled on every bar.
- Caption rewritten per `docs/writing_advice.md` — takeaway-led, self-contained:
  *humans win by balance, not by maxing any single axis* (they are absent from the
  surprise leaders but never collapse on a dimension).

## Files modified / created

- `src/plot_twist/scripts/run_realism.py` (new), `configs/plot_twist/realism.yaml` (new)
- `src/plot_twist/scripts/make_tc_barplot.py` (scorecard redesign; +Inter, +Other group,
  +zoomed facets)
- `configs/plot_twist/tc.yaml` (realism_scores path), `docs/tracks/plot_twist/cost_log.md`
- Overleaf repo `papers/pt2cb-iclr-2027/` (separate git, pushed): `figures/tc_scorecard.png`,
  `figures/fig_scorecard.tex`, `sections/03_benchmark.tex`, `sections/04_results.tex`

## Key decisions / insights

- **Realism is a 4th equal-weighted dimension**, not a filter — it catches the sci-fi
  escape hatch without discarding stories, and it is what restores the human ceiling.
- Wide figures shrink apparent font at `\textwidth`; the fix is a fixed near-square
  canvas + explicit margins, not `bbox_inches="tight"`.
- Zooming the facet axes to the data range is honest here because every bar carries its
  exact value label.

## Open questions / next steps

- **Thinking-vs-non-thinking ablation** (§4 CREATE-style probe): `llm.py` already has the
  `reasoning` param; pick ~5–6 toggleable models, generate ON vs OFF, compare TC + facets.
- Refresh AGC / Arena-CW / EQ-Bench-CW / MMLU-Pro correlations on the **4-facet** overall.
- Update the findings report to the 4-dimension metric + realism + DeepSeek story.
- Phase 4: CSAM method runner (still pending — benchmark/§3–§4 is the work done so far).
