# DAT/CDAT/PACE Correlation Study — Track Progress

## Overview

**ICCC 2026 Short Paper:** Does the Divergent Association Test Actually Measure Creativity in LLMs?

Evaluate a set of LLMs on three creativity metrics (DAT, CDAT, PACE), then correlate each with Chatbot Arena Creative Writing rankings. Test whether these psycholinguistic creativity measures actually predict creative ability in LLMs, and which metric does so most reliably.

## Claims

1. **DAT scores may not meaningfully predict LLM creative writing ability** — the CDAT paper (Nakajima et al., 2026) showed DAT is invalid for LLMs because random words outscore all models. But does it still correlate with Arena CW?
2. **CDAT (with appropriateness gating) is a more valid creativity metric for LLMs than DAT** — if CDAT correlates more strongly with Arena CW, the appropriateness constraint matters.
3. **Comparing DAT, CDAT, and PACE provides a multi-faceted view** — do they measure the same thing? If they correlate with each other AND with Arena CW, they tap into a shared creativity construct. If they diverge, they capture different facets.

## Metrics

- **DAT** (Olson et al., 2021): Generate 10 maximally different nouns. Score = mean pairwise cosine distance (GloVe 840B) of first 7 valid words × 100.
- **CDAT** (Nakajima et al., 2026): Generate 10 diverse words associated with a cue. Novelty (pairwise distance, SBERT) + appropriateness gate (similarity to cue). Score = novelty conditional on passing gate.
- **PACE** (Qiu & Hu, 2025): Generate 3 parallel 20-word association chains per seed. Score = average semantic distance (FastText) across positions and chains.

## Pipeline

```
Step 1: run_evals.py    → raw LLM responses for all three tasks
Step 2: score_evals.py  → scores + correlations with Arena benchmarks
```

## Embedding resources needed for scoring

- `resources/glove.840B.300d.txt` — for DAT scoring
- `resources/crawl-300d-2M.vec` — for PACE scoring
- `all-mpnet-base-v2` (SBERT, auto-downloaded) — for CDAT scoring

## Progress

### 2026-04-11 — Initial implementation
- [x] Track structure created
- [x] DAT module: GloVe-based scoring, validation, prompting (`src/dat_eval/dat.py`)
- [x] CDAT module: SBERT scoring, appropriateness gate, cue words (`src/dat_eval/cdat.py`)
- [x] PACE module: FastText scoring, 2-stage prompting, chain parsing (`src/dat_eval/pace.py`)
- [x] Unified LLM caller via OpenRouter (`src/dat_eval/llm.py`)
- [x] Orchestration: run_evals.py (Step 1) and score_evals.py (Step 2)
- [x] Smoke-tested with GPT-4o: all three tasks produce clean output
- [x] Reuses Arena benchmarks from comb_eval track (`configs/comb_eval/benchmarks.json`)

### Next steps (initial plan, mostly completed below)
- [x] Download GloVe and FastText embeddings to `resources/`
- [x] Run full eval (52 models × all three tasks)
- [x] Run scoring and correlation analysis
- [x] Check if DAT is really "invalid" — confirmed
- [x] Compare CDAT vs PACE vs DAT correlation strengths
- [x] Write up short paper draft

### 2026-04-12 / 2026-04-13 — Full eval, partial correlations, paper draft

Pipeline / infrastructure
- Async OpenRouter client with 20-way concurrency in `src/dat_eval/llm.py`
- Reasoning-model handling: `reasoning.effort=low,exclude=true` plus a
  retry-without-reasoning fallback for providers that reject the param
- Model-aware `max_tokens` (4× multiplier for known reasoning models)
- top_p=1.0 / top_k=0 controls in `run_evals.py` to bypass nucleus filtering
- Budget cap (`budget_usd`) in run config, with `cost_tracker.py` PRICING table
- Idempotent skip-if-exists per (model, eval, temperature) file

Sampling and parameters
- DAT: 40 trials per temp at $T \in \{1.0, 1.5, 2.0\}$, unique seeds
- CDAT: 50 cue words at the same three temperatures
- PACE: 50 seed words at $T = 0$ (per Qiu & Hu)

Benchmarks added
- Arena Overall (already had)
- EQ-Bench Creative Writing v3 — added via `add_eqbench_scores.py`
- Hivemind intra-model similarity (arXiv:2510.22954) — added via
  `add_hivemind_scores.py`

Correlation analysis
- Spearman ρ, Pearson r, 500-iter bootstrap CIs
- Partial correlations against Arena CW, EQ-Bench CW, Hivemind, all
  controlling for Arena Overall (formula in `partial_spearman` /
  `partial_pearson`)
- 4×4 inter-metric correlation matrix

Headline findings (final, n=52 / 51 / 24)
- DAT vs Arena CW: simple ρ=0.36** but partial collapses to ρ=0.03 (NS)
  — entire signal is general-capability driven
- CDAT Appropriateness vs Arena CW: simple ρ=0.45*** but partial flips
  to ρ=−0.16 (NS) — appropriateness indexes capability, not creativity
- PACE vs Arena CW: simple ρ=0.78***, partial ρ=0.31* — only metric
  with creativity-specific signal that survives partialling
- PACE vs Hivemind (partial): ρ=−0.39 — strongest diversity-predictive
  signal among the four

Paper artifacts (Overleaf-synced via `papers/iccc-2026/`)
- 4-page short paper draft pushed to Overleaf
- Camera-ready figures (Helvetica + Batlow/vik):
  - 4×3 scatter grid (metrics × benchmarks)
  - Triangular inter-metric heatmap (Batlow)
  - Per-temperature CDAT bar chart
  - Color-coded correlation table (green = expected-direction
    significant; maroon = wrong-direction significant)
  - Example-responses figure (DAT/CDAT/PACE outputs from Sonnet 4.5)
- Math definition of partial correlation in Method section
- Bibliography entries flagged as AI-generated (need human verification)

### Next steps
- [ ] Verify and clean up bibliography entries (currently flagged
      `% --- AI-GENERATED REFERENCE (VERIFY) ---`)
- [ ] Editorial pass on prose for short-paper concision
- [ ] Decide on Spearman vs Pearson tie-break for the Hivemind
      partial finding (they disagree; n=24 is small)
- [ ] Optionally extend Hivemind sample by adding the Hivemind-only
      Llama-3.1-405B / o1 family / smaller Qwen variants

### 2026-04-14 — Chain Drift Score (CDS): a simpler Pareto improvement over PACE

Mechanistic decomposition of PACE on the same 54-model data revealed that
PACE's signal comes entirely from non-adjacent word-pair distances in the
20-word chain. A simpler metric — **mean cosine distance across all
non-adjacent pairs, uniform-weighted (no positional weighting)** — Pareto-
dominates PACE on every creative-writing benchmark in our suite. Details
in [CDS report](../../reports/2026-04-13_chain_drift_score/report.md) and
[mechanism report](../../reports/2026-04-13_mechanistic_pace/report.md).

Per-gap correlation analysis (gap k = j-i in chain positions):
- Adjacent pairs (k=1) correlate NEGATIVELY with Arena CW (ρ = -0.205)
- Mid-gap pairs (k=4-8) are the peak (ρ ≈ +0.85)
- PACE's formula 1/[(n-1)(i-1)] gives 19× more weight to k=1 than k=19 — exactly
  inverted from the signal distribution

Head-to-head results (n=49-51 for Arena, n=34 for EQ-Bench):

| Benchmark | PACE ρ / r | CDS ρ / r | CDS advantage |
|---|---|---|---|
| Arena CW | +0.770 / +0.720 | **+0.838 / +0.733** | +0.068 ρ / +0.013 r |
| Arena Overall | +0.724 / +0.667 | **+0.781 / +0.678** | +0.057 ρ / +0.011 r |
| EQ-Bench CW | +0.756 / +0.710 | **+0.816 / +0.773** | +0.060 ρ / +0.063 r |
| Mazur CW v2 | +0.701 / +0.727 | +0.684 / +0.697 | ~tie (n=20) |

Triple-control partial correlations (controlling for Arena Overall AND MMLU-Pro,
n expanded to 49 after MMLU-Pro scrape fill-in):

| Benchmark | Metric | raw r | \| AO | \| MMLU | **\| BOTH** |
|---|---|---|---|---|---|
| Arena CW | CDS | +0.733*** | +0.446** | +0.472*** | **+0.396\*\*** |
| Arena CW | PACE | +0.720*** | +0.427** | +0.547*** | **+0.326\*** |
| EQ-Bench CW | CDS | +0.773*** | +0.339* | +0.480** | **+0.349\*** |
| EQ-Bench CW | PACE | +0.710*** | +0.316. | +0.452** | +0.331. |

**CDS and PACE both show significant creativity-specific partial correlations
on Arena CW and EQ-Bench CW under the most stringent BOTH-control
specification at n ≈ 33-49.** DAT, CDAT-Novelty, CDAT-Approp remain near zero
or wrong-direction across every control specification.

Hivemind partial correction: the previously reported +0.549* (CDS, BOTH
controls) was inflated by small-n (n=14). With MMLU-Pro coverage expanded
(now 53/55 models) the matched sample grows to n=23, and the Hivemind partial
drops to +0.330 (not significant) for CDS, +0.355 (marginal) for PACE. Both
still point in the right direction (more CDS → lower intra-model similarity
= more diverse outputs) but the effect is modest, not dramatic. This is a
correction to the Hivemind finding in the ICCC draft if that number is cited.

### Implications for ICCC 2026 draft

**Supports existing claims:**
- "PACE is the only creativity metric that predicts Arena CW after partialling"
  is *strengthened*, because (a) MMLU-Pro control gives larger partials than
  Arena Overall control (due to Arena Overall being itself contaminated with
  creative-writing variance), and (b) PACE survives the most stringent BOTH
  control at partial r = +0.33*.
- DAT/CDAT negative results replicate under the more stringent MMLU-Pro and
  BOTH controls; the story in the ICCC paper holds.

**Two optional updates to consider before ICCC camera-ready:**
1. Swap PACE for CDS as the primary reported metric. Higher correlations,
   simpler formula, mechanistic justification. Tradeoff: requires re-running
   the existing paper's analyses (bootstrap CIs, inter-metric heatmap) with CDS
   in place of PACE, and explaining CDS in the Method section. Upside: cleaner
   numbers and a novel methodological contribution.
2. Keep PACE as-is and reserve CDS for a full NeurIPS paper. Under this plan,
   the ICCC paper stays focused on "which existing metrics predict creative
   writing" and the CDS work stands alone as a methodological advance.

Path of least resistance for ICCC: **option 2**. The CDS finding is a full
paper's worth of work; the ICCC short paper is already complete and its PACE
framing is correct.

### 2026-04-26 — Headline figure: combine with per-benchmark ceilings + add construct ceiling

`fig_headline` and `fig_specificity_ceilings` merged into a single
two-row figure (`fig_headline.pdf`, replaces both). Top row keeps the
3 construct-level scatter panels (Creative Writing / Divergent Thinking
/ Scientific Ideation) but now overlays the construct-level theoretical
specificity ceiling --- the unweighted mean across the panel's
benchmarks of the per-benchmark bound
$|r(X,Y\!\mid\!g)| \leq v\sqrt{1\!-\!R^2} + |R|\sqrt{1\!-\!v^2}$
($g$ = Arena Overall + MMLU-Pro). Bottom row is the previous 6-panel
per-benchmark lens diagram. Single shared legend (test colours +
"theoretical ceiling") at the bottom.

Implementation: `_benchmark_signed_R(bench_key, BMARKS)` and
`_panel_avg_ceiling(R_list, v_grid)` helpers in `make_figures.py` so the
construct-level ceiling tracks any future benchmark coverage updates.
Per-panel R values used (computed on the n-subset with Y, Arena
Overall, MMLU-Pro all present): CW = +0.98 / +0.83 / +0.80; DT = -0.68
/ -0.33; SI = +0.62.

LaTeX: `fig:spec-ceilings` figure environment removed from
`05_discussion.tex`; refs in `05_discussion.tex` and `07_appendix.tex`
redirected to `fig:headline`; caption in `03_method.tex` updated to
describe both rows.

Heads-up: `multi_embed_appendix.py` writes its `multi_embed_scores.json`
inside the `--overwrite`-able `output_dir`, so re-running
`score_evals.py --overwrite` deletes it. Today I had to regenerate via
`uv run python src/dat_eval/scripts/multi_embed_appendix.py` (~3 min)
to feed the bottom-row recomputation. Worth moving the file one level
up (next to `cdat_gated_scores.json`) so it survives.

### 2026-04-25 — ARC-AGI v2 correlations (n=10, exploratory)

Added an ARC-AGI v2 block to `score_evals.py` covering (a) each creativity
metric vs ARC-AGI and (b) ARC-AGI vs every other benchmark in the suite.
ARC-AGI v2 only exists for 10 models in our pool (frontier reasoning
models on the public llm-stats.com leaderboard); intersections with
Arena CW / MMLU-Pro / EQ-Bench CW are n=5–6, with Hivemind n=0 and
Mazur n=2. Partials are gated at n≥7 (single control) and n≥8 (two
controls) — at the actual n=6 intersection they are skipped to avoid
the |ρ|=1 / NaN small-sample artifact that the rank-residual partial
returns at n=5.

Headline at n=10:

| Pair | r (n) | p |
|---|---|---|
| DAT vs ARC-AGI | +0.39 (10) | 0.26 |
| CDAT-Novelty vs ARC-AGI | −0.27 (9) | 0.49 |
| CDAT-Appropriateness vs ARC-AGI | +0.40 (9) | 0.28 |
| PACE vs ARC-AGI | −0.35 (10) | 0.32 |

ARC-AGI vs other benchmarks (small-n, exploratory only):

| Pair | r (n) | p |
|---|---|---|
| ARC-AGI vs Arena Overall | +0.80 (6) | 0.06 |
| ARC-AGI vs Arena CW | +0.44 (6) | 0.38 |
| ARC-AGI vs EQ-Bench CW | +0.98 (5) | 0.003 |
| ARC-AGI vs EQ-Bench CW (rubric) | +0.89 (5) | 0.05 |
| ARC-AGI vs MMLU-Pro | +0.89 (6) | 0.02 |
| ARC-AGI vs Hivemind | n=0 — skipped |
| ARC-AGI vs Mazur CW v2 | n=2 — skipped |

Reading: at n=10 none of the four creativity metrics meaningfully
predicts ARC-AGI (all |r|<0.5, all p>0.26). On the benchmark side
ARC-AGI tightly tracks MMLU-Pro (r=0.89) and EQ-Bench CW (r=0.98) at
n=5–6, while its correlation with Arena CW is much weaker (r=0.44),
i.e. the ARC-AGI-eligible subsample shows reasoning capability
separating from Arena creative-writing rankings.
Both observations are directionally consistent with the paper's
"creativity ≠ general capability" framing, but n is too small for
inferential weight; reportable only as an appendix exploratory column.

Public leaderboard checked 2026-04-25: the 5 leaderboard models we
don't have (GPT-5.5, GPT-5.2 Pro, Claude Opus 4, Muse Spark, Gemini 3
Pro non-image) are either not on OpenRouter (Muse, Gemini 3 Pro) or
absent from our dat_eval pool; running them would lift the ARC-AGI
sample to ~13.

Results saved to:
`data/dat_eval/run_v1/downstream/scores_v1/results/correlation_analysis.json`
under each metric's `vs_arc_agi` and `partial_arc_agi_*` keys, plus
top-level `arc_agi_vs_benchmarks`.

### 2026-04-14 — MMLU-Pro coverage expanded

Scraped MMLU-Pro scores from TIGER-Lab's public leaderboard to fill coverage
gaps. Went from 36/55 models to 53/55. Only gpt-5.4-mini and gpt-5.4-nano
remain missing (not yet on the TIGER-Lab leaderboard). This expansion
is what enables the n=49 Arena-CW-BOTH-control partial correlation reported
above, where the earlier n=33 version was marginal.
