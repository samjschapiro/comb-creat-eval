# Camera-ready recompute: MMLU-Pro-only specificity, the RAT, and Kombine as a fourth construct

*2026-10-03 · dat_eval track (with kg_creat data) · NeurIPS 2026 camera-ready*

**Question.** For the camera-ready of "Do Semantic Distance Tests *Actually* Predict Creativity in LLMs?" there were four questions:
- What do validity and specificity look like when specificity controls for MMLU-Pro alone, as promised to reviewer F63x?
- What do they look like on the current data?
- What happens with the RAT added as a test?
- What happens with the Kombine benchmark added as a fourth construct, combinatorial creativity?

**Answer in one paragraph.** Dropping Arena Overall from the control set changes the story for creative writing. DAT now has significant specificity on all three creative-writing benchmarks (+.38\*, +.50\*\*, +.68\*\*). PACE, which the submitted paper called "mostly a capability proxy", now has significant specificity on Arena CW (+.41\*) and Mazur CW (+.52\*). On divergent thinking, scientific ideation and combinatorial creativity, no test reaches significant specificity. PACE gains significant validity on LiveIdeaBench (+.44\*, n = 25), so "no test predicts scientific ideation" now holds for specificity only. Kombine behaves like creative writing: every test except CDAT/CDAT-N is valid on it (DAT +.66\*\*, PACE +.70\*\*\*, RAT +.58\*\*, CDAT-A +.54\*), and none keeps significant specificity. As a whole, Kombine correlates with the creative-writing benchmarks at r = .67–.79. In a separate analysis, blending originality turns out to be the part of Kombine that existing benchmarks miss, and only CDAT predicts it beyond capability.

## Data & sampling

- **Models (tests):** 54 instruction-tuned LLMs from 10 providers. Test scores come from the paper's own pipeline: `data/dat_eval/run_v1/downstream/scores_v1/results/multi_embed_scores.json`, regenerated 2026-07-27 after the CDAT gate fix. Each test is z-scored within each of GloVe, FastText and SBERT and then averaged.
  - DAT and CDAT pool T = 1.0, 1.5 and 2.0; PACE uses T = 0.
  - RAT is zero-shot strict accuracy on 30 Bowden & Jung-Beeman items (pilot_v1 + expansion_v1).
  - It is a convenience sample of models reachable by API.
- **Benchmarks:** `configs/comb_eval/benchmarks.json` (NoveltyBench now n = 33 values, 13 of them self-run). Kombine overall comes from `data/kg_creat/kombine_test30/scores/composite.json` (35 models).
- **Control:** MMLU-Pro (TIGER-Lab). It is taken from `benchmarks.json` plus five exact-name matches from `data/dat_eval/mmlu_pro_raw.json`: Kimi K2, GLM-4.5-Air, Gemini 3.1 Pro, Gemini 3 Flash, GPT-5.2.
- **Unit and test:** the model. Two-sided Pearson t-test; df = n − 2 for validity and n − 3 for specificity.
- **Per-benchmark n (validity/specificity, any test):** Arena CW 54/40, EQ-Bench CW 35/27, Mazur CW 21/18, Hivemind 25/21, NoveltyBench 27/24, LiveIdeaBench 28/22, Kombine 20/18. Per-cell n is in `data/dat_eval/neurips_camera_ready/cells.json`.
- **Power:** at n ≈ 18–24 with one control, only |r| ≳ 0.55–0.62 is detectable. Nulls on divergent thinking, scientific ideation and Kombine rule out large effects only.
- **What this frame is not evidence about:**
  - Base models.
  - Models newer than the pool. Kombine's newest 15 models have no MMLU-Pro score and fall out of every Kombine specificity cell, and only 20 of the 35 Kombine models overlap the paper's test pool.
  - Creative constructs beyond the four measured.
- **Why the numbers differ from the rebuttal.** The rebuttal's NoveltyBench and LiveIdeaBench numbers came from a script that required Arena Overall even for validity, giving smaller n. The numbers here follow the paper's convention: validity uses every model with the test and the benchmark.

## Findings

### 1. With MMLU-Pro as the only control, DAT and PACE both keep creative-writing-specific signal

| test | Arena CW v / s | EQ-Bench CW v / s | Mazur CW v / s |
|---|---|---|---|
| DAT | +.47\*\*\* / **+.38\*** | +.73\*\*\* / **+.50\*\*** | +.60\*\* / **+.68\*\*** |
| PACE | +.71\*\*\* / **+.41\*** | +.71\*\*\* / +.30 | +.76\*\*\* / **+.52\*** |
| CDAT-A | +.54\*\*\* / **+.37\*** | +.48\*\* / +.18 | +.24 / +.04 |
| RAT | +.62\*\*\* / +.15 | +.44\* / +.08 | +.51\* / +.28 |

n (v/s) = 54/40, 35/27, 21/18 at most per benchmark. The submitted paper controlled for Arena Overall too, which shares a method with Arena CW. That absorbed most of the creative-writing variance and made PACE look like a capability proxy. Under MMLU-Pro alone, the claim "PACE is mostly a capability proxy" no longer holds. Only the RAT shows the pattern of high validity with near-zero specificity.

### 2. No test has significant specificity on divergent thinking, scientific ideation or Kombine

- **Best by v + s:** CDAT-N on both divergent-thinking benchmarks (Hivemind +.24/+.35, NoveltyBench +.35/+.35); PACE on LiveIdeaBench (+.44\*/+.28) and on Kombine (+.70\*\*\*/+.22).
- **Specificity nulls:** none of these specificities is significant (all p > .10).
- **"CDAT is the best predictor of divergent thinking"** becomes "CDAT-N ranks highest, not significant", as promised in the rebuttal. Gated CDAT itself is at +.25/+.29 and +.26/+.26.

### 3. Kombine behaves like a creative-writing benchmark

- **Validity:** DAT +.66\*\* (n = 20), PACE +.70\*\*\* (20), RAT +.58\*\* (20), CDAT-A +.54\* (19), CDAT +.04 (18), CDAT-N −.27 (19).
- **Specificity:** all between −.26 and +.35, none significant (n = 16–18).
- **Construct overlap:** Kombine overall correlates with Arena CW +.67\*\* (17), EQ-Bench CW +.71\*\* (13), Mazur CW +.79\* (9), MMLU-Pro +.61\*\* (20).
- **Small-n pairs:** the correlation with NoveltyBench is −.90\*\* but rests on n = 7, so it should not be read as a finding. Hivemind and LiveIdeaBench overlap on only 6 models.

### 4. Blending originality is the part of Kombine the other benchmarks miss (separate analysis)

`docs/logs/2026-10-03/kombine_as_semantic_distance_benchmark.md` scores the tests against Kombine's individual criteria. That analysis uses DAT and CDAT at T = 1.0 from one embedding, so its numbers are not interchangeable with Table 1.
- **Separate from capability and creative writing:** blending originality and blending emergent originality correlate with MMLU-Pro at −.34 and −.03, and with Arena CW at −.39 and −.15 (n = 15–17).
- **Only CDAT predicts it:** CDAT is the only test whose specificity on originality is significant. The three cells surviving Benjamini–Hochberg over 88 tests are association originality +.71\*\* (n = 18), analogy emergent originality +.64\*\*, and blending originality +.64\*\*.
- **What this suggests:** the overall Kombine score adds little beyond creative writing, but inventiveness under blending is a candidate construct that the current six benchmarks miss.

## What changed in the paper (local, `papers/iccc-2026`, not yet pushed)

- **Table 1 moved to the appendix** (new section "Validity and Specificity for Every Test--Benchmark Pair", `app:correlations`). The results now centre on statistical significance, so the best-test green shading was removed as ad hoc; bold still marks p < .05.
- **Table 1 body:** it is now generated (`tables_neurips/main_result_body.tex`, by `src/dat_eval/scripts/neurips_camera_ready.py`).
  - It adds a RAT row and a Combinatorial Creativity / Kombine column.
  - Specificity is MMLU-Pro only, and every number is recomputed on the current data.
  - The caption in `main_result.tex` is unchanged.
- **Headline figure (`figures/neurips/fig_headline.pdf`):** panel (a) has four constructs and panel (b) seven benchmarks.
  - All points and ceilings are computed live from the same cells as Table 1; the submitted figure had them hard-coded.
  - The ceilings use R = the benchmark's correlation with MMLU-Pro.
  - Labels are placed automatically so they don't collide.
- **Figure 3 (`figures/neurips/fig_benchmark_correlations.pdf`):** every cell is coloured by r, but the value is printed only for correlations with p < .05, without stars. Kombine is added to (a) and the RAT is in (b).
- **New filenames:** both figures live in `figures/neurips/`, so the shared `figures/fig_*.pdf` used by the ICML and preprint versions are untouched.
- **Text:**
  - §4 now defines specificity with MMLU-Pro only (k = 1).
  - The frontier example in §4 and in the appendix now reads Arena CW R = .87, cap ≈ .49; NoveltyBench R ≈ −.32.
  - The two `\includegraphics` paths point to the new figures.

## Still owed (author)

- **Results prose (§5)** quotes the submitted numbers, and three of its claims no longer hold:
  - PACE as a capability proxy (finding 1).
  - "CDAT best predicts divergent thinking" (finding 2).
  - "Every test's validity on LiveIdeaBench is near zero" (PACE +.44\*).
- **Contribution 3 in the intro** (PACE r ≈ .73 → r|g ≈ .15) is no longer true.
- **Captions:**
  - Figure 3: lists benchmarks and tests without Kombine and the RAT, and doesn't mention that only significant cells are shown.
  - Headline figure: the gold star and the "three constructs" framing.
  - Table 1: n under each header now means the most models with any test, and the "Green cells" sentence describes shading that no longer exists.
- **Kombine needs a benchmark description and a citation** in §3, a fourth construct in the abstract and intro ("three constructs" appears in several places), and a column in the appendix per-model benchmark table.
- **Preliminaries** quotes benchmark–capability correlations against Arena Overall. That is still factually right, but the MMLU-Pro correlations are now the ones that matter. LiveIdeaBench's "r = 0.36 / 0.59" is stale: the current values are r = .24 / .18.

## Reproduce

```
.venv/bin/python -m src.dat_eval.scripts.neurips_camera_ready     # table body, both figures, cells.json
.venv/bin/python -m src.dat_eval.scripts.kombine_as_benchmark    # Kombine criteria analysis (finding 4)
```
