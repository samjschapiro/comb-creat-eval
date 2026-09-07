# 2026-09-07 — Kombine: pool 30→35, factuality judge migration, thinking-effort study

Multi-day session (2026-09-05 → 09-07). Expanded the model pool to 35, replaced the factuality judge
across the whole pool after discovering it was silently leaving a quarter of all paths unjudged, ran a
rigorous test of the analogy-vs-blending utility claim, and built a six-configuration thinking-effort
study end to end (elicit → score → analyse → camera-ready figures).

## Headline results

- **(#1a) Analogy beats blending by 10 points, not 14.** Macro means over 35 models: analogy
  `U_an` 54.0%, blending `U_bl` 43.8%. Over the 1020 matched model×item cells the gap is **+10.0 pts**
  (analogy 54.3%, blending 44.3%; McNemar exact **p = 2.9e-6**, OR **1.55**, 95% CI [1.29, 1.88]).
  **No item is impossible**: all 30 anchor pairs had at least one model produce both a valid analogy
  and a valid blend, so the gap is not item difficulty.
- The advantage is concentrated on hard items: hardest tercile **+27.0 pts** (p=3.9e-13), middle
  **+15.5**, easiest **−12.2** (blending is *ahead* on the easiest third). Per-item difficulty is
  essentially **uncorrelated across tasks** (r = +0.14, n=30) — a pair that is hard to analogise is
  not the pair that is hard to blend.
- **(#1b)** Models find a valid generic space only **43.8%** of the time, but conditioned on finding
  one they describe a coherent blend **97.1%** of the time. Abstraction is the bottleneck.
- **(Effort study) More thinking buys more output, not better output.** See below.

## Pool expansion 30 → 35, and non-OpenRouter routing

- Added 4 Anthropic models plus `gpt-6-astra-flex` as its own pool entry.
- **New `src/kg_creat/providers.py`** — elicitation routes that deliberately bypass `LLM_BASE_URL`
  so the OpenRouter budget guards stay intact while running on the user's own keys:
  - `OpenAICompatibleProvider` (LiteLLM gateway): reasoning effort is a **top-level**
    `reasoning_effort`, not OpenRouter's `extra_body.reasoning`; models the gateway has not registered
    as reasoning-capable need `allowed_openai_params` to force it through.
  - `AnthropicProvider`: **must stream** (the SDK refuses non-streaming calls >10 min); thinking blocks
    are captured as the trace; `temperature` is unsupported by SDK 1.4; new models reject
    `thinking.type.enabled` (use `output_config.effort` / adaptive).
- `run_elicit.py` gained provider support, `max_tokens_absolute`, and an **all-failed guard** — it now
  raises rather than writing an empty result set when every draw errors at the API.
- Elicitation ran on the user's LiteLLM/Anthropic keys; **OpenRouter was used for scoring only**.

## Factuality judge: gpt-oss-120b → claude-haiku-4.5

The scoring artifact of the session. `gpt-oss-120b` was failing to return a parsable verdict on a large
fraction of paths, which `score.py` recorded as `channel="unjudged"` and the aggregate treated as a
utility failure — silently depressing association and analogy utility.

- **New `src/kg_creat/scripts/rejudge_factuality_haiku.py`** — re-judges every path, preserving the
  original verdict as `factual_gptoss` / `channel_gptoss` via `setdefault`, so re-running is safe and
  never clobbers the original. Takes `--scores-dir`.
- Main pool re-judged; 2,039 unjudged paths → 0. Hallucination rate rose 21% → **27.1%** (the honest
  number; the old one was deflated by unscored paths).
- **The effort study carried the same bug, and it scaled with the independent variable** — unjudged
  rates 27.3 / 41.0 / 47.0% (sol low/med/high) and 76.9 / 89.1 / 86.5% (astra). Reporting that
  uncorrected would have manufactured a spurious "effort hurts association" finding. Re-judged: 2,433
  calls, 2.11M in + 1.17M out, **$6.36**, 382 s at concurrency 32, unjudged now ~1%.
- Decision (user): **stick with claude-haiku-4.5 everywhere** — "for factuality bigger is better."

## Thinking-effort study

`gpt-5.6-sol` and `gpt-6-astra-flex` × {low, medium, high}, 6 configs, scored in **one pooled pass** so
that pool-relative originality is comparable across effort levels (scoring each level alone would make
each level's originality relative to itself).

- `configs/kg_creat/effort/effort_{low,medium,high}.yaml` + `effort_score.yaml`.
- **New `analyze_effort_study.py`** — per-config utility/originality per task, with the unjudged rate
  printed as a standing data-integrity check.
- **New `plot_effort_composite.py`** — three camera-ready figures (paper serif, Okabe–Ito
  colourblind-safe) in `data/kg_creat/effort_study/figures/`: `fig_effort_composite`,
  `fig_effort_dimensions`, `fig_effort_delta`, plus `effort_composite.json`.

**Result: effort does essentially nothing to the composite.**

| config | assoc | analogy | blend | overall | 95% CI |
|---|---|---|---|---|---|
| sol · low | 49.14 | 78.61 | 63.79 | 63.85 | [59.1, 67.9] |
| sol · medium | 46.70 | 71.76 | 67.24 | 61.90 | [56.7, 66.5] |
| sol · high | 44.48 | 73.89 | 71.86 | 63.41 | [58.3, 67.8] |
| astra · low | 44.89 | 70.00 | 64.86 | 59.92 | [54.3, 65.0] |
| astra · medium | 44.41 | 66.37 | 53.75 | 54.84 | [48.6, 60.3] |
| astra · high | 46.62 | 70.34 | 61.64 | 59.53 | [53.6, 64.5] |

- Paired high−low overall: sol **−0.43** [−6.2, +5.4], astra **−0.38** [−8.1, +7.0].
- Of 38 effort contrasts, **only three exclude zero**, all on sol's association — and they are *one*
  effect, not three: surprise and originality are **utility-gated**, so a utility drop drags both down
  mechanically.
- That association drop is itself a **path-length artifact**. Effort lengthens chains (sol 4.81 → 4.95
  → 5.52 hops) and a path is factual only if *every* triple is. Per-triple factuality is near-flat
  (sol 93.9 → 92.2%, z=2.74 p=0.006 but only 1.7 pts; astra 93.8 → 94.2%, p=0.20). Modelling path
  utility as `p_triple ^ mean_hops` predicts the observed rate within 1–3 pts in all six configs.
- **Manipulation check passes**: mean reasoning tokens 1,786 → 4,981 → **13,076** (sol, 7.3×) and
  1,011 → 6,526 → **14,779** (astra, 14.6×). The models genuinely thought an order of magnitude
  longer and scored the same, which makes the null much stronger.
- **Reasoning trace TEXT is not available.** The LiteLLM gateway strips it
  (`merge_reasoning_content_in_choices: false`); only token counts come back. `providers.py` does try
  `reasoning_content` then `reasoning`. Astra was excluded from traces by user decision.
- Volume rises sharply with effort: sol yields 428 → 556 → **802** valid association paths.

## Bugs found and fixed

- **`generic_ok` vs `blend_integration`.** The blend gate is `generic_ok`; `blend_integration` (scope)
  is a *separate* panel field. Three places used scope as the gate. Fixed — frontier generic-space rate
  40.4% → 46.7%. The two fields disagree on **22.2%** of blends (n=1033).
- **`compute_composite` silently dropped `em_originality`** when a follow-up scorer had not been run
  pool-wide, quietly reducing the composite from 6 dimensions to 4 on two tasks. Added a guard that
  refuses unless `--allow-dropped-dims`. It fired for real on the effort study, where
  `rescore_split_originality` had never been run; ran it (judge-free, local MLX, $0) and recomputed.
- **`analyze_inventive_multiples` was silently failing** on `sorted()` over a provider set containing
  `None`; stale JSON was reported as fresh. Fixed the `_provider` fallback.
- **Bootstrap over the wrong sampling frame.** Association is posed over a *different* 30 anchor pairs
  than analogy/blending (union 60), so resampling the union jointly let each task's item count drift
  binomially. Now stratified within task, and paired across effort levels for the delta figure.
- **Misleading per-panel autoscaling** in the effort figures (a 2-point originality range read as a
  collapse). All composite panels now share one y-scale; each dimension row shares one across tasks.
- **A stuck background waiter.** `until ! pgrep -f "scripts/score.py"` never exits, because the
  pattern string appears in the waiter's *own* argv, so `pgrep` matches itself. Use
  `while kill -0 <PID>` instead.
- **`model_names.py` had `BRAND` defined twice and `DISPLAY` three times.** All copies were
  byte-identical so nothing misbehaved, but an edit to the first block would have been silently
  discarded. Deduped.

## New shared module

**`src/kg_creat/model_names.py`** — single source for `LOGO_SLUG`, `BRAND`, `DISPLAY` (35 entries) and
`_provider`, with no matplotlib dependency. Created because the map had been copy-pasted across four
scripts and had drifted nine models behind.

## Other new analysis scripts

- `test_utility_analogy_vs_blending.py` — `U_an` vs `U_bl`, impossible-item drop, McNemar, stratified
  by provider and by item-difficulty tercile.
- `analyze_task_dissociation.py` — matched-cell 2×2, McNemar, panel-reliability disattenuation.
- `test_shared_method.py` — cross-encoder matrix (predictor in one embedding space, outcome in another)
  to answer the shared-method caveat.
- `compute_icc.py` — ICC(2,3) (Shrout–Fleiss) from stored per-judge verdicts; feeds Table 9.
- `rescore_originality.py` — judge-free pool-wide base-originality recompute (originality is
  pool-relative, so adding models requires rescoring *all* models).
- `analyze_blend_integration.py` — scripts the previously one-off blend memo numbers.
- `make_paper_multiples_figure.py` — stacks the multiples matrix over the landscape.

## Paper (`papers/kg_creat-iclr/`)

Pushed earlier in the session: 35-model tables, provider logos in the appendix tables with
shrink-to-fit so nothing overflows horizontally, Table 9 (inter-judge reliability) regenerated,
de-floated prompt boxes so the blending prompt no longer runs past the page bottom, author/affiliation
spacing, and the rebuilt `fig_facet_corr_reduced`.

**Staged locally, NOT pushed** (awaiting approval):
- `#1a` heading "14\% better" → "10 points better"; body 58.5/44.0 → **54.0 / 43.8** plus the
  matched-cell test as its own clause.
- `#1b` 42.8% → **43.8%**, 96.8% → **97.1%**.
- Facet-figure caption "30 models" → **35 models**.
- `05_benchmark.tex`: factuality judge `gpt-oss-120b` → **`claude-haiku-4.5`**.

## Open questions / next steps

1. **The judge justification is an open hole.** The old sentence justified `gpt-oss-120b` by citing
   reliability evidence from `wadhwa2026create`. That citation does not transfer to haiku, so it was
   removed rather than re-attributed — which leaves the judge choice unjustified. Needs a sentence
   pointing at our own Table 9 numbers, written or approved by the user.
2. `content/15_embedding_robustness.tex` and its `main.tex` `\input` are still in `stash@{0}`; its
   rank-stability numbers are from the n=30 era and need regenerating for 35.
3. Optional: add the reasoning-token curve to `fig_effort_composite` as an explicit manipulation-check
   panel.
4. The effort study has 81 residual unparsed paths and a few failed high-effort API calls
   (sol 88/90, astra 83/90 with usage recorded).
