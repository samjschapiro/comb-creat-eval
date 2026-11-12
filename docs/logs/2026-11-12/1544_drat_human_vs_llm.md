# DRAT human study vs the LLM battery: diversity, DRAT-N/DRAT-A, correlation structure

*Session 2026-11-11 → 2026-11-12 · new_tests track*

## Summary

This session looked at the final DRAT human package (Babak Hemmatian, Stony Brook IRB2026-00448: 240 analysed people) next to the 40-model LLM run of the same battery. The headline is that the tests form a multi-dimensional profile in people but load mostly on one capability axis in LLMs. On the DRAT, that axis pushes models toward relevance (DRAT-A) and away from novelty (DRAT-N). The analyses are consolidated in `docs/reports/2026-11-12_drat_human_vs_llm/figures/`. The memo the user asked for (findings write-up in `docs/`) is **not written yet**.

## Tasks completed

- **Data check.** Validated Babak's packages `data/DRAT_data_for_Sam_20261027/` and `data/DRAT_data_for_Sam_20261110/`, and summarised the existing analysis outputs in `data/new_tests/drat_human/`.
  - Analysed sample: `sample == "main" & exclude == False`, 240 people (236 with Block A).
- **Framing for a Nature Machine Intelligence submission.** The narrative: the same tests measure creativity in people but capability in LLMs, building on the NeurIPS paper (arXiv 2605.13450; jagged creativity, point vs distributional creativity).
- **SCTT response-diversity analysis** (new; no length adjustment, by the author's rule).
  - Files: `analyze_sctt_diversity.py` + `src/new_tests/sctt_diversity.py`, config `sctt_diversity_20261111.yaml`.
  - Measures (SBERT distance): within-set, across-call (LLMs only), and population spread between individuals.
  - Results:
    - Within-set variety: humans .497 vs LLMs .378; the median LLM is at the 9th percentile of people.
    - LLMs vary less across calls than within one call (.348 vs .378; 90% of models; ρ = .88).
    - Population spread: humans .58 [.57, .60] vs LLMs .39 [.38, .42], on all 9 scenarios.
  - Caveat: human within-set SBERT distance partly measures phrasing. It correlates −.36 with SCTT creativity, because shared sentence stems lower it and terse fragments raise it.
  - Against the paper's benchmarks (corrected: the main metrics have n = 17–20, not the 10–16 I first quoted):
    - Hivemind intra-sim: across-call −.53 (p = .03, BH .07), population −.69 (BH .01).
    - NoveltyBench utility: n.s.
  - The NoveltyBench DeBERTa classifier sits at its ceiling within sets, and its capability link (−.66 with MMLU-Pro) does not appear with SBERT. The author questioned why it is needed; dropping it is proposed.
- **Point vs distributional plot** (`plot_point_vs_distributional.py`).
  - x = DAT, y = within-set SBERT variety. The author rejected the first y-axis (population distinctiveness).
  - Humans ρ = −.04; LLMs +.32 (+.42 without Llama-3.2-1B).
- **Figure consolidation.** `docs/reports/2026-11-12_drat_human_vs_llm/figures/` holds 01–12 (latest version of each) and a README mapping each figure to its source run. The 10-27 and 11-04 versions are superseded.
- **Link to the NeurIPS paper.**
  - The RAT as a capability proxy is confirmed with a human baseline: RAT–ability .29 humans vs .81 LLMs.
  - LLMs lead on the capability-loaded tests (RAT 97th, DRAT relevance 98th, DAT 68th percentile of humans) but not on SCTT creativity (40th).
- **ARC as general ability.**
  - Local `benchmarks.json` has only ARC-AGI-2 (6 battery models). From the ARC Prize leaderboard, 19 of the 40 battery models have ARC-AGI-1, all from OpenAI, Anthropic and Meta.
  - ARC-AGI-1 vs MMLU-Pro: ρ = .93 on the 11 shared models.
  - DAT–ARC = .91 (n = 19).
  - Recommended as a robustness check, not a replacement. Not added to the repo; the parsed leaderboard was left in the session scratchpad.
- **Figure 05, anchor similarity (humans).**
  - GloVe anchor similarity explains about half of item difficulty: survivors ρ = .72, rated difficulty ρ = −.77, 187 anchor sets.
  - Relation type adds to this: causal and constitutive anchors are easier than their cosine predicts.
  - Panel (a) is partly circular because the gate uses GloVe; panel (b), rated difficulty, is the clean evidence.
- **Correction: DRAT relevance as a count is mostly how many words were typed.**
  - People type 7.2 words on average (SD 2.8); LLMs type 9.95. The count correlates .77 with words typed in humans.
  - Using the **share** of relevant words, the DRAT-A–DRAT-N trade-off is −.59 in humans vs −.63 in LLMs. The earlier −.15 for humans was an artifact of the count.
  - What does differ is the relation with ability:
    - share vs ability: +.01 humans vs **+.53** LLMs;
    - DRAT-N vs ability: +.18 vs **−.51**.
  - Why DRAT-N falls with ability in LLMs: share of relevant words plus gated distance explain R² = .88 of DRAT-N, and ability adds nothing beyond them. Weak models drift off-topic (Llama-3.2-1B: 34% relevant, DRAT-N 84). The sign follows the instruction: the DAT says "be different", the DRAT says "relate to all anchors".
- **Showed the DRAT prompt** (`drat_prompt()` in `src/new_tests/human_battery.py`): the participant-app wording verbatim, T = 1.0, no system prompt. The prompt asks for words related to *all* anchors, while the scoring gate only needs *any* one anchor.
- **Relabelled the comparison figure:** "DRAT ungated" → **DRAT-N**, "DRAT relevance" → **DRAT-A** (`compare_human_llm_battery.py`). Reran it; the numeric outputs are byte-identical.
- **Correlation matrices, human vs LLM** (permutation test; exploratory; not in the repo).
  - Patterns differ: p = .004 for the 6 tests, p < .001 with ability.
  - Overall intercorrelation is higher in LLMs only once ability is included: mean |r| +.17, p = .01; first component 54% vs 38%.
  - DAT, RAT and SCTT converge in LLMs, while the DRAT decouples from them.
  - The DRAT's internal structure is the same in both populations (A–N −.59/−.63, score–N .74/.80), which makes it the candidate anchor for measurement invariance.
- **DRAT-N/DRAT-A weight curves** (new `plot_drat_weight_curves.py`, figure 12). For composite = w·z(DRAT-N) + (1−w)·z(DRAT-A), Spearman ρ against each criterion, with bootstrap bands.
  - **Humans:** best weight ≈ 0.65 (SCTT creativity .37), close to Claire and Roger's 0.8/0.2 for the AUT. Every criterion is positive above w ≈ 0.5.
  - **LLMs, battery:** sign flips around w = 0.5 (MMLU-Pro +.55 → −.51).
  - **LLMs, all NeurIPS benchmarks:**
    - creative writing (Arena CW, EQ-Bench CW) prefers DRAT-A;
    - output diversity prefers DRAT-N (Hivemind +.73 at w = 0.7);
    - Kombine is weakly positive (+.22);
    - Mazur and LiveIdeaBench (n = 10 each) are uninformative.
- **Expanding the battery overlap** (assessed, **not run**; the author declined for now). Of 43 models with Mazur, LiveIdeaBench or Kombine scores outside the battery:
  - 20 are on OpenRouter, but the 5 Gemini models return 403 there;
  - o3-mini, o4-mini and GPT-4-turbo exist only as `:batch` variants.
  - Running the 15 that work now: Kombine 15 → 27, LiveIdeaBench 10 → 13, Mazur unchanged; roughly $25–40.

## Files created or modified

- `src/new_tests/sctt_diversity.py`, `src/new_tests/scripts/analyze_sctt_diversity.py`, `configs/new_tests/sctt_diversity_20261111.yaml`, `scripts/new_tests/run_sctt_diversity.sh`
- `src/new_tests/scripts/plot_point_vs_distributional.py`, `configs/new_tests/point_vs_distributional_20261111.yaml`, `scripts/new_tests/run_point_vs_distributional.sh`
- `src/new_tests/scripts/plot_drat_weight_curves.py`, `configs/new_tests/drat_weight_curves_20261112.yaml`, `scripts/new_tests/run_drat_weight_curves.sh`
- `src/new_tests/scripts/compare_human_llm_battery.py`: labels DRAT-A and DRAT-N
- `docs/reports/2026-11-12_drat_human_vs_llm/figures/`: figures 01–12 + README
- `docs/tracks/new_tests/progress.md`, `docs/research_context.md`, `docs/structure.md`, `README.md`
- Memory: `no-length-control.md`, `local-python-env.md`

## Decisions and insights

- Never adjust for response length in creativity or diversity analyses (author's rule).
- DRAT-A should be the share of relevant words, not the count, wherever it is compared across people or populations.
- The SBERT within-set measure needs an idea-level counterpart before the human side of "point vs distributional" carries weight.
- Ideas from the author's meeting notes (Claire, Roger) to carry forward:
  - frame the paper as measurement invariance;
  - use the DRAT as the cross-population task;
  - find an optimal DRAT-N/DRAT-A composite against an external criterion;
  - the AUT is missing and reviewers will flag it.

## Open questions and next steps

- Write the findings memo (`docs/reports/2026-11-12_drat_human_vs_llm/report.md`).
- Switch DRAT-A to the share in `analyze_drat_human_psychometrics.py`, `compare_human_llm_battery.py` and `score_human_battery_llm.py`, and rerun those downstream steps.
- Pending author go-ahead:
  - drop the NoveltyBench classifier from the SCTT diversity analysis (SBERT only);
  - split its scoring from the analysis.
- Add the matrix-comparison test and a cross-validated composite scan to the pipeline.
- Relabel figures 07 and 08 and the diversity correlates to DRAT-N/DRAT-A.
- Optional:
  - run the 15 available models to grow the Kombine and LiveIdeaBench overlap;
  - add ARC-AGI-1 as a second ability measure;
  - run the AUT on the LLMs;
  - check whether LLMs show the same anchor-similarity difficulty curve as people.
