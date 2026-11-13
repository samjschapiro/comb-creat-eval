# 2026-11-13 15:01: rebuttal thread triage, A7nz replies, test–retest scored

**Context.** The user shared the full OpenReview thread (cGKy, En17, A7nz; all 3, weak accept) and noted
that cGKy's review is largely LLM-written.

**Done**
- `docs/tracks/kg_creat/rebuttal_iclr_2027.md`:
  - A7nz section written;
  - the missing cGKy points added (Q1, Q4, Q5, ICC / 66%, human baseline, rubric comparability, version
    effect);
  - cGKy triaged into priority and short-answer points;
  - S1 model list corrected;
  - reviewer renamed eN17 → En17;
  - Q1 filled with the test–retest result.
- **OpenRouter budget check.** $41.11 left. A fresh same-item resample of 10 models × 3 tasks would cost
  about $33–45 (too tight). Scoring the already-elicited retest30 was about $20–35, so it was chosen.
  The LiteLLM gateway is unreachable (DNS).
- **Scored retest30** with `configs/kg_creat/kombine_retest30_panel_score.yaml`, after switching its
  factuality judge from gpt-oss-120b to claude-haiku-4.5 to match the paper.
  - A monitor kill-switch watched the key (kill below $3).
  - Actual spend about $24; the ledger records $54.95 at its own list prices.
  - Then ran `rescore_split_originality`.
- **New `src/kg_creat/scripts/analyze_retest.py`.** Recomputes test30 originality over the same 10-model
  pool, reusing the `robustness_originality_pool` machinery. Reports per-model differences with item
  bootstrap CIs, plus Pearson, Spearman, ICC(A,1) and item SE.
- **Report** `docs/reports/2026-11-13_kg_creat_rebuttal_test_retest/report.md`:
  - **Overall:** r = 0.92, ICC 0.92, |Δ| 2.7 points, 0/10 models with a CI excluding 0.
  - **By task:** blending 0.95, analogy 0.87, association 0.65.
- Updated progress, `research_context` and structure.

**Caveats.** 7 of 10 models are Claude. Three models' sampling conditions changed between the sets. The
retest mixes item and draw variance, so it is an upper bound on resample noise.

**Next.** S2 when the ratings arrive; trim cGKy's replies; optional human check of the oracle generic
spaces.
