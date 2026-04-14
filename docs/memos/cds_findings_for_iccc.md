# CDS findings and their relevance to the ICCC 2026 paper

**Date:** 2026-04-14
**Status:** log of cumulative findings from the mechanistic PACE / CDS / triple-control partial correlation work. Relates to the [ICCC 2026 short paper](../tracks/dat_eval/progress.md) and reports under [docs/reports/](../reports/).

## One-paragraph summary

Mechanistic decomposition of PACE identified that its creativity-predictive signal comes entirely from *non-adjacent* word-pair distances in the 20-word chain, not from per-step local novelty. A simpler metric (Chain Drift Score, CDS = mean cosine distance over non-adjacent pairs, uniform-weighted) Pareto-dominates PACE on Arena CW, Arena Overall, and EQ-Bench CW. With MMLU-Pro coverage expanded to 53/55 models, both CDS and PACE show statistically significant creativity-specific partial correlations on Arena CW (CDS: r = +0.40\*\*, n=49) and EQ-Bench CW (CDS: r = +0.35\*, n=33) under the stringent BOTH-control specification (Arena Overall AND MMLU-Pro jointly residualized). DAT, CDAT-Novelty, and CDAT-Approp have zero or wrong-direction partial correlations anywhere except DAT-on-Mazur. This strengthens (does not undermine) the ICCC paper's existing PACE finding.

## Findings organized by relevance to ICCC

### Direct supports for claims already in the ICCC draft

**1. PACE's partial correlation with Arena CW survives stricter controls than just Arena Overall.**
The ICCC draft reports partial ρ = 0.31\* (Arena CW | Arena Overall, n=52). With MMLU-Pro as an alternative control, PACE partial r = +0.547\*\*\* on Arena CW (n=49). Under BOTH controls jointly: partial r = +0.326\* (n=49). The PACE-is-creativity-specific claim holds across capability proxies.

**2. DAT and CDAT results replicate under the expanded control specifications.**
- DAT vs Arena CW: partial r ≈ 0 under all three control specifications (Arena Overall, MMLU-Pro, BOTH)
- CDAT-Novelty: partial r ≈ 0 across all specifications
- CDAT-Approp: partial r slightly negative or near zero across all creative-writing benchmarks after any capability control

The ICCC paper's claim that these metrics' raw correlations with Arena CW are capability artifacts is robust.

**3. Arena Overall is itself contaminated with creative-writing variance.**
MMLU-Pro as a control consistently gives *larger* partial correlations than Arena Overall. For CDS vs Arena CW:
- Partial | Arena Overall: r = +0.446\*\*
- Partial | MMLU-Pro: r = +0.472\*\*\*
- Partial | BOTH: r = +0.396\*\*

This suggests Arena Overall — which partially incorporates creative-writing performance via its scoring methodology — over-controls when used as the sole capability proxy. MMLU-Pro is a cleaner proxy (pure academic knowledge, no creative-writing component), and it reveals slightly stronger creativity-specific signal than Arena Overall does. The ICCC draft uses Arena Overall as the control; the findings hold regardless, but the MMLU-Pro version is slightly stronger.

### Corrections to the ICCC draft

**1. Hivemind partial correlation at small n was inflated.**
The draft may cite PACE vs Hivemind partial ρ = -0.39 (n=24). With MMLU-Pro coverage expanded and BOTH-control partials computed, the honest numbers at n=23 are:
- CDS partial r on Hivemind diversity: +0.330 (not significant)
- PACE partial r on Hivemind diversity: +0.355 (p = 0.096, marginal)

Both still point in the correct direction (lower Hivemind = more diverse outputs; positive partial = creativity-aligned). The magnitude is modest, not dramatic. If the ICCC draft cites stronger Hivemind numbers based on n=14 BOTH-control, that should be softened to the n=23 estimates above or omitted.

### New findings not in the ICCC draft (candidates for NeurIPS paper)

**1. Chain Drift Score (CDS): a simpler metric that outperforms PACE.**

Definition: $\mathrm{CDS}(c) = \frac{1}{|\mathcal{P}_n|} \sum_{(i,j) \in \mathcal{P}_n} d_{\cos}(e(w_i), e(w_j))$
where $\mathcal{P}_n = \{(i,j) : 1 \le i < j \le n,\ j - i \ge 2\}$ is the set of non-adjacent chain-position pairs. For a 20-word PACE chain, $|\mathcal{P}_{20}| = 171$.

CDS beats PACE on:
- Arena CW: ρ = +0.838 vs +0.770, r = +0.733 vs +0.720
- EQ-Bench CW: ρ = +0.816 vs +0.756, r = +0.773 vs +0.710
- PACE becomes redundant in hierarchical regression: adding PACE to CDS adds ΔR² = 0.000 (p = 0.88) on Arena CW

**2. Mechanistic interpretation: "sustained drift" — not per-step novelty.**
Per-gap correlation analysis (gap k = j - i between chain positions) on the same 20-word chains:

| Gap k | Arena CW ρ of mean distance at that gap | Interpretation |
|---|---|---|
| 1 (adjacent) | -0.205 | **Negative** — larger local jumps are worse for creative writing |
| 4-8 (peak window) | +0.82 to +0.85 | Peak creativity signal |
| 19 (full chain span) | +0.296 | Drops off (chain coherence breaks down at extreme spans) |

PACE's per-pair weight formula is $\frac{1}{(n-1)(i-1)}$, which depends only on the later position $i$, not on the gap. This puts heaviest weight on pairs involving position 2 (including the seed-to-first-word edge, which is negatively correlated with Arena CW) and lightest weight on long-span pairs. CDS's uniform weighting partially corrects this.

Cognitive interpretation: CDS measures "sustained associative drift" — small coherent local steps that compose into large cumulative semantic displacement — rather than per-step local novelty. This aligns with Mednick's (1962) associative theory of creativity (flat associative hierarchies → reach distant concepts in fewer steps) and with the general character of good creative prose (local coherence + global distance).

**3. The DAT vs CDS comparison is the cleanest evidence for "emergent vs instructed divergence."**
DAT and CDS are structurally identical (mean pairwise cosine distance across a model-generated word set) but differ in *task context*:
- **DAT**: the model is explicitly asked for unrelated words.
- **CDS**: the model is asked to chain associations; divergence is a byproduct.

Empirically they behave completely differently:
- DAT vs Arena CW: ρ = +0.271, **r = -0.102** (n=51)
- CDS vs Arena CW: ρ = +0.837, r = +0.733 (n=51)
- DAT vs CDS direct correlation: ρ = +0.414, r = -0.006 (n=54) — nearly uncorrelated on Pearson

Hierarchical regression on Arena CW:
- Y ~ DAT alone: R² = 0.010
- Y ~ CDS alone: R² = 0.537
- Y ~ DAT + CDS: R² = 0.542 (CDS adds +0.531 R²; DAT adds +0.005)

The thesis this supports: **explicit-instruction creativity measurements collapse into instruction-following capability; emergent-behavior measurements capture genuine creative variation.**

### Findings that are negative results (informative but not in either paper's main story)

- **C-PACE (lexical / semantic constraints layered onto PACE)** — soft composite adds incremental R² to PACE (+0.12-0.21 on Arena columns) but does not beat PACE as a standalone metric at n=44. Report: [docs/reports/2026-04-13_c_pace_negative_result/](../reports/2026-04-13_c_pace_negative_result/).
- **Circle construction (structural constraint: close the chain back to seed)** — same failure mode. valid_circle_rate adds ΔR² = +0.16 to PACE on Arena CW but correlation drops with n; no standalone advantage over PACE. Report: [docs/reports/2026-04-13_circle_construction_negative_result/](../reports/2026-04-13_circle_construction_negative_result/).

Generalization from both: adding any explicit rule or structural constraint to PACE's prompt converts the measurement from emergent-creativity to instruction-following-under-constraint, which collapses into capability. This is the reason CDS works — it doesn't add anything; it just scores PACE's chains more cleanly.

## Partial-correlation methodology note

For reviewers concerned about the partial correlation methodology:

Given metric X, target Y, and controls $Z_1 = $ Arena Overall, $Z_2 = $ MMLU-Pro, the "BOTH" partial Pearson r is computed via:

1. Fit OLS $X = \beta_0 + \beta_1 Z_1 + \beta_2 Z_2 + \epsilon_X$, get residuals $\hat{e}_X = X - \hat{X}$
2. Fit OLS $Y = \gamma_0 + \gamma_1 Z_1 + \gamma_2 Z_2 + \epsilon_Y$, get residuals $\hat{e}_Y = Y - \hat{Y}$
3. Report $r_{XY \cdot Z_1 Z_2} = \mathrm{Pearson}(\hat{e}_X, \hat{e}_Y)$

Implemented in `partial_corr` in the mechanistic-PACE analysis script. The BOTH specification strips out everything jointly predictable from any linear combination of Arena Overall and MMLU-Pro, leaving only the portion of metric and target that survives this combined capability control. A significant partial correlation under BOTH means the metric-target relationship is not an artifact of any linear combination of these two capability proxies.

## Recommendation

Two options for how the CDS finding relates to the ICCC paper:

1. **ICCC stays as-is with PACE; CDS becomes a separate NeurIPS paper.** Cleanest split. The ICCC short paper focuses on "which existing metrics predict creative writing" and has a coherent story already. CDS + mechanism + triple-control partial correlations are a full methodological paper that doesn't fit the ICCC short-paper format.

2. **Swap PACE for CDS in the ICCC draft.** Would strengthen the headline numbers but requires rerunning bootstrap CIs, inter-metric heatmap, and rewriting the Method section to define CDS. The mechanism story doesn't fit in the short paper's space.

My recommendation: **option 1**. The mechanistic work and the NeurIPS-length paper are the right home for CDS; the ICCC draft is complete and its PACE framing is correct.

## Artifacts

- Code: `src/dat_eval/scripts/analyze_pace_mechanisms.py`
- Config: `configs/dat_eval/analyze_pace_mechanisms.yaml`
- Data: `data/dat_eval/run_v1/downstream/pace_mechanisms_v1/results/`
- Benchmark updates: 17 new MMLU-Pro scores added to `configs/comb_eval/benchmarks.json`
- Reports: [chain_drift_score](../reports/2026-04-13_chain_drift_score/), [mechanistic_pace](../reports/2026-04-13_mechanistic_pace/)
- Figures: `docs/reports/2026-04-13_chain_drift_score/figures/{fig_metric_grid, fig_partial_corr_heatmap, fig_partial_residual_scatter}.pdf`
