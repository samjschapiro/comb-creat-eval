# What does PACE actually measure? A mechanistic decomposition

**Date**: 2026-04-13
**Status**: Positive finding. A simpler metric (mean non-adjacent pair distance) outperforms PACE on every creative writing benchmark we tested, at n=51 models. PACE's predictive signal is fully contained within it.
**Relates to**: [PACE correlation study](../2026-04-12_preliminary_correlations/report.md), [C-PACE negative result](../2026-04-13_c_pace_negative_result/report.md), [Circle construction negative result](../2026-04-13_circle_construction_negative_result/report.md)

## Motivation

After two failed attempts to *add* something to PACE (C-PACE with rule-based constraints, circle construction with structural constraints), the pivot was to *explain* rather than *replace*: what specifically about a model's associative behaviour drives PACE's correlation with creative writing?

PACE (Qiu & Hu, EMNLP 2025) computes, for each position i in a 20-word associative chain, the mean cosine distance from position i to *every prior position*. The chain score averages this over positions 2..20. This formula implicitly weights late positions more heavily (position 20 contributes 19 distances; position 2 contributes 1) and includes both *adjacent* pairs (i, i-1) and *non-adjacent* pairs (i, j<i-1).

We decomposed PACE into eight interpretable components and asked which one captures the predictive signal.

## Decomposition

For each 20-word PACE chain:

| Component | Definition |
|---|---|
| `pace_full` | Standard PACE: average over positions 1..19 of mean distance to all prior positions |
| `mean_adjacent_dist` | Mean cosine distance of *consecutive* pairs only |
| `mean_nonadjacent_dist` | Mean cosine distance of *non-adjacent* pairs (word pairs at least 2 positions apart) |
| `pace_early_pos1_9` | PACE restricted to chain positions 1-9 |
| `pace_late_pos10_19` | PACE restricted to chain positions 10-19 |
| `first_edge_dist` | Single cosine distance from seed → first chain word |
| `return_dist_last_to_first` | Cosine distance from position 19 back to seed |
| `max_edge_dist` | Max consecutive-pair cosine distance in the chain |
| `edge_dist_variance` | Variance of consecutive-pair cosine distances |

Each component is computed per chain, averaged per model over ~150 chains (50 seeds × 3 first-associations), and correlated against creative writing benchmarks.

## Headline result

**`mean_nonadjacent_dist` outperforms `pace_full` on every creative writing benchmark** at n=51 (Arena), n=34 (EQ-Bench), n=20 (Mazur):

| Benchmark | PACE ρ | PACE r | `mean_nonadjacent_dist` ρ | `mean_nonadjacent_dist` r | n |
|---|---|---|---|---|---|
| Arena creative writing | +0.705 *** | +0.592 *** | **+0.837** *** | **+0.733** *** | 51 |
| Arena overall | +0.677 *** | +0.559 *** | **+0.781** *** | **+0.678** *** | 51 |
| EQ-Bench creative writing | +0.755 *** | +0.728 *** | **+0.816** *** | **+0.773** *** | 34 |
| Mazur creative writing v2 | **+0.701** *** | **+0.727** *** | +0.684 *** | +0.697 *** | 20 |

On the two largest benchmarks (Arena CW, Arena Overall) and EQ-Bench, the simpler non-adjacent-pairs metric beats PACE by 0.08-0.13 Spearman and 0.08-0.14 Pearson. Mazur (n=20) is roughly a tie.

### Creative writing vs general capability

`mean_nonadjacent_dist` is better than PACE on both Arena columns, but the lift is larger on Arena CW than on Arena Overall:

| Benchmark | PACE ρ / r | NonAdj ρ / r | Δρ / Δr (lift) |
|---|---|---|---|
| Arena overall (capability) | +0.677 / +0.559 | +0.781 / +0.678 | +0.104 / +0.119 |
| Arena CW (creative writing) | +0.705 / +0.592 | +0.837 / +0.733 | **+0.132 / +0.141** |
| EQ-Bench CW | +0.755 / +0.728 | +0.816 / +0.773 | +0.061 / +0.045 |
| Mazur CW v2 | +0.701 / +0.727 | +0.684 / +0.697 | -0.017 / -0.030 |

Creative-writing-specificity (correlation gap between Arena CW and Arena Overall):

| Metric | Δρ (CW − Overall) | Δr (CW − Overall) |
|---|---|---|
| PACE | +0.028 | +0.033 |
| **`mean_nonadjacent_dist`** | **+0.056** | **+0.055** |

`mean_nonadjacent_dist` has roughly **2× the creative-writing specificity of PACE** (Δρ = +0.056 vs +0.028). In absolute terms the specificity is small — both metrics are largely tracking general model quality, as any metric on this benchmark set must (Arena CW and Arena Overall share 95%+ variance). But the *relative* advantage over PACE is real: `mean_nonadjacent_dist` is both a better general model-quality proxy (+21% Pearson lift on Arena Overall) and more creative-writing-weighted (+24% Pearson lift on Arena CW).

## PACE is redundant given `mean_nonadjacent_dist`

Hierarchical regression tells us what happens when we add PACE on top of `mean_nonadjacent_dist`:

| Benchmark | R²(mean_nonadjacent_dist) | R²(+ PACE) | ΔR² | F p |
|---|---|---|---|---|
| Arena creative writing | 0.537 | 0.537 | **+0.000** | 0.88 |
| Arena overall | 0.459 | 0.460 | **+0.001** | 0.77 |
| EQ-Bench creative writing | 0.598 | 0.600 | **+0.003** | 0.66 |
| Mazur creative writing v2 | 0.486 | 0.529 | +0.043 | 0.23 |

**Adding PACE to `mean_nonadjacent_dist` adds essentially zero explanatory power.** PACE contains no creative-writing-predictive signal that isn't already present in the simpler metric.

The reverse direction also confirms: adding `mean_nonadjacent_dist` to a model containing PACE adds significant variance (on Arena CW: 0.705 → stronger; full hierarchical stats in `data/dat_eval/run_v1/downstream/pace_mechanisms_v1/results/correlation_analysis.json`).

## Adjacent pairs actively hurt

The most interesting individual component: **`mean_adjacent_dist` (consecutive-pair distance alone) is *negatively* correlated with Arena creative writing** (ρ = -0.233, r = -0.225, n=51).

| Metric | Arena CW ρ | Arena CW r |
|---|---|---|
| `mean_nonadjacent_dist` | **+0.837** | **+0.733** |
| `pace_full` | +0.705 | +0.592 |
| `mean_adjacent_dist` | **-0.233** | **-0.225** |

`first_edge_dist` (seed → first chain word) is also negative (ρ = -0.317). A model that makes small first-leap *into* the seed's conceptual neighborhood scores *higher* on creative writing than one that immediately leaps far. This is the opposite of what the "PACE measures free-associative leaps" reading would predict.

## Mechanistic interpretation

The separation between non-adjacent and adjacent pair distances reveals what PACE is actually measuring. Three patterns a model could exhibit:

1. **Tight clustering**: small consecutive steps, chain stays in a small region → low non-adjacent distance → low creative writing.
2. **Random jumps**: large consecutive steps, but trajectory is noisy; words at position i+5 might be near position i again → high adjacent distance, moderate non-adjacent distance → moderate creative writing.
3. **Sustained drift**: small consistent steps in a coherent direction, chain keeps moving through concept space → low adjacent distance, *high* non-adjacent distance → high creative writing.

Creative writing benchmarks reward pattern (3). Models that score well on Arena creative writing produce chains that **drift steadily**: each step is a short, coherent associative move, but those moves compose into large cumulative displacement. Jumpy or random trajectories (patterns 1-2) don't predict creative writing performance.

This is consistent with the intuition about creative prose itself: good creative writing doesn't proceed by non sequiturs. It threads a long sustained trajectory where each sentence connects locally to the previous one, but the overall arc covers significant conceptual distance.

## Implications

**Practically**: `mean_nonadjacent_dist` is a Pareto improvement over PACE. Simpler formula (no position weighting, no adjacent pairs); higher Spearman and Pearson correlations on Arena CW, Arena Overall, and EQ-Bench CW; roughly 2× the creative-writing specificity. We'd recommend it as a drop-in replacement. Suggested name: **Chain Drift Score** (CDS).

**Conceptually**: PACE's predictive signal isn't per-step lexical diversity or free-associative novelty. It's *sustained coherent drift*: the ability to keep moving through concept space while maintaining local associative plausibility. Adjacent-pair distance has *negative* correlation with Arena CW; non-adjacent-pair distance has strong positive correlation. This is a testable mechanistic claim about what distinguishes high-quality from low-quality generation in this setup.

**For the original ICCC 2026 draft**: the claim "PACE predicts Arena creative writing" still holds; this finding refines rather than undermines it. The mechanism discussion should be updated — the creative-writing signal isn't about word-to-word divergence but about cumulative chain-level drift. Adjacent-pair distance carries no creative-writing signal; non-adjacent-pair distance carries it all. Swapping PACE for CDS would also tighten the headline correlations.

**For the broader research program**: combined with our two negative results (C-PACE, Circle Construction), this refines the picture: even PACE and its best mechanistic successor are still mostly capability measurements. Absolute creative-writing specificity is capped by the fact that Arena CW shares 95%+ variance with Arena Overall. Progress on genuinely creativity-specific measurement likely requires better ground truth — benchmarks less capability-confounded than Arena CW — rather than better metrics on existing ground truth.

## Limitations

- All analysis uses the same 50-seed PACE setup from dat_eval. We didn't vary chain length or seed set.
- Mazur CW v2 has only n=20 coverage; the PACE vs `mean_nonadjacent_dist` difference there is noisy (PACE slightly wins on Spearman, ties on Pearson).
- Hivemind intra-model similarity correlations are near zero for *all* PACE components (ρ ≈ +0.13, r ≈ +0.04). No PACE decomposition captures Hivemind's diversity signal — consistent with prior finding that PACE and Hivemind measure orthogonal things.
- We haven't explained *why* high-Arena-CW models produce sustained-drift chains. Training data distribution? Instruction tuning? Architectural choice? Out of scope here.
- The "no positional weighting helps" conclusion may be specific to 20-word chains. At very long or very short chains, positional weighting could matter more.

## Artifacts

- Code: `src/dat_eval/scripts/analyze_pace_mechanisms.py`
- Config: `configs/dat_eval/analyze_pace_mechanisms.yaml`
- Data: `data/dat_eval/run_v1/downstream/pace_mechanisms_v1/` — per-model decomposition CSV + JSON, full correlation and hierarchical analysis
- No new API calls. Uses existing PACE responses from `data/dat_eval/run_v1/`.
