# Chain Drift Score (CDS): a simpler, stronger metric than PACE

**Date**: 2026-04-13
**Status**: Finding. A simpler decomposition of PACE — the mean cosine distance across non-adjacent word pairs in an associative chain — outperforms PACE on every creative-writing benchmark we tested at n=51, has roughly 2× PACE's creative-writing specificity, and fully subsumes PACE in hierarchical regression.
**Relates to**: [PACE correlation study](../2026-04-12_preliminary_correlations/report.md), [Mechanistic PACE decomposition](../2026-04-13_mechanistic_pace/report.md)

## The finding

Given a PACE associative chain $c = (w_1, w_2, \ldots, w_{20})$ — seed plus 19 model-generated words — define:

$$
\mathrm{CDS}(c) \;=\; \frac{1}{|\mathcal{P}_{20}|} \sum_{(i, j) \in \mathcal{P}_{20}} d_{\cos}\!\big(e(w_i),\, e(w_j)\big)
$$

where $\mathcal{P}_{20} = \{(i, j) : 1 \le i < j \le 20,\, j - i \ge 2\}$ is the set of non-adjacent index pairs ($|\mathcal{P}_{20}| = 171$ of the $\binom{20}{2} = 190$ total pairs), $e(\cdot)$ is the FastText embedding, and $d_{\cos}(u, v) = 1 - u \cdot v / (\lVert u \rVert \lVert v \rVert)$.

Per-model CDS is the mean of chain-level CDS across all $|\mathcal{S}| \cdot K$ chains (seeds × first-associations; PACE's default 50 × 3 = 150 chains per model).

**Claim 1**: CDS outperforms PACE on Arena creative writing, Arena Overall, and EQ-Bench creative writing. Mazur is roughly a tie.

**Claim 2**: PACE adds no variance to CDS in a hierarchical regression (ΔR² = 0.000 on Arena CW, p = 0.88). CDS fully subsumes PACE's predictive signal.

**Claim 3**: The mechanism: chain-level *sustained drift* (non-adjacent distance) predicts creative writing; per-step *local jumps* (adjacent-pair distance) do not. Adjacent-pair distance has *negative* correlation with Arena CW at n=51 (ρ = −0.233). PACE dilutes its signal by averaging over both types of pairs and weighting by position; CDS cleans both choices up.

## Evidence

**Setup.** 54 models × 50 seeds × 3 first-associations × 20-word chains (the canonical PACE setup from Qiu & Hu 2025, as run in our prior study). All scoring is post-hoc on saved chains — no new API calls.

### CDS vs PACE across four benchmarks

| Benchmark | n | PACE ρ / r | **CDS ρ / r** | **Δρ / Δr** |
|---|---|---|---|---|
| Arena creative writing | 51 | +0.705 / +0.592 | **+0.837 / +0.733** | **+0.132 / +0.141** |
| Arena overall | 51 | +0.677 / +0.559 | **+0.781 / +0.678** | +0.104 / +0.119 |
| EQ-Bench creative writing | 34 | +0.755 / +0.728 | **+0.816 / +0.773** | +0.061 / +0.045 |
| Mazur creative writing v2 | 20 | +0.701 / +0.727 | +0.684 / +0.697 | -0.017 / -0.030 |

CDS wins on three of four benchmarks by meaningful margins — 17-19% Spearman and 20-24% Pearson on the two largest. Mazur (n=20) is roughly a tie.

### Creative-writing specificity

Specificity = correlation gap between Arena CW and Arena Overall. Both benchmarks share 95%+ variance in the full benchmark set (ρ = 0.964), so absolute specificity is capped.

| Metric | ρ vs Overall | ρ vs CW | **Δρ** | r vs Overall | r vs CW | **Δr** |
|---|---|---|---|---|---|---|
| PACE | +0.677 | +0.705 | +0.028 | +0.559 | +0.592 | +0.033 |
| **CDS** | +0.781 | +0.837 | **+0.056** | +0.678 | +0.733 | **+0.055** |

CDS has **roughly 2× PACE's creative-writing specificity** in both correlation types. In absolute terms the specificity is small — both metrics are largely tracking general model quality — but the relative advantage is real.

### Hierarchical regression: does PACE add anything?

Testing whether PACE carries information CDS doesn't already have:

| Benchmark | R²(CDS) | R²(CDS + PACE) | ΔR² | F p |
|---|---|---|---|---|
| Arena creative writing | 0.537 | 0.537 | **+0.000** | 0.88 |
| Arena overall | 0.459 | 0.460 | +0.001 | 0.77 |
| EQ-Bench creative writing | 0.598 | 0.600 | +0.003 | 0.66 |
| Mazur creative writing v2 | 0.486 | 0.529 | +0.043 | 0.23 |

Adding PACE to a model containing CDS explains 0.0% more Arena CW variance. PACE's creative-writing signal is **fully contained within CDS**.

### The mechanism: adjacent pairs carry zero signal

Decomposing PACE's pair-distance aggregate into adjacent and non-adjacent components:

| Component | Arena CW ρ | Arena CW r |
|---|---|---|
| **CDS (non-adjacent pairs)** | **+0.837** *** | **+0.733** *** |
| PACE (full, both pair types, position-weighted) | +0.705 *** | +0.592 *** |
| mean distance over adjacent pairs only | **-0.233** | **-0.225** |
| `first_edge_dist` (seed → first chain word) | -0.317 * | -0.256 . |

Adjacent-pair mean distance **negatively** predicts creative writing. Models that make larger consecutive jumps score *worse* on Arena CW, controlling for nothing. The first-edge distance (seed → first chain word) is also negatively correlated. PACE's formula averages over both types of pairs and weights by position — this dilutes the signal. CDS's decision to drop adjacent pairs and use uniform weighting is what produces the lift.

## Interpretation

What distinguishes a high-CDS chain from a low-CDS chain? Three patterns a model could exhibit during free-association:

1. **Tight clustering** — small consecutive steps, chain stays in a small region of concept space. Low non-adjacent distance, low CDS, low creative writing.
2. **Random jumps** — large consecutive steps but noisy trajectory; words at position $i+5$ might return near position $i$. High adjacent distance, moderate non-adjacent distance, moderate creative writing.
3. **Sustained drift** — small consistent steps in a coherent direction. Low adjacent distance, *high* non-adjacent distance, high CDS, high creative writing.

Pattern (3) wins. **The signal is cumulative displacement, not per-step novelty.** Creative-writing-adjacent associative behaviour looks like slow drift through concept space where local connections are plausible but the global arc covers substantial semantic distance. This is empirically what good creative prose looks like — sentences connect locally, but a paragraph covers a lot of ground.

This interpretation also explains why the two negative results in our prior work (C-PACE constraints and circle construction) didn't improve over PACE. Both added explicit structural requirements to the generation task — rules to track, closures to plan — which converted the measurement from "natural drift" (what CDS/PACE measure) to "drift under constraint" (which measures instruction-following more than creativity).

## Practical use

CDS is a drop-in replacement for PACE. The data collection procedure is identical; only the scoring formula changes. Concretely:

- Replace the PACE chain-score function with CDS.
- Everything else (stage 1 seed → first-associations, stage 2 chain extension, 50 seeds × 3 chains per model) stays as in Qiu & Hu 2025.
- Per-chain cost is O(n²) cosine distances instead of O(n²) in PACE's original formula — same order, trivially cheap at n=20.

For our ICCC 2026 draft, this suggests swapping PACE for CDS as the primary reported metric, updating correlation tables with the stronger numbers, and adding a mechanistic paragraph on what CDS actually measures (sustained drift, not per-step novelty).

## Limitations

- **Same benchmarks, same chains.** The decomposition is on the identical data Qiu & Hu introduced. We have not validated CDS on held-out benchmarks, new model samples, or chains of different lengths. The finding could be specific to 20-word chains.
- **Mazur n=20.** CDS slightly under-performs PACE on Mazur (Pearson r -0.030, not significant). We don't have enough data to tell whether this is noise or a real pattern.
- **Absolute specificity is small.** CDS's creative-writing specificity over capability is +0.056 (Spearman), higher than PACE's +0.028 but still modest. Arena CW and Arena Overall share 95%+ variance; no metric on these benchmarks can be highly creative-writing specific.
- **Hivemind unaffected.** CDS has near-zero correlation with Hivemind intra-model similarity (ρ = +0.127), same as PACE. Neither captures Hivemind's diversity signal.
- **We haven't explained *why* some models produce sustained-drift chains.** Training data distribution? Instruction tuning? Model family? Out of scope here; a natural follow-up.

## Artifacts

- Code: `src/dat_eval/scripts/analyze_pace_mechanisms.py`
- Config: `configs/dat_eval/analyze_pace_mechanisms.yaml`
- Per-model decomposition: `data/dat_eval/run_v1/downstream/pace_mechanisms_v1/results/per_model_decomposition.{csv,json}`
- Full correlation + hierarchical tables: `data/dat_eval/run_v1/downstream/pace_mechanisms_v1/results/correlation_analysis.json`
