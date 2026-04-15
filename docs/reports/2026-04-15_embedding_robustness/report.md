# Embedding robustness of semantic-distance creativity metrics

**Date**: 2026-04-15
**Status**: Finding. The main-paper claims have opposite embedding-robustness profiles: the PACE × Arena CW creativity-specific signal holds under FastText only, while the PACE × Hivemind diversity signal we withdrew as fragile survives under all three embeddings we tested. DAT and CDAT capability-proxy diagnoses are embedding-robust.
**Relates to**: [Preliminary correlations](../2026-04-12_preliminary_correlations/report.md), [ICCC paper](../../../papers/iccc-2026/)

## TL;DR

Semantic-distance creativity metrics score word sets via cosine distance in an embedding space. We rescored all three metrics (DAT, CDAT, PACE) on the same raw LLM responses under three embeddings (GloVe 840B, FastText crawl-300d-2M, SBERT `all-mpnet-base-v2`) and recomputed the joint partial correlation controlling for Arena Overall Elo and MMLU-Pro.

Two twists. **(1) PACE's creativity-specific signal on Arena CW — the paper's headline positive result — only survives under FastText; it collapses under GloVe and SBERT.** **(2) PACE's diversity signal on Hivemind — which the paper withdrew as fragile — survives under *all three* embeddings.** The implication: what we thought was the embedding-robust finding (creative writing) is actually embedding-specific, and what we thought was the fragile cross-check (diversity) is the robust one.

DAT and CDAT's capability-proxy collapse replicates cleanly under all three embeddings on every benchmark. That part of the main story is secure.

## Context

The main paper evaluates whether three semantic-distance creativity metrics carry creativity-specific signal after partialling out general capability (Arena Overall Elo and MMLU-Pro, jointly). Every score is computed as a cosine distance between embedded words, and each metric was originally defined with a specific embedding in mind:

- DAT → GloVe 840B 300d (Olson et al.\ 2021)
- CDAT → SBERT `all-mpnet-base-v2` (Nakajima et al.\ 2026)
- PACE → FastText crawl-300d-2M (Qiu & Hu 2025)

A score is really a joint property of *metric* and *embedding*. The main paper reports a single score per metric under its original embedding. This report asks: how much do the reported conclusions depend on that embedding choice?

## Method

For every LLM in our set, the raw responses (word lists for DAT/CDAT; chain word sequences for PACE) were already saved to disk. We rescored all three metrics under all three embeddings without any new LLM calls. Key implementation choices:

- The same word lists were scored under each embedder; for DAT, the original GloVe-vocabulary word-validity filter was kept constant so the *word sets* are identical across embedders and only the *distance computation* varies.
- PACE's native filtering (chains $\ge 3$ words, drop chain scores $\le 0$, drop seed scores $\le 0$) was preserved so the rescored FastText values reproduce the main-paper numbers exactly (within 1e-4).
- For each (task × embedding × benchmark) triple we computed both Pearson $r$ and Spearman $\rho$ for the simple correlation, plus the joint partial controlling for Arena Overall and MMLU-Pro simultaneously (same specification as the main-paper headline).
- Benchmarks: Chatbot Arena CW, EQ-Bench CW, Mazur V2, Hivemind diversity ($= 1 - \mathrm{intra\_sim}$).

Code: [`src/dat_eval/scripts/multi_embed_appendix.py`](../../../src/dat_eval/scripts/multi_embed_appendix.py). Scores: [`data/dat_eval/run_v1/downstream/scores_v1/results/multi_embed_scores.json`](../../../data/dat_eval/run_v1/downstream/scores_v1/results/multi_embed_scores.json).

## Findings

### Simple correlations are embedding-stable

Each metric's raw correlation with each benchmark is broadly consistent across embeddings: PACE's Arena CW simple $r$ is $0.75$ (GloVe), $0.77$ (FastText), $0.63$ (SBERT); Spearman $\rho$ similarly $0.79 / 0.80 / 0.71$. DAT's Arena CW correlation is $r \approx 0.30$–$0.43$ across embeddings. CDAT Appropriateness is $r \approx 0.52$–$0.56$. Whatever these metrics are capturing at the raw-correlation level does not depend strongly on embedding choice.

### Joint partial × Arena CW: PACE's positive result is FastText-specific

The claim we make loudest in the main paper — "PACE is the only semantic-distance test with creativity-specific signal on Arena CW after capability partialling" — does not generalise beyond FastText.

| Metric | GloVe | FastText | SBERT |
|---|---|---|---|
| DAT | $r=+.12$ / $\rho=+.10$ | $r=-.10$ / $\rho=+.00$ | $r=+.16$ / $\rho=+.04$ |
| CDAT Nov. | $r=+.07$ / $\rho=+.08$ | $r=+.25$ / $\rho=+.26$ | $r=+.08$ / $\rho=+.13$ |
| CDAT App. | $r=-.06$ / $\rho=-.10$ | $r=-.16$ / $\rho=-.28$ | $r=-.11$ / $\rho=-.17$ |
| **PACE** | $r=-.02$ / $\rho=+.01$ | **$r=+.33^{*}$ / $\rho=+.29^{*}$** | $r=-.04$ / $\rho=-.01$ |

$n = 48$–$49$ depending on metric coverage. Significant cells ($p<.05$) are bolded.

Only one cell in the grid is significant: PACE × FastText. Under GloVe and SBERT the PACE joint partial is indistinguishable from zero. DAT's residual and CDAT's are flat or negative under every embedding, as expected.

### Joint partial × Hivemind diversity: PACE's diversity signal is embedding-robust

The claim we *withdrew* in the main paper — "PACE predicts output diversity" — actually holds up as the most robust single result across the grid.

| Metric | GloVe | FastText | SBERT |
|---|---|---|---|
| DAT | $r=+.21$ / $\rho=+.25$ | $r=+.13$ / $\rho=+.23$ | $r=+.32$ / $\rho=+.37$ |
| CDAT Nov. | $r=+.25$ / $\rho=+.21$ | $r=+.16$ / $\rho=+.10$ | $r=+.18$ / $\rho=+.14$ |
| CDAT App. | $r=-.20$ / $\rho=-.30$ | $r=-.17$ / $\rho=-.20$ | $r=-.17$ / $\rho=-.19$ |
| **PACE** | **$r=+.48^{*}$ / $\rho=+.61^{**}$** | $r=+.36$ / **$\rho=+.46^{*}$** | **$r=+.42^{*}$ / $\rho=+.47^{*}$** |

$n = 23$. The PACE row is significant in the expected direction under every embedding, on at least one coefficient; Spearman $\rho$ is significant under all three. Effect sizes are large ($|\rho| \ge 0.46$ everywhere).

### Joint partial × EQ-Bench CW and Mazur V2

Both of these are under-powered but interesting for diagnostic purposes.

**EQ-Bench CW ($n=32$–$33$)**: DAT's Pearson partial is significant under GloVe ($r=+.47^{**}$) and SBERT ($r=+.59^{***}$) but not FastText — a genuine residual DAT → EQ-Bench signal that is embedding-present but inconsistent. PACE is marginal across all three embeddings ($r \in [.23, .33]$, $\rho \in [.11, .30]$). CDAT flat.

**Mazur V2 ($n=20$–$21$)**: DAT under FastText is a strikingly large partial ($r=+.63^{**}$, $\rho=+.73^{***}$) that does not generalise to the other embeddings; at this sample size it should be read as a small-sample artifact or an embedding-specific capture of whatever Mazur's graders reward. PACE is marginal across embeddings ($\rho=+.40$ to $+.53$).

### Summary grid

Count of significant-in-expected-direction cells per metric, pooling benchmarks and coefficient choices (12 total cells per metric = 3 embeddings × 4 benchmarks × 2 coefficients, minus a few non-applicable ones):

| Metric | Sig. cells / total | Pattern |
|---|---|---|
| DAT | 6 / 24 | Concentrated in EQ-B. and Mazur; not Arena or Hivemind |
| CDAT Nov. | 1 / 24 | Marginal on Mazur GloVe only |
| CDAT App. | 0 / 24 | Never survives; trends wrong-direction on Hivemind |
| PACE | 8 / 24 | Spread across Arena CW (FastText only), Hivemind (all 3 embs), Mazur (one cell) |

## Interpretation

### The PACE × Arena CW result is weaker than the main paper suggests

Under FastText alone, the joint partial is $r = +0.33$, $p = 0.022$ — a real but modest effect. Under GloVe, SBERT, the partial is near zero. Three plausible readings, which we can't fully distinguish from this data:

1. **FastText captures the signal; other embeddings smooth it out.** FastText's subword composition picks up stylistic / informal / morphological association variance that GloVe (lemmatised-ish word-level) and SBERT (sentence-optimised) don't. PACE was designed for FastText, and the creativity signal lives in the associative texture FastText preserves. If so, "PACE + FastText" is the real unit of analysis and our claim should be scoped accordingly.

2. **FastText introduces noise that happens to correlate with capability-adjacent creative-writing variance.** Subword embeddings have known instabilities at the margin, and the $n=49$ partial is in the $.01 < p < .05$ replication-fragile range. The other embeddings may be giving us the "true" answer (no creativity-specific signal) and FastText may be the outlier.

3. **Both embeddings see something different and neither is wrong.** Under an embedding-as-measurement framing, different embedders measure different things. Creativity-as-associative-texture might be a FastText measurement; creativity-as-topical-spread might be a GloVe measurement. Our benchmarks may or may not privilege one.

Choosing between these requires either (i) running on more embeddings with known theoretical differences, (ii) decomposing what specifically about FastText differs, or (iii) defining "creativity" well enough to say which embedding ought to see it.

### The Hivemind diversity result is stronger than the main paper credits

The main paper withdrew the PACE × Hivemind diversity claim because the MMLU-Pro-alone partial was near zero while the Arena-Overall-alone partial was significant, and we treated the disagreement as "sensitive to proxy choice, don't trust." But the *joint* partial — which simultaneously controls for both proxies and is the specification we use throughout the rest of the paper — was $\rho = +0.46^{*}$ in the original run and replicates at $r \ge +0.36$ and $\rho \ge +0.46$ under every embedding we tried in this report. This is the single most consistent positive cell across the entire grid.

The reason the single-proxy-MMLU-Pro partial disappeared is visible in the benchmark-proxy correlation structure: Hivemind diversity correlates $-0.67$ with Arena Overall and $-0.59$ with MMLU-Pro. Arena Overall carries the relevant variance; MMLU-Pro carries less. Partialling on MMLU-Pro alone leaves the capability confound largely intact; partialling on Arena Overall or on the joint stack removes it. The joint specification is the right one, and under it PACE predicts output diversity robustly.

We suggest the main paper revisit the Hivemind framing: the claim wasn't actually as fragile as the MMLU-Pro-alone comparison made it look.

### The DAT/CDAT null results are robust

Across 3 embeddings × 4 benchmarks × 2 coefficients — 24 cells — DAT's joint partial is significant at $p<.05$ in the expected direction in 6 cells, all of which are concentrated on EQ-B. CW and Mazur V2 (small-$n$, embedding-inconsistent). CDAT Novelty: 1 cell out of 24. CDAT Appropriateness: 0 cells (and 5 cells in the *wrong* direction). The main paper's diagnosis — DAT and CDAT function as capability proxies, not creativity-specific metrics — holds under every embedding we tested.

## Implications for the ICCC paper

Two recommended edits to the main paper, one mandatory and one worth discussing:

**1. Scope the PACE × Arena CW claim to FastText.** The abstract and intro should read "PACE (under FastText) retains a creativity-specific Arena CW signal" rather than "PACE is the only metric that survives." Add the embedding-robustness appendix (already drafted) and cross-reference from Discussion.

**2. Unwithdraw (or partially re-embrace) the Hivemind diversity finding.** The joint-partial Hivemind result replicates under all three embeddings; it's the paper's most robust positive cell empirically. The reason we hedged — single-proxy-MMLU-Pro disagreement — turns out to reflect a specific structural fact (Hivemind correlates much more with Arena Overall than with MMLU-Pro) that is itself a finding, not a validity threat. Re-framing options: (a) reinstate the diversity claim with the embedding-robustness data as support, or (b) keep it withdrawn but explain why the joint partial is robust where the single-proxy partials disagree.

**3. Keep the DAT/CDAT null results as-stated.** They replicate under every embedding we tested and the main-paper language is correct.

## Limitations

- Three embeddings is a small sweep. A wider set (Sentence-T5, BGE-large, e5, OpenAI text-embedding-3-large) would tell us more about what specifically about FastText matters.
- Partial correlations at $n \le 25$ (Hivemind, Mazur) are suggestive rather than definitive — even if the sign and approximate magnitude are stable, the $p$-values at these sample sizes should be read generously.
- We did not vary the word-validity filter: DAT still uses GloVe-vocabulary validation even when distances are computed with SBERT. An alternative design (let SBERT score all generated words regardless of vocab) could change the DAT numbers under SBERT specifically.
- We did not rerun the simpler CDS metric (from the [Chain Drift Score report](../2026-04-13_chain_drift_score/report.md)) under multiple embeddings. Given its mechanistic relationship to PACE, its embedding-robustness profile is a natural follow-up.
