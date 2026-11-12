# new_tests — progress

## Goal

Determine whether existing preference-optimization methods for LLM
creativity (CrPO and predecessors) actually teach creativity, or
whether they teach distributional fit to small-c psychometric tasks.
If they don't teach creativity, identify what does — likely a
question of preference-data construction more than of loss form.

The pivot point for this track: creativity is the ability to produce
novel and appropriate responses *outside* the training-distribution
shape. A method that helps only on its training distribution has not
taught creativity by definition. So the relevant evaluation is just
benchmark performance on the three creativity constructs the field
already cares about (creative writing, divergent thinking, scientific
ideation), measured directly. There is no separate "transfer"
question.

## Status — 2026-11-12 (DRAT-N / DRAT-A, human vs LLM correlation structure)

Log: `docs/logs/2026-11-12/1544_drat_human_vs_llm.md`. All current figures are collected in
`docs/reports/2026-11-12_drat_human_vs_llm/figures/` (01–12, with a README mapping each to its run). The
findings memo is **not written yet**.

- **Naming.** DRAT-N = ungated distance (`glove_dist_all`). DRAT-A = relevance. The comparison figure (09) uses
  these labels.
- **Use the share of relevant words for DRAT-A, not the count.**
  - The count (`glove_n_surv`) correlates .77 with words typed in humans (7.2 words vs the LLMs' 9.95).
  - As a share, the A–N trade-off is −.59 in humans vs −.63 in LLMs. The −.15 human figure in
    `compare_20261110` is a count artifact.
  - The pipeline still uses the count, so the switch is owed in psychometrics, compare and score.
- **What differs between populations is the relation with ability:**
  - DRAT-A share vs ability: +.01 humans vs +.53 LLMs;
  - DRAT-N vs ability: +.18 vs −.51.
  - In LLMs, DRAT-N is explained (R² = .88) by the share of relevant words and gated distance. Weak models drift
    off-topic (Llama-3.2-1B: 34% relevant). The sign follows the instruction ("relate to all anchors"), so
    DAT–ability is +.56 while DRAT-N–ability is −.51.
- **Correlation matrices** (exploratory permutation test, scratch only):
  - patterns differ (p = .004 for the 6 tests);
  - higher overall intercorrelation in LLMs only with ability included (mean |r| +.17, p = .01; first component
    54% vs 38%);
  - DAT, RAT and SCTT converge in LLMs while the DRAT decouples;
  - the DRAT's internal structure matches across populations, making it the measurement-invariance anchor.
- **Weight curves** (`plot_drat_weight_curves.py`, config `drat_weight_curves_20261112.yaml`, output
  `data/new_tests/human_battery_llm/drat_weight_curves_20261112/`). Composite w·z(N) + (1−w)·z(A).
  - Humans: best w ≈ 0.65 for SCTT creativity (.37), and every criterion is positive above 0.5.
  - LLMs: the sign flips around 0.5. Creative-writing benchmarks prefer A; output diversity prefers N
    (Hivemind diversity +.73 at 0.7, n = 16). Kombine +.22 (n = 15). Mazur and LiveIdeaBench (n = 10) are
    uninformative.
  - Best weights are picked on the same data, so treat the peaks as optimistic.
- **ARC-AGI-1** covers 19 of 40 battery models (OpenAI, Anthropic and Meta only). ρ = .93 with MMLU-Pro on 11
  shared models. Robustness check only; not added to `benchmarks.json`.
- **Anchor similarity (humans, fig 05)** explains about half of item difficulty (survivors .72, rated difficulty
  −.77). Causal and constitutive anchors are easier than their cosine predicts.
- **Battery expansion** (assessed, not run): 15 models are runnable on OpenRouter today. They would take Kombine
  from 15 to 27 and LiveIdeaBench from 10 to 13, at about $25–40. Gemini returns 403 via OpenRouter. o3-mini,
  o4-mini and GPT-4-turbo are batch-only. Most Mazur and LiveIdeaBench models are deprecated.

**Next:**
1. Write the memo.
2. Switch DRAT-A to the share and rerun the downstream steps.
3. With the author's go-ahead: drop the classifier and split diversity scoring from analysis.
4. Add the matrix test and a cross-validated composite scan to the pipeline.
5. Relabel figures 07 and 08.
6. Later: AUT on the LLMs; an idea-level SCTT diversity scorer.

## Status — 2026-11-11 (SCTT response diversity, humans vs LLMs)

New analysis `analyze_sctt_diversity.py` (config `configs/new_tests/sctt_diversity_20261111.yaml`, wrapper
`scripts/new_tests/run_sctt_diversity.sh`, output `data/new_tests/human_battery_llm/sctt_diversity_20261111/`).
It scores the nine common SCTT scenarios (3 responses each) for 240 people (10 Nov package, exclusions
applied) and the 40 battery LLMs (3 calls per scenario). Scorers are SBERT all-mpnet-base-v2 cosine distance
and the NoveltyBench DeBERTa distinct classifier. Scores are raw, with **no length adjustment** (author's rule).

- LLM call handling matches `prep_sctt_judging_llm.py`: first 3 items kept. Of 1,080 calls, 37 were dropped
  (22 had fewer than 3 items, 9 unparsed, 6 were 429 errors) and 10 were truncated. Logged in `llm_call_log.csv`.
- **The classifier hits its ceiling within a set** (humans .996, LLMs 1.000), so it is uninformative there.
  It is only useful across LLM calls (.946; reliability .88).
- **Within-set SBERT distance is reliable but picks up phrasing.** Spearman-Brown is .81 for humans and .69 for
  LLMs. In people it correlates *negatively* with SCTT creativity (−.36), RAT (−.28) and Raven's (−.16).
  - The least-diverse human sets repeat a sentence stem ("The water in one lake is warmer than...") while
    giving three distinct ideas.
  - The most-diverse sets are terse fragments.
  - Treat it as a phrasing measure, not idea diversity, until an idea-level scorer exists.
- **Across-call diversity in LLMs partly tracks the paper's output-diversity benchmarks.** Corrected 2026-11-12:
  the main metrics cover 17–20 battery models, so the earlier n = 10/16 figures were secondary metrics.
  - Hivemind intra-sim (n = 17): across-call −.53 (p = .03, BH .07); population spread −.69 (BH .01).
  - NoveltyBench utility (n = 20): not significant (within .14, across .35, population .11).
  - Secondary metrics, smaller n: NoveltyBench distinct .85 (n = 10), Hivemind diversity .72 (n = 16).
  - Classifier vs MMLU-Pro: ρ = −.66 (n = 26), but SBERT across-call shows no such link (−.27, n.s.). It is
    classifier-specific. The author questioned the need for the classifier; dropping it is proposed.
- **LLMs vary less across calls than within one call:** SBERT .348 vs .378, lower in 90% of models. But the two
  correlate at .88, so on this scorer they are not "jagged".
- **Population spread:** humans .582 [.568, .595] vs LLMs .394 [.375, .416]. Every scenario shows the gap, and
  the median model is less distinctive than every person (median percentile in the human distribution: 0).

## Status — 2026-05-09 (paper polish + LIB pool refresh)

DRAT scoring pool grew to **n = 20** on LiveIdeaBench after adding the
per-facet LIB rows (originality / flexibility / feasibility / fluency /
clarity) to `configs/comb_eval/benchmarks.json`. Re-ran the
utility-gate ablation (`scratch/drat_paper_comparison/compute_gate_ablation.py`)
on the new pool. Headline result strengthens:

- **$\max$ gate (used) on LIB**: $v = +0.57^{**}$ ($p = .008$),
  $s = +0.50^{*}$ ($p = .018$). Sig on both axes.
- $\min$ on LIB: $v = +0.32, s = +0.21$ — both lose significance at
  the new $n$.
- $\mathrm{avg}$ on LIB: $v = +0.44, s = +0.34$ — validity now $p < .10$
  rather than $p < .05$; specificity $p = .12$.

The earlier "validity is roughly invariant to the gate" claim no
longer holds. New picture: $\max$ is the only gate to clear
significance on either axis on LIB, and clears both. The cross-anchor
coverage requirement (enforced by $\min$ in the strict reading) costs
too much variance; the per-word "near *any* anchor" reading
($\max$) preserves enough variance for specificity to surface.

The (k, vocab) ablation in `04_drat.tex` cites pre-refresh values
(k=4 scientific terms: $v=+0.56^{*}, s=+0.35^{*}$; monotonic
$+0.33 \to +0.50 \to +0.56$ for validity at expert k=2,3,4); the
latest `ablation_results.json` shows k=4 scientific-terms at
$v=+0.57^{**}, s=+0.50^{*}$ at $n=20$. Worth aligning the paragraph
numbers next session.

Naming: the "expert curated" anchor bank is now called the
"scientific-terms" bank throughout the paper and the plot legend
(commit b6a7b5a → 1cb410a in `papers/iccc-2026/`).
The full 30 quadruples for both banks (scientific terms +
ConceptNet relation-distant) are now documented in the appendix
under `\subsection{DRAT Anchor Banks}` (`app:anchor_banks`).
The RAT prompt template from `src/new_tests/rat.py:rat_prompt` is
now in the Prompts appendix.

## Status — 2026-05-06 (PM)

DRAT pilot run end-to-end: Phase 1 (6 cheap LIB-17 models) + Phase 2
(4 mid-tier) + Phase 3 (6 expensive/reasoning) gives **n = 16 of 17
LIB-17 coverage** (qwq-32b skipped per dat_eval config — its
reasoning tokens are baked into the fine-tune and can't be disabled).
Both default and analogical prompt elicitations run; multi-embedding
rescoring (SBERT, GloVe, FastText) wired up via `rescore_drat.py`.

**Headline empirical findings, n = 16, analogical prompt, gated $n_{\min}=3$:**

- **r(DRAT, DAT) = −0.65 (p = 0.006)** under GloVe — strongest signal
  in the analysis. DRAT measures something genuinely distinct from
  DAT, not "DAT with anchors". Mechanism: DAT rewards pure scatter,
  DRAT rewards controlled bridging within the A∪B universe.
- **r|g(DRAT, LIB) = +0.28 (p = 0.31)** under SBERT, +0.24 (p = 0.39)
  under FastText. Matches published CDAT/DAT specificity on LIB
  (+0.21, +0.24) but not significant — same wall the dat_eval paper
  hit at n=17.
- **r|g(DRAT, NoveltyBench Utility) = +0.72 (p = 0.11)** at n=7.
  Effect size matches CDAT's +0.60 from the paper, sign is robust
  across all three embeddings, but n is small.
- **r|g(DRAT, Arena CW) ≈ +0.23**, **r|g(DRAT, MMLU-Pro) low**,
  **r|g(DRAT, Hivemind) ≈ −0.10**. Hivemind is small-negative, not
  the strong negative we saw at n=10.
- **DRAT × CDAT-A: +0.49 (p = 0.056)**, DRAT × PACE: ~0 — DRAT shares
  appropriateness signal with CDAT-A but is independent of PACE.

**Headline design findings:**

- Analogical prompt framing ("metaphorically applied to both A and
  B") elicits qualitatively richer responses than the default
  ("connects A and B") — visibly Gentner-style structure mapping.
  Mean DRAT scores collapse (50→25 range) but model rankings under
  analogical correlate better with LIB. The prompt + score are
  *more* aligned, not less, because the embedding gate filters in
  favor of grounded analogies (firewall, network, mechanism) and
  against vague metaphors (river, labyrinth).
- Embedding choice meaningfully affects results. SBERT and FastText
  give the most defensible LIB specificity numbers; GloVe is most
  affected by token-level co-occurrence quirks.
- Anchor-bank construction has a substantial researcher-degree-of-
  freedom problem: 1/3 of pairs in the v1 bank have $\tau > 0.35$
  (high), driving gate failures concentrated on broad/abstract
  anchors. Pre-registered τ-prescreening is needed before any
  paper-defensible run.
- $n_{\min}=3$ is more discriminating than $n_{\min}=5$ on this
  bank; n_min=5 turned the score into a binary gate-pass-rate
  proxy and lost the within-task variance.

## Status — 2026-05-06 (AM)

DRAT (Divergent Remote Association Test) design and smoke test. The
test is a hybrid of RAT and DAT designed to bridge convergent and
divergent thinking measures into a single vocab-space test, motivated
by the dat_eval finding that no existing test predicts LiveIdeaBench
specifically. Design doc in
[drat_design.md](drat_design.md). Smoke-test pipeline in
`src/new_tests/drat.py`, `src/new_tests/scripts/run_drat_smoke.py`,
`configs/new_tests/drat_smoke.yaml`,
`scripts/new_tests/run_drat_smoke.sh`.

Smoke test passes on `gemini-2.0-flash-lite-001` over 3 hand-crafted
anchor pairs. With the explicit divergence-and-bridging prompt, all
three pairs produce non-zero DRAT scores in the 57–73 range,
matching typical DAT magnitudes.

Next: Phase 1 pilot — six cheap LIB-17 models, $K=30$ pre-registered
anchor bank, all three embeddings (GloVe, FastText, SBERT). Plan in
[drat_design.md §Decision rule](drat_design.md). The
gating engineering — anchor-bank constructor, multi-embedding
support, full async eval runner — is still TODO.

## Status — 2026-05-03

- Read [iccc-2026 paper](../../../papers/iccc-2026/sections/) end to
  end. Compiled [survey.md](survey.md) covering Nagarajan
  Roll-the-Dice, Schapiro comb-creat, Wadhwa CREATE, plus ~30
  adjacent benchmarks.
- Pulled apart CrPO ([arXiv 2505.14442](https://arxiv.org/abs/2505.14442))
  in detail. Two structural observations recorded in
  [proposals.md](proposals.md):
  1. The CrPO loss multiplies the standard DPO loss by a scalar
     `w(x, y^w) = Σ_d λ_d · s_d^w` that does not depend on θ. The
     creativity scores therefore enter only as per-pair learning-rate
     scheduling; they do not change the gradient direction at any
     pair, and the model converges to the standard DPO stationary
     point with biased pair sampling. The mechanically-cleaner
     alternative is to put the creativity contrast `c^w − c^l`
     *inside* the sigmoid (Cre-DPO).
  2. MuCE-Pref is built from ~12 templated psychometric tasks with
     short undergraduate-pool free-text responses. The CrPO paper
     reports that vanilla DPO on MuCE-Pref already beats GPT-4o on
     the held-out MuCE suite — strongly suggesting the held-out suite
     and the training set share a distribution, and that the headline
     CrPO result reflects distributional fit, not creativity.
- Logged the central claim as [H1 in docs/HYPOTHESES.md](../../HYPOTHESES.md):
  *CrPO does not improve creativity.* Apparent gains on MuCE-style
  held-out tasks reflect distributional fit, not transferable
  creative competence. Falsifiable by running released CrPO
  checkpoints on the six creativity benchmarks dat_eval already uses.
- Downloaded MuCE / MuCE-SFT / MuCE-Pref to
  `data/new_tests/muce/` (gitignored). Schema fact worth flagging:
  MuCE-Pref stores the four auto-metrics for the *chosen* response
  only. To train any contrast-based variant of CrPO (Cre-DPO,
  GRPO-Cre, etc.) the rejected side has to be rescored first.

## Next steps

1. **Test H1.** Run the released CrPO checkpoints (Llama-3.1-8B and
   Mistral-7B variants from
   [CNCL-Penn-State on HF](https://huggingface.co/collections/CNCL-Penn-State/crpo-67d0b11ff358430823dbb3df))
   plus their SFT-only and DPO-only ablations against the six
   creativity benchmarks (Arena CW, EQ-Bench CW, Mazur CW, Hivemind,
   NoveltyBench Utility, LiveIdeaBench), plus a held-out MuCE slice
   as a control. Report benchmark scores and per-benchmark deltas
   relative to the matched base model. Resolution criteria are
   pinned in [H1](../../HYPOTHESES.md#h1--crpo-does-not-improve-creativity).
2. **If H1 supported (CrPO does not improve creativity):** the work
   becomes about *what would* improve creativity. The most likely
   answer is preference-data construction. Candidate datasets to
   try, with the same matched base models for clean comparison:
   - Creative-writing pairs mined from r/WritingPrompts / LitBench /
     Arena CW battle pairs.
   - Divergent-thinking pairs from Hivemind / NoveltyBench prompts
     via best-of-K with rubric judges.
   - Scientific-ideation pairs from LiveIdeaBench-style keyword
     prompts via best-of-K with the LiveIdeaBench multi-judge
     rubric.
3. **If H1 falsified (CrPO does improve creativity in some
   meaningful way):** then the loss-form discussion in
   [proposals.md](proposals.md) becomes the primary direction —
   train Cre-DPO and check whether the contrast-inside-the-sigmoid
   form gives a clean improvement over CrPO on the same benchmarks.
4. (Either way, deferred.) GRPO-Cre — online policy gradient with
   `c` as the reward. Strictly stronger than offline preference
   methods if the reward signal is right; only worth pursuing once
   we know what the right reward signal is.

## Design constraints

- Released CrPO checkpoints used as-is for the H1 test. No
  retraining required for step 1.
- For any retraining (steps 2–3), match CrPO's training setup
  (LoRA r=128, α=256, 1-epoch SFT, then preference optimization)
  so any benchmark delta is attributable to the data or loss
  change, not auxiliary differences.
- Reference reward / surprise / embedding models held fixed
  (Skywork-Reward-Gemma-27B-v0.2, Gemma-2-27B, jina-embeddings-v3)
  to match CrPO when we want a controlled comparison.
- API spend on benchmark scoring respects `budget_usd` in run
  configs; reuse the OpenRouter async client from `dat_eval`.
