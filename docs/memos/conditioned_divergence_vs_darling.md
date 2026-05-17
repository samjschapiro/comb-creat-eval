# Conditioned divergence vs DARLING/DivPO — mechanism delta + next-attempt-prediction framing

Written after the `full_run_v1` MCNS-RL null result and a verified read of the DARLING and DivPO method sections. Purpose: pin down whether S-conditioned preference optimization is genuinely distinct from the 2025 diversity-RLHF wave, and record the "next-attempt prediction" reframe and the post-alignment "DT-recovery" idea. Companion to [mcns_dpo.md](./mcns_dpo.md) and [preference_optimization_for_novelty.md](./preference_optimization_for_novelty.md).

## Verified mechanism delta

Facts below are from the method sections of DARLING (arXiv 2509.02534) and DivPO (arXiv 2501.18101), not the abstracts.

**DARLING.** Diversity of generation $y_i$ is the average pairwise distance to the *other $n-1$ concurrent rollouts to the same prompt*: $\mathrm{Div}(y_i \mid y_1,\dots,y_n) = \frac{1}{n-1}\sum_{j\neq i} d(y_i, y_j)$. Quality and diversity are combined **multiplicatively**: $r_{\text{darling}}(x, y_i \mid y_{1:n}) = r(x,y_i)\times \mathrm{Norm}(\mathrm{Div}(y_i\mid y_{1:n}))$. Optimized with GRPO (token-level averaging, std-normalization removed from the advantage, KL to a pretraining-checkpoint reference). The "learned partition function" is a binary semantic-equivalence classifier (ModernBERT-base / Qwen3-Embedding-4B) trained on NoveltyBench annotations (~78% test accuracy). The model never sees the other rollouts; diversity enters only through the reward.

**DivPO.** Sample $N$ responses, score with a reward model, split by a quality threshold $\rho$ into chosen/rejected sets. Pick $y_c = \arg\max_i D(y_i, Y^{\text{chosen}})$ (most diverse among high-quality) and $y_r = \arg\min_i D(y_i, Y^{\text{rejected}})$ (least diverse among low-quality), then a standard DPO loss on $(y_c, y_r)$. Diversity measures: model-probability, word-frequency, or LLM-judge — *not* embeddings. Offline or online (online stronger). The model never conditions on the other responses; diversity is used only to *select pairs*.

**Earlier error, corrected:** DARLING is *multiplicative* (quality × diversity), not additive. The "additive joint vs. gated" distinction from the early memo therefore does **not** separate these methods — DARLING is already gated-like. The distinguishing axis is conditioning, not the combination form.

| | DARLING | DivPO | S-conditioned DPO (proposed) |
|---|---|---|---|
| Comparison set | batch of $n$ concurrent rollouts (same prompt) | pool of $N$ samples (same prompt) | prior attempts $S$ |
| How the set is used | shapes the **reward** (GRPO advantage) | shapes **pair selection** | placed in the model's **input context** |
| Model conditions on the set? | No | No | **Yes** (train and inference) |
| What the model learns | unconditioned $\pi(y\mid x)$ | unconditioned $\pi(y\mid x)$ | conditioned $\pi(y\mid x, S)$ |
| Steerable at inference? | No | No | Yes (via what is placed in $S$) |
| Objective stationarity | online/on-policy | offline or online | stationary if $S$ drawn from data |

**The one genuine, verified distinction.** In DARLING and DivPO the comparison set influences the *training signal* (reward; pair selection) but is never an input to the model, so both yield an *unconditioned, fixed, non-steerable* policy whose diversity is an emergent weight property. The proposed method places $S$ in the context, so the model learns a *conditioned operation* and diversity becomes an inference-time-steerable capability. Narrow, but real, and unoccupied in the literature checked so far.

## The next-attempt-prediction reframe

Lift the autoregressive unit from *token* to *attempt*. The model predicts attempt $a_{k+1}$ given $(x, a_1,\dots,a_k)$. Standard next-token prediction is trained by **imitation** (maximize likelihood of the actual continuation — mode-seeking on the data distribution). Next-*attempt* prediction is trained by **anti-imitation**: produce the next attempt that is *unlike* the prior ones, subject to an appropriateness floor $A(x,y)>\tau$.

Implied target, reusing the DPO-with-context derivation (the Bradley–Terry-over-KL-optimal-policy argument is agnostic to the structure of the conditioning variable, so $c=(x,S)$ substitutes for $x$ verbatim):

$$
\pi^{*}(y \mid x, S) \;\propto\; \pi_{\text{ref}}(y \mid x, S)\,\exp\!\big(\alpha\, N(y; S)/\beta\big)\,\mathbb{1}[A(x,y)>\tau],
$$

where $N(y;S)$ is novelty of $y$ relative to $S$ (e.g. mean embedding distance to elements of $S$).

Why this is the load-bearing reframe rather than relabeling:

- **Generalization argument.** Token-level NTP generalizes because "predict next given context" is a context-invariant operation learned over hugely varied contexts; the skill lives in the transition operator, not in memorized content. If "produce next attempt unlike the shown set" is trained over enough varied $(x,S)$, it should generalize OOD for the *same structural reason*. DARLING/DivPO cannot make this argument — they do not model an attempt sequence; they collapse the set to a scalar.
- **Deployment.** A model that has learned the attempt-level transition executes it *in context* at inference — no per-prompt search loop; $S$ is just context. This is the resolution to the "open-ended search isn't a conversational model" objection.

**Honest caveat (do not overclaim).** This is NTP in *structure* (autoregressive over attempts, context-conditioned) but **not** in *objective*: there is no ground-truth "correct different next attempt", so plain MLE is impossible — a preference/reward ($N(y;S)$, gated by $A>\tau$) is still required to define "good and different". The analogy is structural, not literal. Consequently the binding constraint is **data**: token-level NTP works because trillions of naturally-occurring sequential tokens exist; *sequences of deliberately-diverse attempts at one prompt do not occur naturally and must be synthesized*. Data synthesis, not the objective, is the main risk.

## Conditioned divergence as a separate post-alignment stage ("DT recovery")

Idea (from discussion): rather than training divergence into the base behavior, make it a *separate stage after standard alignment*:

```
pretrain → SFT → RLHF/alignment → [conditioned-divergence stage]
```

Why this is attractive:

- **No trade-off against alignment.** Alignment objectives are mode-seeking, which is correct for the majority of graded tasks. If divergence is a capability *conditioned on $S$ being present in context*, the model behaves normally (aligned, mode-seeking) when $S$ is absent and switches to divergent mode only when invoked. $S$ acts as a mode switch. You are not fighting alignment; you are adding a dormant conditioned capability.
- **Pipeline-shaped.** It is just another post-training stage, like DPO after SFT. Deployment profile unchanged; single forward pass; conversational.
- **Targets a documented failure.** RLHF is known to reduce output diversity / divergent-thinking capacity. A post-alignment stage that restores divergent-thinking *on demand* directly addresses that.

**The recovery claim, stated at defensible strength.** Weak (defensible) form: after the stage, the model regains the *capacity* to produce divergent-thinking-test-passing outputs *when prompted with $S$*; alignment behavior is untouched when $S$ is absent. Strong (risky, likely overclaim) form: the stage globally restores unconditioned pre-alignment diversity while keeping alignment gains. Default to the weak claim; the strong claim must be separately demonstrated, not assumed.

**Evaluation of the recovery claim.** Held-out DAT / CDAT / PACE (per the project's "creativity = OOD generalization" stance, evaluated on prompts/cues disjoint from training), measured before vs after the stage, *plus* alignment-benchmark retention with $S$ absent. Report DT-test correlations with Pearson $r$ only if any are computed. **Do not** import the validity/specificity framework here — that framework is for evaluating creativity *tests* against benchmarks across a model population, not for before/after benchmarking of a single trained model. The recovery eval is a plain paired before/after on held-out DT tests plus a retention check.

This also unifies the two tracks: the `dat_eval` evaluation machinery becomes the adjudicator of whether the conditioned-divergence stage recovers DT, which is a cleaner story than "a new diversity-training method works."

## Open questions / risks

1. **Data synthesis is the binding constraint** — how to generate $(x, S, y_w, y_l)$ tuples whose $S$-distribution matches inference-time use (sequential self-conditioning shifts $S$ as the model changes — the off-policy-in-$S$ problem). Mitigations: randomize $S$ construction (base / paraphrase / strong-model / empty / varying $m$), iterate.
2. **Degenerate divergence** — high $N(y;S)$ via fixed contrarian style; caught by the $A>\tau$ gate only if it manifests as off-topic, not if it manifests as low-quality-but-on-topic. Needs a second held-out embedder + judge-quality at eval.
3. **Is "recovery" real or just a conditioned escape hatch?** The weak claim is the escape hatch; whether anything stronger holds is open.
4. **$S$ is a set** — concatenation is order/size-sensitive; fine for small $m$ with shuffling, needs set-encoding/summarization for large $m$.

## Cheap decisive next step

Zero-shot probe (no training, no GPU, API only): does a strong instruct model, given $(x, S)$, already produce appropriate responses far from $S$, with the effect scaling in $|S|$ on held-out prompts? This tests whether the attempt-level anti-imitation transition exists *in context* before any training — i.e., whether the NTP analogy holds zero-shot. If yes, the transition is learnable and the data-synthesis investment is justified; if a capable model cannot do it even in-context, the line dies for ~$10.
