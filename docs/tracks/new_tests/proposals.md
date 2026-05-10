# Proposals

This track explores a **preference-optimization method** that
improves on CrPO ([Ismayilzada et al., 2025](https://arxiv.org/abs/2505.14442))
and is evaluated against the three constructs from
[dat_eval](../dat_eval/progress.md) using the existing validity /
specificity / frontier pipeline.

---

## Headline proposal — Cre-DPO

### Critique of CrPO that motivates the change

CrPO modifies DPO by multiplying the standard DPO loss by a weighted
sum of creativity scores computed on the *winner only*:

```
ℒ_CrPO = -E[ ( λ_d · δ^w + λ_n · ν^w + λ_s · ξ^w + λ_q · γ^w ) · ℓ_DPO ]
```

with `δ, ν, ξ, γ ∈ [0, 1]` for diversity, novelty (DSI), surprise
(perplexity under Gemma-2-27B), and quality (Skywork reward). All
four scores are precomputed at data preparation time and attached to
each preference pair as a fixed scalar.

The structural problem: the multiplier `w(x, y^w) = Σ_d λ_d · s_d^w`
does not depend on θ. Therefore at every preference pair the
gradient is

```
∇_θ ℒ_CrPO = - w(x, y^w) · ∇_θ ℓ_DPO
```

i.e. **the per-pair gradient direction is identical to vanilla
DPO's gradient direction**. CrPO is per-pair learning-rate
scheduling. At convergence the model satisfies the standard DPO
stationary point with biased pair-sampling. The creativity scores
never shape the learned implicit reward — only the speed of getting
to the DPO solution.

A few subsidiary CrPO weaknesses (independent of the structural
issue above):

- **Train/test metric overlap.** CrPO trains on (novelty, diversity,
  surprise, quality) and evaluates on the same metrics. NoveltyBench
  is the only somewhat-independent eval. No construct-level
  evaluation against creative-writing or scientific-ideation
  benchmarks.
- **No scientific-ideation evaluation at all.** The construct on
  which dat_eval found the cleanest gap (LiveIdeaBench) is not
  tested.
- **Surprise = perplexity under another LLM** is a textbook bad
  creativity proxy: it spikes on typos, OCR noise, jargon, code-
  switching. The toxicity regression CrPO authors flag is most
  likely surprise-driven.
- **MuCE = small-c psychometric distribution** (587 prompts across
  25 psychological assessments). Generalization beyond AUT-shaped
  prompts is unverified.
- **Single reward model for quality** (Skywork-Reward-Gemma-27B-v0.2)
  carries known biases (length, formatting, sycophancy) into the
  "quality" channel.
- **Single-completion preference pairs.** Creativity is a property
  of a *set* of generations; one-vs-one preferences throw that
  structure away.
- **Offline DPO when rewards are automatic.** All four creativity
  scores are computable at sampling time. There is no labeling cost
  argument against online preference optimization.

### Cre-DPO loss

Move the creativity contrast inside the sigmoid:

```
ℓ_Cre = -log σ( β · [ log π(y^w|x) / π_ref(y^w|x)
                     − log π(y^l|x) / π_ref(y^l|x) ]
              + α · ( c^w − c^l ) )
```

with `c = Σ_d λ_d · s_d` a weighted creativity composite over the
same dimensions CrPO uses. Reduces to vanilla DPO when α = 0;
reduces to CrPO-style reweighting when α = 0 and the loss is
multiplied externally by a precomputed scalar.

### Why this is structurally different from CrPO

The gradient of `−log σ(z)` w.r.t. θ, where `z = β · Δ_π + α · Δ_c`,
is

```
∇_θ ℓ_Cre  =  −σ(−z) · β · ∇_θ Δ_π
```

At any single pair the direction is still proportional to `∇_θ Δ_π`
(same as DPO and same as CrPO). The structural difference is at the
**stationary point**: as the loss saturates, the implicit reward gap
satisfies

```
β · Δ_π  ≈  α · Δ_c  +  const
```

i.e. the model learns an implicit reward function whose contrast
between winner and loser tracks the creativity contrast `Δ_c`, plus
a constant absorbed by σ saturation. CrPO has no such structure;
its stationary point is the vanilla DPO stationary point.

A second (more practical) difference: the σ-shaped multiplier
`σ(−z)` in Cre-DPO is *adaptive* — it depends on both the creativity
contrast and the model's current implicit reward gap. Pairs where
the model has already achieved an implicit reward gap matching the
creativity gap saturate and stop training. CrPO's static `w(x, y^w)`
multiplier never adapts.

### Close priors so we know we're not reinventing

- **NVIDIA's RPO** (Adler et al., Nemotron-4) adds `η · (r^w − r^l)`
  inside the sigmoid (or as an MSE target). Same structural form,
  using a generic reward.
- **IPO** (Azar et al.) replaces the log-sigmoid with a squared loss
  targeting a specific gap τ; can be adapted to use a creativity-
  derived τ.
- **DPO-Positive / cDPO** family does margin-style modifications.

The novelty of Cre-DPO is therefore *not* the loss form. It is
(a) the choice of `c` as a creativity composite, (b) construct-
aligned evaluation against the dat_eval pipeline (validity +
specificity + frontier on the three constructs), and (c)
demonstration on scientific ideation, the construct dat_eval
identified as the cleanest gap.

### Open decisions

1. **Single α vs per-dimension α_d.** Per-dimension is more flexible
   but more hyperparameters. CrPO's ablation showed
   novelty/diversity/surprise peak at λ ≈ 0.5 and quality at λ ≈ 1.0;
   per-dimension is probably worth keeping.
2. **Unified c vs per-construct c.** Train one model with a single
   creativity composite, or train three (one per construct: creative-
   writing, divergent-thinking, scientific-ideation) and merge or
   route. The dat_eval finding that no test predicts all three
   constructs argues for per-construct heads.
3. **Preference-pair source.** Default: re-use MuCE-Pref for direct
   comparability with CrPO. Stretch: synthesize a scientific-ideation
   preference set from LiveIdeaBench-style prompts via best-of-K
   self-play with a multi-judge rubric.
4. **What `c` actually is.** CrPO's four channels (DSI, diversity,
   surprise, quality) are not all defensible:
   - Replace surprise = perplexity-under-Gemma-2-27B with a
     calibrated alternative (e.g. KL between policy and a topic-
     controlled reference, or held-out NLL with length normalization).
     The perplexity surprogate is the most likely culprit for the
     toxicity regression CrPO authors observed.
   - Quality: keep Skywork as a baseline but consider an ensemble
     to control for single-RM bias.
   - Add a fifth channel for **constraint satisfaction** when
     pair prompts impose hard constraints (CREATE / comb-creat-style
     factuality / connectivity checks). This is the channel that
     should carry signal on scientific ideation specifically.
5. **GRPO-Cre as V2.** Sample K responses per prompt, advantage =
   `(c_i − μ_K) / σ_K`, run policy gradient. The creativity score
   then directly selects which sample the policy is pushed toward,
   rather than reweighting labeled pairs. Treat as a separate paper
   after Cre-DPO results.

### Predicted profile

If the structural argument is right, Cre-DPO's gains over CrPO
should show up most cleanly on **specificity-headroom-rich**
benchmarks — i.e. ones with low R between benchmark and the
capability stack (NoveltyBench at R ≈ −0.33; LiveIdeaBench at R ∈
[0.36, 0.59]) — where vanilla DPO has the most room to fail to
exploit the creativity signal. On Arena CW (R ≈ 0.98), the frontier
caps specificity at ≈ 0.20 for any test, so Cre-DPO and CrPO and DPO
should all converge there.

### Risks

1. **Cre-DPO may collapse to CrPO empirically** if `Δ_c` and
   `ℓ_DPO` are highly correlated on MuCE-Pref (which they should be
   by construction — pairs were filtered to have ≥5 reward margin
   and full annotator agreement on creativity ratings). The
   structural-difference argument is strongest when `Δ_c` and the
   human label are *not* perfectly aligned. Mitigation: run a noisy-
   pair ablation where some MuCE labels are swapped, and check that
   Cre-DPO degrades more gracefully than CrPO.
2. **α / β interaction.** The two parameters are entangled (both
   scale the implicit reward gap); needs a 2-D sweep, not 1-D.
3. **Safety regression.** If we keep CrPO's surprise channel, we
   inherit its toxicity regression. Replace surprise before training,
   not after.

---

## Demoted proposals

The four proposals from the initial scoping are no longer
standalone tracks. They are retained here as *candidate evaluation
tasks or diagnostic analyses* for Cre-DPO.

### Construct-Crossing — *evaluation analysis*

Take a held-out LiveIdeaBench response set and decompose each
response on three orthogonal axes (semantic distance, constraint
satisfaction, axiom-modification depth). Per-model 3-vector. After
training Cre-DPO, run the decomposition on Cre-DPO outputs vs CrPO
vs DPO-only, and report which axis carries the gain. This is the
fastest way to localise *where* a method's improvements live within
the scientific-ideation construct.

### KG-CDAT-Sci — *candidate constraint-satisfaction channel*

Constrained multi-hop pathfinding on a domain knowledge graph
(Wikidata-bio, ACL anthology citation graph, MeSH). Inclusion /
exclusion constraints + endpoint pair as prompt. Score = comb-creat
× CREATE-style novelty × utility × distinctiveness. This is the most
natural source of a *constraint satisfaction* channel for `c` (open
decision 4 above). Not a standalone test — but a candidate addition
to the creativity composite.

### Axiom-Mod — *long-tail evaluation task*

A minimal, hard-verifier (SAT/SMT) test for transformational
creativity: present an axiom system and observations the system
cannot explain; score whether the model proposes a minimal axiom
modification that resolves the contradiction. Defer until Cre-DPO
results are in. If Cre-DPO makes meaningful progress on
LiveIdeaBench specificity, Axiom-Mod becomes a natural follow-up
benchmark to test whether the gains transfer to a strictly
verifiable transformational-creativity task.

### Seed-Diverge — *diagnostic for mode-collapse confound*

Score each model's diversity gain from per-sample seed-conditioning
in context (Nagarajan-style) vs from temperature alone. If the
diagnostic shows that CrPO and Cre-DPO models have very different
seed-conditioning behaviour, that's evidence the methods affect
latent diversity capacity differently. Cheap; worth running as a
sanity-check after Cre-DPO training, not before.

---

## Recommended sequencing

1. **Immediate (no training).** Run the CrPO-vs-baseline replication
   evaluation through the dat_eval pipeline. Score the released
   CrPO checkpoints (Llama-3.1-8B, Mistral-7B) plus SFT-only and
   DPO-only baselines on Arena CW, EQ CW, Mazur CW, Hivemind,
   NoveltyBench, and LiveIdeaBench. Report validity and specificity.
   This is publishable on its own as a workshop-sized result and
   motivates Cre-DPO; it also establishes whether the structural
   argument carries empirical weight before we commit GPU.
2. **Specify Cre-DPO formally.** Lock in the four open decisions
   above. Pre-register hypothesis (Cre-DPO improves specificity over
   CrPO on benchmarks with substantial R-headroom; matches CrPO on
   capability-saturated benchmarks).
3. **Train Cre-DPO** on Llama-3.1-8B and Mistral-7B with the
   matched-CrPO setup.
4. **Evaluate** through the same dat_eval pipeline. Side-by-side
   Cre-DPO vs CrPO vs DPO vs SFT vs the GPT-4o / Claude / Gemini
   tier from CrPO's reported numbers, on the full construct stack.
5. **Run Construct-Crossing** on the trained checkpoints to localise
   where gains live; run **Seed-Diverge** as a mode-collapse sanity
   check.
6. **(V2.)** GRPO-Cre.
