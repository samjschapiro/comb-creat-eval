# MCNS-DPO: Minimal Criteria Novelty Search for preference optimization

A focused proposal: replace the DPO fitness signal with **novelty conditional on a minimal criterion of appropriateness**, in the Lehman & Stanley (2010) MCNS sense. Novelty is *the* objective; appropriateness is *the* constraint. This is the most committed version of the gated-DPO sketch in [preference optimization for intrinsic novelty](./preference_optimization_for_novelty.md) and the strongest form of the Stanley-style modifications to DPO.

## Setup

- Prompt $x$, response $y$, reference policy $\pi_\text{ref}$, trained policy $\pi_\theta$.
- Behavior characterization (BC): $\varphi : \mathcal{Y} \times \mathcal{X} \to \mathbb{R}^d$, e.g. an SBERT embedding of $y$ (optionally concatenated with an embedding of $x$).
- Minimal criterion (appropriateness): $A : \mathcal{Y} \times \mathcal{X} \to \mathbb{R}$, with threshold $\tau$. Source: a reward model, an LLM judge, or a rule.
- Archive $\mathcal{A}_t$: a set of past responses (with their BCs and prompts) that have **passed the minimal criterion**. The archive only contains feasible individuals — this is the structural commitment of MCNS.
- $k$-nearest-neighbor novelty against the archive:

$$
N(y, x;\, \mathcal{A}_t) \;=\; \frac{1}{k} \sum_{a \,\in\, \mathrm{kNN}_k\!\big(\varphi(y, x),\, \mathcal{A}_t\big)} d\!\big(\varphi(y, x),\, \varphi(a)\big),
$$

with $d$ a distance in BC space (cosine or Euclidean).

## The MCNS reward

The entire optimization signal is:

$$
r_\text{MCNS}(y, x) \;=\; N(y, x;\, \mathcal{A}_t) \cdot \mathbb{1}\!\big[\, A(y, x) > \tau \,\big].
$$

Appropriateness contributes nothing positive — it gates novelty. Two responses that both pass the criterion are ranked purely by novelty. A response that fails has reward zero regardless of how novel it is.

## Two ways to plug this into training

### M1. Derived-preference DPO

Stay in the DPO framework but derive preference labels from $r_\text{MCNS}$ rather than from human pairs.

For each prompt $x$, sample $K$ responses $y_1, \ldots, y_K \sim \pi_\theta(\cdot \mid x)$. Score each on appropriateness $a_i = A(y_i, x)$ and novelty $n_i = N(y_i, x; \mathcal{A}_t)$. Set:

$$
r_i \;=\;
\begin{cases}
n_i & \text{if } a_i > \tau \\
0   & \text{otherwise.}
\end{cases}
$$

Construct preferences:

$$
y_w \succ y_l \;\;\Longleftrightarrow\;\; r_w > r_l.
$$

Cases:

- both pass the criterion: ranked by novelty.
- exactly one passes: the passing one wins (this is where appropriateness signal enters).
- both fail or are tied: drop the pair.

Loss is vanilla DPO on these dynamically labeled pairs:

$$
\mathcal{L}_\text{MCNS-DPO} \;=\; -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_\text{MCNS}^t} \!\left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)} \;-\; \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)} \right) \right].
$$

Training loop:

```
for t in iterations:
    D_MCNS_t = {}
    for prompt x in batch:
        y_1, ..., y_K  ~  π_θ(·|x)
        for each y_i:
            a_i = A(y_i, x)
            n_i = N(y_i, x; A_t) if a_i > τ else 0
            r_i = n_i if a_i > τ else 0
        build pairs from r_i ranking  →  D_MCNS_t
        A_{t+1} = A_t  ∪  { (y_i, x) : a_i > τ }
    gradient step on L_MCNS-DPO over D_MCNS_t.
```

This is an **iterative / online** DPO, not an offline one — the archive has to grow as the model changes. Pure offline MCNS-DPO is incoherent: a frozen archive of one generation's responses freezes the only thing the model is supposed to be novel against.

### M2. Direct-reward RL (more faithful to original NS)

Skip the preference reduction and use $r_\text{MCNS}$ as the scalar reward in PPO / REINFORCE with KL to $\pi_\text{ref}$:

$$
\mathcal{L}_\text{MCNS-RL} \;=\; -\,\mathbb{E}_{x,\, y \sim \pi_\theta(\cdot \mid x)}\!\left[ r_\text{MCNS}(y, x) \right] \;+\; \beta \cdot \mathrm{KL}\!\big( \pi_\theta(\cdot \mid x) \,\big\Vert\, \pi_\text{ref}(\cdot \mid x) \big).
$$

This is what Lehman & Stanley (2010) actually do: selection on $r_\text{MCNS}$. We lose DPO's offline convenience but gain the correct semantics — every gradient step is "make $r_\text{MCNS}$ larger," not "make this particular pair labeling more confident."

## Soft-gate variant

Hard gates have zero-gradient regions and $\tau$-sensitivity. Replace the indicator with a sigmoid:

$$
r_\text{MCNS}^{\text{soft}}(y, x) \;=\; N(y, x;\, \mathcal{A}_t) \cdot \sigma\!\left( \tfrac{A(y, x) - \tau}{T} \right).
$$

- As $T \to 0$: recover hard MCNS.
- As $T \to \infty$: recover additive $A + N$ (a trade-off, which we explicitly rejected).

$T$ is a knob controlling strictness — small $T$ preserves Stanley's "novelty is the only objective" intent, large $T$ softens it into a weighted combination. Worth ablating, but the default should be small $T$.

## Why MCNS-DPO is not just gated-DPO

Both gate on appropriateness, but the structural difference matters:

| Aspect | Gated-DPO (scheme G in [preference-optimization memo](./preference_optimization_for_novelty.md)) | MCNS-DPO |
|---|---|---|
| Number of preference signals | Two: appropriateness pairs, novelty pairs | One: derived from $r_\text{MCNS}$ |
| Where appropriateness lives | A separate loss term | Inside the reward, as an indicator |
| Required labels | App pairs + nov pairs, both supplied | Per-response scores $a_i$ and $n_i$ |
| Philosophical commitment | Both appropriateness and novelty are objectives | Novelty is *the* objective, appropriateness is *the* constraint |
| Archive | Not required | Required (and grows over training) |
| Anti-mode-collapse | Implicit | Explicit (archive distance penalizes duplicates) |

The philosophical commitment is the substantive point. Lehman & Stanley's 2010 argument is that in deceptive domains you *replace* fitness with novelty and use fitness as a feasibility check, rather than mixing them. Gated-DPO hedges; MCNS-DPO commits.

## Design choices to lock down

### 1. Archive scope

Three options, in order of NS-faithfulness:

- **Global archive** across all prompts. Closest to original NS. Conflates "novel response to *this* prompt" with "novel response overall." Requires $\varphi$ to include $x$ to disambiguate.
- **Per-prompt archive**: archive responses to each prompt over training; novelty is within-prompt only. Cleanest match to CDAT semantics. Suffers if any given prompt is rare in training.
- **Per-prompt-cluster archive**: cluster prompts (by embedding), one archive per cluster. Compromise; recommended default.

### 2. Behavior characterization $\varphi$

- **SBERT embedding of $y$**: direct match to DAT/CDAT metric. Recommended default.
- **SBERT + style features**: concatenate length, sentiment, register features. More expressive, more hyperparameters.
- **Learned BC**: train an encoder to discriminate human-judged-distinct responses. Most expressive, most engineering. Defer.

### 3. Source of $A$

- **Reward model**: cheapest, but inherits RM's appropriateness vs quality conflation. Use only the "is-this-on-topic" head if the RM has one.
- **LLM judge** with a narrow appropriateness rubric: more controllable, slower.
- **Rule-based**: where applicable (e.g., constraint-satisfaction prompts). Cleanest gate but limited domain.

### 4. Threshold $\tau$

Calibrate against a held-out prompt set: set $\tau$ so that, under $\pi_\text{ref}$, a target fraction (e.g. 80%) of responses pass. Re-calibrate periodically if $\pi_\theta$ drifts.

### 5. Archive size and update rule

Unbounded archive grows linearly with training. Two bounded variants:

- **Random eviction** to fixed size $M$: simple, slightly biased.
- **Novelty-thresholded admission**: only admit $y$ to the archive if its novelty exceeds a threshold $\rho$ (recommended in original NS). Caps archive size implicitly.

## Evaluation

Per the project framing (creativity is the OOD competence), every result must be on **held-out prompts** not seen during training and not represented in the archive.

Primary metrics:

- **Held-out novelty** under the same BC: average within-prompt pairwise distance among $K$ samples from $\pi_\theta(\cdot \mid x)$ for $x$ in the held-out set.
- **Appropriateness retention**: fraction of held-out responses passing $A(y, x) > \tau$, vs the same fraction under $\pi_\text{ref}$ and under vanilla-DPO baselines.
- **DAT / CDAT / PACE** on held-out prompts.

Secondary, the **Stanley headline test**: judge-rated quality of held-out responses, vs vanilla DPO trained on the same prompts with the same compute. If MCNS-DPO matches or beats vanilla DPO on judge quality *despite never optimizing for it*, that is the strong-form result of this research direction.

## Relationship to other schemes

- **Replaces** scheme G (gated DPO) in the [preference-optimization memo](./preference_optimization_for_novelty.md) as the preferred "appropriateness as constraint" formalization.
- **Extends** scheme H (conditional novelty) by adding the archive and the criterion gate.
- **Compatible** with scheme E (tempered reference): a tempered $\pi_\text{ref}$ in the KL term lets the policy roam further while $r_\text{MCNS}$ does the steering.
- **Compatible** with POET-style prompt curricula (co-evolve the prompt set with the model: grow the prompt set as the model saturates appropriateness on existing prompts).

## Open questions

1. Does novelty alone (with hard gating) avoid the mode-collapse failure of vanilla DPO without needing local-competition (NSLC)?
2. How does archive scope (global vs per-prompt) affect held-out creativity-test scores?
3. Does the strong-form result (matching judge quality without optimizing for it) hold on creative-writing benchmarks, or only on narrow divergent-thinking tasks where novelty and quality are nearly aligned by construction?
