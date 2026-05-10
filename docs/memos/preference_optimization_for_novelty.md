# Preference optimization for intrinsic novelty

Brainstorm of ways to modify DPO (and DPO-adjacent objectives) so that
training pushes the policy toward *intrinsic* novelty — novelty defined
relative to the model's own distribution, not to an external corpus.

## Setup

- Prompt $x$, preferred response $y_w$, dispreferred response $y_l$.
- Reference policy $\pi_\text{ref}$ (the SFT model DPO starts from).
- Trained policy $\pi_\theta$.
- DPO implicit reward:
  $$
  r_\theta(x, y) \;=\; \beta \, \log \frac{\pi_\theta(y \mid x)}{\pi_\text{ref}(y \mid x)}.
  $$
- DPO loss:
  $$
  \mathcal{L}_\text{DPO}
  \;=\;
  -\,\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}
  \Big[\, \log \sigma\!\big( r_\theta(x, y_w) - r_\theta(x, y_l) \big) \,\Big].
  $$

## What "intrinsic novelty" can mean

Pick one (or several) intrinsic-novelty score $N(y \mid x)$:

- **N1 — surprise under reference**: $N_1(y \mid x) = -\log \pi_\text{ref}(y \mid x)$.
  Free to compute (no extra model). Already implicit in $r_\theta$ but
  cancels symmetrically in the DPO comparison.
- **N2 — self-novelty across related prompts**: low likelihood under
  $\pi_\theta(y \mid x')$ for related $x'$. Requires neighbour prompts.
- **N3 — within-prompt diversity**: expected semantic distance between
  two samples from $\pi_\theta(\cdot \mid x)$. Requires on-policy
  sampling.
- **N4 — appropriateness-gated novelty (CDAT-style)**: novelty
  measured *only when appropriateness passes a threshold*. Structural,
  not additive.

## Approaches

### A. Vanilla DPO (baseline)

$$
\mathcal{L}_A
\;=\;
-\,\mathbb{E}\!\left[
  \log \sigma\!\left(
    \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_\text{ref}(y_w \mid x)}
    \;-\;
    \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_\text{ref}(y_l \mid x)}
  \right)
\right].
$$

No novelty signal beyond what the preference labels already encode.
Use as the control.

### B. Novelty-aware pair construction (data-side)

Keep DPO loss intact. Build/filter pairs so that $y_w$ is both higher
quality and higher novelty than $y_l$. Optionally weight pairs by the
novelty gap $\Delta_N \,=\, N(y_w \mid x) - N(y_l \mid x)$:

$$
\mathcal{L}_B
\;=\;
-\,\mathbb{E}\!\left[\, w(\Delta_N) \cdot \log \sigma\!\big( r_\theta(x, y_w) - r_\theta(x, y_l) \big) \,\right].
$$

- Pure data engineering, no objective change.
- Limit: only as good as the offline novelty proxy.

### C. Novelty margin in the sigmoid (loss-side)

Add the novelty gap as a margin term:

$$
\mathcal{L}_C
\;=\;
-\,\mathbb{E}\!\left[
  \log \sigma\!\Big(
    r_\theta(x, y_w) - r_\theta(x, y_l)
    \;+\; \alpha \,\big( N(y_w \mid x) - N(y_l \mid x) \big)
  \Big)
\right].
$$

- Closely related to IPO/SLiC-style margins.
- With $N = N_1$, this reduces to scheme D below.
- Tune $\alpha$ separately from $\beta$.

### D. Asymmetric KL — decoupled chosen/reference coefficient

Replace the implicit reward with

$$
\tilde r_\theta(x, y)
\;=\;
\beta \log \pi_\theta(y \mid x) \;-\; \gamma \log \pi_\text{ref}(y \mid x),
\qquad \gamma > \beta,
$$

then plug into the DPO sigmoid. Equivalent to scheme C with $N = N_1$
and $\alpha = \gamma - \beta$.

$$
\mathcal{L}_D
\;=\;
-\,\mathbb{E}\!\left[
  \log \sigma\!\big( \tilde r_\theta(x, y_w) - \tilde r_\theta(x, y_l) \big)
\right].
$$

- Breaks the "DPO = RLHF under KL constraint" closed form (the
  regularizer is no longer KL). That's intentional: KL toward
  $\pi_\text{ref}$ is precisely what suppresses novelty.
- One scalar to tune.

### E. Tempered reference

Replace $\pi_\text{ref}$ with a tempered/smoothed reference

$$
\pi_\text{ref}^{T}(y \mid x) \;\propto\; \pi_\text{ref}(y \mid x)^{1/T},
\qquad T > 1,
$$

(or a mixture with uniform). Plug into vanilla DPO. Rare $y$ get less
suppression, so probability mass can move toward novel modes without
fighting the reference.

- Cheap, no extra terms.
- Effect is global (changes the reference everywhere), not targeted at
  high-novelty examples.

### F. Self-novelty regularizer (on-policy)

Add a within-prompt diversity bonus computed from on-policy samples:

$$
\mathcal{L}_F
\;=\;
\mathcal{L}_\text{DPO}
\;-\;
\lambda \cdot \mathbb{E}_{x}\!\left[
  \mathbb{E}_{y, y' \sim \pi_\theta(\cdot \mid x)} \, d(y, y')
\right],
$$

with $d$ a semantic distance (sentence-embedding cosine, DAT-style
pairwise, etc.).

- Directly fights mode collapse.
- No longer pure DPO — needs sampling at training time.
- Closest objective to DAT's "pairwise distance" score.

### G. Gated DPO (CDAT-inspired, additive → gated)

Maintain two preference signals per prompt:

- appropriateness pairs: $y_w^a \succ y_l^a$
- novelty pairs:         $y_w^n \succ y_l^n$

Gate the novelty term on a per-example appropriateness check (RM score
above threshold $\tau$, or both responses being on-topic):

$$
\mathcal{L}_G
\;=\;
\mathcal{L}_\text{DPO}^{\text{(app pairs)}}
\;+\;
\lambda \cdot
  \mathbb{1}\!\left[\, A(y_w^n, x) > \tau \;\land\; A(y_l^n, x) > \tau \,\right]
  \cdot \mathcal{L}_\text{DPO}^{\text{(nov pairs)}}.
$$

- The structural import from CDAT: appropriateness is a *precondition*,
  not a co-dimension. Linear combinations $\mathcal{L}_\text{app} + \lambda \mathcal{L}_\text{nov}$
  admit middling trade-off solutions; gating does not.
- Requires an appropriateness signal (RM, judge, or rule).

### H. Conditional novelty (CDAT-inspired, on-policy)

Define novelty as diversity *conditional on the prompt*: with $k$
on-policy samples $y_1, \ldots, y_k \sim \pi_\theta(\cdot \mid x)$,

$$
N_\text{cond}(x)
\;=\;
\frac{1}{k(k-1)} \sum_{i \neq j} d(y_i, y_j).
$$

Use $N_\text{cond}(x)$ in any of schemes C, F, or G. Adds a per-prompt
appropriateness gate $A(y_i, x) > \tau \;\; \forall i$.

- Rules out the failure mode where the policy adopts one "exotic"
  style and applies it to every prompt.
- This is the closest direct analog of CDAT's score.

### I. Population-level appropriateness gate (CDAT t-test analog)

Outer-loop control: periodically evaluate $\pi_\theta$ on held-out
prompts; only keep the novelty-pressure term active while a
population-level appropriateness test passes against a baseline (e.g.
$\pi_\text{ref}$ or a random baseline). If appropriateness drops below
baseline, freeze the novelty term (or revert to the last passing
checkpoint).

$$
\lambda_t \;=\; \lambda \cdot \mathbb{1}\!\left[\, \text{test\_passes}(\pi_{\theta_t},\, \pi_\text{ref}) \,\right].
$$

- A stop-condition, not a per-step loss change.
- Prevents the well-known DPO failure mode where the chosen-side
  reward runs away.

## Practical recommendations

1. **Baseline**: vanilla DPO (A) on the same data, same $\beta$.
2. **First non-trivial scheme**: D (asymmetric KL). One scalar, no
   extra data, directly addresses the structural reason DPO is
   novelty-neutral.
3. **First CDAT-style scheme**: G (gated DPO). Needs an
   appropriateness label per response (RM or judge).
4. **First on-policy scheme**: H (conditional novelty + per-prompt
   appropriateness gate). Most expensive but cleanest analog of CDAT.

## Evaluation

Per the project framing: creativity is the OOD competence. Every
scheme above must be evaluated on **held-out prompts**, not on the
training prompts where the novelty score was computed. A scheme that
only improves novelty on training prompts has not taught creativity.

Suggested held-out metrics:

- DAT / CDAT / PACE on prompts disjoint from the training set.
- Per-prompt diversity of on-policy samples (semantic-embedding
  pairwise distance, distinct-n).
- Appropriateness retention vs $\pi_\text{ref}$ (RM win-rate, or judge
  appropriateness score) — to confirm the gate is working.
