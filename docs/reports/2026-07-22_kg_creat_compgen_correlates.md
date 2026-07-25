# Comp-gen benchmarks as correlates for kg_creat — lit review

**Date:** 2026-07-22 · **Track:** kg_creat (ICLR paper) · **Status:** review, no runs

**Question.** The paper frames combinatorial creativity against compositional generalization
([04_background.tex](../../papers/kg_creat-iclr/content/04_background.tex) §Aspects of Comparison). If we want an
empirical arm — *does comp-gen ability predict combinatorial-creativity ability across models?* — which comp-gen
benchmark do we correlate against?

**Provenance caveat.** Produced by a fan-out search → fetch → adversarial-verification pipeline that was stopped
during final synthesis. 57 claims received 3-vote verification; 14 were refuted. Claims below are marked
[V] verified, [R] refuted-as-originally-stated, or [U] unverified. **Spot-check any number before it enters the
paper** — the unverified tier is where fabricated specifics tend to hide (two were caught: a "1–100 words" range
that maxes at 12, and a model count matching no version of its source).

---

## 1. Headline: the three obvious candidates all failed verification

The search phase nominated Ordered CommonGen, Compositional GSM, and Skill-Mix as the top-3. Verification broke
all three **as dependent variables for a cross-model correlation**. This is the main result of the review.

### Ordered CommonGen — Sakai, Kamigaito & Watanabe, ACL 2025
[arXiv:2506.15629](https://arxiv.org/abs/2506.15629) · [ACL Anthology](https://aclanthology.org/2025.acl-long.1508/)

Given a concept set plus a required **order**, generate a covering sentence — structurally the closest published
analogue to our ordering constraint, 36 models in one table. Looked ideal. Three independent verifiers refuted it:

- **[R] The headline metric is non-monotone in capability.** `Ordered Rate` is a *conditional ratio*
  (Coverage-w/order ÷ Coverage-w/o-order; verified arithmetically: 74.44/98.91 = 75.26). Small denominators inflate
  weak models. **Qwen2-0.5B scores 57.34, above GPT-4o at 55.16**; Phi3-mini (62.04) also beats GPT-4o; Mistral-large
  (27.19) sits at the Llama3.2-1B floor. A DV where a 0.5B model outranks GPT-4o cannot carry a capability correlation.
- **[R] Not non-saturated.** The "best model only ~75%" claim covers Table 1's non-reasoning cohort only. Appendix C
  Table 7: **o1-mini = 85.48 Ordered Rate**. A Sept-2024 small reasoning model already clears the supposed ceiling.
- **[R] Unstable and method-confounded.** One in-context exemplar swings per-model scores ±19 points and inverts
  rankings (Qwen2.5-72B 51.73→40.66). Open models were run **4-bit quantized**, proprietary via API.
- **[V] Zero frontier coverage.** All 36 models pre-2025; no Claude of any version, no GPT-5.x, Gemini 2.5/3,
  DeepSeek-R1. No live leaderboard; a frozen June-2025 table.

**Verdict:** do not borrow its numbers. *If* self-run with a fixed protocol, use **Coverage-w/order** (the raw rate),
never Ordered Rate. The task design remains the best conceptual match to our ordering constraint — worth citing as
related work regardless.

### Compositional GSM — Hosseini et al., ICLR 2025
[arXiv:2410.01748](https://arxiv.org/abs/2410.01748)

Chained GSM8K pairs; headline "reasoning gap" `Δ = S_comp − S1×S2`. The difference-score design is genuinely
attractive (it partials out single-hop competence). But:

- **[R] The saturation rationale is fabricated attribution.** The strings `saturat*` and `ceiling` occur **zero times**
  in the full text. The paper's stated rationale for subtracting `S1×S2` is an independence baseline, not ceiling
  avoidance. The authors explicitly disclaim benchmark intent: *"our goal is not to present yet another reasoning benchmark."*
- **[R] Δ compresses at the top — backwards for our use.** Δ is bounded by `1 − S1·S2`, so as scores → 1 it collapses.
  The paper's own finding: *"cost-efficient and smaller LLMs exhibit a much larger gap than closed-source frontier LLMs."*
  Spread lives in the weak tail, not among frontier models. Worse, Δ is a residual: a weak model with S1=S2=0.3 posts
  Δ≈0 and ranks alongside GPT-4o.
- **[R] Not reproducible.** No code/data release, no GitHub, **not on HuggingFace**. The paper never says *which* 1,200
  of GSM8K's 1,319 test items were used, nor the item-level pairing procedure. Cost is also 3 test sets × 1,200 =
  **3,600 calls/model**, not 2,400.
- **[V] Panel is 2024-era**: ~17 base models; `Claude` = 0, `GPT-5` = 0, `DeepSeek` = 0, `o1` = 0 occurrences.

**Verdict:** cite the *gap-metric idea* (it is the right instinct for controlling capability), don't use the benchmark.
If we want a difference score, build it into kg_creat itself — we already have the matched-bundle machinery for exactly
this (`ΔR_emit`, `Δsat` vs baseline), which is strictly better than importing someone else's residual.

### Skill-Mix — Yu, Kaur, Gupta, Brown-Cohen, Goyal & Arora, ICLR 2024
[arXiv:2310.17567](https://arxiv.org/abs/2310.17567) · [skill-mix.github.io](https://skill-mix.github.io/) · [github.com/LeoYu/skill-mix](https://github.com/LeoYu/skill-mix)

Conceptually our closest sibling: compose *k* random skills into short text, LLM-graded — a novelty × satisfaction
structure much like ours, with difficulty tunable via *k* and contamination low by construction (N^k combinations).

- **[R] The cited k=4 config is a floor effect, not good spread.** Ratio-of-full-marks sorted: `.00 .00 .00 .01 .01
  .02 .03 .08 .09 .52` — median .015, 7/10 models in [.00,.03], one outlier (GPT-4) carrying the range. The paper's
  own §5.1: *"all models saturate on or before k = 3 with GPT-4 grading."* For 50 models this yields mass ties at zero.
- **[R] Grader-dependent, not a stable per-model quantity.** LLaMA-2-70B-Chat scores **.40 under a LLaMA-2 grader vs
  .00 under GPT-4**; the authors concede the grader favors its own family. The original GPT-4 grader is deprecated.
- **[V] All models pre-2024**; no borrowable frontier numbers.

**Verdict:** the *k*-knob is the saving grace — retuned to k=5–7 with a modern fixed grader, spread could be restored.
But grader drift is a live confound for cross-model comparison, and it is the most expensive option (LLM judge per item).
Strong **related-work** cite; a self-run only if we accept judge cost and pin one grader.

---

## 2. What actually survived — ranked shortlist

### 1. BeyondBench (Hard/Medium tiers) — ICLR 2026 · [arXiv:2509.24210](https://arxiv.org/abs/2509.24210) [V 2026-07-22]
Srivastava, Hussain, Bi, Roy, Pitre, Lu, Ziyadi & Wang. [Site](https://ctrl-gaurav.github.io/BeyondBench/) ·
[GitHub](https://github.com/ctrl-gaurav/BeyondBench) · [OpenReview](https://openreview.net/forum?id=mIKqVWGjwI)
(OpenReview title differs — "Benchmark-Free Evaluation of Reasoning in Language Models"; renamed for camera-ready.)

Best on every *statistical* criterion, weakest on *construct* validity — see the caveat below.
- **[V] Coverage:** 101 models (85 open, 16 closed), 0.5B–141B, multiple quantization schemes. Interactive
  leaderboard, PyPI package, GitHub → we can generate our own ~50-model scores.
- **[V] Contamination:** structurally ruled out, not merely audited — problems generated algorithmically at eval time
  from a space of **>10^15 unique instances** per task, deterministically verified. Strongest contamination story found.
- **[V] Structure:** 44 tasks / 117 variations in three tiers — Easy (29 tasks, arithmetic + statistics),
  Medium (5 tasks, 49 variations, sequence patterns), Hard (10 tasks, 68 variations, NP-complete + constraint
  satisfaction). **Use Hard/Medium only**; Easy saturates.
- **[V] Spread:** Hard suite non-saturated — Gemini-2.5-Pro ~56%, Qwen2.5-72B ~33%, Llama-3.3-70B ~27%; GPT-5 family
  declines 16.81–43.95 points without tool use. *Number drift:* the abstract gives 56.21/27.16/33.37 and the body
  56.38/26.91/33.60 — cite whichever version you pull.
- **⚠ Construct-fit caveat (the real weakness).** BeyondBench measures **algorithmic reasoning**, not compositional
  generalization as [04_background.tex](../../papers/kg_creat-iclr/content/04_background.tex) defines it
  (systematicity / productivity / substitutivity / localism / overgeneralization). The genuine overlap is
  **productivity** — its polynomial→exponential complexity scaling is a length/depth-generalization probe. It tests
  **no systematicity**. Its constraint-satisfaction tasks arguably map onto our `sat`/execution axis rather than
  ideation, but that is a different claim from "comp-gen predicts combinatorial creativity." Decide deliberately;
  a reviewer who takes the §4 taxonomy seriously will press on this.

### 2. ARC-AGI-2 — [arcprize.org/leaderboard](https://arcprize.org/leaderboard) · [arXiv:2505.11831](https://arxiv.org/abs/2505.11831) [U]
- **The only actively-maintained frontier table** — updated with each release (Opus 4.5 Thinking 37.6% @ $2.20/task;
  Gemini 3 Pro base ~31% @ $0.81/task). Private eval set → low contamination.
- **Two caveats:** small models (Llama-3.1-8B, Flash-Lite) score ~0 → **floor effect at our low end** (pair ARC-AGI-1
  for the bottom of the range); and the board mixes scaffolded systems with bare models — **use only the "Base LLMs"
  rows** to stay apples-to-apples with our prompted setup.
- Construct caveat: abstraction/systematic generalization, not comp-gen in the linguistic sense we define in §4.

### 3. AgentCoMa — ACL 2026 · [arXiv:2508.19988](https://arxiv.org/abs/2508.19988) [U]
- **61 LLMs** in-paper — the largest comp-gen panel found — across base/instruct/reasoning-distilled.
- Mixes commonsense + math reasoning, giving a **~30% average compositional drop**, larger than same-type multi-step
  benchmarks → good spread.
- **Frictions:** HF dataset (`LisaAlaz/AgentCoMa`) is **gated** (login + email/username sharing) and the license is not
  stated — verify terms first. The public leaderboard is thin (16 self-reported entries) and skews open-weight; frontier
  GPT/Claude/Gemini entries absent, so we'd generate those ourselves.

### 4. Ordered CommonGen, self-run with `Coverage-w/order` [R on the published metric]
Kept only because it is the **closest structural analogue to our ordering constraint** and is trivially cheap
(short prompts, string-based scoring, no judge, 4,608 items from 24 permutations of 4-concept sets). If used: fix one
prompt protocol, no quantization, and never use `Ordered Rate`.

### Cite-but-do-not-use (with the reason to state in the paper)
| Benchmark | Why excluded |
|---|---|
| **SCAN** | Scaffold-determined, not a model property: same base model scores ~16% naive vs **99.7%** with least-to-most ([arXiv:2205.10625](https://arxiv.org/abs/2205.10625)) and 100% with SKiC. 20-word vocabulary → memorization near-certain. [V] |
| **CFQ / MCD, COGS** | Published "LLM numbers" are pipeline scores (Drozdov et al., [arXiv:2209.15003](https://arxiv.org/abs/2209.15003): input parsing + decomposition + dynamic exemplar selection around code-davinci-002). They measure prompt engineering, not the model, and differ per paper. [V] |
| **SLOG / ReCOGS / COGS-vf** | All published numbers are from **trained** models on the fixed 32,755-example COGS split ([arXiv:2310.15040](https://arxiv.org/abs/2310.15040)). No prompted-LLM table exists; zero-shot scores would mostly measure logical-form format compliance. [U] |
| **Plain CommonGen** | Order-agnostic coverage is ceilinged on modern LLMs (this is Sakai et al.'s own motivation). [V] |
| **Skills-in-Context tasks** | A *prompting-method* paper ([arXiv:2308.00304](https://arxiv.org/abs/2308.00304)), not a benchmark; tasks all borrowed. Useful as the citation for *why we must fix one prompting protocol across all models*. [V] |
| **CryptoX / CryptoBench** | [arXiv:2502.07813](https://arxiv.org/abs/2502.07813). Broader coverage than first reported (27→33+ models incl. 4 Gemini and DeepSeek-R1 [R]), but it measures compositional *reasoning* by wrapping existing items in cipher layers — weak construct fit for comp-gen, and no maintained leaderboard. |
| **Open LLM Leaderboard v2 (BBH, MuSR)** | Frozen ~13 Mar 2025, open-weights only → never contains GPT-5.x / Claude 4.x / Gemini 3. If used anyway, pull *per-subtask* scores (`dyck_languages`, `word_sorting`, `tracking_shuffled_objects`) from the `results` dataset — the 23-subtask BBH average is not a comp-gen axis. [U] |

---

## 3. The methodological finding that matters more than the choice

**A high correlation will prove nothing unless we control for general capability.** This came through every
methodology source and should shape the arm's design before we pick a benchmark at all.

- **[V] ρ ≈ 0.73 is the null, not the result.** Epoch AI, over 17 benchmarks (≥5 shared models): median pairwise
  Spearman **0.73**, and *cross-domain* pairs (0.68) correlate nearly as strongly as *within-domain* pairs (0.79).
  A ρ≈0.7 between any comp-gen benchmark and kg_creat is the expected baseline for two arbitrary benchmarks.
  → [epoch.ai/data-insights/benchmark-correlations](https://epoch.ai/data-insights/benchmark-correlations)
- **[V] Individual benchmarks correlate with a single latent capability factor at median ρ = 0.90.** Partial that out.
  The **Epoch Capabilities Index** is the ready-made covariate — IRT-style, stays informative after individual
  benchmarks saturate, needs only 4 evals/model, frontier-prioritized, `pip install epochai`, CC-BY.
  → [epoch.ai/benchmarks/eci](https://epoch.ai/benchmarks/eci) · data: [epoch.ai/benchmarks/use-this-data](https://epoch.ai/benchmarks/use-this-data)
- **[U] Report eigenvector rotation, not just ρ.** "The Rise and Fall of G in AGI" ([arXiv:2604.09911](https://arxiv.org/html/2604.09911)),
  39 LLMs × 14 benchmarks, PC1 = 90% of variance. Its recipe for a *new* benchmark: show how it rotates the leading
  eigenvector and whether it raises **effective dimensionality** (their worked example: adding GPQA-Diamond moved
  eff-dim 1.3 → 1.9, exposing a second factor). This is the strongest available argument that kg_creat is not
  redundant with *g* — and a far better headline than a raw correlation.
- **[U] n = 8 is nowhere near enough; ~50 is defensible.** AGC-Bench ([arXiv:2607.01152](https://arxiv.org/html/2607.01152)),
  83 models, shows a creativity factor surviving partialling-out of fluid reasoning (ρ=+0.55) and MMLU-Pro (ρ=+0.62),
  with bootstrap subsampling to 60/83 for eigenvalue stability. Also a direct precedent for our whole design.
- **[U] Use 2–3 benchmarks of different provenance and report separately.** Sun et al., CoNLL 2023
  ([arXiv:2310.17514](https://arxiv.org/abs/2310.17514)): 6 approaches × 4 comp-gen datasets × 18 splits — the
  benchmarks **rank models differently**, and *shared data source* predicts ranking agreement better than *shared
  definition of compositionality*. A null against any single suite would be uninterpretable. Do not pool.
- **[U] Discriminant validity, not just convergent.** Construct-validity review ([arXiv:2511.04703](https://arxiv.org/pdf/2511.04703))
  → include one theoretically *related* benchmark (comp-gen) **and** one difficulty-matched *unrelated* one; the
  convergent correlation only counts if the discriminant one is materially lower (psychometrics convention: gap > 0.40).
- **[U] Watch for family clustering.** Epoch's two-factor decomposition (general capability + "Claudiness", a
  training-recipe style axis) is a specific threat to us: **embedding-remoteness novelty is exactly the kind of measure
  that loads on a style/verbosity/diversity axis rather than on ability**. Check the relation survives *within* model
  family, or report family-level random effects.
  → [epoch.ai/gradient-updates/benchmark-scores-general-capability-claudiness](https://epoch.ai/gradient-updates/benchmark-scores-general-capability-claudiness)

---

## 4. Recommendation

1. **Correlate against BeyondBench (Hard) + ARC-AGI-2 Base-LLM rows** — different provenance, both non-saturated,
   both with real frontier coverage, per Sun et al.'s don't-pool rule. Add AgentCoMa if its license clears.
2. **Report the semi-partial correlation after regressing both axes on ECI**, alongside the raw ρ — and state the 0.73
   null explicitly so a reviewer cannot read ρ=0.7 as a finding.
3. **Make eigenvector rotation / effective dimensionality the headline**, not the raw correlation. That answers the
   abstract's actual question ("is creativity any different from generalization?") in a way a correlation cannot.
4. **Do not borrow published numbers for the classic suites.** Every one is either scaffold-determined, trained-model,
   or pre-frontier. The cite-but-not-use table above is the paper-ready justification.
5. **Prefer our own `Δ` over an imported one.** The matched-bundle design already gives within-`(u,v)` deltas that
   control difficulty far better than Compositional GSM's residual.

## Open items before citing

- ~~Verify BeyondBench~~ **done 2026-07-22** (arXiv abstract + project site; all claims held, minor number drift noted).
  Still to verify: AgentCoMa and the ARC-AGI-2 leaderboard numbers — both [U].
- Decide whether BeyondBench's construct fit (algorithmic reasoning ≈ productivity, no systematicity) is acceptable,
  or whether the shortlist needs a systematicity-probing benchmark alongside it.
- Confirm arXiv 2604.09911 and 2607.01152 exist as described (recent preprints, surfaced only via search snippets).
- Check AgentCoMa's license/gating terms.
- Confirm ECI covers enough of our ~50-model OpenRouter set (complete-case batteries shrink fast: 39 models → 19–22
  with full coverage, per the G-factor paper).
