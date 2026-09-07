# Process-level diversity: what data exists, and what it lets us ask

**Written 2026-07-29.** Starting point for an RL study aimed at the TwistBench finding that LLM
creative *process* is homogeneous (207/207 traces twist-first, 0/207 plot-gating). This memo is
deliberately data-first: what we already own, what exists in the world, what does not exist, and
only then which study designs the data can actually support.

Companion to [conditioned_divergence_vs_darling.md](./conditioned_divergence_vs_darling.md),
[mcns_dpo.md](./mcns_dpo.md), and
[../tracks/plot_twist/reasoning_trace_analysis.md](../tracks/plot_twist/reasoning_trace_analysis.md).

---

## 1. What we already own

All counts below are read off the files, not from prior write-ups.

### plot_twist (`data/plot_twist/`, 181 MB)

| Asset | Unit | n | Fields |
|---|---|---:|---|
| `llm_twists/stories/` | model stories | **2,190** across **73 models** | story, temp, sample idx |
| `annotations/annotations.json` | annotated stories | **2,070** | `setup`, `reveal`, rubric `scores`, `why_scored` |
| `twist_class/classified.json` | classified reveals | **2,053** | `reveal`, 8-code taxonomy, surprise, coherence |
| `structural/structural.json` | structurally extracted | **97** | `t_mod`, `t_mod_frac`, `preservation`, `structural_ptc`, `dsi`, 2 extractors |
| `thinking/stories/` | traced generations | **216** across **9 models** | `story`, `reasoning_trace`, effort level, token counts |
| `thinking/downstream/trace_moves/` | move-coded traces | **207** | 10 boolean reasoning moves |
| `thinking/downstream/trace_anchor/` | anchor-coded traces | **207** | `twist_first`/`plot_first`/`interleaved`, `plot_gating`, `emergent_clue` |
| `prompt_methods/stories/` | prompting-intervention | **318** across **20 models** | be-creative, in-context-regen |
| `human_twists/texts/` | human stories | **35** (18 used as STRONG gold) | public-domain + expert |
| `predict/pilot_v1/` | twist-prediction pilot | 4 predictors × **105** stories | embedding similarity of predicted vs actual reveal |

Two of these matter more than the rest.

**The twist taxonomy (n=2,053) is the only large process-adjacent label set that exists anywhere.**
It codes what kind of assumption the reveal overturns. The distribution is itself the homogeneity
result, stated in a form nobody else has:

| code | n | share |
|---|---:|---:|
| IDENTITY | 696 | 33.9% |
| ONTOLOGICAL | 541 | 26.4% |
| NONE | 293 | 14.3% |
| FACT_OBJECT | 269 | 13.1% |
| ORCHESTRATION | 177 | 8.6% |
| MORAL | 56 | 2.7% |
| TEMPORAL | 16 | 0.8% |
| NARRATOR | 5 | 0.2% |

Two codes cover 60% of all twists produced by 73 models. Three codes are effectively unused. This
is a directly measurable target: a method works if it moves this distribution toward uniform without
losing the realism gate.

**The 207 traces are the only creative reasoning traces we have, and they are exhausted.**
100% twist-first, 0% plot-gating, at every effort level. As a *finding* this is strong. As
*training data* it is worthless — there are zero positive examples of the behavior we want. You
cannot supervise toward a class with no members.

### creativity_rl (`data/creativity_rl/`, 12 MB)

Infrastructure, not data. Faithful reimplementations of DARLING (union-find over a semantic
equivalence classifier) and DivPO, plus GRPO/DMPO runners, an archive, and NoveltyBench and
Hivemind eval harnesses run locally. The one headline result on record is a **null**: MCNS-RL
`full_run_v1` did not beat baselines.

The stale `progress.md` badly understates this track — it stops at 2026-05-15 and does not
mention DARLING, DMPO, or conditioned GRPO at all.

---

## 2. What exists externally

Grouped by the role it could play.

### Prompts and stories (plentiful, cheap)

- **WritingPrompts** (Fan et al. 2018) — 300K stories over 97,223 Reddit prompts. Already the
  TwistBench seed source. Average **2.8 stories per prompt**, but ~53 prompts have **≥50 human
  responses** each (≈2,650 stories). That subset is the important part — see §3.
- **ROCStories**, **STORIUM** (structured story cards), **StoryWars**, **GPT-WritingPrompts**.
- **WHODUNIT** (arXiv 2502.07747) — mystery narratives with ground-truth culprit; already scouted
  in the plot_twist design as a validation set.
- **Annotated Mystery Narratives** (JOHD 2025) — clue/reveal patterns annotated by literary
  scholars. New since the plot_twist design was written; worth a look for reveal-structure labels.
- **TRIPOD** — movie synopses annotated with turning points.

### Quality / reward signal (solved, off the shelf)

- **LitBench** (arXiv 2507.00769) — **43,827 human preference pairs** from r/WritingPrompts plus a
  2,480-pair debiased test set. Trained reward models hit 78% agreement vs 73% for the best
  zero-shot judge (Claude-3.7-Sonnet). This is the single most useful external asset: it removes
  the need to build a creative-writing quality reward from scratch.
- **StoryAlign** (arXiv 2605.04831) — reward models specifically for story generation.

### Process / reasoning chains (almost nothing)

- **COIG-Writer** (arXiv 2510.14763) — **1,665 triplets** over 51 genres: reverse-engineered
  prompt, creative reasoning chain, final text. CC BY-SA 4.0. **Chinese only.** The reasoning
  chains are *reconstructed post-hoc by annotators from finished texts*, not observed while
  writing. Their reported findings are relevant on their own: process supervision works but needs
  ≈1:12 creative-to-general data or performance collapses (62.75% → 35.78%), and lexical diversity
  correlates *negatively* with creative quality.
- **Writing-process corpora outside creative writing**: ArgRewrite V.2 (student argumentative
  revisions), NewsEdits (news revision histories), and a keystroke-level corpus of 61 junior
  researchers' essays. All non-fiction, all revision-focused. None captures the generative decision.

### Multiple human attempts at one premise (rare, and the key gap-measurement asset)

- **TimeTravel** (from ROCStories) — 29,849 counterfactual rewritings; validation and test sets
  carry **3 independent human endings per story** across 1,871 stories each.
- The **WritingPrompts ≥50-response subset** above.

These two are the only sources where many humans respond to one fixed premise. Nothing else
measures what human process diversity even looks like.

---

## 3. What does not exist

Stating this plainly because it determines everything downstream.

1. **No observed creative process, for anyone.** Every corpus is finished text, model text, or a
   post-hoc reconstruction. COIG-Writer is the closest and it is reverse-engineered. Human
   *think-aloud* protocols exist in the writing-studies literature (Emig onward) but as a research
   method, not a released corpus at usable scale.
2. **No dataset of process diversity.** Nobody has "here are N structurally different ways to
   approach the same premise, labeled by approach." Not in creative writing, not anywhere.
3. **No positive examples of plot-gating.** Our own trace corpus is 0/207. Neither we nor anyone
   else has a single recorded instance of a model letting the premise veto its twist.
4. **No human reference for process homogeneity.** We can say models are homogeneous. We cannot
   currently say whether humans are less so, because our 35 human texts have no process attached.
   The WritingPrompts ≥50 subset and TimeTravel are the only ways to build that reference, and both
   give it at the level of the *decision* (what twist / what ending), not the *procedure*.

---

## 4. Prior work that already occupies part of this space

Two papers matter a lot and both are recent.

**DPWriter** (arXiv 2601.09609, Jan 2026) — closer to the obvious method than is comfortable.
Semi-structured long CoT with five tagged planning fields (`<goal> <info> <struct> <lang> <pres>`),
segment-wise branching that samples K continuations per planning segment and keeps the diverse ones,
plus a group-aware diversity reward. Trained on 43K writing instructions assembled from DeepWriting,
WritingPrompts, CreateSet, and COIG-Writer, with plans generated by GPT-4.1. Backbones Qwen3-4B-Base
and Llama-3.2-3B-Instruct. Baselines include GRPO, GRPO-Unlikeliness, DARLING, and GAPO. Reports
+15% embedding diversity and +9.9% n-gram diversity over GRPO at equal quality.

The important detail: **they branch at the plan but the reward is n-gram novelty over final
outputs.** Plan diversity is a *means* to surface diversity, never a measured end. Their own stated
limitation is that "whether diversity benefits creativity remains an open question."

**Are We Measuring Strategy or Phrasing?** (arXiv 2606.29985, June 2026) — the math-domain version
of our question, and it reports three things we should treat as established until refuted:

- Surface diversity metrics are unreliable proxies for approach-level diversity.
- Under diversity-aware RLVR, surface diversity is preserved while **approach-level diversity
  declines**.
- Training against an **LLM-judge diversity reward causes the policy to exploit judge preferences
  rather than broaden its approaches**.
- They conclude direct optimization of approach-level diversity is an open problem.

That third point is the most useful thing in this memo. It is empirical evidence, from a domain
with verifiable rewards, that the naive version of our study fails. Any design that scores process
diversity with an LLM judge should be assumed hacked unless proven otherwise.

Related: "When Reasoning Narrows the Move" (2607.19523) on diversity collapse in game play, and the
broader RLVR diversity-collapse literature (pass@k degradation, entropy collapse).

---

## 5. What the data permits

Constraints that follow from §1–§4, in rough order of how binding they are.

- **Open weights only.** Frontier providers serve summarized CoT (TwistBench caveat 3). Any
  process signal read off a trace requires models whose CoT we own. That caps us at the scale
  `creativity_rl` already targets (1.5B–8B), and DPWriter's 3–4B choice is the same constraint.
- **No supervised process target is available.** Zero positive examples means SFT-toward-plot-first
  is off the table. The signal has to be *unsupervised* (entropy/partition within a group),
  *structural* (a predicate that can be checked without a judge), or *synthesized*.
- **LLM-judge process rewards are known to get hacked** (2606.29985). Rules out the most obvious
  design.
- **Quality reward is solved** — LitBench, plus TwistBench's own validated realism gate. This is
  not where effort should go.
- **The measurement target already exists and is cheap.** The 8-code taxonomy over 2,053 reveals is
  a working process-adjacent metric with a known baseline distribution. Its ceiling is that 8 coarse
  codes may be too blunt to detect real change, and the classifier's reliability at the rare codes
  (TEMPORAL n=16, NARRATOR n=5) is unestablished.
- **A human process-diversity reference is buildable but not free.** Requires running the taxonomy
  classifier over the WritingPrompts ≥50-response subset (≈2,650 human stories) and/or TimeTravel's
  3-endings-per-story sets. Cheap in dollars, and it is the only way to know what target to aim at.

---

## 6. Designs the data supports, ranked

Not commitments — the space of studies these constraints leave open.

**A. Measurement-first paper: is process diversity separable from output diversity in creative
writing?** Port the 2606.29985 question from math to creative writing, where we uniquely have the
labeled corpus. Run the taxonomy and anchor classifiers over assets we already hold — the 318
prompting-intervention stories (including in-context-regen, the one intervention that reached human
parity on output diversity) and the DARLING run logs. If interventions that raise output diversity
leave the taxonomy distribution flat, that is the gap, established for a few dollars and no GPU.
Add the human reference from WritingPrompts ≥50 to make it a comparison rather than an observation.
Lowest cost, highest certainty, and it is the prerequisite for anything else.

**B. Structural process reward.** Make process legible by forcing a typed intermediate artifact so
process properties become checkable predicates instead of judgments — which axiom was flipped,
whether it was present in the plan before the twist was named, whether prior artifacts survive. This
is the CSAM leg from the dropped plot_twist method paper, repurposed: it failed as a prompting
method, but its value was always that it makes the twist *inspectable*, and inspectable is exactly
what a non-hackable reward needs. `structural/structural.json` already has `t_mod` and
`preservation` computed for 97 stories, so the extractor exists. Main risk: emission order is not
decision order — a model can write the plan to contain the axiom it already picked. Mitigable by
freezing the plan and sampling several flips from it, which also makes the premise a real
constraint.

**C. Data synthesis.** The binding constraint named in
[conditioned_divergence_vs_darling.md](./conditioned_divergence_vs_darling.md) — "sequences of
deliberately-diverse attempts at one prompt do not occur naturally and must be synthesized" — is the
same constraint §3 arrives at from the data side. One premise, K structurally different approaches,
gate-filtered, is both the training data and the missing dataset. This is where a dataset
contribution would live, and it is downstream of A telling us the taxonomy can detect what we want.

**D. Twist affordance.** Not every premise supports several good twists; rewarding diversity on
premises that afford one is training for degradation. Affordance is measurable from data we already
have — the entropy of taxonomy codes across 73 models on each premise. Small, and useful to whatever
design wins.

Note that A, C, and D all reuse existing data and cost roughly nothing. Only B needs GPUs.

---

## 7. Open questions, cheapest first

1. **Is the 8-code taxonomy sensitive enough to detect a real shift?** Its inter-rater reliability
   was never established, and half the codes are near-empty. Check before anything is built on it.
2. **Do output-diversity interventions move it at all?** Answerable today from `prompt_methods/`
   and `darling_logs/`.
3. **Are humans actually more process-diverse?** Answerable from WritingPrompts ≥50 and TimeTravel.
   If humans are *also* concentrated on IDENTITY and ONTOLOGICAL, the framing changes completely and
   we should know that early.
4. **Where do trainable open models (≤8B) rank on TwistBench?** If they sit at the floor, improvement
   is unmeasurable. We have 73 models scored, so this is a lookup, not an experiment.
5. **Why is the predict-the-twist pilot so flat?** Four very different predictors all land at
   0.69–0.72 mean similarity. Either predictor skill genuinely does not vary, or embedding similarity
   is saturated and insensitive. The second would matter for
   [predict_the_twist.md](../tracks/plot_twist/predict_the_twist.md).
6. **Is DPWriter's plan-branching already the method?** Needs a full read of the paper, not the
   abstract. The distinction we would be claiming — rewarding plan-level diversity as an end rather
   than using it as a means to surface diversity — is narrow, and needs to be verified as real
   before it is built on.

---

## References

- Fan et al. 2018, *Hierarchical Neural Story Generation* (WritingPrompts), arXiv 1805.04833
- Fisher et al. 2025, *LitBench*, arXiv 2507.00769
- *COIG-Writer*, arXiv 2510.14763
- *DPWriter: RL with Diverse Planning Branching for Creative Writing*, arXiv 2601.09609
- *Are We Measuring Strategy or Phrasing?*, arXiv 2606.29985
- *When Reasoning Narrows the Move: Diversity Collapse in LLM Game Play*, arXiv 2607.19523
- Qin et al. 2019, *Counterfactual Story Reasoning and Generation* (TimeTravel), arXiv 1909.04076
- *WHODUNIT*, arXiv 2502.07747
- *StoryAlign*, arXiv 2605.04831
- Li et al. 2025, *DARLING*, arXiv 2509.02534; Lanchantin et al. 2025, *DivPO*, arXiv 2501.18101
