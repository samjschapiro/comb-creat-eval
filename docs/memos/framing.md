# Paper framing memo (2026-04-25, updated)

Synthesized from a literature deep-dive on DAT/CDAT/PACE for LLMs, the
broader landscape of LLM creativity evaluation, and the human-psychometric
tradition on creativity-vs-g. Updated 2026-04-25 to fold in LiveIdeaBench
results (n=17, scientific ideation as a 3rd construct).

## Theme

> Semantic-distance creativity tests for LLMs need construct-validity
> audits that treat general capability as a confound. We provide the first
> such audit, and the result reshapes when these tests should and shouldn't
> be used.

## Three claims

**Claim 1 (primary, strong, actionable).** The three semantic-distance tests
in widest LLM use today (DAT, CDAT, PACE) function largely as
general-capability detectors when applied to creative-writing benchmarks --
strongest case: PACE's $r = 0.75$ with creative-writing benchmarks
collapses to $\le 0.20$ once Arena Overall + MMLU-Pro are residualised
out. **Implication:** do not use PACE or gated CDAT as creativity metrics,
and do not use Arena CW (which is ~98% capability) to validate creativity
tests; these are the constructions that have been moving into RL reward
signals.

**Claim 2 (diagnostic, strong).** What an SD test is good for depends on
the construct, and one of them is wrongly labelled. CDAT (ungated
novelty) is the only SD test viable for divergent thinking; DAT is the
only one viable for creative writing (and only on EQ-Bench / Mazur). On
**scientific ideation** (LiveIdeaBench, n=17, exploratory), DAT and CDAT
both show modest positive specificity (~0.27) -- the divergent-association
tests do appear to carry idea-quality signal that the creative-writing
benchmarks miss. CDAT's appropriateness facet (CDAT-A) is structurally
a convergent-thinking measure -- its sign flips on both diversity
benchmarks, exactly as a quality/coherence-favouring metric should.

**Claim 3 (constructive, hedged, novel).** A covariance-PSD bound
prescribes the maximum attainable specificity per benchmark. Existing SD
tests sit 0.3-0.6 below the ceiling on every benchmark except Arena CW
(which is ceiling-saturated by construction). New creativity tests for
LLMs should target this gap by *explicitly decoupling from capability* --
the bound makes "decouple from capability" a quantitative design objective,
not a vague aspiration.

## Gap the paper fills

A first-of-its-kind **construct-validity audit of SD creativity tests for
LLMs**: 52+ models x 3 embeddings x 6 benchmarks (3 creative writing, 2
divergent thinking, 1 scientific ideation), capability residualised on a
2-proxy stack, plus a covariance-PSD bound on attainable specificity. The
combination doesn't exist anywhere in the LLM literature, and is the
natural extension of the human-psychometric fluency-confound tradition
(Forthmann/Silvia/Benedek/Beaty) to machines.

## Novelty positioning -- distinguish from

- **Chen 2023 / Bellemare-Pepin 2024 / Wang 2025 (Nature Human Behaviour)
  / "Has creativity peaked?" 2025** -- administer DAT to LLMs, never
  residualise on capability.
- **PACE (Qiu 2025)** -- claims discriminant validity by being more
  correlated with Arena CW (0.74) than Arena Overall (0.66) or MMLU-Pro
  (0.51); we show this is misleading because Arena CW is itself ~98%
  Arena Overall.
- **CDAT (Nakajima 2026)** -- shows random baselines beat SOTA LLMs on
  raw DAT (a confound argument in the opposite direction); does not run
  capability residualisation.
- **Ilic & Gignac 2024 LLM g-factor** -- documents one g-factor across
  591 LLMs accounting for ~85% of variance; we show this g pollutes
  *specifically* the SD creativity tests.
- **LiveIdeaBench (Ruan et al. 2024, arXiv 2412.17596)** --
  qualitatively observes scientific ideation can dissociate from g
  (QwQ-32B-preview beats larger frontier models on Originality). We use
  it as a 6th benchmark (n=17, exploratory) and find that DAT and CDAT
  both carry modest creativity-specific signal there, in contrast to
  the creative-writing pattern. So far the only benchmark on which a
  semantic-distance test looks like a *creativity*-specific predictor.
- **"Measuring what Matters" (arXiv 2511.04703, 2025)** -- argues
  construct-validity failure is systemic across 445 benchmarks; we
  instantiate the argument for SD creativity tests at scale.
- **Forthmann et al. 2019-2025** (psychometric fluency confound) /
  **Karwowski et al. 2017** (threshold-theory reappraisal) /
  **Acar 2021 / Rossiter 2020** (forward-flow validity critique in
  humans) -- our LLM residualisation is the machine analogue of their
  partial-correlation moves on humans.

## RL-reward stakes

The construct-validity question is **no longer purely a measurement
concern**: SD-style scores are now propagating into LLM training:

- **Creative Preference Optimization (CrPO)** -- Ismayilzada et al.,
  EMNLP Findings 2025 (arXiv 2505.14442). Injects creativity-dimension
  signals (DAT/AUT-style) into DPO via the MuCE dataset.
- **Flow of Reasoning** -- arXiv 2406.05673. Uses divergence as a
  fine-tuning objective.
- **Jointly Reinforcing Diversity and Quality in LM Generations** --
  arXiv 2509.02534.
- **Evaluating the Diversity and Quality of LLM Generated Content** --
  arXiv 2504.12522.

## Concrete paper-level changes to implement the framing

1. **Abstract.** Replace `[Explain 3 main findings succinctly]` with the
   three claims above. Mention the RL-reward downstream concern. Close on
   the bound as a roadmap.

2. **Intro -- new para on training stakes.** SD scores propagating into
   training raises the stakes (CrPO; Flow of Reasoning).

3. **Intro -- novelty positioning para.** Distinguish from the prior work
   listed in "Novelty positioning" above.

4. **Background.** Add anchoring to:
   - Forthmann 2019/2025 on equal-odds / fluency confound
   - Karwowski reappraisal of threshold theory
   - Acar 2021 / Rossiter 2020 on forward-flow critique
   - Ilic & Gignac 2024 on LLM g-factor
   - "Measuring what Matters" (arXiv 2511.04703) on construct-validity
     failure across LLM benchmarks

5. **Discussion.** Frame the headroom argument as *the first quantitative
   target the field has had for new SD-test design, comparable to
   Forthmann's equal-odds correction in the human literature.* Cite CrPO
   and note the bound also gives a target for any creativity reward
   signal.

6. **Limitations.** Honest about Mazur ($n=21$) and NovBench ($n=11$);
   the bound holds in the population but the sampled $R$ for NovBench is
   noisy. Honest about the partial-vs-full residualisation choice.

7. **Contributions list.** Three bullets mirroring the three claims.

## Key citations to add (not yet in main.bib)

- **CrPO** (Ismayilzada et al. 2025, arXiv 2505.14442)
- **Flow of Reasoning** (arXiv 2406.05673)
- **Ilic & Gignac 2024** -- "Evidence of interrelated cognitive-like
  capabilities in LLMs" (arXiv 2310.11616)
- **Burnell et al. 2023** -- "Revealing the structure of language model
  capabilities"
- **LiveIdeaBench** (Nature Communications 2026, Ruan et al.)
- **CreativityPrism** (arXiv 2510.20091)
- **Measuring what Matters: Construct Validity in LLM Benchmarks**
  (arXiv 2511.04703)
- **Forthmann et al. 2019/2025** on the equal-odds / fluency confound
- **Karwowski et al. 2017** -- threshold-hypothesis reappraisal
- **Acar et al. 2021 / Rossiter 2020** on forward-flow validity in humans
- **Organisciak et al. 2023** -- "Beyond semantic distance" (LLM-judged
  scoring outperforms cosine on AUT)
- **Artificial Hivemind / NeurIPS 2025 Best Paper** -- Wegner et al.,
  arXiv 2510.22954
- **Creative Short Story Generation in Humans and LLMs**
  (arXiv 2411.02316)

## Status of the implementation

- Headline figure (fig:headline) is now 3 panels (CW / DT / SI).
- Specificity-ceilings figure (fig:spec-ceilings) is now 6 panels.
- Table 1 highlights the best test per benchmark via green cells;
  Overall block sits at the top; LiveIdeaBench column added.
- Appendix A.2 contains the bound proof.
- CDAT-NxA was removed (2026-04-25) for clarity.
- Section 2.2 has qualitative descriptions of all six benchmarks in
  paragraph form (CW / Output Diversity / Scientific Ideation
  subsubsections). Equations only for Hivemind and NovBench.
- Section 3.3 now discusses idiosyncrasies of the three embedding
  models (static vs contextual; subword info; each test's "home"
  embedding).

Next steps before submission: implement the abstract / intro /
discussion edits below.
