# NeurIPS 2026 (Submission 13188) — Rebuttal drafts

Status: drafting. Scope reminder — **DRAT/RAT are out of scope for this submission.**
Do not reference them in any response.

---

## F63x — "Why not simply use the creative achievement benchmarks?"

> If the ultimate goal is to design effective tests to evaluate machine creativity, why not
> simply use the benchmarks of creative achievement employed here? ... especially considering
> that these tests are not even good predictors of human creativity.

### Draft response

Thank you — this is the right question to ask of the paper, and we agree the motivation
deserves to be stated explicitly rather than assumed. We will add it to §1.

**Tests and benchmarks are not competing instruments for the same job.** A benchmark is the
*criterion*: the outcome we actually care about. A test is a cheap proxy administered when the
criterion is unavailable, too slow, or too expensive to observe. This is exactly why
psychometric testing exists for humans: the criterion (creative achievement) manifests
longitudinally over a career and cannot be observed at the moment a decision is made, so the
field validates short instruments against it. Our paper does not assume the proxies work — it
asks whether they do, and reports that for scientific ideation they do not.

**Cost, measured.** Administering the full DAT + CDAT + PACE battery to one model takes ~450
short API calls producing ~90K generated tokens, runs in minutes, and is scored on CPU with
public embedding models. Scoring one model on NoveltyBench requires 1,100 prompts (100
curated + 1,000 WildChat) × 10 generations = **11,000 long-form responses (~5.3M generated
tokens, ~60× the text)**, then a fine-tuned DeBERTa partitioner and a 27B-parameter reward
model (Skywork-Reward-Gemma-2-27B) over every response, which needs a ≥48 GB GPU in bf16.
On the same non-frontier open models, generation alone cost ~$0.9/model for NoveltyBench
versus ~$0.015/model for the full test battery — before any GPU scoring, which the tests do
not need at all. Arena Creative Writing additionally requires human raters.

**Availability, not only cost.** Arena CW cannot be administered at all to a model that has
not been publicly deployed and accumulated community votes. The same holds for every
leaderboard-derived benchmark. This rules the criterion out precisely where a creativity
measure would be most useful: during training, per checkpoint, per decoding configuration,
per ablation, or on an internal model. A test that runs in minutes on any endpoint is usable
in those settings; a benchmark is not.

**Benchmark scores are also not portable across pipelines — we verified this first-hand.**
In preparing this response we attempted to self-administer NoveltyBench to enlarge n. Matching
the published scale required resolving three separate pipeline issues (special-token pollution
in the reference decode path; 8-bit quantization pushing the reward model outside its
calibration band; and the prompt set being the 1,100-prompt union under pure temperature-1
sampling rather than curated-only under provider default top-p/top-k). Only after all three
did we land within 1.8% of the paper's reported utility. A semantic-distance test, by
contrast, is a fixed prompt plus a public embedding model, and reproduces exactly. We are
happy to add this as an appendix note; it is concrete evidence for why the criterion is not a
drop-in measurement tool.

**On "not even good predictors of human creativity."** We agree, and we say so in §2.1 — the
DAT correlates with the AUT at only r ≈ .32–.51, and AUT scores with creative achievement at
r ≈ .17–.22. That is the premise of the paper, not a counterargument to it. These tests are
already being used to make direct claims that LLMs are more or less creative than humans; the
appropriate response is to measure whether they support such claims, not to assume they do.
Our specificity criterion is precisely the tool for deciding when a cheap proxy is trustworthy,
and our answer for scientific ideation is negative — a result the field needs before it relies
on these tests further.

**Revision.** We will add a short paragraph to §1 making the proxy-vs-criterion distinction
and the cost/availability argument explicit, with the per-model cost figures above, and add
the reproduction account as an appendix note.

---

## F63x — Creativity vs creative achievement; can machines be creative?

> the authors seem to equal creativity with creative achievement, which some creativity
> researchers may not agree with. Moreover, it is not uncommon for some researchers working on
> human creativity to take the view that machines cannot be creative as they are not able to
> engage in creative processes.

### Draft response

We appreciate the reviewer raising this, and we agree the paper should state its commitments
explicitly rather than leave them implicit.

**We do not equate the two, and we will make the distinction explicit.** Our intended
structure is the standard psychometric one: creativity is the latent construct (output that is
both novel and appropriate, following Boden and Maher); the benchmarks are *criterion
measures* — observable outcomes used to establish criterion-related validity. The criterion is
never the construct. This is the same relation as in human work, where the DAT is validated
against the AUT and against measures of achievement, and no one takes AUT scores to *be*
creativity. Our claim is about the predictive relation between a test and a criterion, not
about the identity of either with the construct.

**Terminology fix.** "Creative achievement" carries a specific meaning in the human
literature (lifetime accomplishment, e.g. as indexed by the Creative Achievement Questionnaire)
that does not transfer to LLMs. We will define each term at first use and, where the human
sense would mislead, relabel to *creative task performance* / *creative output quality*, which
is what our six benchmarks actually measure.

**Process versus product.** Following the standard four-facet framing (person, process, press,
product), our study lies entirely in the *product* facet. We make no claim that LLMs engage in
a creative process, possess creative intent, or are creative agents. Every claim in the paper
is behavioral and can be restated without commitment on that question: *test score X predicts
(or fails to predict) output-level quality Y in this population, beyond what general
capability g already predicts.* We will add this scope statement to §2 and to the limitations.

**Our results are compatible with — and arguably useful to — the process-skeptic view.** A
researcher who holds that machines cannot be creative because they cannot engage in creative
processes should be particularly interested in evidence that these tests do not measure what
they are taken to measure. We already argue in §2.1 that comparative claims of the form "LLMs
are more/less creative than humans" presuppose that the tests measure the same construct in
both populations, and that this measurement invariance has not been established. The
skeptic's position sharpens that argument rather than conflicting with it.

**Revision.** We will add a short "Definitions and scope" paragraph to §2 fixing: creativity
(construct), creative task performance (criterion), creativity test (proxy instrument),
validity, and specificity; plus the product-facet scope statement and a limitations sentence
acknowledging that the paper is neutral on whether LLMs engage in creative processes.

---

## F63x — "No single test predicts all constructs well" is unsurprising

> The results suggest that "no single test predicts all constructs well", which is not
> surprising. ... We don't really need all the results presented here to know that, as there is
> simply no general-purpose creativity measure. It would be great if the authors could relate
> their findings to creativity theories and discuss the implications in practice.

### Draft response

We take the reviewer's point that domain specificity is well established in creativity theory,
and we agree the paper should engage with that literature directly. We would push back,
respectfully, on the inference that the results are therefore unnecessary.

**The applied LLM literature proceeds as though a general-purpose measure does exist.** The
reviewer's premise is standard among creativity researchers, but it is not the operating
assumption of the work that actually administers these tests to models. The practice takes two
forms, neither of which establishes the predictive relation it presupposes.

*Form 1 — a single semantic-distance test, a conclusion about creativity.* Chen & Ding (2023)
administer only the DAT, describing it as "an objective measurement of creativity," and
conclude that "advanced large language models have divergent semantic associations, which is a
fundamental process underlying creativity," reporting that "GPT-4 outperforms 96% of humans,
while GPT-3.5-turbo exceeds the average human level" [`Chen2023ProbingAssociation`]. Wang et
al. (2026, *Nature Human Behaviour*) likewise administer a single task — the DAT — to 9,198
humans and 215,542 model observations, and state the result at the level of the construct:
"human creativity on average is slightly higher than that of LLMs" [`wang2025large`]. In both
cases a human-percentile claim about *creativity* rests on one instrument whose validity in
this population has never been estimated.

*Form 2 — parallel measurement with no validity link.* Bellemare-Pepin et al. measure both the
DAT and creative writing under "identical, objective scoring," and report that "LLMs can
surpass average human performance on the Divergent Association Task, and approach human
creative writing abilities" [`Bellemare-Pepin2024DivergentLLMs`]. This is the most careful
design in the group, and it still leaves the key quantity unmeasured: whether DAT scores
*predict* the creative-writing scores. The two are reported side by side as facets of
"divergent creativity," which presupposes exactly the predictive relation that was never
estimated — and that our results show does not hold uniformly across constructs.

We want to be clear that this is not a charge of carelessness. Chen & Ding explicitly flag the
concern in their limitations, noting that "measuring creativity is also controversial [and]
requires evaluations from multiple perspectives." But flagging the uncertainty is not resolving
it, and until now there has been no evidence base against which to resolve it. Supplying that
evidence base — which test predicts which construct, and whether it does so independently of
general capability — is what this paper is for.

**Note (do not cite Cropley as an example here).** `cropley2023artificial` argues *our* side:
the abstract states that "while both forms of ChatGPT show a capacity for verbal divergent
production that exceeds human means, a range of factors call into question the 'creativity' of
generative AI." His 2025 follow-up is titled "Why Generative AI Has Limited Creativity."
Citing him as an over-generalizer would be a factual error, and F63x is confidence 5.
He is usable as *support* for the paper's motivation instead.

**The direction is unsurprising; the content does not follow from it.** "No test is general"
does not tell you *which* test predicts *which* construct, and none of our main results is
derivable from it:

1. The DAT is the best predictor of creative writing while the CDAT is the best predictor of
   divergent thinking. The ordering is an empirical fact, not a corollary.
2. PACE has high validity on creative writing (r ≈ 0.73) that collapses to non-significant
   specificity (r|g ≈ 0.15). This is a *different* failure mode from lack of generality: a test
   can be the strongest raw predictor and still be a capability proxy. Nothing in the domain
   specificity position predicts this, and it cannot be detected without the specificity
   criterion.
3. No existing test reliably predicts scientific ideation — a null across all tests, not a
   pattern of construct-by-construct specialization.
4. The validity-specificity frontier separates shortfall that is *structural* from shortfall
   that is *remediable*. Arena CW is capability-loaded at R = 0.98, which caps specificity at
   ≈ 0.20 for a hypothetical perfect-validity test; no test design can beat that. Knowing which
   benchmarks admit headroom is actionable and is not implied by "creativity is domain
   specific."

**The scientific-ideation null runs against the theory that motivates these tests, not with
it.** The associative tradition invoked to justify semantic-distance measures — Mednick's
associative basis of the creative process, Koestler's bisociation, Thagard on creative
combination — is grounded substantially in accounts of *scientific* discovery. If distant
association is the mechanism underlying insight, semantic-distance tests ought to predict
scientific ideation at least as well as creative writing. We find the inversion: they predict
creative writing, and scientific ideation not at all. We think this is genuinely surprising on
the field's own theoretical commitments, and we will make that argument explicitly rather than
leaving the null to speak for itself.

**Relation to creativity theory (as requested).** Our results are the machine-population
analogue of the domain-specificity position in human creativity research — Baer's argument that
divergent thinking tests mislead precisely because creative ability does not transfer across
domains [`baer2011ttct`] predicts the construct-by-construct pattern we observe. We will add a
paragraph making this connection. It comes with a machine-specific wrinkle worth stating: for
LLMs the dominant confound is not only domain but *general capability*, and at a magnitude with
no human analogue. Divergent thinking correlates with intelligence at r ≈ 0.37 in humans
[`gerwig2021relationship`], whereas our creative-achievement benchmarks correlate with general
capability up to r = 0.98. Domain specificity alone therefore under-describes the machine case,
which is why specificity rather than validity is the load-bearing criterion here.

**Implications in practice (as requested).** We will add these as explicit takeaways:
(i) a test score should be reported against a named construct, not as "the creativity of model
M"; (ii) specificity should be reported alongside validity, since high validity is compatible
with zero creativity-specific signal; (iii) no current semantic-distance test should be used as
a proxy for scientific ideation ability; (iv) tests should be validated against benchmarks with
low capability loading, since capability-saturated benchmarks cap attainable specificity
regardless of test design.

---

### Citations to add (NEED HUMAN VERIFICATION before use)

Not currently in `bib/main.bib`. All are real and standard, but verify details:

- Rhodes, M. (1961). An Analysis of Creativity. *Phi Delta Kappan*, 42(7), 305–310. — four-facet framing
- Runco, M. A., & Jaeger, G. J. (2012). The Standard Definition of Creativity. *Creativity Research Journal*, 24(1), 92–96. — novelty + appropriateness
- Amabile, T. M. (1982). Social psychology of creativity: A consensual assessment technique. *JPSP*, 43(5), 997–1013. — product-facet assessment
- Carson, S. H., Peterson, J. B., & Higgins, D. M. (2005). Reliability, validity, and factor structure of the Creative Achievement Questionnaire. *Creativity Research Journal*, 17(1), 37–50. — the "creative achievement" term we are disambiguating from

Already in bib and usable here: `Boden2004TheMechanisms`, `Maher2010EvaluatingSystems`,
`Simonton2004CreativityZeitgeist`, `Dietrich2004TheCreativity`, `wallas1926art` (process side).

---

### Source check on the four "general-purpose use" citations (verified 2026-07-27)

| cite key | what it actually does | usable as an example? |
|---|---|---|
| `Chen2023ProbingAssociation` | DAT **only**; "an objective measurement of creativity"; "GPT-4 outperforms 96% of humans"; concludes LLMs have associations that are "a fundamental process underlying creativity." Has a limitations note conceding creativity measurement "requires evaluations from multiple perspectives." | **Yes — strongest.** arXiv:2310.11158 |
| `wang2025large` | DAT (single task; abstract says only "an established creativity task"); N=9,198 humans, 215,542 model obs; concludes "human creativity on average is slightly higher than that of LLMs." | **Yes.** The construct claim floats free of the unnamed instrument. |
| `Bellemare-Pepin2024DivergentLLMs` | DAT **and** creative writing, identical objective scoring; reports both as facets of "divergent creativity"; never tests whether DAT predicts the writing scores. | **Yes, as the "parallel measurement, no validity link" case.** Do not call it a single-test overreach — it isn't. |
| `cropley2023artificial` | Abstract: DAT scores exceed human means, but "a range of factors call into question the 'creativity' of generative AI." 2025 follow-up: "Why Generative AI Has Limited Creativity." | **No — he agrees with us.** Cite as support, not as an example of the problem. |

Optional fourth example, **not currently in `main.bib`** (would need an entry): Hubert, Awa &
Zabelina (2024), *Scientific Reports* 14:3440, "The current state of artificial intelligence
generative language models is more creative than humans on divergent thinking tasks" — AUT +
Consequences + DAT, concluding AI "was robustly more creative along each divergent thinking
measurement." The university press release generalized this to "AI Outperforms Humans in
Standardized Tests of Creative Potential," which is itself the phenomenon we are describing.

**Bib hygiene — two stale entries a confidence-5 reviewer could notice:**
- `Bellemare-Pepin2024DivergentLLMs` is a `@techreport` in our bib; it is now published in
  *Scientific Reports* (2026), doi 10.1038/s41598-025-25157-3.
- `wang2025large` is listed as year 2025, pages 1–10 (advance online); the version of record is
  *Nature Human Behaviour* 10(3):531–540, 2026.

### Provenance of the cost figures

Computed 2026-07-27 from local data, not estimated:

- Test battery: `data/dat_eval/run_v1/*/` — mean over 75 model dirs of DAT/CDAT/PACE
  `raw_response` fields: ~447 calls/model, ~358 KB generated text/model (~90K tokens at 4 chars/token).
- NoveltyBench: `data/new_tests/noveltybench_skywork/gen/{curated,wildchat}/phi-4/generations.jsonl`
  — 1,000 WildChat prompts × 10 generations = 20.4 MB, plus 100 curated × 10 = 0.89 MB;
  11,000 responses, ~21.3 MB (~5.3M tokens).
- Dollar figures: ~$8 OpenRouter for 9 non-frontier open models' NoveltyBench generation
  (`docs/logs/2026-07-24/nb_selfrun_validated_expansion.md`), scaled by the token ratio for the
  battery.
- Pipeline-reproduction account: same log (three fixes, final utility 3.69 vs paper 3.76, −1.8%).
