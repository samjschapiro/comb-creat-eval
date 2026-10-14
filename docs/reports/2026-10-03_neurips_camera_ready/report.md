# NeurIPS 2026 camera-ready: what the rebuttal committed us to

*2026-10-03 · dat_eval track · camera-ready plan for Submission 13188*

**Question.** The paper "Do Semantic Distance Tests *Actually* Predict Creativity in Large Language Models?" was accepted to NeurIPS 2026 as a poster (decision 24 Sept 2026). What must change between the submitted version and the camera-ready to honour what we told the reviewers and the AC?

**Answer in one paragraph.** The Overleaf NeurIPS files have been reset to the submitted version, without the DRAT. We owe 21 changes plus bib fixes. Six need new analysis or new runs; the rest are writing. The biggest is switching specificity to control for MMLU-Pro alone, because it changes every specificity number and flips some conclusions. Second is the CDAT gate bug fix, which shifted the whole CDAT row of Table 1 after submission. The RAT goes into this paper as a fourth test, and only the DRAT is out of scope. One item needs a decision before the writing starts: how to word the training-checkpoint use case without contradicting the promise to drop "these tests are necessary."

## Where the paper stands

- **Overleaf repo:** `papers/iccc-2026` (remote `git.overleaf.com/69dc0aaf7d0d6aa7082a44af`). The NeurIPS files are `main_neurips.tex`, `sections_neurips/`, `tables_neurips/`.
- **Submitted version:** commit `cac1df9` (2026-05-07 11:48 UTC). Its abstract matches the OpenReview abstract, and every DRAT edit came after it.
- **Reset pushed today:**
  - `da43097` restores the submitted sections and tables, with the DRAT section and tables removed.
  - `a64ef62` brings `main_neurips.tex` up to camera-ready: `styles/` and `bib/` paths, `final` option, new email.
  - `e1ccdff` changes the affiliations to numbers (¹ AI Innovation Institute, Stony Brook University; ² Knowledge Lab, UChicago; ³ UIUC). It also adds the `main` track option, which the camera-ready footer needs.
- **Reusable text:** the June version (`df90b61`, synced from ICML) already has the RAT section and the related-work paragraph on association benchmarks. Lift those parts without the DRAT.

## Status (updated 2026-10-03, later)

Done locally (not yet pushed), details in [the recompute report](../2026-10-03_neurips_mmlu_specificity_and_kombine/report.md):
- **Item 1:** specificity is MMLU-Pro only in Table 1, both figures, the §4 definition and the frontier example.
- **Item 2:** the CDAT row is recomputed on the gate-fixed data.
- **Item 7:** Figure 3 now prints only significant correlations, with stars.
- **Item 8:** the RAT is now a row in Table 1 and in both figures.
- **New:** Kombine is added as a fourth construct (combinatorial creativity) to Table 1 and both figures.

**2026-10-14 (local, not yet pushed):**
- §5 results prose rewritten for the recomputed numbers, including the promised power sentence for scientific ideation (n = 20–22, |r| ≥ .56–.58).
- Intro contributions and conclusion updated to four constructs, with "divergent thinking" renamed "output diversity" to match the author's abstract.
- Kombine benchmark paragraph added to §3, citing arXiv 2509.21043 (BibTeX fetched from arXiv).
- Stale LiveIdeaBench correlations and the Hivemind typo fixed.
- Table 1 and both figures reordered: creative writing, output diversity, combinatorial creativity, scientific ideation.

Still open from these: the §5 results prose and contribution 3 (several claims flipped), the captions, and a Kombine description in §3.

## Data & sampling

- **Source of the commitments:** the OpenReview thread for Submission 13188, saved as a PDF on 2026-10-03. Only text actually posted counts as a commitment. Drafts that were never posted (`docs/tracks/dat_eval/rebuttal_neurips_2026.md`) are listed separately as optional.
- **Reviewers:** rEB8 (5, accept, confidence 2), F63x (2, reject, confidence 5; lowered after discussion), 6Cbj (5, accept, confidence 3; raised after rebuttal and asked that "all the promised modifications are prominently integrated"). The AC recommended accept.
- **Model sample:** 54 instruction-tuned LLMs from 10 providers, a convenience sample of what was reachable by API. Per-benchmark n is smaller because each benchmark covers only some models. The replication unit for every correlation is the model.
- **Per-benchmark n as quoted in the rebuttal:** Arena CW 39–40, EQ-Bench CW 24–27, Mazur CW 17–18, Hivemind 21, NoveltyBench 23 (was 10), LiveIdeaBench 20–21.
- **What the numbers below are.** They are copied from the posted rebuttal and the track log, not recomputed for this report. Each one must be regenerated from the scripts before it goes into the paper. Most are not significant at these n.

## Changes owed

### A. Needs new analysis or runs

| # | change | promised to | notes |
|---|---|---|---|
| 1 | **Compute specificity controlling for MMLU-Pro only.** Arena Overall shares a method with Arena CW, so drop it from the control set. | F63x (summary §7.1: "plan to stick with this moving forward") | Every specificity number and the frontier's R change. Some conclusions flip: DAT on Arena CW goes from +0.05 to +0.38* (n=40); PACE on Arena CW from +0.11 to +0.41* (n=39). Do this first. |
| 2 | **Update the CDAT row of Table 1** for the BH NaN bug fix in `cdat_gate.py`. | F63x, 6Cbj (implicitly) | The bug made gated CDAT effectively a T=1.5-only measure ([log](../../logs/2026-07-27/cdat_gate_bh_nan_bug_fix.md)). The submitted CDAT numbers are stale everywhere, not just on NoveltyBench. The camera-ready should credit the NoveltyBench drop (+0.60 → +0.03) to both the bug fix and the larger n; the rebuttal mentioned only the n. |
| 3 | **Report DAT and CDAT at each temperature** (T = 1.0, 1.5, 2.0) instead of averaging. Add the correlations between temperatures (r = .50–.88) and list each benchmark's temperature. | F63x | The posted reply chose reporting all three temperatures over picking the best one. |
| 4 | **Run DAT and CDAT at T = 0.7** and add the results to the appendix. | F63x ("happy to include") | Needs new runs; estimate the OpenRouter cost first. |
| 5 | **NoveltyBench at n = 23.** Update all numbers. Change "CDAT is the best predictor of divergent thinking" in the abstract and §5 to a ranking that says it is not significant. | F63x, 6Cbj | Correct CDAT specificity at n=23 is **+0.03**. The +0.14 in the reply to 6Cbj was CDAT's validity, pasted by mistake. Also note that CDAT-N outranks CDAT on the two divergent-thinking benchmarks together (v+s 0.74 vs 0.50). |
| 6 | **Power analysis for every null.** LiveIdeaBench at n=21 can only detect \|r\| ≥ 0.56 (or \|r\|g\| ≥ 0.58). r = .40 needs n ≈ 48; r = .30 needs n ≈ 85. Replace the scientific-ideation claim with the sentence promised in the rebuttal (below). | 6Cbj, F63x, AC point 5 | NoveltyBench at n=23: 0.56 for validity, 0.58 with 2 controls. The sentence must use the final control set from item 1. |
| 7 | **Add significance stars to Figure 3** for the correlations between tests and between benchmarks. | F63x | The tables are in the reply of 29 Jul. |

The promised scientific-ideation sentence: "We find no evidence of a large or moderate, capability-independent association between any existing test and scientific ideation; at our current coverage (n=21) the analysis is sensitive only to |r| ≥ 0.56, so a small association cannot be excluded. However, the same tests detect large associations with creative writing at similar sample sizes, so the absence of significant correlations for scientific ideation indicates a weaker (if any) relationship, not merely low power."

### B. Can reuse text from the June version

| # | change | promised to | notes |
|---|---|---|---|
| 8 | **Add the RAT as a test** (30 Bowden & Jung-Beeman items) with its validity and specificity table. | 6Cbj, F63x, AC point 4 | **In scope** (author, 2026-10-03; only the DRAT is out). **Partly done (`2daa18f`):** intro now has the convergent/divergent framing and the RAT paragraph, and the RAT task definition is in Preliminaries, all copied from the preprint. Still to do: Figure 3 caption (the figure already shows the RAT), Table 1, results text, abstract and contributions. Take the RAT parts of the June text and leave out the DRAT. The RAT also has to go into every analysis the other tests get: MMLU-Pro-only specificity (item 1), Table 1, Figure 3 and the power analysis. Rebuttal numbers at n = 17–39: validity +0.65***, +0.55**, +0.50* on the creative-writing benchmarks with specificity near zero; −0.51* on Hivemind; −0.13 n.s. on LiveIdeaBench. |
| 9 | **Related work:** Mednick's RAT, Labyrinth of Links, MM-OPERA, Only Connect Wall, CresOWLve. | 6Cbj ("detailed commentary about all four") | Already written in `df90b61`. |

### C. Writing only

| # | change | promised to |
|---|---|---|
| 10 | **Labelled definition in §2:** "the creativity of a system is its ability to produce artifacts that are novel and appropriate." State the difference between a test score and creative achievement at first use. | F63x, rEB8, AC point 1 |
| 11 | **Scope:** claims are about the product, not the process (Rhodes 1961). Creative achievement as measured here covers Little-C to Pro-C at most (Kaufman & Beghetto 2009). | F63x, AC point 6 |
| 12 | **Terminology:** replace "creativity tests" with "semantic distance tests" throughout. | F63x |
| 13 | **Framing:** make no claim that the tests are necessary. The contribution is checking whether tests already in use predict creative achievement (Wang 2026; Chen & Ding 2023). Soften "headroom to design more effective creativity tests." | F63x |
| 14 | **Future Work:** state the uses we have in mind (measuring creativity at checkpoints during training; a proxy where human raters are unavailable). | F63x |
| 15 | **Move the greedy algorithm** from the appendix (Fig. 5 / Alg. 1) to the main body as motivation. Say it *substantially exceeds the mean* human and LLM scores (94.1 vs 78.4 / 83.75), not that it beats all humans. This also answers 6Cbj's point that the tests can be gamed. | F63x, 6Cbj |
| 16 | **Theory:** relate the findings to convergent and divergent thinking (Dietrich 2004). DAT and PACE mostly measure divergent thinking. | F63x, 6Cbj |
| 17 | **PACE temperature:** stop saying PACE "requires" T = 0. F63x showed its paper varies temperature from 0 to 1 when comparing with humans. Say instead that we follow the setting it uses to compare LLMs. | F63x (final justification) |
| 18 | **Why PACE counts:** it is a semantic distance test because of how it is scored (forward flow; Gray 2019, Beaty 2021), though it asks for words without telling the model to maximize distance. | F63x, AC |
| 19 | **CDAT explanation:** rewrite old lines 212–218. CDAT-N is novelty over all temperatures, and CDAT-A is raw appropriateness. CDAT is novelty averaged only over temperatures that pass the gate (Welch t-test against random nouns, BH-FDR across models, adjusted p < .001). Put the same text in the Table 1 caption. | F63x, AC point 3 |
| 20 | **Table 1:** make CDAT the green cell for Hivemind (v+s +.25/+.10 beats DAT's +.33/+.01). Define the best-test rule (highest v+s) in the main text. Fix "CDAT-A has non-positive specificity throughout"; it is positive on EQ-Bench CW (+.05) and LiveIdeaBench (+.17). | F63x |
| 21 | **Construct-local reading and limitations.** | 6Cbj Q1 and Q4, AC point 6 |
|    | – Creative writing benchmarks correlate with each other at r = .70–.85, and the two divergent-thinking benchmarks at .52. LiveIdeaBench correlates significantly with nothing, including both capability measures. Cite Baer, and say plainly that the findings do not extend to untested constructs such as analogy. | |
|    | – Limitations: instruction-tuned versus base models, and the direction we would expect for each benchmark. Cite Springer 2026, Li 2026 and Li 2025. | |

### D. Bibliography

- `Bellemare-Pepin2024DivergentLLMs` is now published: *Scientific Reports* (2026), doi 10.1038/s41598-025-25157-3.
- `wang2025large`: version of record is *Nature Human Behaviour* 10(3):531–540 (2026).
- Add Rhodes (1961), Kaufman & Beghetto (2009), Runco & Jaeger (2012), Gray et al. (2019), Beaty et al. (2021), Bowden & Jung-Beeman (2003), Mednick (1962/1968), and the five related-work papers. Also add Springer 2026, Li 2026 and Li 2025.
- Do not cite `cropley2023artificial` as an example of over-claiming from the DAT. He argues the same thing we do.

## Decisions for the author

1. **Use case vs "not necessary"** (items 13 and 14). Both were promised to F63x. One way to word it: the tests are in use, and *if* they are valid they would be cheap enough to run at training checkpoints.
2. **Main-text conclusions after item 1.** Once specificity controls only for MMLU-Pro, DAT gains significant specificity on all three creative-writing benchmarks. The abstract's findings (1)–(3) should be re-read against the new table before any prose is written.

## Optional, from drafts that were never posted

From `docs/tracks/dat_eval/rebuttal_neurips_2026.md`:
- Four practical takeaways:
  1. Report a test score against a named construct.
  2. Report specificity alongside validity.
  3. Don't use any current test as a proxy for scientific ideation.
  4. Validate tests against benchmarks that depend little on general capability.
- The point that LLM creative-writing benchmarks track general capability far more than divergent thinking tracks intelligence in humans (up to r = .98 vs r ≈ .37).

## Suggested order

Item 1, then 2, then 5 and 6 (they depend on the final numbers), then 3, 4 and 7. Item 8 (the RAT) goes through items 1, 5, 6 and 7 together with the other tests. After that come the writing items, then the bib. The writing should come last, because its claims depend on the new numbers.
