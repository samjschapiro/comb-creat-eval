# NeurIPS 2026 camera-ready: captions, rebuttal writing, greedy algorithm, appendix layout

*Session 2026-11-03 · dat_eval track · paper repo `papers/iccc-2026` (Overleaf)*

## Summary

This session closed most of the rebuttal writing items for the NeurIPS camera-ready.
- Every stale caption is fixed.
- Rebuttal items #10, #11, #12, #15 and #20 are done.
- Results now stress the output-diversity null, and the Kombine paragraph is shorter.
- The Discussion gains a paragraph on mechanistic interpretability.
- CreativityNeuro and Spark are cited.
- The appendix is laid out one section per page.

Every push was approved by the author and placed on top of the fresh Overleaf head. The main text now runs about half a page onto page 11, and something has to be cut.

## Tasks completed

**Pulls and author edits.** We pulled four author edits from Overleaf:
- `23183dd` (Oct 25): removed the stale "graph pathfinding" Schapiro2025 sentence and its bib entry.
- `04a0d72`: reworded the abstract and un-commented the acknowledgements.
- `52e1edc`: an intro caption tweak.
- `97eb392`: "To start, before turning…" in §2.

**Captions (pushed `7af57fe`):**
- Figure 3 (examples): adds the RAT.
- Correlations figure:
  - MMLU-Pro only, with "output diversity" in place of "divergent thinking".
  - Kombine and the RAT added.
  - n = 6–40 for benchmarks and 64–68 for tests; values shown only where p < .05.
- Headline figure:
  - Removed "R in the title".
  - The gold star means positive and significant on both axes.
  - The diamond marks the maximum of v + s.
- Table 1:
  - No green cells.
  - The RAT appears in the Overall block only.
  - Specificity n is smaller because some models lack MMLU-Pro.
  - The CDAT gate wording from item #19 is in.
- Appendix per-model tables: RAT and Kombine columns named; Arena Overall removed.

**#12.** "Creativity test(s)" is gone from the NeurIPS text. It is now "semantic distance tests", or just "test" where the claim is general (the bound, its intuition, a Discussion heading).

**#15, greedy algorithm (pushed `7af57fe`):**
- Moved from a footnote into the §2 text: it "substantially exceed[s] the mean human and LLM scores (94.1 vs. 78.4 and 83.75)".
- The algorithm (ruled style) and the plot sit side by side at the top of page 4.
- The correlations figure moved to page 5.
- The appendix keeps the details: vocabulary, 120 seeds, standard deviations.

**#10 and #11 (pushed `6ce7098`):**
- §2 opens with Definition 1: creativity is the ability to produce artifacts that are novel and appropriate (Runco & Jaeger 2012).
- It separates a test score from creative achievement.
- Scope: the product, not the process (Rhodes 1961); little-c to Pro-c, not Big-C (Kaufman & Beghetto 2009).
- All three references were added to the bib.

**Results (pushed `6ce7098`):**
- **Output diversity:** the heading is now "No test is a significant predictor of output diversity". It names the DAT null (Hivemind r = +.33 with specificity +.01; NoveltyBench +.07 with +.10). One highly divergent DAT response, whether memorized or algorithmic, is not the same as continually generating distinct responses.
- **Kombine:** cut to two sentences. PACE is the best predictor (r = +.70\*\*\*) but has no significant specificity (+.22). Kombine mostly needs general capability (r = +.61 with MMLU-Pro).

**Discussion (pushed `6ce7098`, `282b533`):**
- New paragraph, "Localizing the algorithms behind test performance", adapted from the preprint's Future Work (Sharkey 2025; Nanda 2023).
- CreativityNeuro (COLM 2026, arXiv 2607.01433) is cited there and in the §2 list of ways to raise test scores.
- Spark (arXiv 2504.20090) is cited for the utility of scientific ideas ("hold up under expert-style review").

**Checks:**
- **#17:** the text never says PACE "requires" T = 0. §4 says it "follows the original setup", which already matches the promise.
- **#20:** the "non-positive" claim is not in the text. CDAT-A specificity is +.37\* (Arena CW), +.18, +.04, +.35 (Kombine), +.14 (LiveIdea), −.38 (Hivemind), −.35 (NoveltyBench).

**Appendix (pushed `97fcfbf`):**
- Each section starts on a new page (`\clearpage`).
- Tables 1 and 2 use `[!h]`, so each sits under its own heading.
- Appendix B gets a short paragraph; it had only ever held the table.

**Figure 5 (pushed `04b395f`; generator `855fa8a`):** cell values are bold at 22 pt with no leading zero (+.87).

## Files

- **Paper (`papers/iccc-2026`, all pushed; Overleaf head `04b395f`):**
  - `main_neurips.tex` (adds a `definition` environment)
  - `bib/main.bib`
  - `sections_neurips/{01,02,04,05,06,08}*.tex`
  - `tables_neurips/{main_result,sd_test_examples,per_test_scores,per_benchmark_scores}.tex`
  - `figures/neurips/fig_benchmark_correlations.pdf`
- **Code:** `src/dat_eval/scripts/neurips_camera_ready.py` (`_sig_triangle` takes `cell_fs`, bold, no leading zero).
- **Docs:** camera-ready plan status (`docs/reports/2026-10-03_neurips_camera_ready/report.md`), the dat_eval progress entry, research context.
- **Local build:** `multirow` and `algorithms` built from CTAN in the session scratchpad. Preview PDF at `scratch/neurips_camera_ready/main_neurips_captions_greedy.pdf`.

## Key decisions

- **Greedy algorithm in the main text** as an algorithm and plot side by side. The plot caption is short; the numbers live in the text and the appendix.
- **Correlation values drop the leading zero** in Figure 5, matching Table 1.
- **The definition and scope go at the top of §2**, as rebuttal item #10 asked.
- **A reference to Algorithm 1** links the output-diversity result and the interpretability paragraph.

## Open / next steps

- **Page limit:** the main text runs about half a page onto page 11 (Discussion end, Limitations, Conclusion). The author decides the cuts.
- **#13–14:**
  - Soften "meaningful room for the design of improved tests" in the abstract and results.
  - Add Future Work uses: training checkpoints, and a proxy where human raters are unavailable.
- **#21:**
  - Construct-local evidence: CW benchmarks inter-correlate at .70–.85; LiveIdeaBench correlates with nothing; no claims about analogy.
  - Base-model direction per benchmark; cite Springer 2026, Li 2026 and Li 2025.
- **#9:** Labyrinth of Links and MM-OPERA in related work.
- **Bib:** Bellemare-Pepin → *Sci. Reports* 2026; Wang → *Nat. Hum. Behav.* 2026, 10(3):531–540; add Mednick, Springer and the two Li papers.
- **Check #16, #18 and #19** in the main text: convergent/divergent ties, why PACE counts, and CDAT-N/CDAT-A definitions in §4.
- **Greedy plot:** the legend (55 models, n = 2082) disagrees with the text (54, n = 2078). One place says GloVe-only, another says averaged over three embeddings.
- **Overview figure** step 1 says "Creativity Test".
- **Submission:** checklist, OpenReview abstract, final PDF.
