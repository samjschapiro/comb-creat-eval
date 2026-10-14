# NeurIPS 2026 camera-ready: DRAT-free reset, MMLU-Pro-only specificity, RAT + Kombine, layout

*Session 2026-10-03 → 2026-10-14 · dat_eval track (with kg_creat data) · paper repo `papers/iccc-2026` (Overleaf)*

## Summary

After the NeurIPS 2026 acceptance (poster, decision 24 Sept), the NeurIPS version of "Do Semantic Distance Tests *Actually* Predict Creativity in LLMs?" was reset to the submitted version without the DRAT. It was then brought most of the way to camera-ready:
- specificity now controls for MMLU-Pro only, as promised in the rebuttal;
- the RAT is added as a test and Kombine as a fourth construct (combinatorial creativity);
- every table and figure is regenerated from one script;
- the results prose is rewritten for the new numbers;
- the figures are relaid out.

All Overleaf pushes were patches on the fresh remote head, each summarized and approved first.

## Tasks completed

**Reset and header (Oct 3)**
- Pulled Overleaf. Identified the submitted version as `cac1df9` (abstract matches OpenReview) and restored `main_neurips.tex`, `sections_neurips/` and `tables_neurips/` to it, deleting the post-submission DRAT and reorganized sections.
- Camera-ready header: `final, main` options (`main` is needed or the footer's `\@trackname` is undefined), `styles/` and `bib/` paths, new email. Affiliations are numbered: AI Innovation Institute (Stony Brook), Knowledge Lab (UChicago), UIUC.

**Rebuttal audit (Oct 3)**
- Read the full OpenReview thread and compiled 21 promised changes plus bib fixes into [the camera-ready plan](../../reports/2026-10-03_neurips_camera_ready/report.md).
- Corrections found along the way:
  - The CDAT NoveltyBench specificity is +0.03; the +0.14 in the reply to 6Cbj was validity.
  - The CDAT gate bug fix (2026-07-27) changed the whole CDAT row.
  - Only the DRAT is out of scope; the RAT is in.

**RAT and abstract (Oct 3)**
- Added the convergent/divergent framing and a RAT paragraph to the intro, and a RAT task definition to §3, taken word for word from the preprint (`sections_jmlr/`).
- Abstract opening replaced with the preprint's, keeping "semantic distance tests". The author later reworded it on Overleaf for four constructs.

**Kombine as a benchmark (Oct 3)**
- `src/dat_eval/scripts/kombine_as_benchmark.py`: DAT/CDAT at T = 1.0, PACE and RAT scored on Kombine and its criteria.
- Finding: the overall score tracks creative writing and capability. CDAT is the only test that predicts Kombine *originality* beyond capability (+.64 to +.71, three cells survive BH over 88).
- See [log](../2026-10-03/kombine_as_semantic_distance_benchmark.md).

**MMLU-Pro-only recompute (Oct 3)**
- `src/dat_eval/scripts/neurips_camera_ready.py` generates the Table 1 body, the headline figure and Figure 4 (inter-benchmark/inter-test) from one set of cells (`data/dat_eval/neurips_camera_ready/cells.json`).
- Specificity uses k = 1 (MMLU-Pro). The RAT is a row and Kombine a construct. The data are current (NoveltyBench n = 23, gate-fixed CDAT).
- Five Kombine models get MMLU-Pro by exact name match from the TIGER-Lab file.
- Headline: DAT has significant specificity on all three creative-writing benchmarks; PACE keeps it on Arena CW and Mazur CW; the RAT is now the capability proxy; nothing has significant specificity elsewhere; PACE has significant validity on LiveIdeaBench.
- See [report](../../reports/2026-10-03_neurips_mmlu_specificity_and_kombine/report.md).
- Author decisions applied:
  - Table 1 moved to the appendix (`app:correlations`) with no best-test shading.
  - Figure 4 colours every cell but prints values only where p < .05, without stars.

**Writing for the new results (Oct 14)**
- §5 results rewritten in the author's register. Changes:
  - The RAT replaces PACE as the capability proxy.
  - The DAT is the only test with significant validity and specificity on all three creative-writing benchmarks.
  - New paragraphs on output diversity and combinatorial creativity.
  - Scientific ideation now carries the promised power sentence (n = 20–22, |r| ≥ .56–.58; exact noncentral-t power, which reproduces the rebuttal's .56/.58 at n = 21).
  - The frontier wording is softened to the abstract's.
- Intro contributions and conclusion updated to four constructs; "divergent thinking" as a construct is renamed "output diversity" to match the author's abstract. Generator, table and figures reordered to match: creative writing, output diversity, combinatorial creativity, scientific ideation.
- Kombine benchmark paragraph added to §3. Kombine is now arXiv 2509.21043 v6, and arXiv's BibTeX was added with `howpublished`, which the `plain` style needs.
- All capability correlations in §3, §4 and the intro are reported against MMLU-Pro only (Arena Overall removed). NoveltyBench is no longer "most capability-independent" (−.32 vs LiveIdeaBench .18). Arena Ovr removed from Figure 4.
- Fixed stale LiveIdeaBench correlations and the `$r - 0.59$` typo.

**Figures and layout (Oct 14)**
- Divergent–convergent Venn diagram (author's image): first half width in §3.1, then moved to a half-width wrap in the intro on page 2 (Figure 2). Caption written at the author's request.
- New overview figure (author's image) at 0.8 width above the abstract, via `\captionof`; removed from the intro wrap.
- Example prompts (Figure 3): RAT added (real prompt; "ice" for cream/skate/water, the common model answer). Full test names, stacked rows, full width at the top of page 3, response boxes tinted with each test's figure colour.
- The full paper now builds locally: `multirow` built from CTAN source in the scratchpad (TeX install untouched), `styles/` on `TEXINPUTS`, pages rendered with PyMuPDF. Used to check layout before each push.

## Files

- **Paper (`papers/iccc-2026`, all pushed; Overleaf head `8524d97`):**
  - `main_neurips.tex`
  - `sections_neurips/{00..08}*.tex`
  - `tables_neurips/{main_result,main_result_body,sd_test_examples}.tex`
  - `figures/neurips/{fig_headline.pdf, fig_benchmark_correlations.pdf, fig_divergent_convergent.png, fig_overview.png}`
  - `bib/main.bib`
- **Code:** `src/dat_eval/scripts/neurips_camera_ready.py`, `src/dat_eval/scripts/kombine_as_benchmark.py`.
- **Docs:**
  - reports `2026-10-03_neurips_camera_ready/` and `2026-10-03_neurips_mmlu_specificity_and_kombine/`
  - log `2026-10-03/kombine_as_semantic_distance_benchmark.md`
  - dat_eval and kg_creat progress entries
- **Memory:** `specificity-mmlu-pro-only`.

## Key decisions

- **Specificity controls for MMLU-Pro only.** Arena Overall is gone everywhere in the NeurIPS text and figures except the appendix per-model table.
- **Results centre on significance at p < .05.** The ad hoc "best test" shading was removed.
- **The construct is "output diversity on open-ended prompts", not "divergent thinking".** Divergent thinking stays as the cognitive mechanism the DAT measures.
- **NeurIPS figures live in `figures/neurips/`** so the ICML and preprint builds, which share `figures/`, are untouched.
- **Captions remain the author's,** except the Venn caption, which the author asked for.

## Open / next steps

- **Captions:**
  - Figure 3 doesn't mention the RAT.
  - Figure 4 still names Arena Ovr and "divergent thinking".
  - Table 1 still has the "Green cells" sentence.
  - The headline-figure caption is stale.
- **Discussion:** the `Schapiro2025CombinatorialAbilities` sentence ("graph pathfinding task") no longer matches arXiv 2509.21043, now the Kombine paper.
- **Analysis owed:**
  - per-temperature DAT/CDAT appendix table
  - T = 0.7 runs (paid; estimate first)
  - RAT and Kombine columns in the appendix per-model tables
- **Remaining rebuttal writing:**
  - creativity definition and scope
  - "not necessary" framing and Future Work
  - greedy algorithm in the main text
  - CDAT and PACE explanations
  - construct-local reading and base models
  - Labyrinth of Links and MM-OPERA
  - bib fixes and the checklist
- **Submission:** OpenReview abstract and PDF upload.
- **Overview figure:** step 1 says "Creativity Test"; the rest of the paper says "semantic distance tests".
