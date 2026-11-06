# NeurIPS camera-ready finished: jagged creativity, code release, project page, base vs instruct

*Session 2026-11-04 to 2026-11-06 · dat_eval track · paper repo `papers/iccc-2026` (Overleaf)*

## Summary

The NeurIPS 2026 camera-ready of "Do Semantic Distance Tests *Actually* Predict Creativity in LLMs?" is finished, apart from the author's OpenReview upload.
- The paper is reframed around **jagged creativity**: a model can produce one highly distant response yet repeat itself across samples.
- The jagged finding is backed by real DAT responses and a new base-vs-instruct experiment.
- The code and data are public on GitHub, and the project page is live at schapiro.ai/semantic-distance-tests.

## Tasks completed

**Reframe and writing (paper):**
- **Results:** condensed to four findings:
  - the output-diversity null, which became jagged creativity;
  - the RAT as a capability proxy;
  - the DAT as the best predictor of creative writing;
  - no single test predicts all constructs.
- **Jagged creativity:** coined as the lead finding. Point creativity vs distributional creativity, cited to Mineault et al. 2026 for jagged intelligence. Written into the abstract, intro, results and conclusion, in bold italics.
- **Author comments:** all 29 draft comments addressed (from the author's screenshots, `SD_comments.pdf`). The §2 preliminaries are now a titled box, which the author renamed "Note on Terminology".
- **Wording and citations:**
  - The RAT abstract sentence was smoothed (the author fixed the final wording).
  - Labyrinth of Links and MM-OPERA are cited (rebuttal #9).
  - TwistBench is cited with its OpenReview URL.
  - A Mednick flat/steep and steep-deviant footnote was added.
- **Page fit:** many rounds of one-line trims kept the main text on page 10. Most recently, these followed the author's Conclusion rewrite, the Table 5 and 6 callouts and the Figure 6/7 moves.
- **Appendix and checklist:**
  - The checklist answers are updated.
  - The LLM-usage appendix is restored.
  - The broader-impacts template text is restored at the author's request.
  - The RAT prompt is Appendix H.5.

**Figures and tables:**
- **Figure 6 (headline):** fonts about 1.45× larger, a two-row legend, more spacing, and moved to the top of page 8.
- **Figure 3:** stays at the top of page 3 via `\afterpage`. Its colours and test order match Figure 2.
- **Figure 7 (jagged):** went from a schematic with made-up words to real DAT words:
  - (a) Mistral 7B, ocean/desert/volcano;
  - (b) GPT-5.4, quasar/algorithm/moss;
  - (c) o3-mini;
  - (d) hypothetical, dashed, because no model reaches it.
  - The axes are mean DAT score and vocabulary size, with "ideal scaling" (a→d) and "actual scaling" (a→b) arrows.
  - Moved beside the output-diversity result on page 9.
- **Appendix G, "Evidence for Jagged Creativity":**
  - An intro paragraph.
  - Table 5: repeated words per model and temperature.
  - Table 6: base vs instruct, with a mean ± SEM row, top-p and top-k in the caption, and ties broken alphabetically so reruns match.
- **Provider logos** in the Appendix D and E tables.

**Data correction:** the Opus 4.6 and Sonnet 4.6 DAT and CDAT responses were truncated by a 256-token cap. Six models were re-run and rescored:
- Tables 1–2, Figures 5–6, Appendix D and the quoted numbers were regenerated.
- No Table 1 significance changed.
- The model count is 55 everywhere.
- The greedy text matches the plot: LLM mean 81.1, SD 3.0, 2,082 trials.

**Bib audit:** all 55 cited entries were checked against arXiv and Crossref and fixed:
- Kapoor et al.: missing author and arXiv ID.
- EscapeBench: title and pages.
- PACE: pages.
- Olson: PNAS volume and DOI.
- Wechsler: 1939 book.
- Chen and Bellemare-Pepin: as arXiv preprints.
- Full author lists.

BibTeX now gives 0 warnings.

**Code and data release:** https://github.com/samjschapiro/do-semantic-distance-tests-actually-predict-creativity-neurips-26 (local `../semantic-distance-creativity`). It reproduces the paper's `cells.json` and Tables 1–2 exactly, and includes the six-model re-runs. The README has badges and the overview figure.

**Project page** at schapiro.ai/semantic-distance-tests. Source is `scratch/neurips_camera_ready/project_page/`, deployed to `samjschapiro.github.io/docs/semantic-distance-tests/`.
- **Header:** NeurIPS logo, institution logos (square KLab logo) and a Paper button linking `files/SD_NIPS.pdf`, which is the final PDF.
- **Interactive parts:** a Results Explorer, a sortable per-model table with provider logos, and a figure carousel.
- **Background:** a faint neuron background, also added to the Kombine page.
- **Latest update:**
  - The overview figure is first.
  - The TL;DR leads with jagged creativity.
  - The abstract is the paper's current one.
  - A new Jagged Creativity section has Figure 7, three headline numbers, GPT-5.4 vs Command R+ example responses and the base-vs-instruct table.

**Base vs instruct experiment (new):**
- **Code:** `src/dat_eval/scripts/run_base_dat.py`, config `configs/dat_eval/base_models_dat_20261105.yaml`, wrapper `scripts/dat_eval/run_base_dat.sh`, output `data/dat_eval/base_models_dat_20261105/`.
- **Setup:** the paper's DAT prompt given to six local base models as a text completion (HF transformers, MPS, bf16), 40 runs at each T ∈ {1, 1.5, 2}, top-p 1 and top-k 0, with GloVe scoring.
- **Results at T = 1.0, five usable pairs:** instruction tuning raises the mean DAT score from 77.4 ± 3.1 to 81.4 ± 2.1, but cuts distinct words in 40 runs from 296 ± 35 to 97 ± 41. The most extreme pairs are Llama 3.2 3B (341 → 13) and Mistral 7B (351 → 15). The instruct model's top word appears in 33.4 of 40 runs, against 5.8 for the base model.
- **Caveats:**
  - Base models give almost no valid lists at T ≥ 1.5.
  - The Gemma base models often fail the task.
  - Gemma 2 2B is excluded because its instruct version has no valid DAT data.
- **Smaller findings:** the Qwen and Mistral pairs gave the first vocabulary comparison (Qwen 298 vs 120, Mistral 351 vs 15).

**Cross-repo:** a memo in `llm_creativity_mech_interp/docs/memos/2026-11-06_base_vs_instruct_dat.md`, with a pointer in `divergent_circuits/progress.md`. It proposes comparing base and instruct models mechanistically on the DAT, and now has the full five-pair results.

## Files modified or created (this repo)

- `src/dat_eval/scripts/run_base_dat.py` (new), `configs/dat_eval/base_models_dat_20261105.yaml` (new), `scripts/dat_eval/run_base_dat.sh` (new).
- `docs/tracks/dat_eval/progress.md`, `docs/research_context.md`, `docs/structure.md`, `README.md` (55 LLMs).
- Untracked, outside git:
  - `scratch/neurips_camera_ready/` (figure scripts, project page source, table generators);
  - `data/dat_eval/base_models_dat_20261105/`;
  - `papers/iccc-2026/`, which syncs with Overleaf.

## Key decisions and insights

- **Jagged creativity is the paper's lead finding.** Table 5 (raw repetition) and Table 6 (instruction tuning shrinks vocabulary while raising the DAT score) give it direct evidence beyond the null correlation.
- **Figure 7 panel (d) is hypothetical.** The author's view is that no model reaches high point and high distributional creativity, so the caption names models only for (a)–(c).
- **Sampling for base models is untruncated:** top-p 1 and top-k 0, the same as the API runs. This is why base models fall apart at high T.
- **Gotcha:** Python's `run_base_dat.py` does not read `.env`. Without `HF_TOKEN` exported, gated models fail as "gated repo" even with access granted. One Gemma 9B download also stalled at 0 bytes and needed a restart, which kept the finished files.
- **Gotcha:** ties in "most repeated word" were broken in random set order, which changed the table between reruns. They are now broken alphabetically.

## Open questions and next steps

- The author uploads the final PDF and abstract to OpenReview.
- Optional: Gemma 2 2B could be paired if its instruct version were re-run.
- Optional: a T = 1.0-only instruct comparison is already used. A matched completion-frame run of the instruct models would separate framing from tuning, as noted in the mech-interp memo.
