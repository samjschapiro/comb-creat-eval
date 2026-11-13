# 2026-11-12 → 11-13 — Kombine rebuttal: co-author rater study (human–judge validity, Exp 2)

## Summary

Built, tested and sent a web platform in which Kombine co-authors independently rate the same 40
human and model answers. The ratings answer reviewers A7nz and eN17 (judge validity with multiple
annotators) and give a human check of embedding originality (ρ). The analysis script is ready for
when the rating files come back.

## Tasks completed

- **Builder** `src/kg_creat/scripts/build_rater_study.py` writes three files to `data/kg_creat/rater_study/`:
  - `items.json`: blind content for the page;
  - `key.json`: hidden; source, prompt, panel votes and ρ fields for each answer;
  - `rate.html`: the page itself, self-contained with the items embedded.
- **Design.**
  - **Size:** 12 entity pairs and 40 answers in total (association 4 × 4, analogy 4 × 3, blending 4 × 3).
  - **Composition:** each pair shows one human answer plus 2–3 model answers. Grouping by pair lets
    originality be judged against the same comparison set ρ uses. The model answers are chosen so each
    group, human included, is about half panel-pass. Task totals are exactly 8/16, 6/12 and 6/12.
  - **Same items for everyone:** every rater sees the same answers, needed for inter-rater reliability.
    Order is shuffled per rater, seeded by their name.
  - **Blinding:** no model names, no human/model labels, no judge verdicts on the page.
  - **Exclusions:** the paper's worked-example items are excluded.
- **Filtering** (after the user flagged incoherent-looking association answers):
  - Every answer already passed parsing and the well-formedness checks:
    - association: right endpoints, continuous, no revisited node;
    - analogy: both paths, a projection, a named invention;
    - blending: a generic space, a name, ≥ 2 links.
  - The odd ones were genuine human answers. Human answers now come only from the **46/75 kept
    participants who pass the pre-registered Breakfast + Lunch comprehension screen**. This is the same
    rule as `analyze_human_vs_models.py`; `--no-screen` turns it off.
- **Questions mirror the judges one to one.**
  - association: every link factual (F), optional pick of the false links, originality 1–5.
  - analogy: valid analogy (pair_sat), mapping used (valid / Q(h)), coherent (V(h)), originality of the
    analogy and of the invention.
  - blending: generic space shared (G), coherent (V), shared slot and emergent (giving Q 1/2/3),
    originality of the generic space and of the invention.
  - Every yes/no question allows Unsure.
- **UI = the human study's UI** (the user's request). The page is a jsPsych 7.3.4 experiment with the
  same plugins and the study's stylesheet vendored verbatim (`rater_study_kombine.css`). Answers are drawn
  as read-only copies of the study's widgets:
  - colour-coded concept chips;
  - entity/relation/entity cells and analogy columns;
  - gold invented concept with a direction arrow;
  - blend source pills, and emergent links below the divider.

  Task intros reuse the study's worked examples. Progress autosaves in the browser and resumes by name;
  the end screen offers a JSON download with a copy-paste fallback.
- **Analysis** `src/kg_creat/scripts/analyze_rater_study.py` (reads `ratings/*.json`, writes `analysis.json`):
  - Krippendorff's α per dimension: nominal for yes/no, ordinal for scope, interval for originality;
  - human–human percent agreement;
  - panel vs rater majority;
  - **the panel vs a held-out rater, compared with the other raters' majority vs that rater**: is the
    judge as close to a human as another human is?
  - Spearman of mean human originality vs ρ, pooled and within pair.
- **Verification.**
  - Headless-Chromium runs went end to end: start, all 40 answers, a mid-study reload and resume,
    comment, download. They gave valid JSON with no console errors and no horizontal scroll at phone width.
  - Unanswered questions block Next.
  - The α implementation matches Krippendorff's published example exactly (0.743 / 0.815 / 0.849).
  - On synthetic raters the analysis recovered the planted signal.
- **Bugs fixed during testing.**
  - The end screen threw before wiring the download button (it read past the last step).
  - For analogies, the column holding the projected concept was guessed from entity overlap. A tie put
    one human's *server → crystal* under the wrong column. Detection now uses exact entities, then shared
    words, and falls back to a neutral "known link → invented" layout.
- **Sent.** The user texted the page to the co-authors on 2026-11-13.

## Files

- New: `src/kg_creat/scripts/{build_rater_study.py, analyze_rater_study.py, rater_study_template.html, rater_study_kombine.css}`
- Generated (gitignored): `data/kg_creat/rater_study/{items.json, key.json, rate.html}`; returned files go to `ratings/`.

## Key decisions

- **A self-contained file rather than an Artifact with a database.** Co-authors span many institutions
  and would need claude.ai accounts. The page needs internet only for jsPsych from unpkg, like the
  human study.
- **Not hosted on schapiro.ai.** The Pages repo is public and the page contains Prolific participants'
  answers.
- **Unsure counts as missing** in all agreement statistics.

## Next steps

- Collect `kombine_ratings_<name>.json` from co-authors into `data/kg_creat/rater_study/ratings/`, then run
  `.venv/bin/python -m src.kg_creat.scripts.analyze_rater_study`.
- Remaining rebuttal work: see `docs/tracks/kg_creat/progress.md` (2026-11-13).
