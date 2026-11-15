# 2026-11-14 23:50: memorised inventions analysis

**Request.** Search for "memorised inventions", like the inventive-multiples analysis, grouped by
provider, input properties and task.

**Done**
- **New `src/kg_creat/scripts/memorized_inventions.py`.** Reuses the infini-gram query and cache layer of
  `corpus_novelty.py`.
  - **Definition:** name attested at least 10 times, in context with an input, and at least 2 properties
    documented with the name.
  - **Null:** one mismatched item (same task) per invention; "specific" means passing with its own
    context and failing the null.
  - About 6,600 new queries, about 2.5 hours; no API spend.
- **Bugs caught by a SMOKE=40 dry run:** formatting None when a task had no positives, and undefined
  Spearman on constant input. Both fixed before the full run.
- **Results:** see `progress.md` (2026-11-14, later) and
  `docs/reports/2026-11-14_kg_creat_memorized_inventions/report.md`.
- **Corrections made to the first report draft after checking against the data:**
  - 15 models (not six) have no memorised inventions;
  - the showcase inventions *liquid franchise* and *narrative species* do occur verbatim (38–45× and
    912×) but are not memorised under the definition.
- **Rebuttal:** En17 Q1 point 5 gained a memorised-inventions bullet.
