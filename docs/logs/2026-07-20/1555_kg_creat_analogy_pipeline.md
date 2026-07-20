# 2026-07-20 — kg_creat: full pipeline + first analogy result; paper callouts

## Summary

Built the **entire kg_creat evaluation pipeline end-to-end** (Wikidata builder → sampler →
prompts → elicitation → judge → scoring → plots) and produced a **first empirical result on the
analogy tier** (200 random pairs × 8 models). Also did early-session edits to the ICLR paper
(Overleaf) and stood up a blind judge-reliability review harness. Committed the track to `main`.

## Tasks completed (proportional — the pipeline build was the bulk)

**Paper (early session, `papers/kg_creat-iclr` on Overleaf):**
- Installed a local LaTeX toolchain (TinyTeX, no sudo) so the paper compiles locally.
- Added `researchquestion` callout boxes around the two guiding questions (Q1 theoretical /
  Q2 empirical) in the intro; verified render.
- Captioned Figure 2 (the six constraint types A–F), renamed label `fig:constraints`.
- Recolored Table 1 check/cross marks to `#388C3C` / `#CE5E5E`; reported the research-box hex.
- Pushed all to Overleaf (rebased over remote edits).

**Environment:** repo's pinned CUDA torch is unusable on Mac → round-1 is torch-free; set up
`.venv` (3.14, non-torch deps) + **`.venv_mlx` (3.12)** for MLX (local model serving via
`mlx-lm`, local embeddings via `mlx-embeddings`, the scorer).

**Pipeline (new `src/kg_creat/`):** `wikidata.py` (REST-BFS builder, frequency-derived relation
vocab + admin/attribute stoplist, domain-tagged seeds), `sample.py` (matched-bundle + random
Regime-B samplers), `prompts.py` (CREATE-aligned, open-vocab), `run_elicit.py` (OpenRouter +
local MLX, budget cap, per-model resume), `judge.py` (gpt-oss-120b factuality/semantic/relation-
constraint), `embed.py` (MLX MiniLM), `aggregate.py`, `score.py`, plotters, review harness.

**Design iterations (several reversed earlier assumptions):** dropped exact-hop-count (variable
length, CREATE-style); tried controlled vocab then **reverted to open vocab**; **frequency-derived**
(not hand-picked) relation vocabulary; **random** (not curated) Regime-B endpoint pairs — "can a
model find an analogy between arbitrary entities"; **domain-tagged seeds** (domain = IV);
judge-based constraint checking; strict analogy validity = exact-relation-match ∧ disjoint ∧
node-distinct ∧ factual ∧ judged.

**Runs / results:**
- Regime-A pilot: 10 matched bundles × Qwen2.5-3B/7B (local, free) → `plot_novelty_utility` 2×2
  (exclusion "handled" vs inclusion/categorical/ordering → the gap; capability-dominated at 3B/7B).
- Judge migration: local 3B → local 32B → **OpenRouter gpt-oss-120b** (CREATE's judge; a local
  32B wrongly rejected the atom::solar-system analogy — motivated the upgrade).
- **Analogy study:** 40 → 200 random pairs; **8-model suite** (Llama-3.1-8B → Sonnet-4.6).
  Sonnet 26.0% ≈ Haiku 25.5% at top; even best ~26%. Per-pair complementary analysis:
  **anchor distance is a weak/structural (not distributional) predictor** (Pearson −0.14).
- Wrote report: `docs/reports/2026-07-20_kg_creat_analogy/`.
- Built blind judge-reliability review harness (60 factuality + 40 analogy items, web UI, auto-log).

## Files created / modified
- `src/kg_creat/`: wikidata, sample, prompts, judge, embed, aggregate, parse, scoring + 11 scripts + vendored CREATE.
- `configs/kg_creat/*.yaml`; `scripts/safety/cost_tracker.py` (+judge/suite pricing).
- `docs/tracks/kg_creat/progress.md` (new 2026-07-20 status) + `assessment.md` (new); `docs/reports/2026-07-20_kg_creat_analogy/`.
- Committed `9b19612` on `main` (`data/` gitignored).

## Key decisions / insights
- **Difficulty is the point** for analogy — random pairs, no curation (hand-curation "destroys the task").
- **Structure-mapping is strict** — same relations, disjoint structures; loose scoring inflated rates ~2×.
- **Embedding distance ≠ analogical mappability** — the useful negative result of the complementary analysis.
- **Judge reliability is the load-bearing untested assumption** — harness built, human pass pending.

## Open questions / next steps
1. Run the blind judge-reliability review → CREATE-comparable agreement/precision/recall.
2. **Scale Regime A (the constraint typology — the paper's intended headline) to frontier models** —
   under-explored vs analogy this session.
3. Fold analogy-success into `score.py` (currently computed in the plotters).
4. Broaden seeds beyond the academia/awards-heavy set; count parse failures as failures for fair model comparison.
