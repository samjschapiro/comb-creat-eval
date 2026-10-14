# Kombine as a fourth construct for the semantic distance tests paper

*2026-10-03 · dat_eval × kg_creat · exploratory*

**Question.** Are we missing a real-world creative construct beyond creative writing, divergent thinking (output diversity) and scientific ideation? We test this by scoring DAT, CDAT, PACE and RAT on the Kombine benchmark (association, analogy, blending), using the paper's two criteria. Specificity controls for **MMLU-Pro only**.

**Script:** `src/dat_eval/scripts/kombine_as_benchmark.py` → `data/dat_eval/kombine_benchmark/results.json`. No API cost.

## Data & sampling

- **Models:** the 35-model Kombine pool (`data/kg_creat/kombine_test30/scores`). It is a convenience sample of frontier and near-frontier API models, newer than the paper's 54.
- **Test coverage:** DAT 33, CDAT 31, PACE 21, RAT 34. With MMLU-Pro as well: 19, 18, 18, 20.
- **MMLU-Pro:** 20 of 35 models have a score: 15 from `benchmarks.json` and 5 matched by exact name in the TIGER-Lab file (Kimi K2, GLM-4.5-Air, Gemini 3.1 Pro, Gemini 3 Flash, GPT-5.2).
- **Missing from every specificity cell:** the 15 newest models have no MMLU-Pro score anywhere (Fable 5/5.1, Opus 4.7/4.8/5, Sonnet 5, GPT-5.6, GPT-6, Grok 4.5/4.6, Gemini 3.7 Flash, Qwen3-Max, among others). **Specificity is therefore measured on older models only, while validity covers the full pool.**
- **Test scores:**
  - DAT and CDAT at T = 1.0 (Kombine runs at 0.9), from one embedding, keeping only models with ≥70% of trials parsed.
  - RAT: strict zero-shot accuracy. GPT-5.6 is dropped (its 43% is a token-cap artifact).
  - PACE at T = 0.
  - **These are not the paper's Table 1 scores** (those average temperatures and three embeddings), so the numbers are not comparable cell for cell.
- **Unit:** the model. Two-sided Pearson t-test; df = n − 2 for validity and n − 3 for specificity.
- **Power:** n ≈ 19 with one control can only detect |r| ≳ 0.60; n ≈ 33 for validity detects |r| ≳ 0.47. Every null below is uninformative about effects of moderate size.

## Findings

**1. All four tests predict Kombine, but mostly through capability.** Validity against the Kombine overall score is DAT +0.66\*\*\* (n = 33), PACE +0.56\*\* (21), RAT +0.53\*\* (34), CDAT +0.31 (31). Controlling for MMLU-Pro, none survives at the level of the overall score (DAT +0.18, CDAT −0.24, PACE +0.12, RAT +0.29; n = 18–20). This is the same pattern the paper finds for creative writing.

**2. The one significant specificity on a task score is RAT on association: +0.48\* (n = 20).** Association asks for a chain of links between two input entities, which is close to what the RAT tests. With p ≈ .03 and one cell out of 16 composite cells, it is suggestive only.

**3. CDAT predicts originality beyond capability.** Across the ungated criteria (88 cells), CDAT's specificity is large on every originality score:

| criterion | CDAT specificity | n |
|---|---|---|
| association originality | +0.71\*\* | 18 |
| analogy emergent originality | +0.64\*\* | 18 |
| blending originality | +0.64\*\* | 18 |
| analogy originality | +0.54\* | 18 |
| blending emergent originality | +0.51\* | 18 |

The first three survive Benjamini–Hochberg over all 88 facet cells (19 survive, threshold p ≤ .010; most of the survivors are validity cells). No other test has a significant specificity on any originality score. This matches the September finding that originality loads on CDAT (`analyze_cognitive_facets.py`, validity only), and now holds after controlling for capability.

**4. Kombine overall is not a new construct; blending originality may be.**
- The overall score tracks creative writing and capability: Arena CW +0.67\*\* (n = 17), EQ-Bench CW +0.71\*\* (13), Mazur CW +0.79\* (9), MMLU-Pro +0.64\*\* (15).
- Its correlations with Hivemind, NoveltyBench and LiveIdeaBench rest on n = 6–7 and say nothing.
- **Blending originality and blending emergent originality do not track capability or creative writing:** MMLU-Pro −0.34 and −0.03, Arena CW −0.39 and −0.15 (n = 15–17).
- Analogy originality, by contrast, does track capability (MMLU-Pro +0.69\*\*).
- So the part of Kombine that existing benchmarks miss is inventiveness under blending, and the CDAT is the only test that predicts it beyond capability.

## What this does and does not show

- **Generality:** one benchmark, 35 models, and specificity measured only on the ~20 older ones. If the newest models were included the results could change, in either direction.
- **Multiple comparisons:** finding 3 is the only result that survives correction. Findings 2 and 4 are exploratory.
- **Use in the paper:** Kombine would be a fourth construct (combinational creativity). It would also need the paper's own test scores (temperature handling, three embeddings) recomputed for the Kombine pool. And the paper would have to cite Kombine, which is unpublished.
