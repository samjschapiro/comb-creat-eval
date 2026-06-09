# plot_twist — judge rubric design

The fixed-rubric LLM judge is the benchmark's primary score (§3) and is used in §4
(frontier eval), H4 (vs DSI), and H5 (human vs LLM). Code:
[src/plot_twist/rubric_judge.py](../../../src/plot_twist/rubric_judge.py).

## v1 (current, shipped)

Four anchored 1--5 dimensions + a boolean:
- **surprise** (= theory's `T_mod` / re-read), **coherence** (= preservation),
  **prose_quality** (the H5 covariate), **overall** (holistic), **twist_present** (H4 detection).
Ensemble of 3 judges, temp 0, median per dimension (majority for twist_present),
CoT-before-JSON, robust parsing, versioned.

## Decisions (locked 2026-06-08)

- **Headline per-story score = surprise $\times$ coherence**, combined as the
  **geometric mean** $\sqrt{\text{surprise}\cdot\text{coherence}}$ (stays on 1--5;
  collapses toward 1 if either is low = "surprising *and* coherent"). `min` and raw
  product are ablation combiners. **OVERALL is kept only as a validation cross-check**
  (does the structural product predict the judge's holistic rating?).
- **Disjoint judge set**: judges are **excluded from the ranked generation pool** to avoid
  self-preference; still **report a self-preference check** on a subset.

## v2 to-dos

- [ ] **Geomean scoring**: benchmark score = `geomean(surprise, coherence)`; store
      OVERALL separately and correlate it against the product (validation). Add `min` /
      `product` as ablations.
- [ ] **Disjoint judge config**: set `judge_models` to a held-out tier (one strong model
      per family: Anthropic / OpenAI / Google) that is NOT in the generation pool; document
      the exclusion; add a self-preference sanity check.
- [ ] **Extract-first**: judge names the **reveal** and the **overturned assumption** before
      scoring (one sentence each); emit them in the JSON (`reveal`, `overturns`). Sharpens
      scoring AND feeds the structural metric.
- [ ] **Few-shot anchors** (biggest reliability lever): one compact exemplar per corner with
      target scores --- true twist (5,5), predictable (low surprise), random (low
      coherence). Keep them short/synthetic to bound context cost.
- [ ] **Anti-recognition / rubric-adherence line**: "rate only what the rubric asks; do not
      credit a story for being a known classic or penalize an unfamiliar one." Pair with a
      rubric-adherence check (the commented Limitations bullet).
- [ ] **Calibrate before ranking**: validate the rubric against a human-rated subset
      (judge--human agreement = the §3 make-or-break number) and **pre-register** before
      seeing model rankings.
- [ ] **Bias controls**: instruct judges to ignore prose for surprise/coherence;
      length/position-randomize; consider length normalization.
- [ ] **Scale resolution** (deferred): keep 1--5 for the benchmark; use **pairwise**
      (Bradley--Terry) for the §5 human study where fine human--LLM gaps matter.
- [ ] **twist_present**: decide whether it stays a separate judgment or is derived
      (e.g., surprise $\ge 2$); needed for H4 presence detection.
