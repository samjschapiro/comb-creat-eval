# 2026-05-06 — DRAT design and smoke test

## Summary

Designed and smoke-tested the Divergent Remote Association Test
(DRAT), a vocab-space creativity test motivated by the dat_eval
finding that no existing test (DAT, CDAT, PACE) predicts scientific
ideation on LiveIdeaBench with significance. The design went
through several iterations driven by user feedback: started as a
two-anchor extension of CDAT (Bridge Associates Test, BAT) with
min-utility scoring, then renamed to DRAT after the framing
crystallized as a hybrid of the Remote Associates Test
(stimulus structure) and the Divergent Association Task (response
format and scoring), then revised to use max-utility plus a
DAT-style mean-pairwise-distance score per the user's proposed
unification of convergent and divergent thinking. Built a
minimal smoke-test pipeline reusing dat_eval's OpenRouter async
client and CDAT's SBERT embeddings; ran on
gemini-2.0-flash-lite over 3 hand-crafted anchor pairs; pipeline
green.

## Substantive design decisions

### Iterations on the score

1. **Initial proposal (BAT, multiplicative).** Per-word bridge
   score $b(w; A, B) = \min(\cos(w, A), \cos(w, B))$, score $=
   \bar{b} \cdot \nu$. Worked examples showed $\nu$ doing most of
   the discriminating work; ratio of genuine bridge-finder to
   generic-hypernyms cheat ≈ 1.78 on word anchors but ≈ 1.25 on
   scientific-concept anchors.
2. **Per-response gate (BAT, gated).** Switched to a per-word
   threshold $\tau$ from a random-noun null distribution, with
   $|S| \geq n_{\min}$ qualification, score $= (|S|/n) \cdot
   \nu(S)$. The $|S|/n$ factor penalizes mixed-bridge/filler
   responses that the multiplicative version overscored.
3. **Renamed to DRAT.** Framing crystallized as a hybrid of RAT
   (anchor structure, reduced 3 → 2 and convergent → divergent)
   and DAT (response cardinality and embedding-based score).
   Detailed Mednick-anchored motivation written into the design
   doc.
4. **Final design (current).** User flipped utility from $\min$ to
   $\max$: a word qualifies if anchored in *either* anchor's
   neighborhood. Score is mean pairwise distance over survivors,
   matching DAT/CDAT convention at 100 × mean. The motivation
   shifted from "scientific ideation" to "unifying convergent and
   divergent thinking in a single test"; the LIB prediction
   remains the empirical target but is no longer the primary
   framing. Worked examples re-derived: spanning strategy
   (5 + 5 across both clusters) strictly dominates pure-bridges
   (RAT-mode), CDAT-mode, and generic cheats.

### Score formula and the off-by-2 episode

User originally wrote $\frac{100}{k(k-1)} \sum_{i<j} d_{ij}$, which
is $50 \times$ mean since $\sum_{i<j}$ has $k(k-1)/2$ terms. I
fixed it once with prefactor $200/(k(k-1))$, which the user then
asked me to revert to $100/(k(k-1))$. Final form sums over ordered
pairs $i \neq j$ (which has $k(k-1)$ terms, so the same $100$
prefactor produces $100 \times$ mean and matches DAT). All worked
examples re-tabulated at the corrected scale.

### Prompt iteration

First prompt was bare ("Give 10 single words that connect A and B").
On the (ecosystem, economy) pair, this produced too few
strongly-anchored words and gate-failed at $n_{\min} = 5$. User
proposed an explicit divergence-and-bridging instruction matching
DAT/CDAT phrasing: *"Please give 10 words that are as different
from each other as possible, in all meanings and uses, and each of
which connects A and B."* Re-running with the explicit prompt:
all three pairs produce non-zero DRAT, with similar magnitudes
(64 → 68, 74 → 73) on the easy pairs and recovery from gate-fail
to 57.5 on (ecosystem, economy). The pilot will use the explicit
prompt.

## Smoke-test results

Gemini-2.0-flash-lite, 3 hand-crafted anchor pairs, SBERT
(all-mpnet-base-v2), CDAT's 50-cue noun pool, $\tau = 90$th
percentile, $n_{\min} = 5$.

| pair | $\tau$ | survivors | DRAT |
|---|---|---|---|
| immune system × supply chain    | 0.27 | 8/10 | 68.01 |
| phase transition × NN training  | 0.22 | 7/10 | 73.10 |
| ecosystem × economy             | 0.34 | 5/10 | 57.53 |

Sanity checks: $\tau$ values in $[0.10, 0.40]$, DRAT in $[0, 200]$,
all pairs non-zero. Magnitudes match typical DAT scores in the
dat_eval pool. Outputs in `data/new_tests/drat/smoke_v1/`.

## Findings worth flagging

1. **Anchor concreteness drives $\tau$.** Concrete pairs (immune
   system × supply chain) get $\tau \approx 0.27$; abstract pairs
   (ecosystem × economy) get $\tau \approx 0.34$ because random
   nouns in the CDAT pool happen to be near-similar to abstract
   concepts. The pre-registered pilot bank should weight toward
   concrete-domain anchors. Failure mode 1 from the design doc's
   stress-test got teeth here.
2. **Spanning strategy dominates pure-bridges.** Per the worked
   examples in the final design (and what gemini-flash-lite
   actually produced), the model that spans both anchor clusters
   with diverse vocabulary scores higher than one that produces
   only bridges. Bridges live in the intersection region, which
   is denser than either cluster, so within-bridge pairwise
   distance is bounded by the intersection's radius. This is the
   design intent of the unified convergent/divergent test.
3. **Prompt sensitivity (failure mode 6) drops with explicit
   instruction.** The bare "connect A and B" prompt elicited
   underspecified responses. The explicit "as different as
   possible, each connecting A and B" prompt elicits the task
   we're scoring. Cross-model variance attributable to prompt
   interpretation should drop in the pilot.

## Files

- `docs/tracks/new_tests/drat_design.md` (NEW). Full design doc:
  motivation grounded in Mednick / RAT / DAT, formal spec
  including the max-utility gate and DAT-style score, stress-test
  with seven failure modes, worked examples on both word-level
  and scientific-concept anchors, decision rule for pilot scaling.
- `src/new_tests/drat.py` (NEW). `drat_prompt`, `compute_tau`,
  `score_drat`. Reuses `SBERTEmbeddings` and `validate_words_sbert`
  from `src/dat_eval/cdat.py`.
- `src/new_tests/scripts/run_drat_smoke.py` (NEW). Async
  orchestration. Reuses `get_async_client`, `call_llm_async`,
  `extract_words_from_response` from `src/dat_eval/llm.py`.
- `configs/new_tests/drat_smoke.yaml` (NEW). Smoke-test config:
  one model (gemini-2.0-flash-lite), 3 hand-crafted pairs,
  CDAT's 50-cue noun pool, $\tau = $ 90th percentile, $n_{\min} = 5$.
- `scripts/new_tests/run_drat_smoke.sh` (NEW). Bash wrapper.
- `docs/tracks/new_tests/progress.md` (UPDATED). 2026-05-06 status
  block at top.

## Environment fix

Hit a broken `huggingface-hub-1.10.1.dist-info` (missing RECORD
file) on first run. Fixed by deleting the partial install
manually and re-running `uv sync`, which restored
`huggingface-hub==0.36.2`. Logged in case it recurs.

## Next session

Phase 1 pilot. Build the gating engineering:

1. Anchor-bank constructor (mines distance-quantile-bounded pairs
   from a scientific concept source; pre-registered output).
2. Multi-embedding support (GloVe, FastText, SBERT) matching
   dat_eval's appendix protocol.
3. Full async eval runner with budget cap and 3-seed sampling.
4. Score script computing Pearson $r$ and semi-partial
   $r(X, Y - \hat{Y}^g)$ vs benchmarks.json for the LIB-17.

Cost target for Phase 1: under \$1 across six cheap models, $K=30$
pairs, $T=1$, $S=3$.
