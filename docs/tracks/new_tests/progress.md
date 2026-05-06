# new_tests — progress

## Goal

Determine whether existing preference-optimization methods for LLM
creativity (CrPO and predecessors) actually teach creativity, or
whether they teach distributional fit to small-c psychometric tasks.
If they don't teach creativity, identify what does — likely a
question of preference-data construction more than of loss form.

The pivot point for this track: creativity is the ability to produce
novel and appropriate responses *outside* the training-distribution
shape. A method that helps only on its training distribution has not
taught creativity by definition. So the relevant evaluation is just
benchmark performance on the three creativity constructs the field
already cares about (creative writing, divergent thinking, scientific
ideation), measured directly. There is no separate "transfer"
question.

## Status — 2026-05-06

DRAT (Divergent Remote Association Test) design and smoke test. The
test is a hybrid of RAT and DAT designed to bridge convergent and
divergent thinking measures into a single vocab-space test, motivated
by the dat_eval finding that no existing test predicts LiveIdeaBench
specifically. Design doc in
[drat_design.md](drat_design.md). Smoke-test pipeline in
`src/new_tests/drat.py`, `src/new_tests/scripts/run_drat_smoke.py`,
`configs/new_tests/drat_smoke.yaml`,
`scripts/new_tests/run_drat_smoke.sh`.

Smoke test passes on `gemini-2.0-flash-lite-001` over 3 hand-crafted
anchor pairs. With the explicit divergence-and-bridging prompt, all
three pairs produce non-zero DRAT scores in the 57–73 range,
matching typical DAT magnitudes.

Next: Phase 1 pilot — six cheap LIB-17 models, $K=30$ pre-registered
anchor bank, all three embeddings (GloVe, FastText, SBERT). Plan in
[drat_design.md §Decision rule](drat_design.md). The
gating engineering — anchor-bank constructor, multi-embedding
support, full async eval runner — is still TODO.

## Status — 2026-05-03

- Read [iccc-2026 paper](../../../papers/iccc-2026/sections/) end to
  end. Compiled [survey.md](survey.md) covering Nagarajan
  Roll-the-Dice, Schapiro comb-creat, Wadhwa CREATE, plus ~30
  adjacent benchmarks.
- Pulled apart CrPO ([arXiv 2505.14442](https://arxiv.org/abs/2505.14442))
  in detail. Two structural observations recorded in
  [proposals.md](proposals.md):
  1. The CrPO loss multiplies the standard DPO loss by a scalar
     `w(x, y^w) = Σ_d λ_d · s_d^w` that does not depend on θ. The
     creativity scores therefore enter only as per-pair learning-rate
     scheduling; they do not change the gradient direction at any
     pair, and the model converges to the standard DPO stationary
     point with biased pair sampling. The mechanically-cleaner
     alternative is to put the creativity contrast `c^w − c^l`
     *inside* the sigmoid (Cre-DPO).
  2. MuCE-Pref is built from ~12 templated psychometric tasks with
     short undergraduate-pool free-text responses. The CrPO paper
     reports that vanilla DPO on MuCE-Pref already beats GPT-4o on
     the held-out MuCE suite — strongly suggesting the held-out suite
     and the training set share a distribution, and that the headline
     CrPO result reflects distributional fit, not creativity.
- Logged the central claim as [H1 in docs/HYPOTHESES.md](../../HYPOTHESES.md):
  *CrPO does not improve creativity.* Apparent gains on MuCE-style
  held-out tasks reflect distributional fit, not transferable
  creative competence. Falsifiable by running released CrPO
  checkpoints on the six creativity benchmarks dat_eval already uses.
- Downloaded MuCE / MuCE-SFT / MuCE-Pref to
  `data/new_tests/muce/` (gitignored). Schema fact worth flagging:
  MuCE-Pref stores the four auto-metrics for the *chosen* response
  only. To train any contrast-based variant of CrPO (Cre-DPO,
  GRPO-Cre, etc.) the rejected side has to be rescored first.

## Next steps

1. **Test H1.** Run the released CrPO checkpoints (Llama-3.1-8B and
   Mistral-7B variants from
   [CNCL-Penn-State on HF](https://huggingface.co/collections/CNCL-Penn-State/crpo-67d0b11ff358430823dbb3df))
   plus their SFT-only and DPO-only ablations against the six
   creativity benchmarks (Arena CW, EQ-Bench CW, Mazur CW, Hivemind,
   NoveltyBench Utility, LiveIdeaBench), plus a held-out MuCE slice
   as a control. Report benchmark scores and per-benchmark deltas
   relative to the matched base model. Resolution criteria are
   pinned in [H1](../../HYPOTHESES.md#h1--crpo-does-not-improve-creativity).
2. **If H1 supported (CrPO does not improve creativity):** the work
   becomes about *what would* improve creativity. The most likely
   answer is preference-data construction. Candidate datasets to
   try, with the same matched base models for clean comparison:
   - Creative-writing pairs mined from r/WritingPrompts / LitBench /
     Arena CW battle pairs.
   - Divergent-thinking pairs from Hivemind / NoveltyBench prompts
     via best-of-K with rubric judges.
   - Scientific-ideation pairs from LiveIdeaBench-style keyword
     prompts via best-of-K with the LiveIdeaBench multi-judge
     rubric.
3. **If H1 falsified (CrPO does improve creativity in some
   meaningful way):** then the loss-form discussion in
   [proposals.md](proposals.md) becomes the primary direction —
   train Cre-DPO and check whether the contrast-inside-the-sigmoid
   form gives a clean improvement over CrPO on the same benchmarks.
4. (Either way, deferred.) GRPO-Cre — online policy gradient with
   `c` as the reward. Strictly stronger than offline preference
   methods if the reward signal is right; only worth pursuing once
   we know what the right reward signal is.

## Design constraints

- Released CrPO checkpoints used as-is for the H1 test. No
  retraining required for step 1.
- For any retraining (steps 2–3), match CrPO's training setup
  (LoRA r=128, α=256, 1-epoch SFT, then preference optimization)
  so any benchmark delta is attributable to the data or loss
  change, not auxiliary differences.
- Reference reward / surprise / embedding models held fixed
  (Skywork-Reward-Gemma-27B-v0.2, Gemma-2-27B, jina-embeddings-v3)
  to match CrPO when we want a controlled comparison.
- API spend on benchmark scoring respects `budget_usd` in run
  configs; reuse the OpenRouter async client from `dat_eval`.
