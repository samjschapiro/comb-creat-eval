# 2026-08-30 — Kombine: scoring pipeline realigned to the paper, cost ledger, first multi-model test run

Large multi-day session (2026-08-28 → 30). Finalized the analogy/blending task formalism in the
paper, rewired the elicitation + scoring code to match it, built a persistent cost ledger, ran the
whole loop (elicit → parse → score) on **6 models at 30 items/task**, and redesigned the human
generation study to the new task structure.

## Paper (Overleaf `papers/kg_creat-iclr/`, pushed throughout)

- **Unified analogy/blending via a projection operator.** New `[Structure Mapping]` definition lifts a
  map to triples/sets, `M[T]`. Analogy invents `h := M[Φ]` (project a source concept's structure).
  Blend uses two independent selective projections `M_u, M_v` into a blended space
  `c' := M_u[Φ_u] ∪ M_v[Φ_v] ∪ Δ`; the **generic space `g`** was reintroduced as a component (a
  *textual* schema, not a triple set) after the user flagged it isn't a triple set in our examples.
- **Blend scoring on `g`.** Utility `U_bl = J^gen` (judge: both inputs instantiate `g`), surprise
  `S_bl = ½(d(u,g)+d(v,g))` (abstraction distance), originality `O_bl = ρ_g(g)`. The double-scope
  **quality score `Q_bl ∈ {1,2,3}`** written as a top-down `cases` block. Surprise instruction was
  *removed from the blend prompt* (scored only) to stop the model over-abstracting the schema.
- **Emergent creativity is reported as SEPARATE dimensions** (originality / coherence / validity or
  scope), never a single aggregate — explicit user instruction.
- **Originality = pool-relative embedding distance ρ (kNN)**, replacing inverse-frequency.
- Restored `tab:scoring` (dimensions × tasks); `tab:examples` (Table 2) rewritten to minimal operator
  form and de-colored; Table 1 (`tab:forms`) blend example → *organism + machine → cyborg*.
- **Judge-prompt appendix** now has factuality `F`, analogy-invention (coherence + validity), and the
  blend judge (`J^gen` / `J^coh` / `Q_bl` scope). All user-reviewed and approved.

## Prompts (`src/kg_creat/prompts.py`)

- Analogy now asks for **one** analogy (not "as many as you can").
- Blending worked example → **cyborg**; generic-space + emergent examples updated.
- **Removed all mid-sentence line breaks** (they became literal newlines in the model-facing text).
- Reward bullets map 1:1 to the scoring dimensions; dropped the SURPRISE bullet from blending.
- **Brevity/concreteness rule**: short recognizable entities; no descriptive clauses, CamelCase, or
  dash-coined compounds. Blend structure capped at **4–6 showcase triples**.

## Scoring pipeline realigned (`judge.py`, `parse.py`, `score.py`, `run_elicit.py`, `dat_eval/llm.py`)

- `parse_blend` reads the new `{triple, from}` structure and keeps the u/v/emergent **tags**;
  `parse_items` now preserves the analogy `invention`/`projected`/`projection`.
- Blend judge → `generic_ok`/`coherent`/`scope` (needs the **tagged** structure, else it can't score);
  new analogy-**invention** judge (coherence + validity). Old polysemy/relational judges retired.
- **Originality** switched to embedding-kNN `ρ`; **surprise** made paper-exact (association = adjacent
  entities, analogy = cross-path aligned pairs `d(a_i,b_i)`, blend = abstraction distance). Analogy
  **utility** = structural (relation-identity, already in `analogy_structural_ok`) ∧ factual — dropped
  the redundant semantic judge.
- **Judge explanations + per-judge verdicts are now PERSISTED** on each record (`blend_judges`,
  `invention_judges`) — no more re-judging just to see the "why".
- Bugs found + fixed while validating: (1) `_majority` bool-coerced the ordinal `Q_bl` → added
  `_majority_val`; (2) `gpt-oss-120b` is a reasoning judge and truncated at low `max_tokens` → bumped
  subjective judges to 3000; (3) blend judge was passed **untagged** structure.

## Cost ledger (`src/kg_creat/cost_ledger.py`, NEW)

Append-only `data/kg_creat/cost_ledger.jsonl` (gitignored): one row per model per phase with **actual**
API token usage and USD from `cost_tracker.PRICING`. `python -m src.kg_creat.cost_ledger` prints the
running total by phase/model. Wired into `run_elicit` (elicit) and `score.py` (score); `call_llm_async`
gained `capture_usage`. Only fresh (non-resume-skipped) runs record, so re-runs never double-count.

## Runs & findings (`data/kg_creat/kombine_test30/`, gitignored)

- **6 models × 30 items/task** (90 prompts each), all three tasks, temp 0.9, default reasoning effort:
  gpt-5-mini, claude-sonnet-4.5, deepseek-chat, gemini-3.7-flash, llama-3.3-70b, grok-4.6. **100% parse**
  everywhere; blend triples land in the 4–6 target; entities concise (dash/CamelCase leakage ~1%).
- Full score loop ran with a single cheap judge (`gpt-oss-120b`). Sensible, model-differentiating
  scores; **grok-4.6 tops utility** across tasks; `Q_bl` mostly 3 (with real spread once the bool bug
  was fixed). Sample high/low analogies + blends inspected with saved explanations.
- **Cost: $10.53 total** (elicit $10.08 + score $0.45). **grok-4.6 alone was $5.43** (870k output/
  reasoning tokens) — heavy reasoning models dominate cost; the pre-run budget cap uses a flat estimate
  and does **not** catch this. Cheapest four models combined were $0.70.

## Human generation study (sibling repo `llm_creativity_mech_interp/.../kombine_generation/`)

Redesigned to the new task structure (committed in that repo, not here): ID page first; per-task intros
shown right before each block; **association = a path of full triples** (head/relation/tail, with a
final relation into the endpoint); **analogy = a full triple in each domain per row (path_a | path_b),
plus the invention as a triple below a divider**; blending example → cyborg; "I can't think of one"
opt-out on the hard invention cell. Served locally at `localhost:8000` for review.

## Files modified (this repo)

- Code: `src/kg_creat/{prompts,judge,parse,cost_ledger}.py`, `src/kg_creat/scripts/{run_elicit,score}.py`,
  `src/dat_eval/llm.py`.
- Configs: `configs/kg_creat/kombine_{smoke_run,test_run,test30_run,test30_more,test30_score}.yaml`.

## Open questions / next steps

- **Factuality judge explanation** isn't persisted yet (only per-triple flags) — small change to
  `judge_factuality_batch`.
- Swap the single `gpt-oss-120b` judge for the **3-model panel** on a real run.
- Add an **actual-cost stop** to `run_elicit` (the flat pre-estimate under-predicts reasoning models —
  grok-4.6 taught us this); decide per-model effort caps before a full ~30-model × 70-item run.
- `pipeline judge/score` still carry some **legacy Regime-A** constraint code paths (categorical /
  relation-constraint judges) unused by the three main tasks.
