# kg_creat — experimental methods (scoring + CREATE-extension pipeline)

How we actually run the study: the scoring operationalization (novelty / diversity /
utility), the key scientific question these serve, and how we extend CREATE's released
code ([Wadhwa et al. 2026, arXiv:2603.09970](https://arxiv.org/abs/2603.09970),
`github.com/ManyaWadhwa/CREATE`, cloned to `resources/repos/CREATE`) into our pipeline.

Companion docs: [design.md](design.md) (task spec + formal constraint predicates),
[constraints.md](constraints.md) (constraint taxonomy + failure-mode grounding),
[novelty_vs_create.md](novelty_vs_create.md) (moat vs CREATE).

## 1. The key scientific question (the dependent variable)

> **How does within-path novelty (and set diversity) modulate as a function of constraint
> type?**

Not "degrade" — *modulate*: some constraints may raise novelty, some lower it; the
per-type pattern is the empirical finding, **not predicted in advance** (the
novelty–utility tradeoff is already well established in the computational-creativity
literature; which constraints move novelty which way is exactly what we measure).

**Measured relative to what is structurally possible.** When novelty shifts under
constraint `t`, decompose it:

- **Structural** — `t` mechanically changes the feasible-path set (novelty was
  impossible/forced). Model-free: enumerate valid paths under `t` on `G_c`
  (`graph.py::enumerate_paths`, already built) and score their `R`/`D` distribution.
- **Model** — the model's *realized* novelty under `t`.

Per constraint type, within matched bundles: `ΔR_struct(t)` vs `ΔR_model(t)`. The **gap**
`ΔR_struct − ΔR_model` is the model's *avoidable* creativity loss (the ideation–execution
gap, quantified). Because the model answers the open KG, `G_c`'s feasible set is a **lower
bound** on achievable novelty — if a novel valid path exists in `G_c` and the model doesn't
produce it, there is no excuse.

**Design consequences** (locked): matched endpoint-bundles (fix `(u,v,h)`, toggle only the
constraint); **difficulty-matching** holds the *quantity* of feasible-path pruning constant
across types (matched feasible-fraction) so ΔR reflects *which* paths `t` removes (high- vs
low-novelty), not *how many*; `D` compared at matched path-count `k`.

## 2. Scoring

### 2.1 Novelty `R` — within-path, embedding distance

Within-path novelty = semantic remoteness of the concepts the path combines (DAT lineage).

$$R(P)=\frac{1}{\binom{h+1}{2}}\sum_{0\le i<j\le h}\big(1-\cos(\phi(x_i),\phi(x_j))\big)$$

**Unit of embedding — a post-hoc ablation, not a pre-commit.** `x_i` is either the bare
concept `(c,c')` or the full triple `(c,r,c')` embedded as a sentence. This is a *scoring*
choice computed downstream on the parsed paths, so it does **not** affect elicitation — we
collect paths once and compute `R` under both. Current lean: **`(c,r,c')` triples** as the
headline, `(c,c')` concepts as the ablation (decided at analysis time from the numbers).
Reporting the finding under *both* is a robustness check: if the per-constraint
novelty-modulation holds under both, it isn't an embedding artifact; if it flips, we need to
know before claiming anything. Axes do **not** need to be orthogonal — we are not treating
"concepts→novelty, relations→utility" as a hard rule.

- `φ` = SBERT (`all-mpnet-base-v2` for concepts / triple-sentences); GloVe/FastText as the
  embedding ablation.
- `u,v` fixed per prompt → the `u`–`v` pair is a constant; optionally drop it so `R` is
  driven by the intermediates where it varies.
- Full ablation grid: unit `{(c,c'), (c,r,c')}` × embedding `{SBERT, GloVe, FastText}` ×
  disambiguated-label `{yes, no}` — all post-hoc.

### 2.2 Diversity `D` — between-path

Set-level distinctness over the `k` returned valid paths — **reuse CREATE's** path-string
(triple-sentence) SBERT embedding + pairwise cosine distance (`creative_utility.py`).
Separate from `R`: `R` = within-path concept spread, `D` = between-path route difference.
Compare at matched `k` (and against the structural feasible-set reference — waypoint
mechanically forces shared `w`, so some D-drop is expected, not a model failure).

### 2.3 Utility `U` — leveled (gate → gate → graded)

$$U(P;x)=\underbrace{[\text{coherent}]}_{\text{L0}}\cdot\underbrace{[\text{factual}]}_{\text{L1}}\cdot\underbrace{\Big(\prod_t (1+\alpha_t n_t)\Big)\,[\text{all }K\text{ satisfied}]}_{\text{L2+}}$$

- **Level 0 — coherence** (exact): well-formed path — endpoints `u,v`, hop count `h`,
  consecutive triples share an entity, node-distinct.
- **Level 1 — factuality** (judge): every triple true (CREATE's gpt-oss-120b).
- **Level 2+ — constraints** (graded): the imposed constraints (inclusion / exclusion /
  categorical / waypoint / …) satisfied, weighted by load `∏_t(1+α_t n_t)`.

Constraints in `K` are **hard** (any violation → `U=0`, not partial credit); the multiplier
rewards the *load carried*. Maps onto the ideation–execution split: **L0–L1 = execution
floor; L2+ = the constraint `sat` axis.** In the per-constraint headline (one constraint per
matched prompt) utility collapses to the binary `sat(t) = coherent ∧ factual ∧
that-constraint-satisfied`; the weighted product is for the secondary aggregate `C`.

### 2.4 Logging requirement (load-bearing)

Persist the **full parsed path with both entities and relations** (CREATE's triple format
already does this). Then every novelty definition, embedding, and the structural reference
are all recoverable post-hoc — nothing about the scoring choices is load-bearing at
collection time. Save raw model responses immediately (never-waste-API-spend), resumable.

## 3. Extending CREATE

### 3.1 What their code is

Nine files. **Their evaluation is a post-hoc scorer on a predictions file** (`{query,
path_prediction: list[str]}`), decoupled from elicitation → **our elicited outputs are
drop-in, and our scoring imports theirs.**

- `prompt.py` — base prompt; connect `entity_a`→`entity_b`, JSON triples in `<answer>`
  tags, relations "1–3 words" (**free-form**). Already imposes one constraint (final
  `rel_b`) and *excludes it from strength scoring* — the germ of "prompt-imposed constraint
  ≠ model novelty."
- `path_evaluator.py` — `Path.parse_path_from_text()` (robust messy-JSON→triples) and
  `check_path_validity()` (structural: endpoints + continuity, substring-based).
- `prompt_bank.py` — `TRIPLE_FACTUAL_CHECKING_PROMPT` (per-triple hallucinated/not),
  `CLASS_SIZE_PROMPT` (specificity σ).
- `creative_utility.py` — SBERT path embeddings → cosine distance → `saturating_drop`
  nonlinearity → greedy quality×diversity → patience-discounted (`γ=0.9`) utility `s(U)`.
- `inference.py` — thin litellm generator.

### 3.2 Reuse / adapt / replace / add

| CREATE module | Move | Why |
|---|---|---|
| `Path.parse_path_from_text()` | **reuse verbatim** | robust parser for messy model JSON |
| `check_path_validity()` | **reuse** as `wf` structural check | endpoints + continuity |
| `TRIPLE_FACTUAL_CHECKING_PROMPT` + `get_factuality` | **reuse verbatim** | the gpt-oss-120b factuality judge |
| `creative_utility.*` (SBERT, greedy, patience) | **reuse** for `s(U)` | our secondary `C` + no-constraint baseline |
| `prompt.get_prompt()` | **adapt** — inject constraint block `K` + controlled relation vocab; keep `<answer>` JSON | diversity/dedup prose already helps `D` |
| `inference.py` litellm | **replace for elicitation** with our OpenRouter runner (budget/resume); **keep litellm for the judge** | need budget cap + resumability; judge stays identical for comparability |
| `CLASS_SIZE_PROMPT` / σ | **replace** with embedding remoteness `R`; keep σ as optional secondary "informativeness" | `R` is judge-free, DAT-lineage |
| — | **add: constraint checker** (predicates on parsed path) | the contribution |
| — | **add: matched-bundle sampler** (needs `graph.py` + Wikidata builder) | replaces their (relation,category) endpoint sampling |
| — | **add: aggregator** → per-constraint `R`/`D`/`sat` + failure channels + modulation profile | the headline |

### 3.3 Data flow (ours = superset of theirs)

```
CREATE:  query → (their inference) → path_prediction.jsonl → parse → {structural, σ, factual} → s(U)
OURS:    bundle(u,v,h,K,relvocab) → (our OpenRouter runner) → path_prediction.jsonl  [SAME SCHEMA]
            └ parse (theirs) → { structural (theirs) + factual (their judge)
                                + constraint-sat (NEW) + novelty R (dat_eval) + diversity D (theirs) }
               → per-constraint novelty/diversity modulation vs structural reference (NEW)
               + s(U) baseline (theirs, secondary)
```

Because the predictions schema is identical, **CREATE runs as the no-constraint cell of
every bundle with its own metric** — an apples-to-apples baseline for free.

### 3.4 Deliberate deviations (design decisions, not oversights)

1. **Controlled relation vocabulary** (vs their free-form) — makes relation-level
   constraints exact; generalizes their `rel_b`-excluded-from-scoring principle. *Ablate*
   free-form vs controlled to bound the side-effect.
2. **Novelty = entity/triple embedding remoteness `R`** (vs judge-graded σ) — judge-free.
3. **Full constraint set `K`** exactly checked (vs their single `rel_b`).
4. **Our elicitation runner** (OpenRouter, budget, resume); **judge untouched** for
   comparability.
5. **Matched-bundle sampling** (vs their relation-category endpoint sampling).

### 3.5 Vendoring + license

Their *dataset* is CC BY-SA 4.0; the *code* has **no license file**. **Resolved 2026-07-05:**
the author gave permission to reuse the code, so it is **vendored** to
`src/kg_creat/vendor/create/` (`path_evaluator`, `prompt_bank`, `creative_utility`,
`inference`, `prompt`) with an attribution `NOTICE.md`; only intra-package imports were made
relative, logic unmodified for comparability. `src/kg_creat/parse.py` bridges CREATE's
`Path.parse_path_from_text` → our `EmittedPath`.

## 4. Where we improve on CREATE for combinatorial creativity

CREATE red-teamed against the creativity definition (novel + useful + surprising):

- **Their "novelty" is specificity, not remoteness.** σ = decreasing in relation
  class-size, so `A —spouse→ B` (class ≈1) scores *maximally strong* — the most obvious
  connection. CC novelty = distance between combined concepts (Mednick remote association;
  the analogy/blending tradition). → `R`.
- **`min`-over-triples is a bottleneck** that punishes compositional structure. → whole-set
  mean-pairwise.
- **No surprise term** (specific ≠ unexpected). Considered adding a non-obviousness /
  endpoint-remoteness term; **dropped for v1** — sticking with within-path novelty only.
- **Endpoint sampling has obvious-connection bias** (within-class pairs). → sample /
  control endpoint remoteness.
- **Utility = factuality + specificity can't express the novelty–utility tradeoff** that
  *defines* CC — no constraint lever. → constraints (our core).
- **Diversity over verbalized strings**, not concepts; hand-tuned `saturating_drop`.
- **Creativity entangled with capability** → report solve-rate separately (comb_eval
  decoupling).

**Keep from CREATE** (don't reinvent): open-KG + judge factuality, greedy quality×diversity
aggregation, the parser + judge, and σ as a *secondary* informativeness signal (subsume,
don't discard).

## 5. Risks / what could break, and build order

- **Format compliance** — invalid JSON / out-of-vocab relations = a *format* failure
  channel, **not** a creativity signal (separate it, like `comb_eval/scoring.py` does).
- **Alias matching** for named waypoint/hub — use `G_c` labels+aliases; judge fallback.
- **Judge reliability** — factuality precision 0.52 on incorrect relations; human
  spot-check is a required result. Constraint-sat is exact (judge-free) — only factuality is
  noisy.
- **Controlled-vocab side-effect** — ablate free-form vs controlled.
- **Cost** — frontier models × bundles × `k` × judge calls; budget cap; sequence KGs.

**Build order (scorer-first, network-free):** (1) vendor `create_compat.py` (parser + judge
+ utility), (2) add the constraint checker + `R`, (3) validate on hand-written toy
predictions (known-valid / known-violating). Then the Wikidata builder + matched-bundle
sampler feeds it real prompts. Rationale: the scorer is testable without spending on
elicitation — same discipline as `graph.py`.
