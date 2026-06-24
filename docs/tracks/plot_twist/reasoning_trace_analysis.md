# Reasoning-Trace Analysis (plot_twist)

How LLMs *reason* on their way to a plot twist, and whether inference-time knobs change it.
This documents the analyses, the data/scripts behind them, the findings, and every figure/table
they feed in the paper (`papers/pt2cb-iclr-2027`).

> TL;DR — Reasoning is **strategically monolithic** and **twist-first**: every trace (207/207)
> fixes the reveal and reverse-engineers a plot to deliver it; none explores a plot forward to see
> what twist it affords. More reasoning effort (and temperature) doesn't change this.

---

## 1. Inputs

- **Traces:** `data/plot_twist/thinking/stories/<model>/<id>__t10__r{low,medium,high}__sNN.json`
  Each record has `story`, `reasoning_trace`, `reasoning_level`, `reasoning_tokens`,
  `completion_tokens`. 9 reasoning-effort-controllable models × {low, medium, high}, fixed
  temperature (1.0) → **207 traces** with non-empty `reasoning_trace`.
- **Per-(model,level) cells:** `data/plot_twist/thinking/downstream/analysis/thinking_cells.json`
  — facet means + `tc_within` (within-model equal-weight z-composite). Produced by
  `run_thinking_analysis.py`.
- **Per-story rubric scores (this slice):** `data/plot_twist/thinking/downstream/rubric/scores/*.json`
  (surprise, coherence, …) — used for the (null) move→score test.

---

## 2. Analyses

### 2a. LLM move-coding  → `data/plot_twist/thinking/downstream/trace_moves/`
Each trace coded by **gpt-4o-mini** (temp 0, strict) on 10 boolean reasoning *moves*:

`frames_constraints, enumerates_tropes, proposes_and_rejects, reveal_first, setup_first,
seeks_max_recontextualization, checks_preservation, plans_specific_clues, outlines_structure,
picks_reveal_vehicle`.

Used for move-frequency + profile clustering. **Move frequencies (n=207):**

| move | freq | | move | freq |
|---|---|---|---|---|
| seeks_max_recontextualization | 99% | | proposes_and_rejects | 80% |
| setup_first | 98% | | enumerates_tropes | 66% |
| checks_preservation | 97% | | outlines_structure | 61% |
| plans_specific_clues | 90% | | picks_reveal_vehicle | 37% |
| frames_constraints | 86% | | **reveal_first** | **0%** |

`reveal_first`=0 / `setup_first`=98 is a coding artifact: the coder read *surface prose order*
(models describe a scene before stating the twist) — it does **not** mean the plot is the anchor
(see 2b). Clustering the binary move-profiles (KMeans) gave **no clean strategy types**
(silhouette rises monotonically with k); the only variation is cosmetic scaffolding, and the
open-vs-frontier gap (open models show *more* moves) is a **CoT-summarization artifact**
(frontier providers serve summarized CoT).

### 2b. Design-anchor classification  → `data/plot_twist/thinking/downstream/trace_anchor/`
gpt-4o-mini codes, per trace: `anchor ∈ {twist_first, plot_first, interleaved}`,
`plot_gating` (does a premise ever get to constrain/veto the twist?), `emergent_clue` (is any clue
exploited from already-written detail vs planted forward to fit the reveal?).

**Result: 207/207 (100%) `twist_first`; 0/207 `plot_gating`; 100% at every effort level.**
`emergent_clue`=73% → "retrofit with opportunistic reuse," not pure forward-planting. The classifier
is not lazily defaulting (emergent_clue is not constant), and it triangulates with 2c.

### 2c. Move-position analysis  → `data/plot_twist/thinking/downstream/move_positions/`
`make_move_positions.py` locates each move by regex in every trace, normalises the char offset to
[0,1] (0 = start, 1 = end), and records median + IQR + occurrence count
(`move_positions_stats.json`). Moves split into a **divergent (surprise-generating)** band early and
a **convergent (coherence-securing)** band late:

| move | phase | median | IQR | n |
|---|---|---|---|---|
| frame the task | framing | 0.05 | [.01,.53] | 141 |
| promise coherence | framing | 0.07 | [.01,.48] | 98 |
| list potential twists | surprise | 0.16 | [.06,.35] | 107 |
| plan setting | surprise | 0.20 | [.09,.41] | 181 |
| restate recontextualization goal | surprise | 0.22 | [.05,.56] | 174 |
| propose, reject, & finalize twist | surprise | 0.33 | [.15,.59] | 147 |
| *— divergent → convergent handoff ≈ 0.4 —* | | | | |
| plan clues to plant | coherence | 0.52 | [.25,.76] | 127 |
| choose a reveal event | coherence | 0.63 | [.39,.80] | 99 |
| verify it coheres | coherence | 0.66 | [.45,.92] | 38 |
| outline full plot | coherence | 0.67 | [.36,.89] | 121 |

**Ordering test** (per-trace, robust — not ceiling-limited): surprise-side moves precede
coherence-side moves (median pos 0.34 vs 0.52; later in **72%** of traces; Wilcoxon **p≈1e-9**).
Note **coherence is *promised* at pos 0.07 but only *verified* at pos 0.66**, and "verify it coheres"
is the **rarest** move (n=38) — *asserted, not verified*.

### 2d. Move→score test (NULL, ceiling)
Joining move-codes to per-story surprise/coherence: **no move predicts its target score** — because
this slice is at ceiling (surprise 4–5, coherence 3–5, sd≈0.5). So the surprise/coherence split is
grounded on **trace structure (ordering) and function**, NOT differential outcome effects.

### 2e. Steelman + cognitive-science framing
Models perform an *informal* SBV (Schapiro–Black–Varshney) operation — identify a load-bearing
assumption, invert it (axiom flip), prefer max-recontextualization (informal `T_mod`), aim for
hindsight-consistency (preservation) — but cut corners at two stages: search a **flat trope library**
instead of a constructed DAG; select by **gut, not `T_mod`**; **assert preservation forward** instead
of verifying it. The moves map onto divergent→convergent process taxonomies (Sawyer's 8 stages;
Mumford; Wallas; Geneplore generate→explore).

---

## 3. Inference-time knobs (do they help?)
Within-model design (mirrors the trace work). **Friedman across levels:**

- **Reasoning effort** (9 models, low/med/high): TC **p=0.90**; no facet improves (surprise 0.67,
  coherence 0.15, diversity 0.90, realism 0.49); if anything realism drifts down.
- **Sampling temperature** (0.9/1.0/1.2): **p=0.012** — 0.9≈1.0, 1.2 lowers the composite.
- **Prompting strategy**: panel currently **placeholder** (pending the real experiment).

---

## 4. Scripts

| script (`src/plot_twist/scripts/`) | produces |
|---|---|
| `make_move_positions.py` | `move_positions_stats.json` (+ a standalone strip figure, now consolidated into the table) |
| `make_effort_temp_boxplots.py` | `effort_temp_boxplots.pdf` — 3-panel (effort \| temperature \| prompting), within-model Overall z |
| `make_thinking_boxplot.py` | `thinking_overall_boxplot.pdf` (effort only; currently superseded by the combined figure) |
| `make_tc_vs_temp.py` | standalone temperature boxplot (superseded by the combined figure) |
| `make_over_time_appendix.py` | `facets_over_time.pdf`, `tc_per_org_over_time.pdf` |
| `run_thinking_analysis.py` | `thinking_cells.json` |

The **move-coding** (2a) and **design-anchor** (2b) classifiers were ad-hoc gpt-4o-mini passes
(run once, outputs cached under `trace_moves/` and `trace_anchor/`); their taxonomies/prompts are in
§2 above. Total classification spend was a few cents.

Run figures with: `cd <repo root> && PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/<script>.py`

---

## 5. Paper artifacts (`papers/pt2cb-iclr-2027`)

- **§5 `sections/05_reasoning_trace_analysis.tex`** — the section; narrative = C1 (effort doesn't
  improve TC) → C2 (closed-ended, twist-first) → bridge (open-endedness; Stanley/Lehman/Clune).
  Prose stubs currently commented; the computed numbers are embedded as comments.
- **`tables/tab_moves.tex`** — CREATE-Table-1-style: each move + a verbatim excerpt (8 models) + a
  **Pos. (median [IQR])** column + `n`, move names colour-coded by phase. (The move-position strip
  figure was consolidated into this table.)
- **`figures/fig_effort_temp.tex` + `effort_temp_boxplots.pdf`** — the inference-time-knobs figure
  in §4 Ablations (panel a = effort carries the reasoning result).

---

## 6. Caveats
- **Ceiling** on this slice → move→score causation untestable; rest on anchor + ordering + structure.
- **No human process traces** (gold stories are finished texts) → twist-first-vs-human is conceptual.
- **CoT summarization** confounds open-vs-frontier move visibility → report the robust anchor/ordering
  results; treat style-clusters as illustrative.
- **Twist-first ≠ inherently bad** (authors plan endings); the flaw is the *conjunction*: trope-library
  retrieval + zero plot-gating + asserted (not verified) preservation.
