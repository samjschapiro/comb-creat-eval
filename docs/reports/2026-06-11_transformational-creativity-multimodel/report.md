# Transformational creativity via plot twists: a 35-model study, with mode-collapse analysis

**Date:** 2026-06-11 · **Track:** plot_twist · **Status:** exploratory findings (not yet pre-registered)

**One-line:** Measuring transformational creativity as `tc = diversity × (surprise·coherence)` over plot
twists, expert humans top a 35-model leaderboard — but the gap is **diversity, not per-story
quality**, and LLMs exhibit severe, *lineage-inherited* mode collapse onto family-default character
names (Emma=Llama, Mara=GPT-5.4, Margaret=Claude) and a shared bestiary of genre-twist clichés
(AI reveal, simulation, ghost, faked death) that the human canon never uses.

---

## 1. Setup

- **Construct.** A plot twist operationalizes Boden's *transformational* creativity: a reveal that
  overturns a foundational assumption in the reader's model of the story (an axiom modification in
  the SBV graphical theory). A good twist is both **surprising** (the assumption mattered) and
  **coherent** (the earlier story still holds).
- **Metric** (paper §3): for a set of stories `T`,
  `Div(T) = mean_{i≠j}[1 − cos(f(T_i), f(T_j))]` over twist embeddings `f` (we embed each story's
  annotated *reveal* with `all-mpnet-base-v2`), and
  **`tc(T) = Div(T) · mean_i[ S(T_i) · Coh(T_i) ]`** where S=surprise, Coh=coherence (1–5).
- **Judge.** Fixed rubric, 3-LLM ensemble (claude-sonnet-4, gpt-4o, gemini-2.5-flash), per-dimension
  **median**. Inter-judge reliability on the variance-rich LLM set: ICC(2,k) 0.81–0.90, GRM marginal
  ρ≈0.87; judges differ mainly in *severity*, agree on ranking (θ↔median ρ=0.95). **Not yet
  human-validated** — the key open hole.
- **Generators.** 35 LLMs (weak → frontier), open-ended prompt ("write a story with a plot twist",
  length-matched 2,000–3,000 words), 3 temperatures × 10 samples = 30 stories/model. Generators are
  disjoint from judges (anti-circularity).
- **Human ceiling.** 18 public-domain "STRONG" twists (genuine reinterpretation), vetted from a
  35-story gold set (2 NONE + 15 borderline dropped; recorded as `twist_type` in the manifest).
- **External anchors.** AGC-Bench `mean_z` (67-dataset creativity composite, n=34 overlap) and, from
  `configs/comb_eval/benchmarks.json`, Arena CW (n=23), EQ-Bench CW (n=17), MMLU-Pro (n=19).

---

## 2. Result 1 — tc leaderboard: humans win on diversity, not quality

Expert humans rank #1 (tc=12.73). The top LLM cluster — claude-sonnet-4.5 (11.71), opus-4.6 (11.52),
sonnet-4.6 (11.12), haiku-4.5 (11.11) — sits within ~1.5. Full leaderboard: `data/plot_twist/tc/tc.json`.
Figure: [`figures/tc_breakdown.png`](figures/tc_breakdown.png).

**The dissociation (the core finding).** On the two per-story quality facets, the best LLMs **match or
beat** humans:

| facet | Expert humans | best LLM |
|---|---|---|
| Surprise | 4.22 | opus-4.5 4.27, gemini-2.5-pro 4.23 |
| Coherence | 4.94 | opus-4.6 5.00, opus-4.5/haiku 4.88 |
| Prose quality | 4.50 | opus-4.5 4.72 |
| **Diversity (Div)** | **0.609** | gemma-3-27b 0.610; **opus-4.5 only 0.483** |

So humans top `tc` almost entirely via **diversity**. The single best per-story craftsman (opus-4.5,
mean S·Coh ≈ 20.9) collapses on diversity (0.483) and drops to #10. **`tc` ≠ general creativity:**
gpt-5.4 has the highest AGC `mean_z` (+0.74) but is only #9 on `tc`; gemma-3-12b/27b punch far above
their AGC (~0). Cross-model correlation `tc`↔`mean_z` = +0.82 (not 1.0), so model-specific
twist ability is real signal beyond general creativity.

---

## 3. Result 2 — external validity: which facet is novel?

Per-facet Pearson r vs four external benchmarks (*** p<.001, ** p<.01, * p<.05):

| facet | AGC mean_z (n=34) | Arena CW (n=23) | EQ-Bench CW (n=17) | MMLU-Pro (n=19) |
|---|---|---|---|---|
| **Coherence** | +0.81*** | +0.89*** | +0.89*** | +0.86*** |
| tc | +0.82*** | +0.87*** | +0.83*** | +0.85*** |
| mean(S·Coh) | +0.79*** | +0.87*** | +0.82*** | +0.84*** |
| **Surprise** | +0.61*** | +0.62** | +0.50* | +0.63** |
| **Diversity** | +0.49** | +0.39 (ns) | +0.32 (ns) | +0.44 (ns) |

There is a clean **validity gradient**, and it maps onto the theory (coherence = preservation/value;
surprise = T_mod; diversity = breadth of transformation):

- **Coherence is a capability/fluency proxy** — strongest correlate of *every* external benchmark.
  Redundant with what Arena/EQ-Bench/MMLU-Pro already measure.
- **Surprise is partly distinctive** — consistently weaker; capability doesn't buy surprising twists.
- **Diversity is the distinctive facet** — orthogonal to per-output quality (Arena/EQ-Bench, ns) and
  to reasoning (MMLU-Pro, p=.058 borderline), yet **convergent** with the broad creativity composite
  (AGC `mean_z`, +0.49 p=.003). Discriminant + convergent validity for the diversity term.

Takeaway: the *quality* half of `tc` is largely re-measuring capability; the **novel signal lives in
surprise + diversity**, which existing per-output and capability benchmarks miss.

---

## 4. Result 3 — mode collapse in LLMs vs the human population

Unit of analysis: one model sampled 30× is *one author*; the 18 human twists are a *population of 18
authors*. We quantify how narrowly each source samples twist-space. Data: `data/plot_twist/collapse.json`.

### 4a. Name collapse — a training-lineage fingerprint
Humans use a distinct protagonist in **94%** of stories (top-name share 12%). Models collapse onto a
single default name, and **the default is inherited across a model family**:

| default name | family | models (share) |
|---|---|---|
| **Emma** | **Llama** | *all 7 Llama variants* — 3.2-3b **100%**, 4-scout 73%, 3.1-70b 73%, 3.3-70b 63%, … |
| **Mara** | **GPT-5.4** | nano **90%**, mini 67%, full 45% |
| **Margaret** | **Claude** | opus-4.5 63%, sonnet-4.6, opus-4.6 (the "Margaret Chen" we first saw in opus) |
| **Elias** | **Gemma/Google** | gemma-4-26b 93%, gemini-2.5-pro 63%, ministral 80% |

llama-3.2-3b names its protagonist "Emma" in **100%** of stories. That collapse is *inherited* across
a lineage (not a per-checkpoint quirk) is the most novel single finding here.

### 4b. The same twist on repeat
Self-duplication (fraction of a source's reveals with a near-twin, cos>0.6, among its own):

| | self-dup | self-sim | uniq-name |
|---|---|---|---|
| **Expert humans** | **0.22** | **0.39** | **0.94** |
| collapsed (granite, llama-3.2-3b, gpt-5.4-nano) | 0.93–1.00 | 0.65 | 0.03–0.13 |
| diverse (gpt-4.1, gpt-4.1-mini, nova-lite) | 0.23–0.53 | 0.40 | 0.70–0.87 |

For the worst models, nearly every story has a structural near-twin of another of its own.

### 4c. What they collapse onto: genre-twist clichés
KMeans archetypes of all reveals: **94% of human twists fall in one cluster** — *social/identity
reinterpretation* ("the man she called father wasn't"; "the necklace was paste"). LLMs spread across a
recognizable set of **tropes the human canon never uses**: "she was an AI/synthetic being",
"it was a simulation / implanted memories", "she's a ghost / dead all along", "he faked his own death",
"fractured identity orchestrated by a voice", doppelgänger/time-loop. LLM "diversity" is largely
**diversity across sci-fi/horror twist clichés**; humans write the mundane-but-literary reveal.

### 4d. Collapse is orthogonal to capability
Not weak-vs-strong: strong models collapse (opus-4.5 → Margaret 63%; the whole GPT-5.4 line → Mara)
while other strong models stay diverse (gpt-4.1 87% unique, opus-4.6 64%, deepseek-v3 67%). Some weak
models *scatter* but incoherently. This squares with **Div ⟂ MMLU-Pro** (§3) — collapse is a distinct
axis, plausibly RLHF/instruction-tuning pulling toward a narrow "safe literary default."

### 4e. The species is broad; each individual is narrow
Pooling all models covers many trope-clusters (diverse in aggregate) — but that's many *different*
narrow models stacked, each on its own attractor. No single model approaches the human population's
spread.

---

## 5. Limitations (red-teamed)

1. **Judge not human-validated** — the make-or-break number; all quantitative headlines rest on a
   3-LLM rubric. The generation prompt also teaches the rubric (mild circularity).
2. **Diversity is embedding/annotator-dependent** — reveals are gpt-4o-mini summaries embedded by one
   model; abstracting names reordered the human-vs-model diversity ranking in a prior check.
3. **No CIs on the leaderboard** — the top (humans vs the sonnet/opus cluster) is plausibly within
   noise; per-(model,temp) cells are n=10. Bootstrap CIs are the next step.
4. **Contamination + survivorship** — human gold = famous PD stories in every model's *and judge's*
   pretraining; the human ceiling is an upper bound, not a contamination-controlled comparison.
5. **Human n=18, STRONG-selected** — the archetype concentration of humans is partly our selection;
   the name-reuse and self-similarity contrasts (4a–4b) do not have this confound and are load-bearing.
6. **External correlations** at n=17–34 are now properly powered; AGC `mean_z` (n=34) is the
   trustworthy column.

---

## 6. Cost & reproducibility

- **Cumulative OpenRouter spend: ~$26.4** (ledger: `docs/tracks/plot_twist/cost_log.md`). Tier-2
  (12 models) added ~$14.6 (gen $6.1 + score $7.6 + annot $0.9).
- **Pipeline** (`src/plot_twist/`): `run_generate` → `run_rubric_stimuli` → `run_annotate` →
  `make_tc_barplot` → `analyze_collapse` / `correlate_dsi`. All per-item durable + resumable
  (stories under `data/plot_twist/llm_twists/stories/<model>/`).
- **Key artifacts:** `data/plot_twist/tc/tc.json` (leaderboard), `collapse.json` (mode-collapse
  metrics), `annotations/annotations.json` (setup/reveal/why per story), figures in `figures/`.

---

## 7. Bottom line

Frontier LLMs can write *a* human-quality plot twist (matching humans on surprise, coherence, and
prose), and `tc` tracks broad creativity benchmarks well — but that tracking is carried by the quality
term, which is mostly capability. The benchmark's distinctive signal is **diversity**, and there LLMs
fail in a specific, structured way: each model is a one-trick author that collapses onto a
**family-default character and a small set of genre-twist clichés**, inherited across its lineage and
independent of raw capability. A population of human authors stays broad and literary; a single LLM,
sampled repeatedly, does not.
