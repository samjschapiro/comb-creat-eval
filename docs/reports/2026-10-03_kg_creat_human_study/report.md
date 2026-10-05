# Kombine human study: first full collection — data quality, human vs. model utility, directional originality

*2026-10-03 · kg_creat track · interim report (collection paused at 92 of 120 slots)*

**Question.** What did the first full run of the human generation study (schapiro.ai/kombine) collect, how
much of it is usable, and how do human answers compare with the 35 benchmark models on the same items?

**Answer in one paragraph.** 118 sessions were collected under the final design (with the quick check);
after screening, 75 are kept, 19 are kept but flagged as likely AI-assisted, and 24 are excluded (19 caught
by hidden trap words, 5 no effort). AI use is the main data-quality problem: about 1 in 3 sessions shows some
AI signal. Kept human answers pass the benchmark's validity check (utility) less often than the typical
model on every task, and on association less often than every model, mostly by breaking the task's rules
(chains that do not end at the target, false links). On the paper's composite, kept humans rank last of 37
entries, yet their valid answers are more original than most models' valid answers, and less surprising.
Human answers are farther from model answers than from each other on all three tasks; model answers, in turn, sit inside the human answers' region
on association and analogy but not on blending, where each group stays away from the other.

## Data size and quality (read this first)

| | count |
|---|---|
| sessions collected under the final design (quick check after each task's instructions) | **118** (131 files; 13 are duplicate uploads) |
| kept | **75** |
| kept but flagged (likely AI, suspected AI, or borderline no effort) — reported separately | **19** |
| excluded: AI use caught by hidden trap words | **19** |
| excluded: no effort (keyboard mashing, nonsense) | **5** |
| median completion time, kept sessions | 50.5 min |
| slots filled toward the 120-slot plan (excluded sessions' slots reopened) | 92 of 120 |

**Usable answers per item** (kept sessions, test items only; the target was 20):

| task | items | answers per item, min / median / max | total |
|---|---|---|---|
| association | 30 | 9 / 14 / 15 | 375 |
| analogy | 28 | 8 / 11 / 15 | 298 |
| blending | 28 | **3 / 8 / 11** | 201 |

Blending is thin because people skip it: of the 75 kept people, 21 answered no test blend at all and 21
answered all of them. Every blending result below rests on few answers per item.

**Recruitment.** Prolific, Cocosci Lab account, study `6abd23503d9299654c1d658e`; desktop only; reward $8.00.
Prolific shows 706 submission records in total: 557 returned (most after the slots filled, when the page
says the study is full), 21 timed out, 20 rejected (AI use, plus 3 people who submitted the completion code
in under 2 minutes without doing the study), 108 awaiting review. The Prolific rejection limit was reached
with 9 AI and 4 no-effort submissions still unrejected; they are excluded from the analysis regardless.
The 8 pilot sessions (design without the quick check; `docs/reports/2026-10-02_kg_creat_human_pilot/`)
are not included.

**What this frame is not evidence about.** One pass of 75 US/UK-English Prolific workers is not a
population estimate of human creativity. Each person answered each item once, while each model answered it
up to 5 times, so every comparison here is of the typical answer, not the best one. The 35 models are pooled
when compared with humans.

## How sessions were screened

Each session was read and given one verdict; the computed signals below are stored next to the verdicts
and agree with them (no session with 2+ trap words was kept, and every trap-word exclusion has 2+).

- **Hidden trap words (exclusion).** Each item carries a hidden instruction, invisible on screen but read
  by AI assistants, to include a specific unrelated word (walnut, lantern, cactus, ...). A session is
  excluded as AI if its own item's word appears in 2 or more answers, or if it filled the hidden honeypot
  field. Excluded AI sessions had a median of 10 hits (2 to 17); 79% also tried to paste. One was a browser
  agent: no keystrokes on any item and the honeypot filled.
- **No effort (exclusion).** Keyboard mashing ("Black holes *fger* ergr") or answers unrelated to the items
  ("Vaccines :: Ethics → Soap Fights Sword").
- **Researcher review.** One session that passed every automatic check was moved to flagged after review: its
  blends reused the "a system that/in which ..." shared-structure template of the AI sessions (e.g. "a structured
  system in which smaller units combine to form a large organized whole" against a flagged session's "A system
  in which smaller parts are organized into a larger, structured whole", same item). It was the only kept
  person scoring near the top models on all three tasks. Moving a session between groups needs no rescoring
  (`regroup_human_scores.py`).
- **Likely AI (flagged, kept separate).** One trap word, or paste attempts with polished AI-style text, or
  text that appeared without typing, or repeated known AI phrasings. AI-written blends recur across
  different participants ("Chronomeal", "Noonrise", "Midday Reset", "breaks the overnight fast", "a
  scheduled refueling ..."): 47% of flagged and 25% of excluded sessions contain one, against 3% of kept.

**The quick check does not catch AI use.** Passing every quick check on the first try was more common among
excluded (71%) and flagged (68%) sessions than among kept ones (40%). It filters confusion, not cheating.
Most no-effort sessions needed at least one retry, but so did many kept ones.

## Findings

### 1. Human answers pass the validity check less often than most models

**Skipped items count as failed (0)** throughout the main analysis, for humans and models alike: an item
someone ticked "I can't think of one" for is a failed answer, not a missing one, so skipping the hard items
cannot raise a score. Kept humans skipped 203 of 1,077 assigned test answers (151 blends, 32% of all blends
assigned; 52 analogies; association has no skip box); the 35 models left 50 answers unusable between them
(`build_skip0_scores.py`). Dropping skips instead flatters humans, mostly on blending (utility 0.22 instead of
0.13, composite 13.2 instead of 7.4); that version is kept as a sensitivity check (`combined_scores/`).

Utility is the benchmark's first gate: an association chain that is well formed, ends at the target and is
factual; an analogy whose two paths form a valid mapping (pair verdict); a blend whose shared abstract
structure the judge panel accepts. Same judges and rules as the models (3-judge panel o3, GPT-5.4,
Haiku 4.5; factuality by Haiku 4.5). Test items only.

| task | humans, kept | humans, kept + passed the Breakfast + Lunch screen | humans, flagged | 35 models: min / median / max | kept humans beat |
|---|---|---|---|---|---|
| association | 0.22 (n = 375) | 0.22 (n = 230) | 0.39 (n = 95) | 0.30 / 0.63 / 0.78 | 0 of 35 |
| analogy | 0.33 (n = 350) | 0.38 (n = 214) | 0.45 (n = 87) | 0.25 / 0.50 / 0.86 | 4 of 35 |
| blending | 0.13 (n = 352) | 0.17 (n = 213) | 0.24 (n = 89) | 0.07 / 0.43 / 0.86 | 1 of 35 |

- **Association fails on the rules.** Of the 375 kept chains, 31% fail on structure (they do not end at the
  target, or the links do not connect) and a further 45% have at least one link the factuality judge marks
  false. The models rarely break structure.
- **The pre-registered comprehension screen changes little.** 46 of the 65 kept people who answered the
  Breakfast + Lunch warm-up passed it (generic-space judge); restricting to them raises analogy and blending
  utility by 4–5 points and leaves association unchanged.
- **Flagged sessions score higher than kept ones on every task**, consistent with AI help, and a reason to
  keep them out of the main human numbers.

### 2. On the paper's composite, kept humans rank last; their valid answers are more original but less surprising

All 35 models and both human groups rescored against ONE originality pool (every human and model answer to the
item), on the 86 items both answered (the human warm-ups and the two worked-example pairs dropped). The
rescoring reproduces the models' stored originality exactly when the pool is models only (19,555 values, max
difference 0). The composite is the paper's: each dimension is gated by utility (0 if the answer failed) and
expressed as a fraction of its maximum.

- **Overall: kept humans 13.7, last of 37** (phi-4 17.2; model median 36.7). Kept humans who passed the
  Breakfast + Lunch screen: 15.7, also last. Flagged humans 22.2. By task, kept humans rank 36th (association
  14.7), 34th (analogy 19.1) and 35th (blending 7.4) of 36.
- The gap comes from utility and from everything gated on it, above all the emergent steps: humans' invention
  utility and integration (analogy) and emergent utility and integration (blending) are below all but 0–2
  models, largely because people skip the invention and the "true of neither" step.

**Among answers that pass utility** (not gated), the picture splits:

| dimension (passed answers only) | humans, kept | model median | models below humans |
|---|---|---|---|
| association originality | 0.354 | 0.321 | 32 of 35 |
| analogy originality (the mapping) | 0.456 | 0.472 | 11 of 35 |
| analogy invention originality | 0.539 | 0.508 | 30 of 35 |
| blend originality (generic space) | 0.523 | 0.378 | 35 of 35 |
| blend emergent originality | 0.533 | 0.478 | 35 of 35 |
| association surprise | 0.632 | 0.668 | 3 of 35 |
| analogy surprise | 0.704 | 0.769 | 1 of 35 |
| blend surprise | 0.757 | 0.802 | 5 of 35 |

When a human answer is valid, it is more original than most models' valid answers on four of five
originality measures, and less surprising (its linked concepts sit closer together) on all three tasks.
Caveat: the shared pool is mostly model answers, so originality here is partly distance from what the models
say — the same thing finding 4 measures directly.

### 3. How performance is distributed: humans are zero-inflated and more varied, not just shifted down

Per-person scores rest on ~14 answers against ~90 per model, so a person's score is noisier for that reason
alone. Humans are therefore compared with model **pseudo-participants**: a model answering a random set of
items the same size as a real person's set (30 per model, 1,050 in all). Each group is fitted with a
zero-inflated beta (a share P(0) of exact zeros, a beta for the rest) and, for pass counts, a beta-binomial,
whose ρ is the share of variance due to real differences between units (0 = all alike), net of sampling noise.

| | P(0) humans / pseudo | mean of the non-zero part, humans / pseudo | SD humans / pseudo |
|---|---|---|---|
| Overall composite | 0.11 / 0.00 | 0.15 / 0.37 | 0.10 / 0.13 |
| association composite | 0.39 / 0.03 | 0.24 / 0.39 | 0.15 / 0.17 |
| analogy composite | 0.24 / 0.04 | 0.25 / 0.42 | 0.17 / 0.20 |
| blending composite | 0.59 / 0.10 | 0.19 / 0.37 | 0.11 / 0.20 |
| analogy invention originality | 0.39 / 0.04 | 0.28 / 0.29 | 0.17 / 0.14 |
| blend originality | 0.59 / 0.10 | 0.17 / 0.19 | 0.10 / 0.11 |

| pass rate (beta-binomial) | humans: mean, ρ | pseudo-participants: mean, ρ |
|---|---|---|
| association | 0.22, 0.14 | 0.57, 0.07 |
| analogy | 0.33, 0.12 | 0.55, 0.08 |
| blending | 0.13, 0.13 | 0.44, 0.09 |

- **Two kinds of human.** A large share of people produce no valid answer at all on a task (39% on
  association, 24% on analogy, 59% on blending), against 3–10% of model pseudo-participants. The rest form a
  continuous distribution that overlaps the models', shifted lower on composite and utility.
- **Originality is where the non-zero parts meet.** Among units with any valid answer, human and model
  originality distributions nearly coincide (analogy invention 0.28 vs 0.29; blend originality 0.17 vs 0.19);
  the composite gap is driven by zeros and by validity, not by how original valid work is.
- **People differ more than models.** Net of sampling noise, between-person variation in pass rate (ρ 0.12–0.14)
  is about twice the between-model variation at matched size (ρ 0.07–0.09).
- Every human vs. pseudo-participant distribution differs (two-sample KS p < 10⁻⁶ on all 19 metrics); per-dimension
  fits for every task in `figures/score_distributions_per_dim.png`.

**Per dimension, over answers that pass utility** (raw values, not gated; beta fits; humans kept vs. all model
answers):

| dimension | n humans / models | mean humans / models | difference [95% CI] | SD humans / models |
|---|---|---|---|---|
| association surprise | 83 / 9,386 | 0.632 / 0.664 | −0.032 [−0.049, −0.015] | 0.077 / 0.093 |
| association originality | 83 / 9,386 | 0.354 / 0.331 | +0.024 [+0.007, +0.040] | 0.078 / 0.053 |
| analogy surprise | 115 / 530 | 0.704 / 0.764 | −0.060 [−0.076, −0.045] | 0.078 / 0.076 |
| analogy originality (mapping) | 115 / 530 | 0.456 / 0.463 | −0.007 [−0.020, +0.008] | 0.073 / 0.058 |
| analogy invention originality | 88 / 530 | 0.539 / 0.507 | +0.032 [+0.018, +0.045] | 0.060 / 0.058 |
| blend surprise | 45 / 435 | 0.757 / 0.802 | −0.045 [−0.078, −0.016] | 0.099 / 0.082 |
| blend originality (generic space) | 45 / 435 | 0.523 / 0.383 | +0.139 [+0.098, +0.182] | 0.143 / 0.096 |
| blend emergent originality | 36 / 434 | 0.533 / 0.482 | +0.051 [+0.018, +0.082] | 0.095 / 0.058 |

Valid human answers are less surprising on every task, more original on four of five originality measures
(analogy mapping originality the exception, no difference), and more spread out on originality (SD up to 1.5x).
Utility, emergent utility and integration are discrete per unit (k of n), so their beta curves in the per-dimension
figure are approximate; the beta-binomial above is the right model for pass counts.

### 4. Directional originality: humans reach places models do not; on blending the two groups avoid each other

For an answer from group X, O(X|Y) is the benchmark's originality computed against group Y only: the mean
distance of the answer's concepts and relations to their 5 nearest neighbours among Y's answers to the same
item (MiniLM embeddings). Every reference pool is cut to the same size per item (median 36–48 element
surfaces), because a larger pool always looks closer, and an answer is never compared with its own producer
(a person's own answer; a model's other draws). Kept humans vs. all 35 models; all answers, passed or not.
95% bootstrap intervals over items.

| task | O(H\|H) | O(H\|L) | O(L\|L) | O(L\|H) |
|---|---|---|---|---|
| association (30 items) | 0.595 | 0.651 | 0.633 | 0.637 |
| analogy (28) | 0.604 | 0.646 | 0.639 | 0.647 |
| blending (28) | 0.634 | 0.666 | 0.618 | 0.658 |

| contrast | association | analogy | blending |
|---|---|---|---|
| O(H\|L) − O(H\|H): humans farther from models than from each other | +0.057 [0.050, 0.063] | +0.042 [0.035, 0.050] | +0.032 [0.025, 0.040] |
| O(L\|H) − O(L\|L): models farther from humans than from other models | +0.004 [−0.001, 0.010] | +0.007 [0.001, 0.014] | +0.040 [0.031, 0.048] |
| O(H\|H) − O(L\|L): humans more spread out than models | −0.038 [−0.048, −0.028] | −0.035 [−0.049, −0.022] | +0.016 [0.007, 0.024] |

- On every task, humans share ideas with each other that the models do not use.
- On association and analogy the asymmetry is one-way: model answers are about as close to human answers as
  to other models' answers, so the models sit inside the region humans cover, and humans also cover ground
  no model reaches.
- On blending it runs both ways: model blends are as far from human blends as human blends are from model
  blends.
- Humans are more alike than models on association and analogy, and less alike on blending, where model
  blends converge on each other.

### 5. What humans and LLMs say differently: everyday and human-centred vs. specific and encyclopedic

Concepts and relations of kept humans vs. the 35 LLMs on the 86 shared items (concept uses: association 783
human / 55,411 LLM; analogy 891 / 5,777; blending 771 / 5,879). Three views
(`analyze_human_llm_language.py`): per-element features, words over-used by each group (log-odds with an
informative Dirichlet prior, Monroe et al. 2008), and k-means clusters (k = 30) of concept embeddings with the
clusters where each group is over-represented.

| | word commonness (wordfreq Zipf), human / LLM | concepts with a rare word (Zipf < 3), human / LLM |
|---|---|---|
| association | 4.47 / 3.89 | 8% / 35% |
| analogy | 4.60 / 4.00 | 5% / 24% |
| blending | 4.55 / 4.09 | 11% / 21% |

- **Concepts.** LLMs reach for named and technical entities: scientists and artists (James Clerk Maxwell, Dorothy
  Hodgkin, Emil Fischer, Salvador Dalí), places (Naples, the Mogao Caves), physics (ions, event horizon, Hawking
  radiation), mechanisms (pin tumbler, quorum sensing), law and politics (public reason, informed consent).
  Humans reach for people and everyday life (people, humans, money, power, rain, football), emotion and value
  (happiness, love, the soul, death), religion and art.
- **Relations.** Humans link with simple causal and helping verbs (creates, brings, helps, allows, makes, needs),
  spatial words (inside, up) and similarity ("like" is the most human-skewed association relation; "sounds
  like" marks wordplay). LLMs link with formal and scholarly verbs (studied, influenced, developed, pioneered;
  enables, employs, shapes; via, transforms, converts, encodes).
- **Chains.** Human association chains are shorter (2.8 vs. 4.5 links) and move by everyday association and
  similarity; LLM chains walk documented facts ("X studied Y, who influenced Z").
- **Framing.** Human blends often address a person ("you", "your", "can", "helps you ..."); LLM blends describe
  mechanisms in the third person ("transforms X via Y").

**By WordNet supersense** (26 noun / 15 verb classes; Ciaramita & Johnson 2003; `analyze_supersenses.py`; first
sense of the phrase or its head word; 95% bootstrap CIs over participants and models), the differences that
exclude zero:

| task | more often human | more often LLM |
|---|---|---|
| association concepts | object (+3.0 pp: sun, rock), state (+2.4), attribute (+1.8), animal (+1.3) | not in WordNet (−6.5: names, technical terms), person (−5.2: Emil Fischer, Galileo), location (−3.6: London) |
| association relations | motion (+7.6: drives, moves, leads to), emotion (+2.3), contact (+2.1: opens, unlocks) | creation (−4.3: wrote, written by), communication (−3.8: directed, named after), consumption (−2.8), cognition (−2.5: studied by) |
| analogy concepts | state (+1.9: life, happiness, peace), feeling (+1.3: emotion, hope) | location (−1.8), plant (−1.0) |
| analogy relations | motion (+6.9: leads to, brings), no verb (+6.0) | change (−4.8: enables, regulates, converts), cognition (−3.6), communication (−3.5), possession, consumption |
| blending concepts | not in WordNet (+6.9: adjectives, odd phrases), person (+3.0: players, youth), animal (+1.5: humans) | communication (−6.3: information, testimony), act (−2.6), quantity (−1.0) |
| blending relations | stative (+7.6: is, is a), no verb (+5.2) | possession (−4.4: distributes), change (−4.0: transforms, converts) |

The supersenses agree with the word-level picture: LLMs name people and places and link through authorship,
study and transformation; humans use everyday objects, states and feelings and link through movement, being and
plain association. First-sense tagging is noisy ("like" is tagged as an emotion verb, though it is mostly used
for similarity), so single small categories should not be over-read.

Caveats: proper names count as rare words, so "specific" and "named" overlap; human word lists rest on a few
hundred uses per task, so single words are noisy (the categories recur on all three tasks); and some plainness
may be effort, since humans type their answers.

### 6. Humans produce no inventive multiples (documented only; left out of the paper by decision, 2026-10-05)

The paper's τ-inventive-multiple test, unchanged (`analyze_human_multiples.py`): an invention's properties are
"relation object" with its name dropped and anchor echoes removed; two inventions on the same item are a
τ-multiple when a one-to-one pairing of their properties at cosine ≥ θ = 0.674 has size ≥ τ. Kept humans vs. the
35 LLMs on the shared items: analogy 298 human / 967 LLM inventions, blending 201 / 965. A pair rate does not
depend on how many inventions an item has, so the three pair types compare directly. "Eligible" = both
inventions carry ≥ τ properties; "per property" = shared / the smaller property count.

| | pairs | τ = 1 | τ = 2 | eligible τ = 2 | per property |
|---|---|---|---|---|---|
| blending, human–human | 675 | 3 (0.44%) | **0** | 0 of 642 | 0.17% |
| blending, human–LLM | 6,928 | 95 (1.37%) | **0** | 0 of 6,755 | 0.46% |
| blending, LLM–LLM | 16,154 | 2,973 (18.4%) | 358 (2.22%) | 2.22% | 4.51% |
| analogy, human–human | 1,509 | 1 (0.07%) | 0 | 0 of 3 | 0.16% |
| analogy, human–LLM | 10,307 | 5 (0.05%) | 0 | 0 of 463 | 0.08% |
| analogy, LLM–LLM | 16,221 | 369 (2.27%) | 10 (0.06%) | 0.10% | 1.29% |

- **Blending is the clean test.** Human blends carry 2.8 properties (LLMs 5.0) and 642 of 675 human pairs could
  qualify at τ = 2. At the LLM–LLM rate about 14 human–human and 150 human–LLM multiples would be expected; there
  are none. Even single shared properties are rare: 0.44% of human pairs against 18.4% of LLM pairs.
- **Analogy cannot really test τ = 2.** Humans wrote one projected property (182 of 298) or none (98), so only 3
  human pairs carry two properties each. At τ = 1, among eligible pairs, humans share a property 0.16% of the
  time against 2.34% for LLMs.
- **Utility-passed answers only:** the same, no human multiple at τ = 2 on either task (human–LLM blending τ = 1
  rises to 3.1%, still far below LLM–LLM 18.5%).
- **What a human–LLM single match looks like:** on Blues + Key, "Expression ... unlocks feelings" against
  several models' "Blues Key ... unlocks hidden emotions"; on analogy E25, "conscience guides personal decisions"
  against GPT-4o-mini's "choice compass guides personal choices". These are the shared stock image of the item,
  one property deep.

So convergence on the same invention is an LLM phenomenon here: humans neither converge with each other nor
with models. Caveats: an item has 3–11 human blends, so a human multiple needs two people on the same item, but
the pair rate already accounts for that; human properties are shorter and plainer, which may lower cosine
matches somewhat; a single human writes one answer per item, like a single model draw.

## Cost

Judging all human answers (kept + flagged + warm-ups, 1,277 answers): **$48.25** — $47.40 on OpenRouter
(o3 $25.94, GPT-5.4 $17.16, Haiku panel $4.25, first-pass factuality $0.05) and $0.85 on the Anthropic key
(Haiku factuality rejudge). Recorded in `data/kg_creat/cost_ledger.jsonl`. Directional originality is free.

## Not yet done

- **Figures for finding 5** (chosen 2026-10-05: word clouds + category bars; mocks made, not yet in the paper):
  `plot_human_llm_language.py` writes `data/kg_creat/human_study/figures/language_wordclouds.{png,pdf}` (2x2:
  human / LLM concepts and relations, whole phrases pooled over tasks, sized by log-odds z; per-panel PDFs
  `_a`-`_d`) and `supersense_bars.{png,pdf}` (2x3: concept and relation supersense shares per task, top 8
  categories by total share, * = 95% CI on the difference excludes zero). Context-aware supersense tagging
  would remove first-sense noise.
- **Utility-gated directional originality** (passed answers only); robustness to a second embedding model.
- **The remaining 28 slots.** Collection is paused; refilling them brings association and analogy to about
  18–19 answers per item, blending to about 10.
- **Cross-session repeat check** against model answers, to catch AI text retyped without trap words.

## Reproduce / where things are

- Raw responses: Drive folder "Kombine study data (private)/responses", downloaded to `data/kg_creat/responses/`
  (gitignored; no Prolific IDs). Screening verdicts: `data/kg_creat/human_study/screening_verdicts.csv`.
- Who counts, coverage: `src/kg_creat/scripts/analyze_human_study.py` → `data/kg_creat/human_study/{sessions,coverage}.csv`, `summary.json`.
- Scoring: `src/kg_creat/scripts/build_human_score_input.py`, then `score.py configs/kg_creat/human_study_score.yaml`
  (new `pool_extra_dirs` option: the models' answers join the originality pool without being rescored), then
  `rejudge_factuality_haiku.py --scores-dir data/kg_creat/human_study/scores`.
- Utility vs. models: `data/kg_creat/human_study/analysis/utility_vs_models.json` (computed from path scores,
  test items only).
- Composite on one shared pool: `src/kg_creat/scripts/rescore_originality_combined.py --pool combined` →
  `data/kg_creat/human_study/combined_scores/` (copies; the models' own score files are untouched), then
  `compute_composite.py` on it, then `analyze_human_vs_models.py` → `analysis/human_vs_models.json`.
- Skips as failures: `src/kg_creat/scripts/build_skip0_scores.py` → `combined_scores_skip0/` (main analysis).
- Distributions: `src/kg_creat/scripts/analyze_score_distributions.py` → `analysis/score_distributions.json`,
  `figures/score_distributions.png`; per-person histograms `plot_human_vs_model_hist.py`.
- Language contrast: `src/kg_creat/scripts/analyze_human_llm_language.py` → `analysis/language_contrast.json`;
  supersenses `analyze_supersenses.py` → `analysis/supersenses.json` (WordNet in `data/nltk_data`, run with
  `NLTK_DATA=data/nltk_data`).
- Directional originality: `src/kg_creat/scripts/analyze_directional_originality.py` →
  `data/kg_creat/human_study/analysis/directional_originality.json` (per-item values included).
