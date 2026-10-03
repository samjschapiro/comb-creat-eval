# Kombine human study: first full collection — data quality, human vs. model utility, directional originality

*2026-10-03 · kg_creat track · interim report (collection paused at 92 of 120 slots)*

**Question.** What did the first full run of the human generation study (schapiro.ai/kombine) collect, how
much of it is usable, and how do human answers compare with the 35 benchmark models on the same items?

**Answer in one paragraph.** 118 sessions were collected under the final design (with the quick check);
after screening, 76 are kept, 18 are kept but flagged as likely AI-assisted, and 24 are excluded (19 caught
by hidden trap words, 5 no effort). AI use is the main data-quality problem: about 1 in 3 sessions shows some
AI signal. Kept human answers pass the benchmark's validity check (utility) less often than the typical
model on every task, and on association less often than every model, mostly by breaking the task's rules
(chains that do not end at the target, false links). On originality, human answers are farther from model
answers than from each other on all three tasks; model answers, in turn, sit inside the human answers' region
on association and analogy but not on blending, where each group stays away from the other.

## Data size and quality (read this first)

| | count |
|---|---|
| sessions collected under the final design (quick check after each task's instructions) | **118** (131 files; 13 are duplicate uploads) |
| kept | **76** |
| kept but flagged (likely AI, suspected AI, or borderline no effort) — reported separately | **18** |
| excluded: AI use caught by hidden trap words | **19** |
| excluded: no effort (keyboard mashing, nonsense) | **5** |
| median completion time, kept sessions | 51 min |
| slots filled toward the 120-slot plan (excluded sessions' slots reopened) | 92 of 120 |

**Usable answers per item** (kept sessions, test items only; the target was 20):

| task | items | answers per item, min / median / max | total |
|---|---|---|---|
| association | 30 | 9 / 14 / 15 | 380 |
| analogy | 28 | 8 / 11 / 15 | 302 |
| blending | 28 | **3 / 8 / 11** | 206 |

Blending is thin because people skip it: of the 76 kept people, 21 answered no test blend at all and 22
answered all of them. Every blending result below rests on few answers per item.

**Recruitment.** Prolific, Cocosci Lab account, study `6abd23503d9299654c1d658e`; desktop only; reward $8.00.
Prolific shows 706 submission records in total: 557 returned (most after the slots filled, when the page
says the study is full), 21 timed out, 20 rejected (AI use, plus 3 people who submitted the completion code
in under 2 minutes without doing the study), 108 awaiting review. The Prolific rejection limit was reached
with 9 AI and 4 no-effort submissions still unrejected; they are excluded from the analysis regardless.
The 8 pilot sessions (design without the quick check; `docs/reports/2026-10-02_kg_creat_human_pilot/`)
are not included.

**What this frame is not evidence about.** One pass of 76 US/UK-English Prolific workers is not a
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
- **Likely AI (flagged, kept separate).** One trap word, or paste attempts with polished AI-style text, or
  text that appeared without typing, or repeated known AI phrasings. AI-written blends recur across
  different participants ("Chronomeal", "Noonrise", "Midday Reset", "breaks the overnight fast", "a
  scheduled refueling ..."): 50% of flagged and 25% of excluded sessions contain one, against 3% of kept.

**The quick check does not catch AI use.** Passing every quick check on the first try was more common among
excluded (71%) and flagged (67%) sessions than among kept ones (41%). It filters confusion, not cheating.
Most no-effort sessions needed at least one retry, but so did many kept ones.

## Findings

### 1. Human answers pass the validity check less often than most models

Utility is the benchmark's first gate: an association chain that is well formed, ends at the target and is
factual; an analogy whose two paths form a valid mapping (pair verdict); a blend whose shared abstract
structure the judge panel accepts. Same judges and rules as the models (3-judge panel o3, GPT-5.4,
Haiku 4.5; factuality by Haiku 4.5). Test items only.

| task | humans, kept | humans, kept + passed the Breakfast + Lunch screen | humans, flagged | 35 models: min / median / max | kept humans beat |
|---|---|---|---|---|---|
| association | 0.23 (n = 380) | 0.23 (n = 235) | 0.37 (n = 90) | 0.30 / 0.63 / 0.78 | 0 of 35 |
| analogy | 0.39 (n = 302) | 0.43 (n = 197) | 0.54 (n = 67) | 0.28 / 0.53 / 0.87 | 7 of 35 |
| blending | 0.24 (n = 206) | 0.27 (n = 152) | 0.29 (n = 59) | 0.11 / 0.41 / 0.83 | 2 of 35 |

- **Association fails on the rules.** Of the 380 kept chains, 31% fail on structure (they do not end at the
  target, or the links do not connect) and a further 45% have at least one link the factuality judge marks
  false. The models rarely break structure.
- **The pre-registered comprehension screen changes little.** 47 of the 66 kept people who answered the
  Breakfast + Lunch warm-up passed it (generic-space judge); restricting to them raises analogy and blending
  utility by 3–4 points and leaves association unchanged.
- **Flagged sessions score higher than kept ones on every task**, consistent with AI help, and a reason to
  keep them out of the main human numbers.

### 2. Directional originality: humans reach places models do not; on blending the two groups avoid each other

For an answer from group X, O(X|Y) is the benchmark's originality computed against group Y only: the mean
distance of the answer's concepts and relations to their 5 nearest neighbours among Y's answers to the same
item (MiniLM embeddings). Every reference pool is cut to the same size per item (median 38–48 element
surfaces), because a larger pool always looks closer, and an answer is never compared with its own producer
(a person's own answer; a model's other draws). Kept humans vs. all 35 models; all answers, passed or not.
95% bootstrap intervals over items.

| task | O(H\|H) | O(H\|L) | O(L\|L) | O(L\|H) |
|---|---|---|---|---|
| association (30 items) | 0.594 | 0.650 | 0.632 | 0.636 |
| analogy (28) | 0.602 | 0.644 | 0.638 | 0.645 |
| blending (28) | 0.629 | 0.663 | 0.616 | 0.654 |

| contrast | association | analogy | blending |
|---|---|---|---|
| O(H\|L) − O(H\|H): humans farther from models than from each other | +0.056 [0.049, 0.063] | +0.043 [0.035, 0.051] | +0.033 [0.027, 0.040] |
| O(L\|H) − O(L\|L): models farther from humans than from other models | +0.004 [−0.002, 0.009] | +0.007 [0.001, 0.014] | +0.038 [0.030, 0.048] |
| O(H\|H) − O(L\|L): humans more spread out than models | −0.038 [−0.049, −0.028] | −0.036 [−0.050, −0.023] | +0.014 [0.006, 0.021] |

- On every task, humans share ideas with each other that the models do not use.
- On association and analogy the asymmetry is one-way: model answers are about as close to human answers as
  to other models' answers, so the models sit inside the region humans cover, and humans also cover ground
  no model reaches.
- On blending it runs both ways: model blends are as far from human blends as human blends are from model
  blends.
- Humans are more alike than models on association and analogy, and less alike on blending, where model
  blends converge on each other.

## Cost

Judging all human answers (kept + flagged + warm-ups, 1,277 answers): **$48.25** — $47.40 on OpenRouter
(o3 $25.94, GPT-5.4 $17.16, Haiku panel $4.25, first-pass factuality $0.05) and $0.85 on the Anthropic key
(Haiku factuality rejudge). Recorded in `data/kg_creat/cost_ledger.jsonl`. Directional originality is free.

## Not yet done

- **Standard originality and the composite for humans vs. models.** Human originality was scored against the
  combined human + model pool, but the models' stored originality uses the older model-only pool;
  recomputing the models against the combined pool is free and comes before any composite comparison.
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
- Directional originality: `src/kg_creat/scripts/analyze_directional_originality.py` →
  `data/kg_creat/human_study/analysis/directional_originality.json` (per-item values included).
