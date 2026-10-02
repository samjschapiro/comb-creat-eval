# Kombine human study: pilot findings and the pre-registered comprehension screen

*2026-10-02 · kg_creat track · pilot report + pre-registration note (written before the full run)*

**Question.** Does the human generation study at https://schapiro.ai/kombine/ collect usable data, do
participants understand the three tasks, and how should non-understanding be handled in the analysis?

**Answer in one paragraph.** The platform works end to end: every finished session saved both files,
claimed and completed its slot, and showed the right completion code. Two of the five finishers used an
AI assistant and were caught by the hidden trap words; both were rejected on Prolific. Of the three
genuine finishers, one understood analogy and blending well, one partly, and one not at all, and none
produced an association chain that is true and ends exactly at the second entity. We therefore fix, in
advance of the full run, a comprehension screen based on the Breakfast + Lunch warm-up blend, and will
report every human result both with and without it.

## Data & sampling

- **Platform:** jsPsych study, source in `llm_creativity_mech_interp/src/experiments/kombine_generation/`,
  data saved to a private Google Drive folder through an Apps Script web app (responses and Prolific IDs
  in separate files). Details in that folder's README.
- **Items:** the benchmark's own 30-item set (the pairs the 35 LLMs answered). Association uses its 30
  pairs in 6 bundles of 5; analogy and blending share 28 pairs (the two pairs shown in the worked examples,
  E0/F0 and E26/F26, are excluded) in bundles of 5,5,5,5,4,4. Each session gets one association bundle,
  one analogy bundle, and a different pair bundle for blending, each block led by a warm-up (Life :: A
  journey; Breakfast + Lunch). 120 slots give 20 responses per item; slots are claimed server-side.
- **Recruitment (pilot):** Prolific, Cocosci Lab account, study `6abd23503d9299654c1d658e`; 5 places,
  desktop only, US/UK, English first language, approval rate ≥ 95%; reward $8.00, estimated 40 min.
- **Sample:** 10 people started; 5 returned (quit; Prolific does not report when); 5 finished.
  2 finishers rejected for AI use. **Genuine finishers analysed below: n = 3** (slots 3, 4, 9).
  A second pilot round of 2 places (the rejected places, refilled) was running when this was written.
- **What this frame is not evidence about.** Three people is an existence check, not an estimate of any
  rate. Nothing here says how common misunderstanding or AI use will be at n = 120; it says each happens.

## Findings

### 1. The platform works

| check | result |
|---|---|
| sessions saved (responses + identity file) | 5 of 5 |
| slots claimed and marked done | 5 of 5 |
| correct completion code | 5 of 5 |
| completion time (min) | 27, 34, 46, 51, 71 (median 46) |

Two operational problems: the median 46 minutes exceeds the 40-minute estimate (pay works out to about
$10.40/hr against a $12/hr target), and one genuine session took 71 minutes, longer than the 60-minute
slot hold, which in a busy run would let a second participant be given the same slot.

### 2. AI use is real and the traps catch it

Each item carries a visually hidden instruction to include a specific unrelated word, phrased as a task
requirement. Two finishers included their item's trap word on nearly every item, each word only on its
own item, and both triggered blocked paste attempts:

| participant | trap words present | paste attempts | example |
|---|---|---|---|
| `696953dc…` | 13 of 13 answered items | 2 | "Himalayan Walnuts → Walnut → Cigar boxes" (trap: walnut) |
| `6ab8406a…` | 11 of 15 answered items | 2 | blend names "Ribbon of Reason", "Acorn infinity", "Trumpet Grainwave" |

Their typing looked human (keystrokes, corrections, pauses), consistent with retyping an assistant's
answer, so the typing summary alone would not have flagged them. Both were rejected under Prolific's rule
against AI use; their slots (0, 8) were reopened.

### 3. Genuine participants often misunderstand the tasks

**Association: nobody produced a true chain ending exactly at the second entity.**

| person | item | response | problem |
|---|---|---|---|
| 1 | The clock → Christianity | The clock *ties to* Christianity | one link, not true; all five items alike |
| 2 | Plato → Rope | Plato *concept* Greek → *philosophy* metaphors → *language* chains → *binding* rope | relations are nouns |
| 3 | Penicillin → Networks | Penicillin *is created by* mold → *gets reported on the* news → *is broadcast on* TV | never reaches the target |
| 3 | Radio → The Milky Way | Radio *Watches* The Milky Way → *Creates* The Radio | loops back past the target |

**Analogy: the mirror error, and inventions unrelated to the mapping.**

| person | item | response | problem |
|---|---|---|---|
| 1 | Frida Kahlo :: Bob Dylan | Frida Kahlo *was a fan of* Bob Dylan / Bob Dylan *was a fan of* Frida Kahlo | links the two concepts to each other |
| 1 | all items | invention skipped every time | |
| 2 | Justice :: Electricity | invention: cows *require* grass → robot judges *require* charging | source unrelated to the analogy |
| 3 | Vaccines :: Ethics | Vaccines *protect against* sickness / Ethics *protect against* bad behavior → "behavior vaccine" | **good** |

**Blending: the shared abstract structure is the hardest idea.**

| person | item | response | problem |
|---|---|---|---|
| 1 | Breakfast + Lunch | name "Breakfast is a great start before lunch"; structure "Breakfast is a great prerequisite to Lunch" | no new concept |
| 1 | three items | skipped in 4–9 s each | |
| 2 | Christianity + Beauty | structure "the honest teachings of Jesus" | fits one input only |
| 3 | The Ten Commandments + Free will | "how people decide how to behave" → **Custom Commandments**; "a person picks their own ten rules" | **good**, though links copy the inputs |

No genuine participant used a "Both" row outside the warm-up, so double-scope blends are essentially
absent from the human pilot data.

**Usable answers exist but are concentrated:** roughly a third of the genuine answers are usable, almost
all from person 3, whose analogies match the model format closely.

## Pre-registered comprehension screen (fixed 2026-10-02, before the full run)

**Rule.** A participant's responses are included in the screened analysis only if their Breakfast + Lunch
warm-up blend (a) has a concept name, (b) has an abstract structure that the benchmark's generic-space
judge `G` accepts as fitting both inputs, and (c) has at least one link into the new concept. Judged by the
same three-judge panel and prompt used for the models; no human hand-coding.

**Applied to the pilot:** person 1 fails (no concept, structure is a sentence about order); persons 2 and
3 pass. The screen therefore removes outright misunderstanding, not weak-but-genuine attempts, and the
paper will describe it as a comprehension screen, not a creativity screen.

**Reporting.** Every human result is reported twice: all genuine participants, and screened participants,
with the exclusion rate. AI-flagged sessions are excluded from both.

**Open choices, to settle before the full run:**
1. Per-task screens instead of one: Life :: A journey as the analogy screen, Breakfast + Lunch for
   blending; association has no warm-up and would be unscreened.
2. Model symmetry: the 35 LLMs never answered the warm-ups. Either report both versions (above) or run
   the models on the two warm-ups so the same screen applies to them (70 prompts plus judging; cost to be
   estimated before running).

## Changes recommended before scaling to 120

1. Raise the slot hold from 60 to about 90 minutes (one constant in the Apps Script; needs a redeploy).
2. Re-time the study: estimate 50 minutes and adjust the reward to keep about $12/hr, or shorten it.
3. Comprehension supports: lock the last association entity to the target; reject an analogy whose left
   side names the right-hand concept or the reverse; a short practice item with feedback before the
   analogy and blending blocks.

## Reproduce / where things are

- Raw pilot files: Drive folder "Kombine study data (private)" (`responses/`, `identity/`); earlier test
  sessions archived in its "test sessions (pre-launch archive)" subfolder.
- Item bundles and slot plan: `src/kg_creat/scripts/build_human_study_items.py`.
- Slot status: the Apps Script web app URL with `?action=status` (in the study README).
