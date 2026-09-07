# Anagram discovery as exploratory creativity — a 14-model study

*2026-07-29 · kg_creat track · judge-free evaluation*

## What this is

The combinatorial-creativity benchmark (analogy, blending, remote association) asks a model to
build a novel *structure between two given entities*. Anagram discovery is its exploratory-creativity
counterpart: a single stimulus, a fixed and finite search space (the letters), and the creative act
is to **explore** that space and surface a rearrangement whose meaning is far from the source. We run
it here as a probe of a different faculty than the combinatorial tasks — and because it is the
framework's only **fully objective, judge-free** task.

Each model is given one word or name and asked to produce *as many distinct anagrams as it can*, most
semantically distant first (the prompt is open-ended — no fixed count). Every candidate is scored
deterministically, with no LLM judge anywhere:

1. **Valid anagram** — uses *exactly* the source's letters (a multiset test, case- and
   space-insensitive). This is the hard floor and it catches the character-level errors LLMs produce.
2. **Meaningful** — every whitespace token is a real word (pyspellchecker lexicon), or the whole
   string is a real named entity (Wikidata fallback, for brands/places/people the lexicon lacks).
3. **Novelty** — cosine distance between the source's and the candidate's sentence embedding: the
   *same* semantic-remoteness measure used by the combinatorial tasks. A distant, valid, meaningful
   anagram is the creative goal.

`utility = valid AND meaningful`; `yield` = usable anagrams per draw.

### Sample frame

| | |
|---|---|
| Anchors | **150** arbitrary G_c entities, 4–13 letters, deduplicated by letter-multiset |
| Domains (reference-only, never shown) | geography 47, music 16, politics 16, art 14, film 14, religion 9, science 9, literature 8, math 5, medicine 5, sports 4, econ 2, phil 1 |
| Models | **14** (frontier → 8B open-weight) |
| Decoding | temperatures {0.7, 0.9, 1.0}, one draw each (M=1: the within-response list *is* the set) |
| Draws | 450/model (**qwen-2.5-72b: 104** — see caveats) |
| Totals | 5,939 draws · 39,058 candidates · 10,911 valid (27.9%) · 7,364 meaningful (18.9%) |
| Scoring | deterministic; **no LLM judge**; Wikidata entity fallback on |

### The exact prompt

The single template administered to every model and anchor (rendered here on the example anchor
`Cleopatra`; `'{u}'` is replaced by the anchor label, and nothing else varies). It is emitted by
`_anagram_prompt` in `src/kg_creat/prompts.py`.

```
Task: You are given ONE word or name: 'Cleopatra'. Rearrange ALL of its letters -- using every
letter exactly once, ignoring spaces and capitalization -- into a NEW, real word or phrase (an
ANAGRAM of 'Cleopatra') whose MEANING is as DIFFERENT and DISTANT from 'Cleopatra' as possible.

An anagram uses exactly the same letters as the original, in a different order. Examples (same
letters, unrelated meaning):
  "stressed" -> "desserts"   (a feeling -> a food)
  "Elvis"    -> "levis"      (a musician -> a jeans brand)
  "listen"   -> "tinsel"     (perception -> a decoration)

BE CREATIVE: the further the anagram's meaning is from 'Cleopatra' -- a completely different domain -- the
better. Every anagram must (1) be a real word or phrase, and (2) use EXACTLY the letters of 'Cleopatra':
the same letters, the same number of each, none added or dropped.

Produce as MANY DISTINCT valid anagrams as you can find, most semantically distant first -- do not stop
at a fixed number; list every one you can. If 'Cleopatra' genuinely has no valid anagram, return an empty JSON
object.

Output requirements (strict):
- Return ONLY a JSON object wrapped in <answer> and </answer> tags. No other text.
- Keys are integers starting from 1; each value is a single string (the anagram).
<answer>{"1": "desserts", "2": "..."}</answer>
```

Note the three in-prompt examples (`stressed→desserts`, `Elvis→levis`, `listen→tinsel`) — llama-3.1-8b
parrots these verbatim as answers (Finding 1 / Model personalities), which the fixed template makes
directly reproducible.

---

## Headline results

Ranked by **distinct yield** (usable anagrams per draw, after collapsing word-order permutations —
see Finding 2 for why the raw number is misleading):

| Model | draws | abstain | raw yield | **distinct yield** | valid | halluc | mean'ful | novelty |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **x-ai/grok-4.5** | 443 | 0.11 | 9.18 | **6.35** | **0.96** | 0.04 | 0.83 | 0.80 |
| deepseek-v3 | 450 | 0.00 | 1.78 | 1.49 | 0.31 | 0.69 | 0.14 | 0.76 |
| gpt-4.1 | 450 | 0.08 | 1.18 | 1.17 | 0.41 | 0.59 | 0.24 | 0.76 |
| claude-sonnet-4.6 | 450 | 0.50 | 0.94 | 0.89 | **0.94** | 0.06 | 0.76 | 0.77 |
| mistral-large-2512 | 450 | 0.00 | 0.87 | 0.80 | 0.11 | 0.89 | 0.06 | 0.74 |
| gemini-2.5-pro | 442 | 0.06 | 0.64 | 0.62 | 0.44 | 0.56 | 0.29 | 0.79 |
| gpt-4.1-mini | 450 | 0.20 | 0.57 | 0.57 | 0.35 | 0.65 | 0.17 | 0.72 |
| gemini-2.5-flash | 450 | 0.03 | 0.53 | 0.50 | 0.21 | 0.79 | 0.08 | 0.77 |
| gemini-2.5-flash-lite | 450 | 0.11 | 0.22 | 0.21 | 0.04 | 0.96 | 0.02 | 0.65 |
| gpt-4o-mini | 450 | 0.41 | 0.19 | 0.19 | 0.24 | 0.76 | 0.14 | 0.75 |
| claude-haiku-4.5 | 450 | 0.83 | 0.18 | 0.18 | 0.52 | 0.48 | 0.49 | 0.75 |
| llama-3.3-70b | 450 | 0.08 | 0.17 | 0.17 | 0.09 | 0.91 | 0.04 | 0.73 |
| qwen-2.5-72b | 104 | 0.02 | 0.09 | 0.09 | 0.03 | 0.97 | 0.01 | 0.57 |
| llama-3.1-8b | 450 | 0.23 | 0.05 | 0.05 | 0.02 | 0.98 | 0.01 | 0.77 |

Three things jump out and are developed below: **(1)** the task is brutally hard and *letter
hallucination* is the dominant failure; **(2)** grok-4.5 is a stark outlier that breaks the
precision/volume tradeoff every other model is stuck on; **(3)** temperature moves almost nothing —
validity is capability-bound — while the *novelty of valid anagrams is uniformly high*.

---

## Finding 1 — The task is hard, and hallucination is the failure mode

Across all 14 models only **27.9% of emitted "anagrams" actually use the right letters.** For 11 of
14 models the *majority* of candidates fail the letter-multiset test — they look like anagrams and
often read fluently, but the letters don't check out. The deterministic floor catches every one.
`valid_rate` spans **96% (grok) → 1.6% (llama-3.1-8b)**, a 60× spread. Validity, not novelty, is the
discriminator here.

**A taxonomy of the 28,147 invalid candidates:**

| Error | Share | What happened |
|---|--:|---|
| Both add & drop | 60% | swapped some letters out for others — `Napoleon II → "AEON PIANO"` (drops an *i* and *l*, adds an *a*) |
| Dropped letters | 28% | fluent but short a letter — `BRICS → "ribs"` (loses the *c*); `Eastern Japan → "PEASANT JANE"` (loses an *r*) |
| Added letters | 11% | one letter too many — `nude → "under"` (adds *r*); `United Nations → "stationed union"` (adds an *o*) |
| Same letters | 1% | just re-cased or re-spaced the source — `Chile → "chile"`, `Spain → "Spain"` |

The pattern is diagnostic: models reason about anagrams at the **word level** (does this look like a
plausible rearrangement?) rather than the **character level** (is the multiset exactly conserved?).
`Napoleon → "no plane"` is the archetype — semantically apt, phonetically anagram-shaped, and wrong
by one *o*. This is exactly the tokenization-level blind spot anagrams are designed to expose, and the
judge-free scorer quantifies it cleanly.

A secondary failure sits *above* the letter floor: **3,547 candidates use the right letters but aren't
real words** — valid-but-meaningless rearrangements. `Salzburg → "blurs zag"`, `Edinburgh → "brined
hug"` (both real-ish but one token fails the lexicon), `Rasheda Ali → "haledarias"`. These are genuine
anagrams that simply don't land on meaning, and the lexicon check separates them from the usable ones.

---

## Finding 2 — grok-4.5 is an outlier, and the precision/volume tradeoff

Every model except grok is stuck on a tradeoff:

- **Precision-abstainers** answer only when sure. **sonnet-4.6**: 94% valid but abstains on half of
  all anchors. **haiku-4.5**: 52% valid, abstains on **83%** — it almost always says "no anagram" and
  is right to. Their usable output is small but clean.
- **Volume-hallucinators** never abstain and spray candidates, most letter-invalid. **mistral-large**
  (0% abstention, 89% hallucination), **deepseek** (0%/69%), **gemini-flash** (3%/79%). They look
  productive and are mostly wrong.

**grok-4.5 breaks the tradeoff**: it abstains rarely (11%), stays 96% letter-valid, *and* produces
the most usable anagrams by a wide margin. It is simultaneously the most precise and the most
prolific — no other model manages both.

**But its raw yield is inflated, and this is an important caveat.** grok games the open-ended "list as
many as you can" instruction by enumerating **word-order permutations** of the same decomposition. Its
single richest draw, `The Buddha` → 42 "meaningful" anagrams, collapses to just **7 distinct
word-sets**, each listed in all six orderings:

```
The Buddha (letters a,b,d,d,e,h,h,t,u) → 42 candidates =
   {bad, he, thud} ×6      {bed, duh, hat} ×6     {bed, had, hut} ×6
   {dad, hub, the} ×6      {be, had, thud} ×6     {dub, had, the} ×6
   {bud, had, the} ×6
```

Collapsing permutations (dedup by sorted word-set) drops grok's yield from 9.18 → **6.35** (a 1.45×
spam factor); deepseek shows a milder 1.19×; every other model is ~1.0 (no spam). Even fully
corrected, grok leads the field by 4×. The lesson for the benchmark is in §"Design implications."

The genuine article is still there under the spam. grok's best coherent anagrams are excellent:
`United Nations → "sedation in nut"`, `Edinburgh → "hung bride"`, `Muslim world → "dim mull rows"`,
`Leo Tolstoy → "tolls ye too"`.

---

## Finding 3 — Temperature-invariant validity, uniformly high novelty

Averaged across models, temperature barely moves any metric:

| T | valid | raw yield | within-response diversity | novelty |
|---|--:|--:|--:|--:|
| 0.7 | 0.335 | 1.25 | 0.658 | 0.731 |
| 0.9 | 0.349 | 1.18 | 0.691 | 0.740 |
| 1.0 | 0.321 | 1.12 | 0.673 | 0.749 |

grok is flat at ~0.95 valid / ~9 yield across all three; sonnet flat at ~0.94 valid / ~50% abstention.
**Anagram validity is a capability floor, not a decoding-temperature effect** — the same
temperature/validity decoupling we saw in the combinatorial tasks. Novelty ticks up a hair with
temperature (0.731 → 0.749) but the effect is tiny.

The more interesting fact: **novelty of *usable* anagrams is uniformly high (~0.65–0.80) across every
model**, strong and weak alike. When a model can produce a valid, meaningful anagram at all, the
"make its meaning distant" instruction lands — `World Bank → "BLAND WORK"`, `Sara Dylan → "yarn salad"`,
`Mount Everest → "Mourns Tee Vet"`, `Hyde Park Gate → "get a dark hype"`. Novelty does **not**
discriminate models; validity does. (Caveat: some of the *highest* novelty scores are an artifact of
fragmentation — chopping into disconnected short words like `Anglicanism → "man in sig cal"` scores
very distant but is barely meaningful; see design implications.)

---

## Model personalities (qualitative)

- **grok-4.5** — the anagrammer. High-volume, high-precision, permutation-happy. Treats it as a
  search problem and enumerates hard.
- **claude-sonnet-4.6** — the craftsman. Abstains on half, but its output is 94% valid and stylish:
  clean two-word anagrams (`Princeton → "cornet pin"`, `World Bank → "DRANK BOWL"`, `Edinburgh → "rub
  hinged"`). Quality over quantity.
- **claude-haiku-4.5** — the cautious one. Abstains 83% of the time; when it does answer it finds
  gems (`Copley Medal → "deploy camel"`, `Anglicanism → "maniac sling"`). Well-calibrated but timid.
- **gpt-4.1 / gemini-2.5-pro** — middle of the pack: real anagrams mixed with ~55–60% letter errors.
- **mistral-large-2512 / gemini-2.5-flash / deepseek** — confident and mostly wrong: never abstain,
  70–90% letter-invalid. deepseek also permutation-spams (`Napoleon → "no el pan" / "pan no el" / "el
  no pan"…`).
- **llama-3.1-8b — the clearest failure, and instructive.** It **parrots the prompt's own few-shot
  examples**: it emitted the literal example words `desserts` / `tinsel` / `levis` / `stressed`
  **174 times** across unrelated anchors — `Napoleon → "desserts"`, `Edinburgh → "desserts"`,
  `Singapore → "tinsel"`. It isn't anagramming; it's copying the demonstration. A textbook
  weak-model instruction-following collapse.
- **qwen-2.5-72b — free-association, not anagramming.** For `Singapore` it returns
  `despising, designing, resigning, presiding, spending, singing…` — real words that *sound or feel*
  related to the source but share the wrong letters. For `Napoleon`: `despots, poodles`. This explains
  its 97% hallucination rate: it treats the task as associative retrieval, not letter rearrangement.
  (Also its smallest sample — n=104 — so read as indicative.)

---

## Error taxonomy (four distinct failure modes)

1. **Letter hallucination** (dominant, all models) — word-level plausibility overriding
   character-level conservation. `Napoleon → "no plane"`.
2. **Word salad** — correct multiset, not real words. `Salzburg → "blurs zag"`.
3. **Example parroting** (weak models) — copying the prompt's demonstrations. llama-3.1-8b →
   `"desserts"` ×many.
4. **Free-association** (qwen) — returning semantically/phonetically related *non-anagrams*.
   `Singapore → "resigning"`.

Only mode 1 is unique to anagrams; modes 3–4 are general weak-model pathologies that this task happens
to surface with unusual clarity because the ground truth is objective.

---

## Abstention and calibration

Abstention is *informative* here, not a cop-out — many anchors genuinely have no common anagram, and a
well-calibrated model should say so. The **hardest anchors** (highest abstention across all models)
are exactly the letter-hostile ones:

`Hajj` (a,h,j,j — two *j*'s, essentially impossible), `zakāt`, `Egypt` (e,g,p,t,y — one vowel),
`Muhammad` (three *m*'s), `Zurich`, `Philippines`, `Czech Republic`. The **easiest** (near-zero
abstention) are vowel-rich and short: `Bel Air`, `nude`, `Dole`, `Spain`, `Iran`, `Chile`, `World
Bank`. The gradient tracks letter tractability closely — evidence that models (the calibrated ones)
are reading the difficulty of the letter set, not guessing blindly.

**Caveat on diacritics:** several hardest anchors carry non-ASCII letters (`zakāt`, `Joseph Gauß`,
`Coyoacán`, `Göttingen`, `Vipassī Buddha`). Our letter test treats `ā`, `ß`, `á`, `ö`, `ī` as distinct
characters that must be reused exactly, which makes these anchors near-impossible and inflates their
abstention. A production version should either transliterate or exclude diacritic anchors.

---

## The gems (best coherent single-phrase anagrams)

Not fragmentations — genuine, readable, distant rearrangements:

| Source | Anagram | Model |
|---|---|---|
| World Bank | **BLAND WORK** / DRANK BOWL | sonnet, gpt-4.1, mistral |
| Mount Everest | **Mourns Tee Vet** | gpt-4o-mini |
| Hyde Park Gate | **get a dark hype** | gemini-2.5-pro |
| United Nations | **sedation in nut** | grok-4.5 |
| Sara Dylan | **yarn salad** | sonnet |
| Copley Medal | **deploy camel** | haiku |
| Edinburgh | **hung bride** | grok-4.5 |
| Princeton | **cornet pin** | sonnet |
| Leo Tolstoy | **tolls ye too** | grok-4.5 |
| Anglicanism | **maniac sling** / manic signal | haiku, gemini-pro |

---

## Design implications for the benchmark

1. **Permutation spam must be collapsed.** The open-ended "list as many as you can" instruction is
   gameable by enumerating word orderings; grok inflates its yield 1.45× this way. **Report distinct
   yield** (dedup by sorted word-set), or cap phrase length, or ask for one best answer per
   semantic target. We already compute distinct yield; make it the headline.
2. **The meaningfulness checker over-credits fragmentation.** Chopping into short real-word tokens
   (`man in sig cal`) both passes the lexicon and scores maximal novelty, gaming both axes at once.
   Options: penalize token count / require a maximum number of words, prefer single-word or
   two-word anagrams, or add a coherence check. The lexicon also admits some questionable short
   tokens (`sig`, `cal`, `hok`) — worth a stoplist.
3. **Novelty doesn't discriminate; keep it as a quality gate, not a ranking axis.** Validity + distinct
   yield rank models; novelty is uniformly high among *valid* outputs and is best used to filter out
   trivial near-source rearrangements, not to score capability.
4. **Exclude or normalize diacritic anchors** (see calibration caveat).

---

## What this says for the paper's combinatorial-vs-exploratory framing

Anagram is a clean foil for the combinatorial tasks:

- It isolates a *different* faculty. A model can be strong at combinatorial structure and weak here:
  the ranking is **not** the combinatorial ranking. grok-4.5 dominates anagrams; sonnet/haiku, strong
  on structured tasks, deliberately abstain. The exploratory axis is not redundant with the
  combinatorial one.
- It is the framework's **objective anchor** — no judge, no rubric, pure letters + lexicon + distance
  — which lets us calibrate how much of the combinatorial signal is judge-dependent.
- The shared novelty metric (embedding remoteness) carries across both, so "distant" means the same
  thing whether the artifact is an analogy, a blend, or an anagram.

The headline for the comparison section: **combinatorial creativity and exploratory creativity are
distinct capabilities** — the task that objectively separates models on letter-space exploration
produces a different leaderboard than the tasks that ask for structure between entities.

---

## Caveats

- **qwen-2.5-72b: n=104 draws** (persistent upstream provider 429s), vs 450 for others. Indicative
  only; not rank-comparable.
- **Wikidata fallback hit rate limits late in scoring**, so a handful of proper-noun anagrams for the
  last-scored models may be undercounted as "not meaningful." This can only *understate*
  meaningful_rate, never inflate it.
- **grok-4.5 (443) and gemini-2.5-pro (442)** had a few API failures; scored on successful draws.
- **Novelty > 1.0** appears for some candidates (cosine distance ranges 0–2 when embeddings
  anti-correlate); this is expected, not an error, but means "novelty" is not a clean [0,1] rate.
- **Meaningfulness checker limitations** as in Design implications §2 — fragmentation over-credit and
  a few dubious short tokens.
- Model IDs reflect what OpenRouter served on 2026-07-29; grok-4.5 routed through a very slow provider
  (66 min for 450 draws).

## Reproduce

```
python src/kg_creat/scripts/sample_anagram.py --gc data/kg_creat/gc_domains_v2 \
    --n 150 --min-letters 4 --max-letters 13 --seed 5 --out data/kg_creat/prompts_anagram
python src/kg_creat/scripts/run_anagram.py configs/kg_creat/run_anagram.yaml --overwrite
.venv_mlx/bin/python src/kg_creat/scripts/score_anagram.py \
    --responses data/kg_creat/responses_anagram --out data/kg_creat/anagram_scores
```

Artifacts: prompts `data/kg_creat/prompts_anagram/`, raw responses
`data/kg_creat/responses_anagram/`, per-candidate scores + `summary.json`
`data/kg_creat/anagram_scores/`.
