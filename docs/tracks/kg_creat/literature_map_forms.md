# Literature map: the three forms of combinatorial creativity

Reference landscape for **remote association**, **analogy-making**, and **conceptual blending** — the
three tasks in the benchmark. Compiled 2026-08-11 from targeted web searches this session plus our
earlier `related_work.md`. Purpose: back the task motivations with real citations (support *and*
refute), and pinpoint the unoccupied cell each task claims.

**Confidence:** ✓ = verified by search this session (title/authors/venue seen); ~ = from memory, verify
before it enters the paper. Candidate BibTeX keys in `[brackets]`.

**The through-line for all three:** the field is dominated by *recognition/completion* over *curated*
items, or *generation* where the model picks a convenient partner / draws on validated bases. None ask
a model to **generate** structure between **two arbitrary, non-validated** entities and score the
**emergent inferences** it licenses. That compound is our cell.

---

## 1. Remote Association

### Cognitive foundations
- ✓ **Mednick 1962**, "The associative basis of the creative process," *Psychological Review* — remote
  association; the Remote Associates Test (RAT); distant associations → more creative. `[associative_basis]`
- ✓ **Benedek, Kenett & Beaty 2023**, "Associative thinking at the core of creativity," *Trends in
  Cognitive Sciences* — modern associative theory; creativity as search over a semantic-memory network. `[benedek2023associative]`
- ✓ **Kenett et al. 2014**, "Investigating the structure of semantic networks in low and high creative
  persons," *Frontiers in Human Neuroscience* — creative people have more flexible semantic networks. `[kenett2014networks]`
- ✓ **Gray et al. 2019**, "Forward flow: A new measure to quantify free thought and predict creativity,"
  *American Psychologist* — mean pairwise distance of chained associates. `[forward_flow]` (in bib)

### Measures / human tasks
- ~ **RAT** (Mednick 1968 norms); computational solvers: comRAT-C (~Oltețeanu & Falomir 2015); RAT as
  semantic-memory retrieval (~Kumar, Schatz, Kenett 2022, *Cognitive Science*).
- ✓ **DAT — Divergent Association Task** (Olson et al. 2021, *PNAS*): name 10 maximally unrelated nouns;
  score = mean pairwise embedding distance. `[olson2021dat]`
- ✓ **DRAT — Divergent Remote Association Test**: RAT×DAT hybrid; first significant predictor of
  *scientific ideation* (from the "Assessing the Creativity of LLMs" line). `[drat]`
- ~ Free-association norms: Small World of Words (De Deyne et al. 2019, *Behavior Research Methods*);
  USF norms (Nelson et al.).

### LLM work
- ✓ LLMs on RAT / divergent semantic association: "Probing the 'Creativity' of LLMs: Can Models Produce
  Divergent Semantic Association?" (arXiv:2310.11158). `[probing_divergent_2023]`
- ✓ "Assessing the Creativity of LLMs: Testing, Limits, and New Frontiers" (arXiv:2605.13450) — DAT/RAT
  battery; no single test predicts scientific ideation. `[assessing_llm_creativity]`
- ✓ **Organisciak et al. 2023**, "Beyond semantic distance: Automated scoring of divergent thinking
  greatly improves with LLMs," *Thinking Skills and Creativity*. `[organisciak2023beyond]`
- ✓ Human–LLM divergent-creativity comparisons: *Sci. Reports* 2025 (arXiv:2405.13012); "Human Creativity
  in the Age of LLMs" (arXiv:2410.03703). `[divergent_humans_llms]`
- ✓ **AssoCiAm** (arXiv:2509.14171) — association-thinking benchmark that circumvents ambiguity. `[associam]`

### Generation over entities (our neighborhood)
- ✓ **CREATE** (Wadhwa et al. 2026) — associative creativity via multi-hop Wikidata paths between
  entities; unified creative-utility metric (quality×diversity). **Our direct baseline.** `[wadhwa2026createtestingllmsassociative]`
- ✓ **Roll the Dice** (Nagarajan et al., ICML 2025) — *sibling-/triangle-discovery* graph tasks; NTP's
  creative limits. (SD/TD in our comparison table.) `[roll_the_dice]`
- ✓ Science-of-science: atypical/distant combinations drive impact (Uzzi et al. 2013, *Science*;
  `surprising_combinations`, `fang2025generalization`).

### Our cell
Generate **non-obvious connection paths** between **arbitrary, non-connectivity-filtered** entities,
under **inclusion/exclusion constraints** (directed association), scored by **emergent inferences**.
CREATE is closest but pre-filters for connectivity and scores path existence + diversity, not emergent
novelty. Human RAT/DAT/forward-flow measure *within-person* association, not open generation over fixed
arbitrary pairs.

---

## 2. Analogy-Making

### Cognitive-science computational models (foundations)
- ✓ **Gentner 1983**, "Structure-mapping: A theoretical framework for analogy," *Cognitive Science*. `[gentner1983]` (in bib)
- ✓ **SME** — Falkenhainer, Forbus & Gentner 1989, "The Structure-Mapping Engine," *Artificial
  Intelligence*. `[falkenhainer1989sme]`
- ~ **ACME** (Holyoak & Thagard 1989, constraint satisfaction); **Copycat** (Hofstadter & Mitchell 1994,
  letter-strings); **LISA** (Hummel & Holyoak 1997/2003, inference + schema induction); DORA, AMBR.
- ✓ **Neural Analogical Matching** (Crouse et al. 2020, arXiv:2004.03573) — neural SME.
- ✓ Review: Gentner & Forbus 2011, "Computational models of analogy," *WIREs Cognitive Science*.

### Recognition / completion benchmarks (most of the field)
- ~ Word/proportional (A:B::C:D): Google analogies (Mikolov et al. 2013 `[mikolov2013]`), **BATS**
  (Gladkova et al. 2016 `[gladkova2016bats]`), SAT (Turney 2005; Ushio et al. 2021), WordRep, MSR.
- ~ Relational similarity: SemEval-2012 Task 2 (Jurgens & Turney).
- ✓ **E-KAR** (Chen et al., ACL Findings 2022) — explainable, knowledge-intensive analogical reasoning
  with rationales. `[chen2022ekar]`
- ✓ **ANALOGICAL** (Wijesiriwardene et al. 2023, arXiv:2305.05050) — 6-level long-text taxonomy. `[analogical2023]`

### Structured / narrative analogy (closest recent neighbors)
- ✓ **ARN — Analogical Reasoning on Narratives** (Sultan & Shahaf, *TACL* 2024, arXiv:2310.00996) —
  *system* analogies between stories, not word pairs. `[arn2024]`
- ✓ **SCAR** — "Beneath Surface Similarity: LLMs Make Reasonable Scientific Analogies after Structure
  Abduction" (Yuan et al., EMNLP Findings 2023) — **400 scientific analogies, 13 fields**; GPT-4
  struggles on deep structure. `[scar2023]`

### Do LLMs *really* reason analogically? (the pro/con debate — good to cite both)
- ✓ **PRO — Webb, Holyoak & Lu 2023**, "Emergent analogical reasoning in large language models," *Nature
  Human Behaviour* 7(9):1526–1541. `[webb2023emergent]`
- ✓ **CON — Lewis & Mitchell 2024**, "Using counterfactual tasks to evaluate the generality of
  analogical reasoning in LLMs" (arXiv:2402.08955) — performance collapses on counterfactual analogies →
  surface reliance, not abstraction. `[lewis2024counterfactual]`
- ✓ Hodel & West response (arXiv:2308.16118); "The Curious Case of Analogies" (arXiv:2511.20344).

### Analogy as a *reasoning method* (not an eval target)
- ✓ **Yasunaga et al. 2024**, "Large Language Models as Analogical Reasoners," *ICLR* (arXiv:2310.01714)
  — self-generated analogical exemplars beat CoT. `[yasunaga2024analogical]`

### Generation / creativity / scientific discovery (our neighborhood)
- ✓ Bhavya et al. 2022 — analogy generation by prompting (model picks the source). `[bhavya2022analogy]`
- ✓ **Cambridge *Design Science*** co-creative design-ideation framework + LLM benchmarking. `[designscience_analogy]`
- ✓ Ding et al. 2023, "Fluid Transformers and Creative Analogies" (*C&C*) — cross-domain analogical
  creativity augmentation. `[ding2023fluid]`
- ✓ "On the Diversity of Analogy Making in LLMs" (arXiv:2608.03233). `[diversity_analogy_2026]`
- ✓ LacMaterial (arXiv:2510.22312) — LLMs as analogical chemists. `[lacmaterial]`
- ✓ **ANALOGYKB** (Yuan & Chen, ACL 2024, arXiv:2305.05994) — million-scale analogy KB mined from KGs;
  recognition **+** generation, but analogies are **validated/extracted**. `[yuan2024analogykb]`
- ✓ Shen et al. 2026 — "Unlocking LLM Creativity in Science through Analogical Reasoning." `[shen2026unlockingllmcreativityscience]` (in bib)
- ✓ Survey: "Modelling Analogies and Analogical Reasoning: Connecting Cognitive Science ↔ NLP"
  (arXiv:2509.09381). `[analogy_survey_2025]`

### Our cell
**Generate** an analogy (a shared relational skeleton) between **two arbitrary, fixed, non-validated**
entities — neither the source given (unlike Bhavya/design-ideation) nor drawn from a validated base
(unlike ANALOGYKB), and generation not recognition (unlike Mikolov→ANALOGICAL, ARN, SCAR). Scored by the
inferences the mapping licenses by transfer.

---

## 3. Conceptual Blending

### Cognitive foundations
- ✓ **Fauconnier & Turner 2002**, *The Way We Think: Conceptual Blending and the Mind's Hidden
  Complexities* — two input spaces → blended space with **emergent structure**. `[the_way_we_think]` (in bib)
- ✓ **Koestler 1964**, *The Act of Creation* — *bisociation*: collision of two self-consistent but
  incompatible frames. `[koestler_creation]` (in bib)
- ✓ Conceptual combination (noun–noun compounds, emergent features): **Hampton 1987** `[hampton1987]`,
  **Wisniewski 1997** `[wisniewski1997]` (in bib) — see also our `emergent novelty` discussion.
- Antanaclasis (rhetoric): one word, two genuine senses — the minimal linguistic blend.

### Computational conceptual blending (systems, pre-LLM)
- ✓ Goguen — category-theoretic account of blending; ~Divago (Pereira & Cardoso).
- ✓ **COINVENT** project — computational concept-invention theory (Confalonieri, Schorlemmer,
  Kutz, Eppe, Plaza, et al.); amalgams (Ontañón & Plaza). `[coinvent]`
- ✓ Confalonieri et al. 2017/2020, "A computational framework for conceptual blending" / "A uniform
  model of computational conceptual blending," *Artificial Intelligence* / *Cognitive Systems Research*.
  `[concept_blending_comp_fr]` (in bib)

### Puns / antanaclasis — the linguistic neighbor
Homographic (same spelling, our antanaclasis mechanism) vs homophonic (same sound) puns.
- ✓ **SemEval-2017 Task 7** — pun detection / location / interpretation. `[semeval2017task7]`
- ✓ **CLEF JOKER** workshop — multilingual pun detection/location/interpretation/translation. `[joker]`
- Generation: ✓ Yu et al. 2018 (neural pun generation) `[yu2018pun]`; Yu et al. 2020 (constrained
  rewriting); AmbiPun (Mittal et al. 2022, arXiv:2205.01825); "Are U a Joke Master?" (ACL Findings 2024).
- Understanding (LLMs): ✓ **Xu et al. 2024**, "'A good pun is its own reword': Can LLMs Understand
  Puns?" *EMNLP* `[xu2024puns]`; "Pun Unintended: LLMs and the Illusion of Humor Understanding"
  (arXiv:2509.12158); "Comparing Apples to Oranges" (arXiv:2507.13335); Survey of Pun Generation
  (arXiv:2507.04793).

### LLM conceptual blending (direct neighbors)
- ✓ **PopBlends** (Wang et al., *CHI* 2023, arXiv:2111.04920) — LLM pipeline for pop-culture conceptual
  blends (divergent→convergent). `[popblends]`
- ✓ **"Conceptual blending in humans and language models"** (*Frontiers in Psychology* 2026) — direct
  human-vs-LM blending comparison. `[blending_humans_lms_2026]`
- ✓ BILLY (arXiv:2510.10157) — merging persona vectors for creative generation. `[billy]`

### Our cell
**Antanaclasis**: given **one arbitrary anchor**, discover a *valid second sense* of the same
orthographic word (a homograph) and build a structured blend, scored by inferences licensed only under
both senses. No large-scale conceptual-blending-*as-creativity* benchmark exists; puns are framed as
**humor**, not blending/creativity (Xu 2024; JOKER); PopBlends/Frontiers are small-scale or free-form.
Abstention is a signal (many anchors have no second sense) — absent from the pun literature.

---

## Citation shortlist for the task motivations (tight, in-text)

- **Association:** `associative_basis` (distance→creativity), `wadhwa2026...` (CREATE baseline),
  `benedek2023associative` (associative theory).
- **Analogy:** `mikolov2013, gladkova2016bats, chen2022ekar` (recognition) · `webb2023emergent` vs
  `lewis2024counterfactual` (the reasoning debate) · `gentner1983` (structure-mapping) ·
  `yuan2024analogykb` (validated-base generation we improve on). SCAR/ARN for a "closest neighbor"
  contrast in Related Work.
- **Blending:** `the_way_we_think, koestler_creation` (foundations) · `hampton1987, wisniewski1997`
  (emergent features) · `xu2024puns` + `semeval2017task7` (pun/antanaclasis neighbor) · `popblends`
  (LLM blending). `lakoff1980` for everyday naming/metaphor.

## Bib status
In bib already: `associative_basis, forward_flow, gentner1983, lakoff1980, the_way_we_think,
koestler_creation, hampton1987, wisniewski1997, concept_blending_comp_fr, wadhwa2026...,
roll_the_dice, shen2026..., surprising_combinations, fang2025...`.
**Need to add** (verified this session): `webb2023emergent, lewis2024counterfactual, mikolov2013,
gladkova2016bats, chen2022ekar, arn2024, scar2023, yuan2024analogykb, olson2021dat, assessing_llm_creativity,
organisciak2023beyond, assassociam, xu2024puns, semeval2017task7, popblends, yasunaga2024analogical`.
Verify ~-marked details (years/venues) before insertion.
