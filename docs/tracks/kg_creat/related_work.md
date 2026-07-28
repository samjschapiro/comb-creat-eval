# kg_creat — related work: analogy & blending benchmarks

Literature review supporting the positioning of our two structured-creativity tasks (analogy,
antanaclasis blending). Compiled 2026-07-27 from prior knowledge + targeted web searches.
**Confidence: citations marked ✓ were verified by search this session; those marked ~ are from
memory and MUST be verified (author/year/venue) before entering the paper.** This is *not* an
exhaustive survey — see "Risky cells" below for where a competitor could still hide.

## The framing that positions us

Every prior benchmark sorts on three axes; the third is the one that isolates our contribution:
1. **Recognition/completion vs GENERATION** — solve `A:B::C:?` / rate a pair, vs produce a new artifact.
2. **Curated/validated pairs vs ARBITRARY pairs** — is the analogy/blend known to exist a priori?
3. **Who chooses the two things being related** — in almost all generation work the model *picks a
   convenient partner*; only we fix **both** endpoints as arbitrary entities and demand the
   structure *between them*. This is the crux.

---

## Analogy

### Recognition / completion (most of the field)
- **Word / proportional analogies** — Google analogy set (~Mikolov 2013), BATS (~Gladkova 2016), SAT
  verbal (~Turney 2005), WordRep: solve `A:B::C:?`. Recognition, curated.
- **Relational similarity** — SemEval-2012 Task 2 (~Jurgens/Turney): rate degree of analogy. Curated.
- **Emergent analogical reasoning in LLMs** — ~Webb, Holyoak & Lu 2023 (*Nature Human Behaviour*);
  ~Mitchell letter-string critiques: solve matrix/letter-string/story analogies. Recognition, curated.
- **ANALOGICAL** ✓ (Wijesiriwardene et al. 2023, arXiv:2305.05050): identify analogical pairs across a
  6-level long-text taxonomy (word→metaphor) in vector space. Recognition, curated.

### Generation (closer — but the model picks the source, or the base is validated)
- **Analogy generation by prompting** ✓ (Bhavya et al. 2022, arXiv:2210.04186, InstructGPT case study);
  long-form analogy extraction/generation (~Illinois/IDEALS): generate an analogy *to explain a given
  target concept* — the model **chooses a familiar source**. Only one endpoint fixed.
- **Design-ideation / co-creative** ✓ (Cambridge *Design Science*; "ALIA"/Analogy Creativity Task):
  generate analogies for a *design problem*, scored on creativity — again the model picks the domain.
- **ANALOGYKB** ✓ (Yuan & Chen, ACL 2024, arXiv:2305.05994) — **closest KG-analogy work**: a
  million-scale analogy KB *mined from knowledge graphs* (same-relation + LLM-filtered analogous-relation
  pairs), used for recognition **and** generation. But the analogies are **validated/extracted** and
  generation draws from that base — *not* synthesis between an arbitrary fixed pair.

### Comparison
| Benchmark | Recog/**Gen** | Curated/**Arbitrary** | Both endpoints fixed? |
|---|---|---|---|
| Google/BATS/SAT/WordRep | recognition | curated | — |
| SemEval-2012 T2 | recognition (rate) | curated | — |
| Webb 2023 / Mitchell | recognition | curated | — |
| ANALOGICAL 2023 | recognition | curated | — |
| ANALOGYKB 2024 | recog. **+ gen.** | **validated** (KG-mined) | no (from base) |
| Bhavya 2022 / long-form | **generation** | target given | **no** (picks source) |
| ALIA / ACT (design) | **generation** | problem given | **no** (picks source) |
| **Ours (analogy)** | **generation** | **arbitrary / non-validated** | **yes** |

**Verdict — strong, clean novelty.** Generation + both endpoints fixed and arbitrary is an
unoccupied cell. Closest threat = ANALOGYKB (opposite direction: it *harvests validated* analogies;
we require *synthesis over non-validated* pairs).

---

## Blending (antanaclasis) — the pun literature is the real neighbor

**Correction to an earlier overclaim:** "first-of-kind antanaclasis benchmark" is *false*. Homographic
puns = polysemy of one word = the mechanism of antanaclasis, and there is a substantial computational
tradition.

### What exists
- **Homographic pun generation** ✓ — Yu et al. 2018 ("A Neural Approach to Pun Generation"), the first
  neural system: generate a *sentence* supporting two meanings of **given target words**; Yu et al.
  2020: lexically-constrained pun rewriting.
- **LLM-era** ✓ — "A good pun is its own reword: Can LLMs Understand Puns?" (EMNLP 2024): pun
  recognition/explanation/generation; **CLEF JOKER** tasks (2023, 2025); **SemEval-2017 Task 7**;
  "A Survey of Pun Generation" (EMNLP Findings 2025).
- The field explicitly separates **homographic (polysemy)** vs **homophone (phonetic)** puns — the same
  axis on which we chose *strict antanaclasis, homographs only, no homophones*.
- Computational **conceptual blending** systems (~Goguen; ~Divago/Pereira; Confalonieri et al. — in our
  bib) — systems, not LLM benchmarks. Only recent hit for LLM blending is *visual* (arXiv:2106.14127).

### How ours differs (four axes)
| | Pun / homographic-pun generation | Our antanaclasis blending |
|---|---|---|
| objective | humor / incongruity (funniness) | conceptual **blending / structure-mapping** (validity + factuality) |
| artifact | a witty **sentence** | a **structured relational blend** (two branches, identical relation skeleton) |
| input | usually **handed** the polysemous target word(s) | **one arbitrary entity**; model must **discover** a second sense (abstention = signal) |
| grounding | free-form text | each sense developed via **factual KG triples** |

**Verdict — real but narrower novelty.** Frame explicitly against the homographic-pun literature
(cite Yu 2018/2020, EMNLP-2024); we recast the same polysemy mechanism as a *factually-grounded
conceptual blend with a shared relational frame, over arbitrary anchors, scored for structure/validity
rather than humor*.

---

## Related generative-over-entities
- **CREATE** ✓ (Wadhwa et al. 2026, in bib): multi-hop Wikidata paths between entities — generation over
  arbitrary pairs, but **association** (path existence), not analogy/blend. Our direct baseline.
- **DAT** ~ (Olson et al. 2021, *PNAS*): name unrelated words — divergent generation, not analogy.

## Risky cells (verify before claiming novelty)
1. **2024–2026 "analogy generation between two GIVEN pairs"** work I may not know — the single most
   likely place a competitor hides. Search specifically.
2. Whether **ACT / ALIA** fix both endpoints (I believe they let the model pick) — confirm.
3. Any **structured / KG-grounded pun or blend** benchmark beyond sentence-level pun generation.
4. Verify all ~-marked citations (Google/BATS/SAT, SemEval-2012, Webb 2023, DAT) for author/year/venue.
