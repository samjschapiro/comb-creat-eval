# 2026-11-14 20:32: rebuttal, corpus contamination check (En17 Q1); test–retest stats explained

**Done**
- **Test–retest report:** added an explanation of Pearson, Spearman and ICC(A,1) (commit d516a67).
- **"Death of the Novel(ty)" checked:** Saakyan, Kim, Muresan & Chakrabarty, ICLR 2026, arXiv 2509.22641.
  About 91% of the most n-gram-novel expressions were not judged creative. So n-grams are used here only
  as a contamination check.
- **New `src/kg_creat/scripts/corpus_novelty.py`:** infini-gram API (free), cached at
  `data/kg_creat/corpus_novelty/cache.jsonl` (about 6.6k queries, 0 failures, about 3 hours throttled).
  - Indexes: OLMo 2 mix (4.6T tokens) to test whether a link exists and to count names verbatim; Dolma
    v1.6 sample (9.2B tokens) for exact co-occurrence.
  - Found while probing: CNF counts on the big index are *sampled* for clauses with more than 500k
    matches, and the magnitudes are unreliable ("bread AND butter" came back at about 1k). A sampled
    nonzero is real; a sampled zero is inconclusive. Hence the two-index rule.
  - The `max_clause_freq` parameter makes single-phrase count queries fail on the Dolma sample index, so
    it is sent only for CNF queries.
- **Results:**
  - Anchor lift median 0.45 (68% of pairs below chance); 57 of 60 pairs co-occur somewhere in 4.6T tokens.
  - 87% of association hops are documented.
  - Names never seen verbatim: 40% (analogy), 32% (blending).
  - Co-occurrence predicts validity on association (+0.47) and analogy (lift +0.49), but no originality,
    multiples or blending outcome.
  - Familiar names get lower emergent originality (within item −0.15 / −0.10). The multiples rate does
    not rise with existing names.
  - The Swanson endpoint contrast failed, because at 4.6T tokens every endpoint pair co-occurs
    somewhere. The report says so.
- **Report:** `docs/reports/2026-11-14_kg_creat_rebuttal_corpus_contamination/report.md`. Rebuttal En17
  Q1 point 5 filled in; A7nz Lu et al. note updated.

**Next.** S2 when the ratings arrive; trim cGKy's replies.
