# 2026-11-11 — Kombine: math-of-combinatorial-creativity talk deck; rebuttal kickoff

## Summary

Wrote the formal definitions behind Kombine as a talk, then built them into a 16-frame
Beamer deck (XeLaTeX, Changa One titles in black, Merriweather body). Each task has a
schema-vs-worked-example slide. Then took stock of which ICLR rebuttal experiments the repo
already holds.

## Tasks completed

- **Math walkthrough (chat).** Gave the full formalism from `papers/kg_creat-iclr/sections/05_benchmark.tex`,
  `media/tab_scoring.tex` and the aggregation rule in `06_results.tex`:
  - preliminaries (entities, relations, triples, judge F, embedding distance, ρ, added elements)
  - association, analogy (structure mapping M, h = M[Φ]) and blending (g, c', Δ, Q levels)
  - utility gate and aggregation
  - τ-inventive multiples
- **Talk deck** `scratch/kombine_math_talk/kombine_math.tex` → `kombine_math.pdf` (16 frames):
  - title; three forms; preliminaries; originality
  - association, then its schema + example
  - analogy I/II, then its schema + example
  - blending I/II, then its schema + example
  - scoring table; aggregation; inventive multiples; multiples statistics
- **Fonts.** Changa One (Google Fonts repo) and Merriweather (upstream SorkinType repo, static TTFs)
  were not installed, so they are vendored in `scratch/kombine_math_talk/fonts/` and loaded by path
  via fontspec. Both are SIL OFL. Changa One is black wherever it appears. The body font is
  Merriweather at Scale 0.9, with serif math (`\usefonttheme[onlymath]{serif}`).
- **Worked examples on the schema slides** use the paper's Table 1 exactly:
  - association: Radio → … → The Milky Way (claude-opus-5)
  - analogy: blue whale :: mattress → whale groomer drone (claude-fable-5)
  - blending: Adam Smith + Bacteria → market colony, rows tagged [uv]/[u]/[v]/[Δ], Q = 3 (claude-opus-4.5)
- **Rebuttal inventory** (see the progress entry).

## Files created

- `scratch/kombine_math_talk/kombine_math.{tex,pdf}`, `scratch/kombine_math_talk/fonts/*.ttf`.
  These are gitignored under `scratch/`; move them to a tracked location if the deck should be kept.

## Key decisions / insights

- **Formalism issues found while writing the talk.** The deck copies the paper's formulas; nothing
  was silently fixed.
  1. Association surprise divides by n−1, but a path has n consecutive pairs.
  2. M is defined only on path entities, yet Φ contains φ and s_j.
  3. Q(c') is undefined when one projection is empty, while the table says {1,2,3}.
  4. The paper's own analogy example projects v→u (vacuum cleaner from the mattress lands on the
     whale), the inverse of M(a_i)=b_i. The example slide draws M undirected to avoid the clash.
- **Beamer gotchas.** A TikZ `.style` with `#1` inside a frame body breaks (move it to the preamble).
  A colour macro used as a bare subscript (`p_\U`) needs an extra brace group.

## Next steps

- Fix the four formalism issues in the paper (they double as rebuttal clarity points).
- Rebuttal: run the experiments not yet done (see progress.md 2026-11-11).
