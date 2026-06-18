# Converting a single-column paper (COLM / NeurIPS) to two-column (ICML)

Notes collected while porting the COLM camera-ready (`overleaf_paper`,
single column) into the ICML workshop/preprint build
(`paper_creativity_neuro_workshop`, two column). The root cause of almost
every issue is the same: **`\textwidth` does not change meaning between the
two layouts, but the usable column does.**

## The core fact

| | single column (COLM) | two column (ICML) |
|---|---|---|
| `\textwidth` | ~6.75 in (the whole text block) | still ~6.75 in (the whole text block, spanning *both* columns) |
| `\columnwidth` | == `\textwidth` | ~3.25 in (one column; `(\textwidth - \columnsep)/2`) |

In a single-column paper `\textwidth == \columnwidth`, so authors reach for
`\textwidth` everywhere and it just works. In two columns those are no longer
equal. Any body content sized to `\textwidth` but placed in a single-column
environment overflows by exactly `\textwidth - \columnwidth` ≈ **252 pt (~3.5
in)** — it runs off the page. This 252.94 pt overfull `\hbox` is the signature
of the bug; if you see it repeated, it's this.

## Pattern 1 — `width=\textwidth` graphics in a plain `figure`

**Symptom:** `Overfull \hbox (252.94pt too wide)`; image visibly bleeds out of
the column, across the gutter, and off the page edge.

```latex
% single-column original — fine there, broken in two columns
\begin{figure}[t]
  \includegraphics[width=\textwidth]{wide_figure.png}
\end{figure}
```

**Fix — pick one based on intent:**
- Figure is *meant* to span the full page → promote the float to the starred
  form. `width=\textwidth` is then correct (now = full page).
  ```latex
  \begin{figure*}[t]                      % spans both columns
    \includegraphics[width=\textwidth]{wide_figure.png}
  \end{figure*}
  ```
- Figure belongs in one column → size it to the column, not the text block.
  ```latex
  \begin{figure}[t]
    \includegraphics[width=\columnwidth]{narrow_figure.png}  % or \linewidth
  \end{figure}
  ```

Rule of thumb: inside a one-column context use `\columnwidth`/`\linewidth`;
reserve `\textwidth` for starred (`figure*`/`table*`) full-page floats.

## Pattern 2 — wide tables in a plain `table`

**Symptom:** same 252 pt-class overfull; a multi-column numeric table that fit
the single column now spills off-page. A 9-column `{ll rrrrrr|r}` table sized
for COLM's 6.75 in column cannot fit ICML's 3.25 in column.

**Fix:**
- `table*` to span both columns (preferred for genuinely wide tables), or
- `\resizebox{\columnwidth}{!}{...}` / `\small`/`\footnotesize` + tighter
  `\tabcolsep` if it must stay in one column, or
- restructure (transpose, split, drop columns).

Note: `table*`/`figure*` floats can only go to the **top or bottom of a page**
(`[t]`/`[b]`/`[p]`) — `[h]`/`[H]` are silently ignored for starred floats.

## Pattern 3 — `wrapfigure` widths are relative to `\textwidth`

**Symptom:** the wrapped figure nearly fills the whole column, leaving a
useless sliver of text beside it; wrapping effectively collapses.

```latex
\begin{wrapfigure}[22]{r}{0.45\textwidth}      % 0.45 x 6.75 = 3.04in
  \includegraphics[width=0.45\textwidth]{fig.pdf}
\end{wrapfigure}
```
`0.45\textwidth` ≈ 3.04 in inside a 3.25 in column → ~94% of the column. There
is no room left to wrap text into.

**Fix:** make the wrap relative to the column, e.g.
`{0.48\columnwidth}` + `width=0.48\columnwidth`, **or** just recast as a normal
single-column `figure` (`width=\columnwidth`). `wrapfigure` is fragile in
two-column mode generally; a plain `figure` is usually the safer port.

## Pattern 4 — narrow columns break long unbreakable tokens

**Symptom:** clusters of moderate overfulls (8–35 pt) in ordinary paragraphs.
Source: inline math (`$(\rho,\alpha)$`, `$\Delta$\,Percentile`), `\textsc{...}`,
`\Cref{...}`, URLs, long compound nouns — none of which TeX can hyphenate.
Fine in a 6.75 in column, they protrude in a 3.25 in one.

**Fixes (cheapest first):**
- `\usepackage{microtype}` — often removes most of these on its own.
- Allow math line breaks where sensible, or wrap a stubborn inline expression
  so it can break.
- Reword to shorten the unbreakable run, or add discretionary breaks `\-`.
- Last resort: `\sloppy` / `sloppypar` for a specific paragraph (loosens
  spacing instead of overflowing).

## Pattern 5 — the title/author block uses a different macro set

This is a *template* difference rather than a pure column-width one, but it
always surfaces during the same port and breaks visibly.

**Symptom:** `Undefined control sequence \@author`, repeated `Missing $
inserted`, author names rendered smushed in math italic (e.g. `FelixSosa`,
`LavR.Varshney`), and a `Missing \icmlcorrespondingauthor` notice in the
footer.

**Cause:** generic / COLM-style author block carried over verbatim:
```latex
\author{Name \thanks{...} \\ Affiliation \AND Other Name \\ ... }
\maketitle
```
ICML doesn't use `\author`/`\maketitle`. It wants its own commands **inside the
`\twocolumn[ ... ]` title block**:
```latex
\twocolumn[
  \icmltitle{...}
  \begin{icmlauthorlist}
    \icmlauthor{Name}{aff1}
    \icmlauthor{Other Name}{aff2}
  \end{icmlauthorlist}
  \icmlaffiliation{aff1}{Institution}
  \icmlaffiliation{aff2}{Institution}
  \icmlcorrespondingauthor{Name}{email}
  \icmlkeywords{...}
]
\printAffiliationsAndNotice{}    % or \printAffiliationsAndNotice{\icmlEqualContribution}
```
Each conference class (COLM `\author`-style, NeurIPS `\author`, ICML
`\icmlauthor`) has its own author API — the block must be rewritten, not
copied.

## Pattern 6 — single- vs two-column structural commands

- The body is wrapped in `\twocolumn[ \icmltitle ... ]`; the title/author block
  lives in the optional argument so it spans both columns. Don't put
  `\maketitle` there.
- Long appendices / wide derivations are often switched back with `\onecolumn`
  (this build does `\onecolumn` before the appendix). Inside an `\onecolumn`
  region, `\textwidth == \columnwidth` again, so `width=\textwidth` figures and
  `table*` are fine there — **the same line that overflows in the body is
  correct in the appendix.** Audit by *region*, not just by grep.
- `algorithm` vs `algorithm*`: a single-column algorithm is fine only if its
  lines fit ~3.25 in. Keep algorithm bodies short-lined, or use `algorithm*`.
  (Our `cn_algo.tex` was rewritten with short `\STATE` lines + a `\ifdefined
  \preprintbuild` wrapfigure variant precisely for this.)

## Pattern 7 — page-limit blowup

Two columns pack ~2x the text per page, but figures/tables sized to the page
(now `figure*`) and the larger relative cost of floats often mean the page
count does **not** halve. Workshop page limits are easy to exceed after a port;
budget a length pass (figure sizing, float placement, trimming).

## Pattern 8 — full-width floats drift away from their reference and leave whitespace

This is the subtlest and most visually damaging failure, and it only appears
*after* you've fixed the overflow (Patterns 1–2) by promoting wide graphics to
`figure*`/`table*`. The content is now correct width but lands in the wrong
place.

**Why it happens:** in two-column mode, a page-spanning float (`figure*`,
`table*`) can only be placed at the **top of a page** — never mid-column,
never `[h]`, never at a column top. LaTeX also defers any float it can't place
within strict area limits (`\dbltopfraction` etc.). So a paper that was a tidy
single column — where a `[t]`/`[h]` figure just dropped in next to its
reference — turns into a pile-up: every wide figure competes for the same
scarce "page-top spanning" slot, the ones that don't fit cascade to later
pages, and they arrive **several pages after the text that cites them**.

**Symptoms (all observed in this port):**
- Figure 2 is cited in §6 (~p4) but typesets on p7; Figures 2–6 all cluster on
  pp7–10 while their references are on pp3–7.
- The teaser/cover figure can't fit at the top of p1 (the title block already
  spans it) → it slides to the top of p2.
- **Near-empty pages.** A float-page (a page LaTeX fills with deferred floats)
  packs only if the floats cover `\floatpagefraction` of it; otherwise you get
  a figure marooned at the top of an otherwise blank page. A small
  single-column figure recast from a `wrapfigure` (Pattern 3) is the classic
  victim — it floated alone onto a ~90%-empty page here.
- Big vertical gaps at the bottom of a page where text "should" have flowed but
  a barrier/float pushed it.

**Fixes, in order of leverage:**

1. **Loosen the float-placement parameters globally** (in the preamble). The
   LaTeX defaults are very conservative for two columns; relaxing them lets
   more float area per page and makes float-pages pack instead of stranding one
   figure:
   ```latex
   \renewcommand{\topfraction}{0.9}          % up from 0.7
   \renewcommand{\bottomfraction}{0.8}
   \renewcommand{\textfraction}{0.07}        % allow nearly-full float pages
   \renewcommand{\floatpagefraction}{0.75}   % a float-page must be >=75% full
   \renewcommand{\dbltopfraction}{0.9}        % SAME, but for figure*/table*
   \renewcommand{\dblfloatpagefraction}{0.7}  % the double-column knobs are
   \setcounter{topnumber}{3}                   % separate — set both sets!
   \setcounter{dbltopnumber}{3}
   \setcounter{totalnumber}{5}
   ```
   The `\dbl*` knobs are the ones that govern `figure*`; forgetting them is the
   most common reason "I loosened the float params and nothing changed."
2. **Use an assertive placement specifier.** `[!t]` tells LaTeX to ignore its
   area restrictions for that float, which usually pulls it back to the first
   page-top after the reference. `[tbp]` gives it more legal slots.
3. **Right-size, then decide span.** A genuinely small figure (the Venn here)
   does not want to be a lonely full-column float — keep it small and let it
   share a page with text via `[t]`, or inline it. Only make something
   `figure*` if it's actually wide; a medium figure forced to span looks
   stretched and short (the teaser: full `\textwidth` but little height →
   squished). Constrain `width=` and center, or split into sub-panels.
4. **Teaser on page 1:** put it inside the title block —
   `\twocolumn[ \icmltitle{...} ... \vskip... \includegraphics ... ]` — rather
   than as a floating `figure*`, if page-1 placement matters.
5. **Place the barrier deliberately.** `\FloatBarrier` (placards package) before
   a section flushes pending floats *there* instead of letting them avalanche
   to the end of the body; use it to keep figures within their section rather
   than after it.

**Verify by eye, not by log.** Overfull boxes don't capture "figure is 3 pages
late" or "page is half blank." Render the body pages to PNG and check each
figure lands on or near the page that cites it, and that no page is mostly
white.

## A practical audit checklist

1. Grep for the hazards:
   `grep -rnE '\\begin\{(figure|table|wrapfigure)\}|width=\\textwidth|0\.[0-9]+\\textwidth' sections tables figures`
2. For each hit, ask: *one column or both?* → `\columnwidth` (plain env) vs
   `\textwidth` (starred env).
3. Compile and grep the log for `Overfull \hbox`:
   - **252.94 pt** repeated → a `\textwidth`-in-single-column float (Pattern 1/2).
   - clusters of 8–35 pt in paragraphs → narrow-column token breaking (Pattern 4).
4. Rewrite the title/author block to the target class's API (Pattern 5).
5. Add `microtype`; do a length/float-placement pass for page limits.
6. Render the first few pages to PNG and eyeball them — overfull boxes that run
   off-page are obvious visually and easy to miss in the log noise.
</content>
