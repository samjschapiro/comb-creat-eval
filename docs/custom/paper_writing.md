# Paper Writing Guidelines (Overleaf)

## Paper Location

Papers live in `papers/<paper-name>/`, where each subdirectory is a separately tracked git repo cloned from Overleaf. These are NOT part of the main repo's git history.

```
papers/
├── my-icml-paper/       # git clone from Overleaf
└── my-workshop-paper/   # another Overleaf repo
```

## Writing Style

- **NEVER use emdashes.** Not em dashes, not en dashes used as em dashes. Just don't.
- **Don't write like a generic AI.** Tell a story like a human would. Be less formal than usual. The reader is a person, not a review committee robot.
- **NEVER write a bibtex entry yourself.** Always copy from a file or directly copy what the user gives. Do not fabricate or reconstruct citations from memory.

## AI-Generated Content

**All AI-generated edits, suggestions, or drafts must be marked with `\ai{}`.**

Add this command to your preamble:

```latex
\usepackage{xcolor}
\newcommand{\ai}[1]{\textcolor{green}{[AI: #1]}}
```

Usage:

```latex
\ai{This paragraph was drafted by AI and needs human review.}
```

This ensures transparency and makes it easy to identify content requiring review before submission.

## Author Comment Commands

Define colored comment commands for each collaborator:

```latex
\newcommand{\authorA}[1]{\textcolor{orange}{[AuthorA: #1]}}
\newcommand{\authorB}[1]{\textcolor{blue}{[AuthorB: #1]}}
\newcommand{\authorC}[1]{\textcolor{red}{[AuthorC: #1]}}
\newcommand{\ai}[1]{\textcolor{green}{[AI: #1]}}
```

## Pre-Submission Checklist

- [ ] No `\ai{}` markers remaining (all AI content reviewed/revised)
- [ ] No author comment commands remaining
- [ ] Anonymous submission requirements met (if applicable)

## Workflow

1. AI suggests edits wrapped in `\ai{}`
2. Human reviews, revises, and removes the `\ai{}` wrapper
3. Use author commands for discussion/feedback
4. Clean all comments before final submission
