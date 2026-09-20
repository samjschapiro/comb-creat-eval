# 2026-09-11 → 09-20 — the Kombine human study platform finalized and live

Session spanning 09-11 to 09-20 (closing tasks run 09-20). All work is in the sibling repo
`llm_creativity_mech_interp/src/experiments/kombine_generation/` (36 commits, 5d163b8 → 709ad31, pushed at
closing) and all of it is live at https://schapiro.ai/kombine/ (verified byte-identical to the source).
Nothing in this repo's code changed; the
`docs/tracks/kg_creat/progress.md` entries for 09-11 and 09-12 were written mid-session.

## Summary

The jsPsych human generation study was rewritten page by page to the wording in the shared Google doc
("Study Items (REVISED)", Tom Griffiths' and Akshay Jagadish's edits accepted), verified word for word
against the doc by rendering each page headlessly, and given the form mechanics the reviewers asked for:
progressive reveal, relation mirroring, an invention-direction arrow, a Both source for double-scope
blend links, opt-outs, easy warm-up items, LLM-use countermeasures, a debug skip bar and a Prolific
completion code.

## Tasks completed

**Wording**
- Six page texts (three overviews, three item prompts) plus the three texts below the line follow the doc.
  "Generic space" never appears to participants; it is "the abstract structure that both concepts share"
  (data key `generic_space` unchanged). Overview titles are "Association Task" etc.; bonus banner on the
  consent page and every overview, now "Your bonus depends on how original and unusual your responses
  are compared with other participants'" (the top-10% / $15 sentence removed at the user's request).
- A word-by-word checker (headless Chrome renders each page with the doc's example pairs; Python diffs
  against the doc text) was run after every doc round; final state: all nine texts match.
- Blending asks for the shared abstract structure before the name; "and/or" wording; the "Links in your
  new concept" sentence removed; emergent prompt "Describe something that is true of your new concept
  but of neither original concept on its own" in the form and the example.

**Worked examples**
- Analogy: a real judge-unanimous LLM invention from the 35-model run (claude-fable-5, The blue whale ::
  The mattress, vacuum cleaner → whale groomer drone, emergent originality 0.58); "epidermis" shown as
  "skin", "ticking" as "sheets". Invention rows carry a direction arrow; invented cells shaded gold.
- Blending: Liquid Franchise trimmed to one Both row (Democracy allocates votes + Banking allocates
  credit → allocates vote-shares), one Banking row, one new link; legend for the gold shading.

**Form mechanics**
- Progressive reveal: the analogy invention appears once every mapping row is filled or skipped; the
  blend's last step once structure, name and every link row are filled. Hidden sections are disabled so
  they are neither validated nor submitted. No invention step when every analogy row is skipped.
- Opt-outs: per analogy row, invention, whole blend ("I can't think of a blend"), emergent.
- Relation typed in one cell mirrors to the row's other relation cells in real time (analogy sides;
  blend source(s) and image).
- Analogy invention rows: required arrow toggle pointing to the invented side, which is shaded;
  `projection[i].direction` recorded with `source` always the true side.
- Blend link rows: required source pills Democracy / Banking / Both; a Both row stacks one source link
  per concept into one link of the new concept (`from: "uv"`, `source_u`/`source_v`), the paper's
  double-scope property. No [u]/[v] tags anywhere. One "+ add a link" button.

**Items**
- 5 association, 4 analogy + Life :: A journey first, 4 blending + Breakfast + Lunch first (15 total).
  Dropped Antibiotic resistance :: A spam filter and Democracy + Banking (the worked example). Easy items
  flagged `is_control`; random insertion was built, tested uniform over slots, then replaced by the
  warm-up ordering at the user's request. Every response carries `position`.

**LLM-use countermeasures**
- Paste and drop blocked on every answer field of a task page with a notice; `paste_attempts` recorded.
- Two visually hidden traps per item (screen-reader-only pattern, in innerText, not aria-hidden): a
  sentence in the prompt and a labelled field (`honeypot`). Wording is a plain task requirement ("One of
  the entities in your answer must be the word {word}"); a different word per item from a 20-word list,
  recorded as `trap_word`. Lesson: Chrome's Ask Gemini ignored traps addressed to AI and traps hidden
  with aria-hidden, and followed them once phrased as task text in the readable page.
- Per-trial typing summary (keys, backspaces, chars, first-key latency, active time, median inter-key
  interval, pauses over 2 s) for screening retyped answers.

**Operations**
- Debug bar (no PROLIFIC_PID): "Skip this page" / "Skip rest of task"; skipped trials flagged.
- Final page shows the Prolific completion code (placeholder `XXXXXXXX`, one constant) and the return
  button opens the completion URL built from it.
- Deploys: `scripts/kg_creat/deploy_study.sh` works to the commit and fails on push here; the user ran
  it from their terminal. Live files were verified byte-identical to the source after each deploy.
  Local site checkout `~/Desktop/Websites/samjschapiro2` fast-forwarded.

## Files

- `llm_creativity_mech_interp/src/experiments/kombine_generation/{index.html, js/experiment.js,
  js/stimuli-data.js, README.md}` (36 commits).
- This repo: `docs/tracks/kg_creat/progress.md` (09-11, 09-12 and this entry), `docs/research_context.md`,
  `README.md`, `docs/structure.md`, this log.
- Memory: `kombine-site-location` (where the study lives, deploy needs the user's terminal, trap lesson).

## Decisions

- Doc is the source of truth for wording; page and doc were reconciled both ways at the user's direction
  (doc updated for the Both-row example and the "final step will appear" sentence; page updated for
  Akshay's phrasing).
- Threshold bonus dropped from the banner pending Prolific rules; originality-relative wording kept.
- Honeypots are a weak defence against browser assistants; the typing summary and an analysis-time
  similarity screen against consumer-model answers are the intended detectors.

## Open

- Data sink still `DATA_SUBMISSION_URL=''`; consent page still says the study is about *evaluating*
  ideas; real completion code; Prolific bonus rules with Catherine.
- Human→model converter must map `uv` rows and the analogy `direction` field before scoring.
- Analysis-time screen against consumer-model answers on the same items (few dollars on the pipeline).
- Akshay's open doc comment "a new one?" on "link that entity to another one".
