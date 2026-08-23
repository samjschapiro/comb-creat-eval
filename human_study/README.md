# Concept Pairs — human study (jsPsych)

A browser experiment that mirrors the three Kombine tasks. On each trial a participant makes an
**association**, **analogy**, or **blend** between arbitrary entities (sampled at random), then rates
**surprise / emergence / confidence** on 7-point scales.

Built on [jsPsych 8](https://www.jspsych.org/). Files:

| file | role |
|------|------|
| `index.html` | loads jsPsych + plugins from CDN, then `style.css` and `experiment.js` |
| `experiment.js` | timeline, entity sampling, the three custom task screens, data shaping |
| `style.css` | visual identity (entity A = coral, B = teal, participant's creation = violet); light + dark |

## Run it locally

jsPsych is loaded from a CDN, so you need internet access at run time. Serve the folder (don't just
double-click `index.html` — the data download and some browsers dislike `file://`):

```bash
cd human_study
python3 -m http.server 8000
# open http://localhost:8000
```

At the end (with no server configured, see below) the data is offered as a **JSON download** and also
dumped to the page.

> Note: this will **not** run as a Claude artifact — the artifact sandbox's CSP blocks the jsPsych CDN.
> The self-contained vanilla-JS mockup was the Claude-artifact version; this is the real, deployable study.

## Configure (top of `experiment.js`)

- `N_PER_TASK` — trials of each task type (default 2 → 6 total).
- `TASK_ORDER` — `"interleaved"` (round-robin, no back-to-back repeats) or `"blocked"`.
- `PROLIFIC_COMPLETION_URL` — set to your study's completion URL to redirect participants when done.
- `DATAPIPE_EXPERIMENT_ID` — set to save to OSF via DataPipe (see below).

Entity pools live in `experiment.js` (`POOL` for association/analogy endpoints, `BLEND` for
polysemy-friendly blend anchors) — edit to taste.

## Saving data

**Default (no config):** local JSON download + on-page dump. Fine for piloting.

**Prolific:** host the folder (GitHub Pages, Netlify, your server), set the study URL to it with
`?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}` (Prolific fills
these in). They're captured into every row. Set `PROLIFIC_COMPLETION_URL` to send participants back.

**DataPipe → OSF (recommended for real collection):** create a free experiment at
<https://pipe.jspsych.org>, then in `index.html` add:

```html
<script src="https://unpkg.com/@jspsych/plugin-pipe@0.5"></script>
```

and in `experiment.js` replace the body of `saveData()` with a `jsPsychPipe.saveData({...})` call
(add it as a final timeline trial so it runs before the Prolific redirect). See the DataPipe docs.

## Data shape

Each task trial stores a tidy `clean` object alongside the raw jsPsych record:

```jsonc
// association
{ "type":"association", "a":"a glacier", "b":"Morse code",
  "path":[{"index":0,"relation":"grinds rock into","entity":"fine dust"}, ...],
  "ratings":{"surprise":6,"emergence":5,"confidence":3} }

// analogy
{ "type":"analogy", "a":"the immune system", "b":"the postal service",
  "a_to":"pathogens", "b_to":"junk mail", "shared_relation":"sorts and rejects unwanted arrivals",
  "ratings":{...} }

// blend
{ "type":"blend", "anchor":"current",
  "sense1":"flow of water", "sense2":"flow of electric charge",
  "shared_frame":"has a direction, a strength, and can be resisted",
  "ratings":{...} }
```

Ratings map to the benchmark constructs: **surprise** ≈ semantic remoteness, **emergence** ≈ emergent
novelty (did the whole license something the parts didn't), **confidence** ≈ self-rated utility/truth.

## Pinning versions

`index.html` uses major-version CDN ranges (`jspsych@8`, `@jspsych/plugin-*@2`). For a locked study,
pin exact patch versions (e.g. `jspsych@8.2.1`) so a future CDN update can't change behavior mid-run.
