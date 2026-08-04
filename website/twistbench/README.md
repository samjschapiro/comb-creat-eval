# TwistBench project page

Static project page for the TwistBench paper, in the style of the Nerfies academic
template (the same convention [creativitybench.github.io](https://creativitybench.github.io/)
follows). No build step, no framework, no external requests — plain HTML/CSS/JS.

```
index.html                 page content
static/css/style.css       all styling
static/js/app.js           sortable leaderboard + story explorer
static/js/orgs.js          per-organization colours + logo paths (see below)
static/figures/*.png       paper figures, downscaled to 2000px
data/                      generated — do not hand-edit
  leaderboard.json         72 ranked sources
  stories_index.json       per-source story metadata (scores, setup/reveal), no prose
  stories/<key>.json       full prose for one source, fetched on first open
```

## Rebuilding the data

`data/` is generated from the run outputs. Regenerate it after any rescoring:

```bash
uv run python src/plot_twist/scripts/build_website_data.py configs/plot_twist/website.yaml --overwrite
```

The script reads `data/plot_twist/tc/tc.json` (the realism-gated leaderboard),
`annotations/annotations.json`, `realism/realism_scores.json`, and the story texts, and it
warns loudly if any leaderboard source ends up with no readable stories.

## Twist markers

The reader highlights the sentence that delivers the plot twist. It comes from two sources,
and in both cases the stored anchor is a **literal substring of the story**, re-verified at
build time — a story whose twist cannot be located gets no marker rather than a guessed one.

- **Human gold (18):** parsed from the paper's own appendix table,
  `papers/pt2cb-iclr-2027/tables/tab_reveal_points.tex`, where the twist sentence of each
  story was annotated by hand. The table's position column disambiguates short anchors that
  occur more than once (`"My father."` in *A Horseman in the Sky*).
- **LLM stories:** located by `run_twist_points.py`, which asks a cheap model for the
  verbatim opening of the twist sentence and matches it back into the prose:

```bash
uv run python src/plot_twist/scripts/run_twist_points.py configs/plot_twist/twist_points.yaml
```

That runner is durable and resumable, and it refuses to exceed `max_calls` in its config.
An embedding-based alternative was tried first and rejected: matching each story's `reveal`
summary against its sentences located the twist within 5% of the true position only 20–28%
of the time (validated against both the hand annotations and the predict pilot), which is
far too imprecise to highlight a specific sentence.

## Organization colours and logos (`static/js/orgs.js`)

Colours are the **exact values the paper's figures use** — the categorical batlow map
(Crameri) indexed by sorted provider name, with the same luminance adjustment and the same
grey "Other" bucket as `make_tc_barplot.py`, and black for the human gold set. Regenerate
them the same way that script does if the provider set changes:

```python
from cmcrameri import cm; from matplotlib.colors import to_hex   # uv pip install cmcrameri
```

Logos are single-path glyphs from [Simple Icons](https://simpleicons.org/) (CC0), inlined
as SVG path data so the page makes no external requests. Brand marks remain the trademarks
of their owners and appear only to identify which lab produced a model. 12 of the 20
organizations have an icon in that set; the rest render as a coloured initial, which is why
the org name is always printed next to the mark — colour and logo are reinforcement, never
the only way to read the row.

## Viewing locally

The page fetches `data/` over HTTP, so opening `index.html` from the filesystem will not
work (the browser blocks `fetch` on `file://`). Serve the directory:

```bash
cd website/twistbench && python -m http.server 8000
# then open http://localhost:8000
```

## Deploying

Everything is relative-path, so the directory can be dropped at the root of a GitHub Pages
repo (e.g. `twistbench.github.io`) or served from any static host as-is.

It is published from the `gh-pages` branch of this repo:
**<https://samjschapiro.github.io/comb-creat-eval/>**

To redeploy after a rebuild:

```bash
bash scripts/plot_twist/deploy_site.sh
```

That copies `website/twistbench/` onto the `gh-pages` branch and pushes it. The generated
`data/` directory (~29 MB) is gitignored on `main` — it lives only on `gh-pages`, since it
is reproducible from the pipeline.

There are no Paper / Code / Data links in the hero yet; add them to `index.html` once the
arXiv and dataset URLs exist.
