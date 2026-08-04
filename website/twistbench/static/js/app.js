/* TwistBench project page: sortable leaderboard + lazy-loading story explorer.
 *
 * Data comes from data/ (built by src/plot_twist/scripts/build_website_data.py):
 *   leaderboard.json      - 72 ranked sources
 *   stories_index.json    - per-source story metadata, no prose
 *   stories/<key>.json    - full prose for one source, fetched on first open
 */

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
));

/* Render story text as paragraphs. Blank lines separate paragraphs in every source;
 * single newlines inside a paragraph are hard wrapping (the Project Gutenberg human
 * texts are wrapped at ~70 columns) and get unwrapped back into flowing prose. */
/* Titles and abbreviations end in a period without ending a sentence — without this the
 * highlight stops dead at `"Mrs.`. */
const ABBREV = /(?:^|[\s"“‘(])(?:Mr|Mrs|Ms|Dr|Prof|Rev|Sgt|Capt|Lt|Col|Gen|St|Sr|Jr|vs|etc|e\.g|i\.e|No|Mt|Ave|Inc|Ltd|Co)\.$/;

/* Index just past the end of the sentence that is still open at `from`. */
function sentenceEnd(p, from) {
  const re = /[.!?…]+["”’')\]]*/g;
  re.lastIndex = from;
  let m;
  while ((m = re.exec(p)) !== null) {
    const stop = m.index + m[0].length;
    const next = p[stop];
    if (next && !/\s/.test(next)) continue;            // mid-token punctuation
    if (ABBREV.test(p.slice(0, stop))) continue;       // "Mrs." is not a sentence end
    return stop;
  }
  return p.length;
}

const asParagraphs = (text, anchor) => {
  const paras = text
    .trim()
    .split(/\n\s*\n/)
    .map((p) => p.replace(/\s*\n\s*/g, " ").trim())
    .filter(Boolean);

  // The twist anchor is a verbatim slice of the story, but the reader has just unwrapped
  // hard line breaks, so normalize the anchor the same way before looking for it.
  const needle = anchor ? anchor.replace(/\s+/g, " ").trim() : null;
  let marked = false;

  return paras.map((p) => {
    const at = needle && !marked ? p.indexOf(needle) : -1;
    if (at < 0) return `<p>${esc(p)}</p>`;
    marked = true;
    const stop = sentenceEnd(p, at + needle.length);
    return `<p>${esc(p.slice(0, at))}<mark class="twist" id="twist-mark"
      >${esc(p.slice(at, stop))}</mark>${esc(p.slice(stop))}</p>`;
  }).join("");
};

let LEADERBOARD = [];
let INDEX = {};
const PROSE_CACHE = {};   // source key -> {story id: text}

/* Organization mark: the lab's logo tinted with its palette colour, or — for labs with no
 * icon in the CC0 set — a coloured disc bearing the initial. ORGS comes from orgs.js. */
function orgMark(source) {
  const key = source === "human" ? "human" : source.split("/")[0];
  const org = ORGS[key];
  if (!org) return `<span class="mark mark-initial" style="background:#8c8c8c">?</span>`;
  if (org.path) {
    return `<span class="mark" style="color:${org.color}" title="${esc(org.name)}">
      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="${org.path}"/></svg>
    </span>`;
  }
  const initial = key === "human" ? "H" : org.name[0].toUpperCase();
  return `<span class="mark mark-initial" style="background:${org.color}"
            title="${esc(org.name)}">${initial}</span>`;
}

const orgColor = (source) =>
  (ORGS[source === "human" ? "human" : source.split("/")[0]] || {}).color || "#8c8c8c";

const orgName = (source) =>
  (ORGS[source === "human" ? "human" : source.split("/")[0]] || {}).name
  || source.split("/")[0];

// ---------------------------------------------------------------- leaderboard

let sortKey = "rank";
let sortAsc = true;

function renderLeaderboard() {
  const q = $("#lb-search").value.trim().toLowerCase();
  const rows = LEADERBOARD
    .filter((r) => !q || r.source.toLowerCase().includes(q) || r.org.toLowerCase().includes(q))
    .sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      const cmp = typeof x === "string" ? x.localeCompare(y) : x - y;
      return sortAsc ? cmp : -cmp;
    });

  // Overall is a z-score; draw a bar proportional to its position in the observed range.
  const zs = LEADERBOARD.map((r) => r.overall);
  const lo = Math.min(...zs), hi = Math.max(...zs);
  const width = (z) => `${Math.max(2, ((z - lo) / (hi - lo)) * 100)}%`;

  $("#lb-table tbody").innerHTML = rows.map((r) => {
    const isHuman = r.source === "human";
    const name = isHuman ? "Expert humans (gold set)" : r.source.split("/").slice(1).join("/");
    return `<tr class="${isHuman ? "human" : ""}" data-key="${r.key}">
      <td class="num">${r.rank}</td>
      <td class="src-cell">${orgMark(r.source)}<span class="src-text">${esc(name)}<span
        class="org">${isHuman ? "public-domain gold set" : esc(orgName(r.source))}</span></span></td>
      <td class="num">${r.overall.toFixed(3)}
        <span class="bar" style="width:${width(r.overall)};background:${orgColor(r.source)}"></span></td>
      <td class="num gated">${r.surprise_gated.toFixed(2)}</td>
      <td class="num gated">${r.coherence_gated.toFixed(2)}</td>
      <td class="num raw">${r.surprise.toFixed(2)}</td>
      <td class="num raw">${r.coherence.toFixed(2)}</td>
      <td class="num">${r.realism.toFixed(2)}</td>
      <td class="num">${r.diversity.toFixed(3)}</td>
      <td class="num">${r.n}</td>
    </tr>`;
  }).join("");

  $("#lb-count").textContent = `${rows.length} of ${LEADERBOARD.length} sources`;
  applyFacetToggle();
}

function applyFacetToggle() {
  const gated = $("#lb-gated").checked;
  document.querySelectorAll(".gated").forEach((el) => el.classList.toggle("hide-col", !gated));
  document.querySelectorAll(".raw").forEach((el) => el.classList.toggle("hide-col", gated));
}

function initLeaderboard() {
  document.querySelectorAll("#lb-table th").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      // First click on a new column sorts descending (best first), except rank/source.
      if (sortKey === key) sortAsc = !sortAsc;
      else { sortKey = key; sortAsc = key === "rank" || key === "source"; }
      document.querySelectorAll("#lb-table th").forEach((h) => h.classList.remove("sorted", "asc"));
      th.classList.add("sorted");
      if (sortAsc) th.classList.add("asc");
      renderLeaderboard();
    });
  });
  $("#lb-search").addEventListener("input", renderLeaderboard);
  $("#lb-gated").addEventListener("change", applyFacetToggle);

  // Clicking a row jumps to that source in the explorer.
  $("#lb-table tbody").addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr || !INDEX[tr.dataset.key]) return;
    selectSource(tr.dataset.key);
    $("#stories").scrollIntoView({ behavior: "smooth" });
  });
}

// ------------------------------------------------------------- story explorer

let currentSource = null;
let currentStory = null;

function sourceOrder() {
  // Order the picker by leaderboard rank so the best sources are on top.
  const ranked = LEADERBOARD.filter((r) => INDEX[r.key]).map((r) => ({ ...r, ...INDEX[r.key] }));
  const seen = new Set(ranked.map((r) => r.key));
  const extra = Object.entries(INDEX)
    .filter(([k]) => !seen.has(k))
    .map(([key, v]) => ({ key, source: v.source, rank: null, org: v.source.split("/")[0] }));
  return ranked.concat(extra);
}

function renderSources() {
  const q = $("#src-search").value.trim().toLowerCase();
  const items = sourceOrder().filter((r) => !q || r.source.toLowerCase().includes(q));
  $("#src-list").innerHTML = items.map((r) => {
    const isHuman = r.source === "human";
    const name = isHuman ? "Expert humans" : r.source.split("/").slice(1).join("/");
    return `<li class="${isHuman ? "human-src" : ""} ${r.key === currentSource ? "active" : ""}"
                data-key="${r.key}" style="--org:${orgColor(r.source)}">
      ${orgMark(r.source)}<span class="src-text">
        <span class="src-rank">${r.rank ?? "–"}</span><span class="src-name">${esc(name)}</span>
        <span class="src-org">${esc(isHuman ? "public-domain gold set" : orgName(r.source))}</span>
      </span>
    </li>`;
  }).join("");
}

function renderStoryList() {
  const entry = INDEX[currentSource];
  if (!entry) return;
  const isHuman = entry.source === "human";
  const onlyGated = $("#only-gated").checked;
  const sort = $("#sort-stories").value;

  // Show exactly the stories the leaderboard scores: for the human gold set that is the
  // 18-story ceiling, not every public-domain text that was collected.
  let stories = entry.stories
    .filter((s) => s.scored)
    .filter((s) => !onlyGated || s.gated);
  stories = stories.slice().sort((a, b) => (
    sort === "id" ? a.id.localeCompare(b.id) : (b[sort] ?? 0) - (a[sort] ?? 0)
  ));

  $("#src-title").textContent = isHuman ? "Expert humans (gold set)" : entry.source;
  $("#story-list").innerHTML = stories.map((s) => {
    const label = isHuman
      ? `${esc(s.title)} <span class="src-org">${esc(s.author)}</span>`
      : `Story ${Number(s.sample) + 1} <span class="src-org">temperature ${Number(s.temp) / 10}</span>`;
    const cut = s.ending === "complete" ? ""
      : `<span class="chip cut">${s.ending === "cut" ? "cut off" : s.ending}</span>`;
    return `<li class="${s.id === currentStory ? "active" : ""}" data-id="${esc(s.id)}">
      <span class="story-title">${label}</span>
      <span class="story-meta">
        <span class="chip s">S ${s.surprise.toFixed(1)}</span>
        <span class="chip c">C ${s.coherence.toFixed(1)}</span>
        <span class="chip ${s.gated ? "gate-pass" : "gate-fail"}">${s.gated ? "realistic" : "gate fail"}</span>
        ${cut}
        <span class="chip">${s.words}w</span>
      </span>
    </li>`;
  }).join("");
  if (!stories.length) $("#story-list").innerHTML = `<li style="cursor:default">No stories match this filter.</li>`;
}

async function loadProse(key) {
  if (!PROSE_CACHE[key]) {
    const res = await fetch(`data/stories/${key}.json`);
    if (!res.ok) throw new Error(`missing prose shard for ${key}`);
    PROSE_CACHE[key] = await res.json();
  }
  return PROSE_CACHE[key];
}

async function selectSource(key) {
  currentSource = key;
  currentStory = null;
  // Keep the URL pointing at the open source so a model's stories can be linked to.
  history.replaceState(null, "", `#source=${key}`);
  renderSources();
  renderStoryList();
  // Open the first story straight away so the reader is never a blank panel.
  const first = $("#story-list li[data-id]");
  if (first) selectStory(first.dataset.id);
  else $("#reader").innerHTML = `<div class="reader-empty"><p>📖</p><p>Pick a story to read it.</p></div>`;
}

async function selectStory(id) {
  currentStory = id;
  renderStoryList();
  const entry = INDEX[currentSource];
  const meta = entry.stories.find((s) => s.id === id);
  $("#reader").innerHTML = `<div class="reader-empty"><p>⏳</p><p>Loading…</p></div>`;

  let text;
  try {
    text = (await loadProse(currentSource))[id];
  } catch (e) {
    $("#reader").innerHTML = `<div class="reader-empty"><p>⚠️</p><p>Could not load this story.</p></div>`;
    return;
  }

  const isHuman = entry.source === "human";
  const heading = isHuman ? esc(meta.title) : `Story ${Number(meta.sample) + 1}`;
  const byline = isHuman
    ? `${esc(meta.author)} · public domain`
    : `${esc(entry.source)} · temperature ${Number(meta.temp) / 10}`;

  $("#reader").innerHTML = `
    <div class="reader-head">
      <h3>${heading}</h3>
      <div class="src">${byline} · ${meta.words} words</div>
      <div class="reader-scores">
        <div>Surprise<b>${meta.surprise.toFixed(1)}</b></div>
        <div>Coherence<b>${meta.coherence.toFixed(1)}</b></div>
        <div>Realism<b>${meta.realism == null ? "–" : meta.realism.toFixed(1)}</b></div>
        <div>Realism gate<b>${meta.gated ? "passes" : "fails"}</b></div>
      </div>
      ${meta.twist_anchor ? `<button class="jump-twist" type="button">Jump to the twist ↓</button>` : ""}
    </div>
    <div class="reader-body">
      <div class="annot">
        <dl><dt>Setup</dt><dd>${esc(meta.setup)}</dd></dl>
      </div>
      <div class="prose">${asParagraphs(text, meta.twist_anchor)}</div>
      ${meta.ending === "complete" ? "" : `<p class="cut-note">${
        meta.words > 3000
          ? `This generation stopped mid-sentence. The prompt asked for 2,000–3,000 words
             and the model ran past it, exhausting the run's 4,500-token cap — failing to
             hold the requested length is itself an instruction-following failure.`
          : `This generation stopped mid-sentence at ${meta.words} words, below the 2,000-word
             minimum the prompt asked for.`
      } The story is served exactly as the model produced it, unfinished.</p>`}

      <div class="annot after">
        <h4>The twist, and how it was scored <a class="spoiler-hint">show</a></h4>
        <dl hidden>
          <dt>Reveal</dt><dd>${esc(meta.reveal)}</dd>
          <dt>Judge note</dt><dd>${esc(meta.why)}</dd>
        </dl>
      </div>
    </div>`;

  // The reveal and the judges' rationale both spoil the story, so they stay folded
  // away below the prose until the visitor asks for them.
  const toggle = $("#reader .after .spoiler-hint");
  const dl = $("#reader .after dl");
  toggle.addEventListener("click", () => {
    dl.hidden = !dl.hidden;
    toggle.textContent = dl.hidden ? "show" : "hide";
  });

  const jump = $("#reader .jump-twist");
  if (jump) {
    jump.addEventListener("click", () => {
      const mark = $("#twist-mark");
      if (mark) mark.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }
}

function initExplorer() {
  $("#src-search").addEventListener("input", renderSources);
  $("#src-list").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-key]");
    if (li) selectSource(li.dataset.key);
  });
  $("#story-list").addEventListener("click", (e) => {
    const li = e.target.closest("li[data-id]");
    if (li) selectStory(li.dataset.id);
  });
  $("#only-gated").addEventListener("change", renderStoryList);
  $("#sort-stories").addEventListener("change", renderStoryList);
}

// ---------------------------------------------------------------------- boot

async function boot() {
  const [lb, idx] = await Promise.all([
    fetch("data/leaderboard.json").then((r) => r.json()),
    fetch("data/stories_index.json").then((r) => r.json()),
  ]);
  LEADERBOARD = lb;
  INDEX = idx;

  initLeaderboard();
  initExplorer();
  renderLeaderboard();
  renderSources();

  // #source=<key> deep-links to a source; otherwise open the top-ranked human gold set.
  const linked = new URLSearchParams(location.hash.slice(1)).get("source");
  selectSource(linked && INDEX[linked] ? linked : "human");
}

boot().catch((e) => {
  console.error(e);
  $("#lb-table tbody").innerHTML =
    `<tr><td colspan="10">Could not load data/. Serve this page over HTTP
     (e.g. <code>python -m http.server</code>) rather than opening the file directly.</td></tr>`;
});
