"""Fetch the public-domain plot-twist gold set from Project Gutenberg.

Pipeline (recorded per story in the output manifest for reproducibility):

  1. RESOLVE   each story to a Gutenberg collection eBook id (configs/plot_twist/
               pd_manifest.json). Stories with status != 'ready' (no verified id)
               are skipped and listed at the end for a human to pin.
  2. DOWNLOAD  the plain-text eBook from the canonical cache URL
               https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt
               (NOT the HTML ebook pages -- PG blocks scraping those). Cached
               under data/plot_twist/human_twists/_raw/ so re-runs are offline.
  3. STRIP     the Project Gutenberg header/footer boilerplate (between the
               '*** START OF ... ***' and '*** END OF ... ***' markers).
  4. EXTRACT   the single story from its collection by locating its title as a
               heading and capturing to the next story heading (a sibling title
               from the same collection, or a generic heading-like line). An
               explicit `start_heading`/`end_heading` in the manifest overrides
               the heuristic.
  5. CLEAN     normalize whitespace; record word_count + sha256 of the cleaned
               text so any later change to a stored story is detectable.

Outputs:
  data/plot_twist/human_twists/texts/<slug>.txt        cleaned story text
  data/plot_twist/human_twists/fetched_manifest.json   per-story provenance

VERIFY after running: word counts are printed; a wrong id or a bad heading match
shows up as a wildly off word_count or a preview that starts mid-story. The
gold set is small -- eyeball every row once.

Usage:
  uv run python src/plot_twist/scripts/fetch_pd_stories.py configs/plot_twist/pd_manifest.json
  ... --dry-run            # resolve + report word counts/previews, write nothing
  ... --only gift-of-the-magi,owl-creek-bridge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

CACHE_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
USER_AGENT = "comb-creat-eval/plot_twist fetch_pd_stories (research; contact maintainer)"
OUT_ROOT = Path("data/plot_twist/human_twists")

_START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
_END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I)
_ROMAN = re.compile(r"^(chapter\s+)?[ivxlcdm]+\b[\.\s]*", re.I)
_LEADNUM = re.compile(r"^\d+[\.\s]+")


def _norm(s: str) -> str:
    """Heading-comparison form: ASCII-fold accents (Desiree -> Desiree), lowercase,
    drop a leading roman numeral / number, strip non-alphanumerics, collapse spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\[\d+\]", "", s)   # drop footnote markers, e.g. "A JURY OF HER PEERS[11]"
    s = s.strip()
    s = _ROMAN.sub("", s)
    s = _LEADNUM.sub("", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def download(gid: int, cache_dir: Path, polite_delay: float = 1.0) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"pg{gid}.txt"
    if cached.exists():
        return cached.read_text(encoding="utf-8", errors="replace")
    url = CACHE_URL.format(id=gid)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode("utf-8", errors="replace")
    cached.write_text(raw, encoding="utf-8")
    time.sleep(polite_delay)  # be polite to PG between fresh downloads
    return raw


def strip_boilerplate(raw: str) -> str:
    s = _START_RE.search(raw)
    e = _END_RE.search(raw)
    body = raw[s.end() : e.start()] if (s and e) else raw
    return body.strip("\n")


def parse_toc(body: str) -> set[str]:
    """Normalized story titles from the collection's 'Contents' list -- the
    authoritative boundary set (robust to ALL-CAPS vs Title-Case heading styles).
    Empty set if no parseable TOC (caller falls back to the heading heuristic)."""
    lines = body.split("\n")
    idx = next((i for i, l in enumerate(lines) if re.match(r"^\s*contents\s*$", l, re.I)), None)
    if idx is None:
        return set()
    titles: set[str] = set()
    blanks = 0
    for l in lines[idx + 1 : idx + 600]:
        s = l.strip()
        if not s:
            blanks += 1
            if blanks >= 4 and titles:
                break
            continue
        blanks = 0
        s = re.sub(r"[\.\s]+\d+$", "", s)   # "Title . . . . 12"
        s = re.sub(r"\s{2,}\d+$", "", s)
        n = _norm(s)
        if n and 1 <= len(n.split()) <= 12:
            titles.add(n)
    return titles


def _is_heading_like(line: str) -> bool:
    """A story-title heading inside a PG collection: a short, mostly-ALL-CAPS line
    not ending in sentence punctuation. We deliberately do NOT accept Title-Case
    lines -- they produce false cuts on ordinary short sentences/dialogue."""
    t = line.strip()
    if not t or len(t.split()) > 8:
        return False
    # allow a trailing period (PG titles often end in "."), reject other sentence punct
    if t[-1] in ",;:!?\"'":
        return False
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 2:
        return False
    upper_ratio = sum(c.isupper() for c in letters) / len(letters)
    return upper_ratio > 0.85


def extract_story(body: str, title: str, siblings: list[str],
                  start_heading: str | None, end_heading: str | None,
                  toc_titles: set[str] | None = None) -> tuple[str, str]:
    """Return (story_text, method). Locate the title heading, then end at the next
    boundary: explicit override > next table-of-contents title > sibling target >
    generic ALL-CAPS heading. Optional start_heading/end_heading override."""
    toc_titles = toc_titles or set()
    lines = body.split("\n")
    norm_lines = [_norm(l) for l in lines]
    want = _norm(start_heading or title)
    wt = set(want.split())

    # start: exact normalized match, else a heading-like line whose tokens are a
    # superset of the title's (handles "The Diamond Necklace" for "The Necklace",
    # or "No. 1 Branch Line: The Signal-Man" for "The Signal-Man"). Prefer the LAST
    # match (the body heading follows the table of contents) with text after it.
    starts = [i for i, nl in enumerate(norm_lines) if nl == want]
    # Prefer ALL-CAPS body headings over Title-Case table-of-contents/index lines
    # that normalize the same (e.g. body "HEARTS AND HANDS" vs TOC "Hearts and Hands").
    caps = [i for i in starts if _is_heading_like(lines[i])]
    if caps:
        starts = caps
    if not starts:
        starts = [
            i for i, nl in enumerate(norm_lines)
            if nl and wt <= set(nl.split()) and len(nl.split()) <= len(wt) + 5
            and _is_heading_like(lines[i])
        ]
    if not starts:
        return "", "NOT_FOUND"

    boundary = (toc_titles | {_norm(s) for s in siblings}) - {want}

    def _end_from(start: int) -> tuple[int, str]:
        if end_heading:
            ehn = _norm(end_heading)
            for i in range(start + 1, len(lines)):
                if norm_lines[i] == ehn:
                    return i, "override"
            return len(lines), "eof"
        words = 0
        for i in range(start + 1, len(lines)):
            words += len(lines[i].split())
            if words <= 30:
                continue
            nl = norm_lines[i]
            # heading position: preceded by a blank OR a section-numeral line
            # ("IX" normalizes to ""), since titles often sit right under a numeral.
            prev_ok = (i == 0) or norm_lines[i - 1] == ""
            next_blank = (i + 1 >= len(lines)) or not lines[i + 1].strip()
            if nl and nl in boundary and prev_ok:
                return i, "boundary_title"
            if words > 60 and nl and _is_heading_like(lines[i]) and prev_ok and next_blank:
                return i, "caps_heading"
        return len(lines), "eof"

    # pick the LAST candidate heading with substantial text after it (the body
    # heading follows any contents/index list), then end at the FIRST boundary.
    start = next((i for i in reversed(starts)
                  if sum(len(lines[j].split()) for j in range(i + 1, min(i + 80, len(lines)))) > 40),
                 starts[-1])
    end, method = _end_from(start)
    text = "\n".join(lines[start + 1 : end]).strip()
    if start_heading:
        method += "+start_override"
    return text, method


def clean(text: str) -> str:
    # collapse 3+ blank lines to one; strip trailing spaces; normalize newlines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated slugs")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    stories = manifest["stories"]
    only = {s for s in args.only.split(",") if s}
    if only:
        stories = [s for s in stories if s["slug"] in only]

    cache_dir = OUT_ROOT / "_raw"
    texts_dir = OUT_ROOT / "texts"

    # group ready stories by collection id (download each collection once)
    by_id: dict[int, list[dict]] = {}
    skipped: list[dict] = []
    for s in stories:
        if s.get("status") == "ready" and s.get("gutenberg_id"):
            by_id.setdefault(s["gutenberg_id"], []).append(s)
        else:
            skipped.append(s)

    records: list[dict] = []
    print(f"{'slug':<22} {'id':>6} {'words':>7}  preview")
    for gid, group in by_id.items():
        try:
            body = strip_boilerplate(download(gid, cache_dir))
        except Exception as ex:  # noqa: BLE001
            for s in group:
                print(f"{s['slug']:<22} {gid:>6} {'ERR':>7}  download failed: {ex}")
            continue
        toc = parse_toc(body)
        sib_titles = [g["title"] for g in group]
        for s in group:
            text, method = extract_story(
                body, s["title"], sib_titles, s.get("start_heading"),
                s.get("end_heading"), toc_titles=toc,
            )
            text = clean(text)
            wc = word_count(text)
            preview = re.sub(r"\s+", " ", text)[:60]
            print(f"{s['slug']:<22} {gid:>6} {wc:>7}  {preview}")
            rec = {
                "slug": s["slug"], "title": s["title"], "author": s["author"],
                "year": s["year"], "type": s["type"],
                "gutenberg_id": gid,
                "source_url": CACHE_URL.format(id=gid),
                "collection": s.get("collection"),
                "translation": s.get("translation"),
                "extract_method": method,
                "word_count": wc,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
            records.append(rec)
            if not args.dry_run and wc > 0:
                texts_dir.mkdir(parents=True, exist_ok=True)
                (texts_dir / f"{s['slug']}.txt").write_text(text, encoding="utf-8")

    if not args.dry_run:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "fetched_manifest.json").write_text(
            json.dumps({"source": "Project Gutenberg", "url_pattern": CACHE_URL,
                        "stories": records}, indent=2, ensure_ascii=False)
        )

    print(f"\nfetched {sum(1 for r in records if r['word_count'] > 0)}/{len(stories)} stories")
    if skipped:
        print(f"\n{len(skipped)} NOT fetched (status != ready / no id) -- pin these:")
        for s in skipped:
            print(f"  [{s.get('status','?'):<12}] {s['slug']:<22} {s.get('collection','')}")
    if args.dry_run:
        print("\n(dry-run: no files written)")


if __name__ == "__main__":
    main()
