"""Build the JSON payload the TwistBench project website reads.

The site is a static page (website/twistbench/index.html); everything it displays comes
from three artifacts written here:

  1. leaderboard.json  - the 72-row headline table (realism-gated z-composite + facets)
  2. stories_index.json - per-source story lists WITHOUT prose (ids, scores, setup/reveal),
                          small enough to ship with the page so the browser can render the
                          picker and per-story score chips immediately
  3. stories/<key>.json - one shard per source holding the full story prose, fetched lazily
                          when a visitor opens that model

    uv run python src/plot_twist/scripts/build_website_data.py configs/plot_twist/website.yaml --overwrite
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, save_config
from src.plot_twist.join import attach_story_text, load_annotations, score_num
from src.plot_twist.predict import locate_anchor


def completion_status(text: str) -> str:
    """Did the generation finish, or was it cut off by the token cap?

    `max_tokens` was 4500 for the benchmark pass against a 2000-3000 word target, so a
    sizeable share of stories stop mid-sentence — for gpt-5.4 the cut lands in a 200-word
    band (sd 65 words), i.e. a hard ceiling rather than an authorial choice. The site
    labels these instead of quietly serving half a story.

      complete - ends on sentence punctuation (allowing closing quotes/markdown emphasis)
      cut      - ends inside a word or clause
      unclear  - ends on something else (list marker, dash, stray symbol)
    """
    tail = re.sub(r"[*_`#~\s]+$", "", text.rstrip())
    if not tail:
        return "empty"
    if re.search(r"[.!?…]+[\"”’')\]]*$", tail):
        return "complete"
    return "cut" if re.search(r"[a-zA-Z]$", tail) else "unclear"


def source_key(source: str) -> str:
    """Filesystem/URL-safe key for a source name ('anthropic/claude-opus-4.6' -> ...)."""
    return re.sub(r"[^a-z0-9]+", "-", source.lower()).strip("-")


def title_key(s: str) -> str:
    """Accent- and punctuation-insensitive title key: the manifest spells a title
    "Desiree's Baby" where the paper table spells it "Désirée's Baby"."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _detex(s: str) -> str:
    """LaTeX fragment -> plain text, enough to match it back against the story prose."""
    for a, b in (("``", '"'), ("''", '"'), (r"\ldots", "…"), ("---", "—"), ("--", "–"),
                 (r"\'e", "é"), (r"\&", "&"), ("~", " ")):
        s = s.replace(a, b)
    s = re.sub(r"\\emph\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return re.sub(r"\s+", " ", s.replace("{", "").replace("}", "")).strip()


def reveal_anchors(table_path: Path) -> dict:
    """Story title -> {anchor, pos}, from the paper's appendix reveal-point table.

    Each human story's twist sentence was annotated there by hand. The quotes may elide
    their middle *or* their start (``…B'' / ``A…B''), so the anchor is the LONGEST
    ellipsis-free run in the quote — the part guaranteed to appear verbatim and
    contiguously in the story. `pos` is the table's own position column (a percentage
    through the text, or "end"), used to disambiguate a short anchor that occurs more
    than once.
    """
    anchors = {}
    for line in table_path.read_text().splitlines():
        if not re.match(r"^\d+\s*&", line):
            continue
        cells = line.rstrip().rstrip("\\").split("&")
        if len(cells) < 4:
            continue
        title = re.sub(r"\s*\([^)]*\)\s*$", "", _detex(cells[1])).strip()
        quote = _detex(cells[2])
        m = re.search(r'"([^"]+)"', quote)                 # the quoted twist itself
        if m:
            quote = m.group(1)
        # Longest run with no elision in it.
        anchor = max((p.strip() for p in quote.split("…")), key=lambda p: len(p.split()))
        pos_cell = cells[3].strip().replace("\\%", "").replace("%", "")
        pos = 1.0 if pos_cell.lower().startswith("end") else (
            float(pos_cell) / 100 if pos_cell.replace(".", "").isdigit() else 1.0)
        if len(anchor.split()) >= 2:
            anchors[title_key(title)] = {"anchor": anchor, "pos": pos}
    return anchors



def human_meta(pd_manifest_path: Path) -> dict:
    """Map human gold story slug -> {title, author, twist_type} for the explorer.

    `twist_type` matters on the site as well as in the analysis: only STRONG stories
    count toward the human ceiling (see src/plot_twist/sets.py), so the explorer has to
    say which of the 35 vetted gold stories are the 18 the leaderboard scores.
    """
    manifest = json.loads(pd_manifest_path.read_text())
    return {
        s["slug"]: {
            "title": s["title"],
            "author": s["author"],
            "twist_type": s.get("twist_type") or "UNVETTED",
        }
        for s in manifest["stories"]
    }


def build_leaderboard(tc_path: Path) -> list[dict]:
    """The headline table, ranked by the realism-gated equal-weight z-composite."""
    rows = json.loads(tc_path.read_text())
    rows = sorted(rows, key=lambda r: -r["overall_eq"])
    out = []
    for rank, r in enumerate(rows, 1):
        out.append({
            "rank": rank,
            "source": r["source"],
            "key": source_key(r["source"]),
            "org": "Human experts" if r["source"] == "human" else r["source"].split("/")[0],
            "n": r["n"],
            "overall": round(r["overall_eq"], 4),
            "surprise": round(r["mean_surprise"], 3),
            "coherence": round(r["mean_coherence"], 3),
            "realism": round(r["mean_realism"], 3),
            "diversity": round(r["div"], 4),
            "surprise_gated": round(r["mean_surprise_g"], 3),
            "coherence_gated": round(r["mean_coherence_g"], 3),
        })
    return out


def build_stories(config) -> tuple[dict, dict]:
    """Group every annotated story under its source, split into a light index and prose shards.

    Returns (index, shards): index[key] = {source, stories:[{id, temp, sample, surprise,
    coherence, realism, gated, setup, reveal, why, words}]}, shards[key] = {id: story text}.
    """
    records = load_annotations(config["annotations_path"])
    realism = json.loads(Path(config["realism_path"]).read_text())
    humans = human_meta(Path(config["pd_manifest_path"]))
    anchors = reveal_anchors(Path(config["reveal_points_path"]))
    gate = float(config["realism_gate"])

    # LLM twist locations, if the localization pass has been run (run_twist_points.py).
    # Optional on purpose: the site builds fine without it, just with no LLM markers.
    twist_path = Path(config["twist_points_path"])
    llm_twists = json.loads(twist_path.read_text()) if twist_path.exists() else {}
    print(f"LLM twist anchors available: {len(llm_twists)}"
          f"{' (run run_twist_points.py to generate)' if not llm_twists else ''}")

    with_text = attach_story_text(
        records, config["llm_stories_dir"], config["human_texts_dir"]
    )
    print(f"Story text resolved for {len(with_text)}/{len(records)} annotated stories.")

    index: dict[str, dict] = {}
    shards: dict[str, dict] = {}
    for r in with_text:
        s, c = score_num(r, "surprise"), score_num(r, "coherence")
        if s is None or c is None:
            continue
        key = source_key(r["source"])
        rv = realism.get(r["id"])
        # Story ids look like "<model_key>__t09__s00"; humans are "<slug>".
        parts = r["id"].split("__")
        meta = {
            "id": r["id"],
            "surprise": s,
            "coherence": c,
            "realism": rv,
            "gated": bool(rv is not None and rv >= gate),
            "setup": r.get("setup") or "",
            "reveal": r.get("reveal") or "",
            "why": r.get("why_scored") or "",
            "words": len(r["text"].split()),
            "ending": completion_status(r["text"]),
        }
        if r["source"] == "human":
            hm = humans.get(r["id"], {"title": r["id"], "author": "", "twist_type": "UNVETTED"})
            meta.update(hm)
            # Only STRONG gold stories feed the human ceiling the leaderboard reports.
            meta["scored"] = hm["twist_type"] == "STRONG"
            # The hand-annotated twist sentence, so the reader can mark it in the prose.
            ref = anchors.get(title_key(hm["title"]))
            if ref:
                found = locate_anchor(r["text"], ref["anchor"], ref["pos"])
                if found:
                    meta["twist_anchor"] = found
        else:
            meta["scored"] = True
            meta["temp"] = parts[1].replace("t", "") if len(parts) > 1 else ""
            meta["sample"] = parts[2].replace("s", "") if len(parts) > 2 else ""
            # Model-located twist. The stored anchor was already verified to be a literal
            # substring of the story when it was produced; re-check here so a stale
            # twist_points.json can never put a marker on the wrong text.
            anchor = llm_twists.get(r["id"])
            if anchor and anchor in r["text"]:
                meta["twist_anchor"] = anchor
        index.setdefault(key, {"source": r["source"], "stories": []})["stories"].append(meta)
        shards.setdefault(key, {})[r["id"]] = r["text"]

    for entry in index.values():
        entry["stories"].sort(key=lambda m: m["id"])
    return index, shards


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for field in ("tc_path", "annotations_path", "realism_path", "llm_stories_dir",
                  "human_texts_dir", "pd_manifest_path", "realism_gate", "output_dir"):
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")

    output_dir = Path(config["output_dir"])
    if output_dir.exists() and not overwrite:
        raise ValueError(f"FATAL: {output_dir} exists. Use --overwrite to replace.")
    (output_dir / "stories").mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    leaderboard = build_leaderboard(Path(config["tc_path"]))
    (output_dir / "leaderboard.json").write_text(json.dumps(leaderboard))
    print(f"Leaderboard: {len(leaderboard)} sources -> leaderboard.json")

    index, shards = build_stories(config)
    if debug:
        keys = sorted(index)[:3]
        index = {k: index[k] for k in keys}
        shards = {k: shards[k] for k in keys}
        print(f"DEBUG MODE: {len(keys)} sources only")

    (output_dir / "stories_index.json").write_text(json.dumps(index))
    for key, texts in shards.items():
        (output_dir / "stories" / f"{key}.json").write_text(json.dumps(texts))

    total = sum(len(v["stories"]) for v in index.values())
    print(f"Stories: {total} across {len(index)} sources -> stories_index.json + "
          f"{len(shards)} prose shards")

    endings = Counter(s["ending"] for v in index.values() for s in v["stories"])
    cut = endings["cut"] + endings["unclear"] + endings["empty"]
    print(f"Endings: {dict(endings)}  ({100 * cut / total:.1f}% did not finish cleanly)")

    scored_humans = [s for s in index.get("human", {}).get("stories", []) if s["scored"]]
    human_marked = sum("twist_anchor" in s for s in scored_humans)
    llm_stories = [s for k, v in index.items() if k != "human" for s in v["stories"]]
    llm_marked = sum("twist_anchor" in s for s in llm_stories)
    print(f"Twist markers: {human_marked}/{len(scored_humans)} human (hand-annotated), "
          f"{llm_marked}/{len(llm_stories)} LLM ({100 * llm_marked / max(len(llm_stories), 1):.0f}%)")
    for s in scored_humans:
        if "twist_anchor" not in s:
            print(f"  UNLOCATED (human): {s.get('title', s['id'])}")

    # The leaderboard is the site's canonical source list; flag anything it names that
    # has no readable stories, so the explorer never silently drops a model.
    missing = [r["source"] for r in leaderboard if r["key"] not in index]
    if missing:
        print(f"WARNING: {len(missing)} leaderboard sources have no stories: {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
