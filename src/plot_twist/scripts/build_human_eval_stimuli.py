"""Build the stimuli, slot plan and server key for the TwistBench human preference study.

Each participant reads one pair, a human gold plot-twist story and a story by the top-ranked
LLM, and says which they prefer, blind to authorship. This script decides what the study shows
and who sees what:

  * Eligibility matches the headline metric. Human stories are vetted STRONG and realism-gated;
    LLM stories are realism-gated and finished cleanly.
  * Length matching is round-robin. Every human story takes its nearest-length LLM story before
    any takes a second, up to `llm_per_human`, and no LLM story is used twice. Plain greedy
    matching let the long stories take the short stories' only neighbours.
  * The slot plan gives every pair `readers_per_pair` slots, half with each story read first.
  * The browser gets opaque story and pair ids only. Real ids, authorship, the reading-check
    answer key and the slot plan go to server/pairs.json, which is never served. An earlier
    payload used real ids such as "anthropic_claude-sonnet-4-5__t12__s03" and listed the human
    story first in every pair, so view-source gave authorship away.
  * The reading checks are audited. The build fails if the correct option is the longest in too
    many items, or if any option uses a colon or semicolon; both were giveaways in the first
    version of the items.

    uv run python src/plot_twist/scripts/build_human_eval_stimuli.py \
        configs/plot_twist/human_eval_stimuli.yaml --overwrite
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, save_config

REQUIRED = ("stories_index_path", "stories_dir", "pd_manifest_path", "llm_source_key",
            "comprehension_items_path", "human_twist_type", "require_gated", "max_words",
            "max_length_ratio", "llm_per_human", "readers_per_pair", "output_dir",
            "experiment_dir")

# If the correct option were no different in length from the distractors, it would be strictly
# the longest in about 25% of items. The first version of the items was at 91%.
LONGEST_MAX_SHARE = 0.40

# Must never appear in what the browser receives, outside story prose and option wording.
CLIENT_FORBIDDEN = ("anthropic", "claude", "sonnet", "__vs__", "author_kind", "human", "llm",
                    "real_id", "correct")


def opaque(prefix, *parts):
    """An id that does not name what it points to. Deterministic, so a rebuild keeps the same ids
    and a study already running against server/pairs.json stays valid."""
    digest = hashlib.sha1("|".join(("twistbench-preference", prefix, *parts)).encode()).hexdigest()
    return f"{prefix}{digest[:12]}"


def eligible_human(index, texts, manifest, config):
    """Vetted-STRONG, realism-gated human gold stories under the word cap."""
    twist_type = {s["slug"]: s.get("twist_type") for s in manifest["stories"]}
    titles = {s["slug"]: (s["title"], s["author"], s["year"]) for s in manifest["stories"]}
    out = []
    for meta in index["human"]["stories"]:
        slug = meta["id"]
        if twist_type.get(slug) != config["human_twist_type"]:
            continue
        if config["require_gated"] and not meta["gated"]:
            continue
        words = len(texts[slug].split())
        if words > config["max_words"]:
            continue
        title, author, year = titles[slug]
        out.append({"id": slug, "author_kind": "human", "words": words, "text": texts[slug],
                    "title": title, "byline": f"{author}, {year}",
                    "surprise": meta["surprise"], "coherence": meta["coherence"],
                    "realism": meta["realism"]})
    return sorted(out, key=lambda s: s["words"])


def eligible_llm(index, texts, config):
    """Realism-gated, cleanly-finished stories from the chosen LLM source, under the cap."""
    key = config["llm_source_key"]
    if key not in index:
        raise ValueError(f"FATAL: llm_source_key '{key}' not in the stories index")
    out = []
    for meta in index[key]["stories"]:
        if config["require_gated"] and not meta["gated"]:
            continue
        if meta["ending"] != "complete":
            continue          # a story cut off mid-word is not a fair comparison
        words = len(texts[meta["id"]].split())
        if words > config["max_words"]:
            continue
        out.append({"id": meta["id"], "author_kind": "llm", "words": words,
                    "text": texts[meta["id"]], "title": meta["id"].split("__", 1)[1],
                    "surprise": meta["surprise"], "coherence": meta["coherence"],
                    "realism": meta["realism"]})
    return sorted(out, key=lambda s: s["words"])


def match_pairs(humans, llms, max_ratio, per_human):
    """Round-robin nearest-length matching, never reusing an LLM story.

    Every human story takes its best remaining match before any takes a second. Within a round
    the longest human stories pick first, since they have the fewest close neighbours. A story
    whose best remaining match is more than `max_ratio` apart in length gets no more matches.
    """
    pool, pairs = list(llms), []
    active = sorted(humans, key=lambda s: -s["words"])
    for _ in range(per_human):
        still = []
        for human in active:
            if not pool:
                break
            llm = min(pool, key=lambda s: abs(s["words"] - human["words"]))
            ratio = max(human["words"], llm["words"]) / min(human["words"], llm["words"])
            if ratio > max_ratio:
                continue
            pool.remove(llm)
            still.append(human)
            h, l = {**human, "key": opaque("s_", human["id"])}, {**llm, "key": opaque("s_", llm["id"])}
            pairs.append({"key": opaque("p_", human["id"], llm["id"]),
                          "real_pair_id": f"{human['id']}__vs__{llm['id']}",
                          "human": h, "llm": l, "length_ratio": round(ratio, 3),
                          # Sorted opaque ids, so position in the payload says nothing about authorship.
                          "story_ids": sorted([h["key"], l["key"]])})
        active = still
    matched = {p["human"]["id"] for p in pairs}
    return (sorted(pairs, key=lambda p: p["key"]),
            [h for h in humans if h["id"] not in matched], pool)


def comprehension_options(story_key, real_id, items):
    """Authored options for one story -> ([{id, text}], correct option id).

    The correct option is authored per story alongside its distractors
    (configs/plot_twist/comprehension_items.json). Option ids are opaque and the browser is
    never told which is right; the experiment shuffles the order per participant.
    """
    if real_id not in items:
        raise ValueError(
            f"FATAL: no comprehension item for '{real_id}'. Every story in the pair pool needs "
            "one in configs/plot_twist/comprehension_items.json.")
    entry = items[real_id]
    if len(entry.get("distractors", [])) != 3:
        raise ValueError(f"FATAL: '{real_id}' needs exactly 3 distractors, "
                         f"got {len(entry.get('distractors', []))}.")
    options = [{"id": opaque("o_", story_key, t), "text": t}
               for t in (entry["correct"], *entry["distractors"])]
    if len({o["id"] for o in options}) != 4:
        raise ValueError(f"FATAL: duplicate option text for '{real_id}'.")
    return options, options[0]["id"]


def audit_comprehension(real_ids, items):
    """Fail on the two giveaways found in the first version of the items. Returns the number of
    items where the correct option is strictly the longest."""
    punct = sorted({rid for rid in real_ids
                    for t in (items[rid]["correct"], *items[rid]["distractors"])
                    if re.search(r"[:;]", t)})
    if punct:
        raise ValueError(f"FATAL: options containing ':' or ';' for {punct}. The first version used "
                         "them almost only in correct answers, which marks the answer.")
    longest = sum(all(len(items[rid]["correct"].split()) > len(d.split())
                      for d in items[rid]["distractors"]) for rid in real_ids)
    share = longest / len(real_ids)
    if share > LONGEST_MAX_SHARE:
        raise ValueError(f"FATAL: the correct option is the longest in {longest}/{len(real_ids)} items "
                         f"({share:.0%}). Chance is about 25%; above {LONGEST_MAX_SHARE:.0%}, picking "
                         "the longest option passes the check without reading.")
    return longest


def slot_plan(pairs, readers):
    """`readers` slots per pair, alternating which story is read first, so every pair (and so
    every story) is read first by exactly half its readers."""
    if readers < 2 or readers % 2:
        raise ValueError(f"FATAL: readers_per_pair must be an even number of at least 2 (got "
                         f"{readers}) so reading order balances within each pair.")
    return [{"slot_id": f"{p['key']}-{i + 1}", "pair_id": p["key"],
             "first_story_id": p["story_ids"][i % 2]}
            for p in pairs for i in range(readers)]


def client_payload(pairs):
    """What the browser receives: stories by opaque id, and pairs of those ids."""
    stories = {}
    for p in pairs:
        for side in ("human", "llm"):
            st = p[side]
            stories[st["key"]] = {"id": st["key"], "words": st["words"], "text": st["text"],
                                  "comprehension": st["comprehension"]}
    rows = [{"pair_id": p["key"], "length_ratio": p["length_ratio"], "story_ids": p["story_ids"]}
            for p in pairs]
    return dict(sorted(stories.items())), rows


def check_client_blind(stories, pair_rows, config_js, header, real_ids):
    """Everything the browser receives, apart from prose and option wording, must be free of
    authorship. Raises rather than writing a payload that would break the blind."""
    skeleton = {
        "stories": {k: {**v, "text": "",
                        "comprehension": {"options": [{**o, "text": ""}
                                                      for o in v["comprehension"]["options"]]}}
                    for k, v in stories.items()},
        "pairs": pair_rows, "config": config_js, "header": header}
    blob = json.dumps(skeleton).lower()
    hits = [t for t in CLIENT_FORBIDDEN if t in blob]
    hits += [r for r in real_ids if f'"{r.lower()}"' in blob]
    if hits:
        raise ValueError(f"FATAL: the browser payload would carry {hits}; it must use opaque ids only.")


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for field in REQUIRED:
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")
    experiment_dir = Path(config["experiment_dir"])
    if not (experiment_dir / "server").is_dir():
        raise ValueError(f"FATAL: experiment_dir {experiment_dir} has no server/ folder. Is the "
                         "study branch checked out there?")

    output_dir = Path(config["output_dir"])
    if output_dir.exists() and not overwrite:
        raise ValueError(f"FATAL: {output_dir} exists. Use --overwrite to replace.")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)

    index = json.loads(Path(config["stories_index_path"]).read_text())
    stories_dir = Path(config["stories_dir"])
    human_texts = json.loads((stories_dir / "human.json").read_text())
    llm_texts = json.loads((stories_dir / f"{config['llm_source_key']}.json").read_text())
    manifest = json.loads(Path(config["pd_manifest_path"]).read_text())
    items = {k: v for k, v in json.loads(Path(config["comprehension_items_path"]).read_text()).items()
             if not k.startswith("_")}

    humans = eligible_human(index, human_texts, manifest, config)
    llms = eligible_llm(index, llm_texts, config)
    source_name = index[config["llm_source_key"]]["source"]
    print(f"Eligible: {len(humans)} human gold stories ({humans[0]['words']}-{humans[-1]['words']} "
          f"words), {len(llms)} {source_name} stories ({llms[0]['words']}-{llms[-1]['words']} words)")

    pairs, dropped, unused = match_pairs(humans, llms, config["max_length_ratio"],
                                         config["llm_per_human"])
    for human in dropped:
        print(f"  DROPPED {human['title']} ({human['words']}w): no LLM story within "
              f"{config['max_length_ratio']}x its length")
    if not pairs:
        raise ValueError("FATAL: no pairs survived length matching")
    if debug:
        pairs = pairs[:2]
        print(f"DEBUG MODE: {len(pairs)} pairs only")

    answer_key = {}
    for p in pairs:
        for side in ("human", "llm"):
            st = p[side]
            st["comprehension"] = {"options": None}
            st["comprehension"]["options"], answer_key[st["key"]] = comprehension_options(
                st["key"], st["id"], items)
    pool_ids = sorted({p[s]["id"] for p in pairs for s in ("human", "llm")})
    longest = audit_comprehension(pool_ids, items)
    slots = slot_plan(pairs, config["readers_per_pair"])

    stories, pair_rows = client_payload(pairs)
    config_js = {"experiment_name": "twistbench_preference", "consent_version": "twistbench_pref_v1"}
    # The header ships to the browser too, so it names no files or sources.
    header = ("/* TwistBench preference study: stimuli. Generated; do not edit by hand.\n"
              "   Story and pair ids are opaque on purpose. */\n")
    check_client_blind(stories, pair_rows, config_js, header, pool_ids)
    js = (f"{header}\nwindow.STORIES = {json.dumps(stories, ensure_ascii=False)};\n"
          f"\nwindow.STIMULUS_PAIRS = {json.dumps(pair_rows)};\n"
          f"\nwindow.EXPERIMENT_CONFIG = {json.dumps(config_js)};\n")

    # Server-side only: maps opaque ids back to real ids and authorship, scores the reading
    # checks and seeds the slot table. Lives in server/, which the deploy never publishes.
    server_key = {
        "llm_source": source_name,
        "readers_per_pair": config["readers_per_pair"],
        "stories": {p[s]["key"]: {"real_id": p[s]["id"], "author_kind": p[s]["author_kind"]}
                    for p in pairs for s in ("human", "llm")},
        "pairs": {p["key"]: {"real_pair_id": p["real_pair_id"], "length_ratio": p["length_ratio"],
                             "stories": {"human": p["human"]["key"], "llm": p["llm"]["key"]}}
                  for p in pairs},
        "comprehension_answer": answer_key,
        "slots": slots,
    }
    analysis_manifest = [
        {"pair_key": p["key"], "pair_id": p["real_pair_id"], "length_ratio": p["length_ratio"],
         "human_id": p["human"]["id"], "human_title": p["human"]["title"],
         "human_words": p["human"]["words"], "llm_id": p["llm"]["id"], "llm_words": p["llm"]["words"],
         "judge_surprise": {"human": p["human"]["surprise"], "llm": p["llm"]["surprise"]},
         "judge_coherence": {"human": p["human"]["coherence"], "llm": p["llm"]["coherence"]}}
        for p in pairs]

    (output_dir / "pairs.json").write_text(json.dumps(analysis_manifest, indent=2))
    (output_dir / "server_key.json").write_text(json.dumps(server_key, indent=2))
    (output_dir / "stimuli-data.js").write_text(js)
    (experiment_dir / "js" / "stimuli-data.js").write_text(js)
    (experiment_dir / "server" / "pairs.json").write_text(json.dumps(server_key, indent=2))
    print(f"Wrote {experiment_dir / 'js' / 'stimuli-data.js'} ({len(js) / 1024:.0f} KB)")
    print(f"Wrote {experiment_dir / 'server' / 'pairs.json'} (server-side only)")

    print("\nComprehension items: REVIEW THESE (* marks the correct answer)")
    for rid in pool_ids:
        e = items[rid]
        print(f"  [{rid}]")
        for text in (e["correct"], *e["distractors"]):
            print(f"    {'*' if text == e['correct'] else ' '} {text}")

    per_human = {}
    for p in pairs:
        per_human[p["human"]["title"]] = per_human.get(p["human"]["title"], 0) + 1
    words = [p["human"]["words"] + p["llm"]["words"] for p in pairs]
    print(f"\nPairs: {len(pairs)} ({len(per_human)} human stories; pairs per story: "
          + ", ".join(f"{t} {n}" for t, n in sorted(per_human.items(), key=lambda x: -x[1])) + ")")
    print(f"LLM stories used: {len(pairs)} of {len(llms)}; unused: "
          + (", ".join(f"{s['title']} ({s['words']}w)" for s in unused) or "none"))
    print(f"Worst length ratio: {max(p['length_ratio'] for p in pairs):.2f}")
    print(f"Reading checks: correct option strictly longest in {longest}/{len(pool_ids)} items")
    print(f"Slots: {len(slots)} ({config['readers_per_pair']} per pair, half with each story first)")
    print(f"Reading load: {min(words)}-{max(words)} words per participant "
          f"(~{sum(words) / len(words) / 200:.0f} min at 200 wpm)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the YAML config")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output_dir")
    parser.add_argument("--debug", action="store_true", help="keep only the first 2 pairs")
    args = parser.parse_args()
    main(args.config, overwrite=args.overwrite, debug=args.debug)
