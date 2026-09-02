"""Build the stimulus file for the TwistBench human preference study.

The study asks people to read two full plot-twist stories — one human gold story, one from
the top-ranked LLM — and say which they prefer, blind to authorship. This script picks the
pair pool and emits it as `js/stimuli-data.js` for the jsPsych experiment.

Two things it enforces, because both are confounds a preference judgment cannot survive:

  * **Same eligibility as the headline metric.** Human stories must be vetted STRONG (a
    genuine reinterpretation twist) and realism-gated, exactly the set the human ceiling in
    the paper is computed on; LLM stories must be gated too.
  * **Length matching.** A 900-word story next to a 3,000-word one is a visible authorship
    cue and a reading-effort confound, so every human story is paired with the nearest-length
    LLM story (greedy, without replacement) and badly-matched pairs are dropped.

    uv run python src/plot_twist/scripts/build_human_eval_stimuli.py \
        configs/plot_twist/human_eval_stimuli.yaml --overwrite
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, save_config

REQUIRED = ("stories_index_path", "stories_dir", "pd_manifest_path", "llm_source_key",
            "comprehension_items_path",
            "human_twist_type", "require_gated", "max_words", "max_length_ratio",
            "pairs_per_participant", "output_dir", "experiment_dir")


# The browser gets ONLY what it must render. Authorship, titles, and the judge's own scores
# stay server-side: on a public URL, view-source is all it takes to break the blind, and a
# participant who knows which story is human is no longer giving a blind preference.
CLIENT_FIELDS = ("id", "words", "text", "comprehension")

# --- comprehension options -----------------------------------------------------------
# The reading check asks "which of these was revealed at the end?", with the correct answer
# being this story's `reveal` annotation and the distractors other stories' reveals. Character
# names are stripped from every option, because a name is learnable from page one: left in,
# the item detects someone who never opened the story but not someone who skimmed it. The
# same transformation is applied to correct answers and distractors alike, so the stripping
# itself can never mark out the right one.

# --- comprehension items ---------------------------------------------------------------
# Distractors are authored per story (configs/plot_twist/comprehension_items.json) rather than
# borrowed from other stories' reveals. Borrowed reveals carry the wrong characters and setting,
# so anyone who read the first page could eliminate them without reading the ending — the item
# tested "did you open this story", not "did you finish it". The correct option is authored in
# the same voice as the distractors for the same reason: otherwise style, not memory, marks it.
#
# The browser is never told which option is right. Each option gets an opaque id and the answer
# key goes to server/pairs.json, exactly as authorship does.

def option_id(story_id, text):
    return hashlib.sha1(f"{story_id}|{text}".encode()).hexdigest()[:12]


def comprehension_options(story_id, items):
    """Authored options for one story -> ([{id, text}], correct_id). Order is fixed here; the
    experiment shuffles per participant."""
    if story_id not in items:
        raise ValueError(
            f"FATAL: no comprehension item for '{story_id}'. Every story in the pair pool needs "
            "one in configs/plot_twist/comprehension_items.json — an unauthored story would fall "
            "back to a weaker check without anyone noticing.")
    entry = items[story_id]
    if len(entry.get("distractors", [])) != 3:
        raise ValueError(f"FATAL: '{story_id}' needs exactly 3 distractors, "
                         f"got {len(entry.get('distractors', []))}.")
    correct = {"id": option_id(story_id, entry["correct"]), "text": entry["correct"]}
    options = [correct] + [{"id": option_id(story_id, d), "text": d} for d in entry["distractors"]]
    if len({o["id"] for o in options}) != 4:
        raise ValueError(f"FATAL: duplicate option text for '{story_id}'.")
    return options, correct["id"]


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
                    "reveal": meta["reveal"],
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
                    "text": texts[meta["id"]], "source": index[key]["source"],
                    "reveal": meta["reveal"],
                    "surprise": meta["surprise"], "coherence": meta["coherence"],
                    "realism": meta["realism"]})
    return sorted(out, key=lambda s: s["words"])


def match_pairs(humans, llms, max_ratio):
    """Greedy nearest-length matching without replacement, longest human story first.

    Longest-first matters: the long tail of human stories has the fewest LLM stories near it,
    so it must get first pick or it ends up matched against whatever is left over.
    """
    pool = list(llms)
    pairs, dropped = [], []
    for human in sorted(humans, key=lambda s: -s["words"]):
        if not pool:
            dropped.append((human, None))
            continue
        llm = min(pool, key=lambda s: abs(s["words"] - human["words"]))
        ratio = max(human["words"], llm["words"]) / min(human["words"], llm["words"])
        if ratio > max_ratio:
            dropped.append((human, ratio))
            continue
        pool.remove(llm)
        pairs.append({"pair_id": f"{human['id']}__vs__{llm['id']}", "human": human,
                      "llm": llm, "length_ratio": round(ratio, 3)})
    return sorted(pairs, key=lambda p: p["pair_id"]), dropped



def stimuli_js(pairs, config, source_name):
    """The pair pool as a JS file the experiment loads before `experiment.js`.

    Every pair ships both stories in full; the experiment picks `pairs_per_participant` at
    random per participant and randomizes which side each story appears on. Responses come
    back keyed by `story_id`; authorship is attached by the server from `pairs.json`.
    """
    payload = [{"pair_id": p["pair_id"], "length_ratio": p["length_ratio"],
                "stories": [
                    {k: p["human"][k] for k in CLIENT_FIELDS},
                    {k: p["llm"][k] for k in CLIENT_FIELDS}]}
               for p in pairs]
    header = (
        "/* TwistBench human preference study — stimulus pairs.\n"
        f"   GENERATED by src/plot_twist/scripts/build_human_eval_stimuli.py — do not edit by hand.\n"
        f"   {len(pairs)} pairs: one vetted-STRONG human gold story vs one {source_name} story,\n"
        "   length-matched. Authorship is deliberately ABSENT — the server attaches it. */\n")
    config_js = {
        "experiment_name": "twistbench_preference",
        "consent_version": "twistbench_pref_v1",
        "pairs_per_participant": config["pairs_per_participant"],
        "llm_source": source_name,
    }
    return (f"{header}\nwindow.STIMULUS_PAIRS = {json.dumps(payload, ensure_ascii=False)};\n"
            f"\nwindow.EXPERIMENT_CONFIG = {json.dumps(config_js, indent=2)};\n")


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for field in REQUIRED:
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")

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
    items = {k: v for k, v in
             json.loads(Path(config["comprehension_items_path"]).read_text()).items()
             if not k.startswith("_")}

    humans = eligible_human(index, human_texts, manifest, config)
    llms = eligible_llm(index, llm_texts, config)
    source_name = index[config["llm_source_key"]]["source"]
    print(f"Eligible: {len(humans)} human gold stories "
          f"({humans[0]['words']}-{humans[-1]['words']} words), "
          f"{len(llms)} {source_name} stories ({llms[0]['words']}-{llms[-1]['words']} words)")

    pairs, dropped = match_pairs(humans, llms, config["max_length_ratio"])
    for human, ratio in dropped:
        why = "no LLM story left" if ratio is None else f"length ratio {ratio:.2f} > {config['max_length_ratio']}"
        print(f"  DROPPED {human['id']} ({human['words']}w): {why}")
    if not pairs:
        raise ValueError("FATAL: no pairs survived length matching")
    if debug:
        pairs = pairs[:2]
        print(f"DEBUG MODE: {len(pairs)} pairs only")

    manifest_out = [{"pair_id": p["pair_id"], "length_ratio": p["length_ratio"],
                     "human_id": p["human"]["id"], "human_title": p["human"]["title"],
                     "human_words": p["human"]["words"], "llm_id": p["llm"]["id"],
                     "llm_words": p["llm"]["words"],
                     "judge_surprise": {"human": p["human"]["surprise"], "llm": p["llm"]["surprise"]},
                     "judge_coherence": {"human": p["human"]["coherence"], "llm": p["llm"]["coherence"]}}
                    for p in pairs]
    (output_dir / "pairs.json").write_text(json.dumps(manifest_out, indent=2))

    answer_key = {}
    for pair in pairs:
        for side in ("human", "llm"):
            st = pair[side]
            options, correct_id = comprehension_options(st["id"], items)
            st["comprehension"] = {"options": options}
            answer_key[st["id"]] = correct_id

    print("\nComprehension items — REVIEW THESE (correct answer first):")
    for pair in pairs:
        for side in ("human", "llm"):
            st = pair[side]
            print(f"  [{st['id'][-18:]:18}]")
            for o in st["comprehension"]["options"]:
                mark = "  * " if o["id"] == answer_key[st["id"]] else "    "
                print(f"  {mark}{o['text']}")
    print()

    js = stimuli_js(pairs, config, source_name)
    (output_dir / "stimuli-data.js").write_text(js)
    experiment_dir = Path(config["experiment_dir"])
    experiment_js = experiment_dir / "js" / "stimuli-data.js"
    if experiment_js.parent.exists():
        experiment_js.write_text(js)
        print(f"Wrote {experiment_js}")
        # The server's authorship key. It must NOT be served to the browser, so it lives in
        # server/ rather than next to the static files.
        key = {"llm_source": source_name,
               "author_kind": {**{p["human"]["id"]: "human" for p in pairs},
                               **{p["llm"]["id"]: "llm" for p in pairs}},
               "comprehension_answer": answer_key}
        (experiment_dir / "server" / "pairs.json").write_text(json.dumps(key, indent=2))
        print(f"Wrote {experiment_dir / 'server' / 'pairs.json'} (authorship key, server-side only)")
    else:
        print(f"WARNING: {experiment_js.parent} does not exist — stimuli written to "
              f"{output_dir / 'stimuli-data.js'} only")

    words = [p["human"]["words"] + p["llm"]["words"] for p in pairs]
    print(f"Pairs: {len(pairs)} -> pairs.json + stimuli-data.js "
          f"({len(js) / 1024:.0f} KB, max length ratio "
          f"{max(p['length_ratio'] for p in pairs):.2f})")
    print(f"Reading load: {min(words)}-{max(words)} words per pair; "
          f"{config['pairs_per_participant']} pairs per participant "
          f"= ~{config['pairs_per_participant'] * sum(words) / len(words) / 200:.0f} min at 200 wpm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to the YAML config")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output_dir")
    parser.add_argument("--debug", action="store_true", help="keep only the first 2 pairs")
    args = parser.parse_args()
    main(args.config, overwrite=args.overwrite, debug=args.debug)
