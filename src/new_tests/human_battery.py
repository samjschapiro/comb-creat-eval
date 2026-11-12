"""The human DRAT study's task battery (DRAT, RAT, SCTT) rendered as LLM prompts.

Wording is taken verbatim from the participant-facing app (samjschapiro/drat_human_experiments,
origin/main b4bd3bf, 2026-10-16: js/sessions/session1.js, session2.js, js/tests/*.js,
js/battery-data.js), with three kinds of change only: HTML stripped; timing/practice/button
sentences that do not apply to a model dropped; and a final line asking for a machine-readable
answer. The main run's later app builds are not on GitHub, so their wording is assumed unchanged.
"""

# SCTT scenarios as shown to participants (battery-data.js; = SCTT preprint Appendix A).
SCTT_ITEMS = {
    "sctt_rq_1": "You travel on a spaceship to a new planet outside of our galaxy. What scientific questions could you ask about this planet?",
    "sctt_rq_2": "You created the first vehicle that can reach an unexplored part of the ocean floor. What scientific questions could you ask about this part of the ocean floor?",
    "sctt_rq_3": "You travel to a jungle that contains no human life and is completely unknown to the scientific community. What scientific questions could you ask about this jungle?",
    "sctt_rq_4": "You discover a remote island with people who communicate without speaking. What scientific questions could you ask about this?",
    "sctt_hyp_1": "You just started going to a new school. You notice that more students sing in the hallways at this school than at your old school. What hypotheses do you have about why that is?",
    "sctt_hyp_2": "You notice that dogs seem to like one of your friends, but cats seem to like another friend. What hypotheses do you have about why that is?",
    "sctt_hyp_3": "You notice that the water in one lake is warmer than the water in another lake even though they both get the same amount of sunlight. What hypotheses do you have about why that is?",
    "sctt_hyp_4": "You look outside one night and see stars disappearing one by one from the sky. What hypotheses do you have about why that is?",
    "sctt_exp_1": "Your cousin claims they can read minds, but it only works when a person doesn't know their mind is being read. How could you test to see if this super power is real?",
    "sctt_exp_2": "You think students at your school are friendlier than students at most schools. How could you test that hypothesis?",
    "sctt_exp_3": "You think some animals have a sense of humor that humans don't usually understand. How could you test that hypothesis?",
    "sctt_exp_4": "You think birds can understand humans but just choose to ignore us because what humans say isn't important to their lives. How could you test that hypothesis?",
}

DRAT_INSTRUCTIONS = (
    "Word Generation with Anchor Words\n\n"
    "On each screen you will see two or three anchor words. Your task is to enter 10 nouns that are as "
    "different from each other as possible, in all meanings and uses of the words, while each noun still "
    "relates to all of the anchor words. A noun can relate to the anchors in any way, literally or "
    "metaphorically.\n\n"
    "Example: for the anchors candle / romance / campfire, two possible nouns are spark (a candle is lit "
    "with a spark, a romance begins with a spark, a campfire throws sparks) and dinner (a candlelit dinner, "
    "a romantic dinner, a campfire dinner). These two nouns are very different in meaning, yet each relates "
    "to all three anchors. Your other nouns should likewise be as different from each other as possible.\n\n"
    "Rules:\n"
    "- Do not use different forms of the same word (e.g., not both “run” and “running”).\n"
    "- Only single words in English.\n"
    "- Only nouns (e.g., things, objects, concepts).\n"
    "- No proper nouns (e.g., no specific people, places, or brands).\n"
    "- No specialized vocabulary (e.g., no technical terms).\n"
    "- Do not use the anchor words or any form of them."
)

RAT_INSTRUCTIONS = (
    "Word Connections\n\n"
    "On each screen you will see three words. Your task is to type the single word that connects all three. "
    "The connecting word forms a compound word or a common phrase with each of them, and it can come before "
    "or after each word.\n\n"
    "Example: sleeping / bean / trash → bag (sleeping bag, beanbag, trash bag)"
)

SCTT_INSTRUCTIONS = (
    "Science Scenarios\n\n"
    "In this task, you will think of creative ideas related to scientific problems. You will read a series "
    "of short scenarios. For each one, type three different responses in the boxes provided.\n\n"
    "The goal is to come up with ideas that are original (unusual, unique, innovative) while also being "
    "scientifically possible (practical and feasible)."
)

SCTT_SECTIONS = {
    "rq": "For each scenario, list three research questions that are creative, different from each other, and scientifically testable.",
    "hyp": "For each scenario, list three hypotheses (possible explanations) that are creative, different from each other, and scientifically plausible.",
    "exp": "For each scenario, describe three ways to test the claim that are creative, different from each other, and feasible.",
}


def parse_list(content):
    """Recover the JSON list a model was asked for; None if no list can be found.

    Tolerates code fences, escaped quotes (`[\\"a\\", ...]`) and trailing prose. Does not repair
    content: a model that declines, or writes something other than a list, gets None.
    """
    import ast
    import json
    import re
    if not content:
        return None
    m = re.search(r"\[.*?\]", content, re.S) or re.search(r"\{.*?\}", content, re.S)  # some models write {"a", "b"}
    if not m:
        return None
    raw = m.group(0)
    if raw.startswith("{"):
        quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', raw.replace('\\"', '"'))
        return [q.strip() for q in quoted if q.strip()] or None
    for candidate in (raw, raw.replace('\\"', '"')):
        for loader in (json.loads, ast.literal_eval):
            try:
                out = loader(candidate)
            except (ValueError, SyntaxError, TypeError):
                continue
            if isinstance(out, list):
                return [str(x).strip() for x in out if str(x).strip()]
    quoted = re.findall(r'"((?:[^"\\]|\\.)*)"', raw.replace('\\"', '"'))
    return [q.strip() for q in quoted if q.strip()] or None


def drat_prompt(anchors):
    return (
        f"{DRAT_INSTRUCTIONS}\n\n"
        f"Anchor words: {' & '.join(anchors)}\n\n"
        "Enter 10 nouns that are as different from each other as possible, while each noun still relates to "
        "all of the anchor words.\n\n"
        "Respond with ONLY a JSON array of exactly 10 nouns, like: "
        '["noun1", "noun2", "noun3", "noun4", "noun5", "noun6", "noun7", "noun8", "noun9", "noun10"]'
    )


def rat_prompt(cues):
    return (
        f"{RAT_INSTRUCTIONS}\n\n"
        f"{' / '.join(cues)}\n\n"
        "What single word connects all three?\n\n"
        "Respond with ONLY the single connecting word in lowercase. No explanation."
    )


def sctt_prompt(item_id):
    section = item_id.split("_")[1]
    return (
        f"{SCTT_INSTRUCTIONS}\n\n"
        f"{SCTT_SECTIONS[section]}\n\n"
        f"{SCTT_ITEMS[item_id]}\n\n"
        "Give 3 responses.\n\n"
        'Respond with ONLY a JSON array of exactly 3 strings, one per response, like: ["response 1", "response 2", "response 3"]'
    )
