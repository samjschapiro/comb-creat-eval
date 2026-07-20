"""LLM judges for the execution axis.

Factuality (Regime A + B): CREATE's per-triple hallucination judge (Prompt K.2,
arXiv:2603.09970 p.33) run verbatim on ``gpt-oss-120b`` for comparability. Semantic
satisfaction (Regime B only): analogy role-correspondence and blending pivot-sense
judges -- these are the E/F ``sat`` axis and, unlike A-D, are judge-based (they need
their own reliability spot-check).

All calls route through the OpenRouter client (``get_async_client_openrouter``) so the
judge is identical whether or not elicitation used a local server, and its spend is
budget-tracked.
"""

from __future__ import annotations

import json
import re

from src.dat_eval.llm import call_llm_async

# --- CREATE factuality prompt K.2 (verbatim, p.33) -----------------------------------
FACTUALITY_PROMPT = """You are a fact-checking expert evaluating a multi-hop factual path for factual accuracy and logical validity.
The path consists of an ordered list of triples. Each triple has the form: (subject, relation, object)
Your task is to evaluate EACH triple in the given path and determine whether it is hallucinated or not. Use the path only for entity disambiguation and contextual grounding. Do not infer unsupported facts.
Evaluation criteria (apply independently to each triple):
1. Entity existence and disambiguation:
- Verify that both the subject and object are real, identifiable entities (people, places, organizations, events, or concepts).
- Use the path context only when necessary to disambiguate entities.
- If either entity is fabricated or incorrectly identified, mark the triple as hallucinated.
2. Relationship validity:
- Verify that the stated relation describes a real, historically valid, currently valid, or plausibly valid relationship between the subject and object.
- If the relationship is plausible and could reasonably have occurred, even if not directly verifiable, mark it as not hallucinated.
- If it is niche information about the subject and only trivia heads would know compared to common knowledge then do not mark it as hallucinated.
3. Hallucination detection:
- Mark the triple as hallucinated if: an entity is fabricated or does not exist; the relation is fabricated, nonsensical, or incorrect; the asserted connection clearly contradicts well-known facts; the triple introduces made-up properties or roles.
4. Directionality handling: If the relation is ambiguous or commonly bidirectional, evaluate both directions. If either direction corresponds to a true or plausible relationship, mark the triple as not hallucinated.
Output instructions:
- Return valid JSON only.
- Include exactly two keys: "explanation" and "judgments".
- Under "explanation", provide a natural-language explanation evaluating the triples in the path.
- Under "judgments", provide the final hallucination judgment for each triple as a list, in the same order as the path.
- Do not include any text outside the JSON object. Do not include markdown or extra keys.
The output must exactly match this schema:
{{ "explanation": "string", "judgments": ["hallucinated" | "not hallucinated", ...] }}
Input:
Path: {path}"""

ANALOGY_JUDGE_PROMPT = """You are evaluating whether a proposed ANALOGY between two concepts is valid.
Concept A: '{u}'   Concept B: '{v}'
Structure for A (ordered triples): {path1}
Structure for B (ordered triples): {path2}
A valid analogy requires that the two structures are relationally parallel AND that entities at
corresponding positions play corresponding roles (a shared relational system), not merely that the
relation words match. Judge holistically.
Return valid JSON only, exactly: {{ "explanation": "string", "valid": true or false }}"""

BLENDING_JUDGE_PROMPT = """You are evaluating whether a path is a valid conceptual BLEND connecting two concepts.
Concept A: '{u}'   Concept B: '{v}'
Path (ordered triples): {path}
A valid blend passes through a PIVOT entity that is invoked in two different, colliding senses -- one
sense linking it to A's domain and another to B's domain -- so the two domains genuinely fuse at the
pivot (e.g. a word/name with two meanings, or an entity bridging two frames). Identify the pivot and
judge whether it truly carries two distinct senses.
Return valid JSON only, exactly: {{ "explanation": "string", "pivot": "string", "valid": true or false }}"""


def format_path(triples: list) -> str:
    """Render triples as an ordered '(s, r, o)' list for a judge prompt."""
    return "[" + ", ".join(f"({t[0]}, {t[1]}, {t[2]})" for t in triples) + "]"


def _extract_json(text: str) -> dict | None:
    """Best-effort parse of the first JSON object in a response."""
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return None


def parse_factuality(response: str, n_triples: int) -> list[bool] | None:
    """Map CREATE's judgments to per-triple factual flags (True = not hallucinated).

    Returns None (a judge-failure channel) if the JSON is unparseable or the judgment
    count doesn't match the number of triples -- we never silently pad/guess.
    """
    obj = _extract_json(response)
    if obj is None or "judgments" not in obj:
        return None
    judgments = obj["judgments"]
    if not isinstance(judgments, list) or len(judgments) != n_triples:
        return None
    return [str(j).strip().lower() == "not hallucinated" for j in judgments]


async def judge_factuality(client, model: str, triples: list) -> list[bool] | None:
    """Per-triple factuality for one path. None on judge failure (unparseable/mismatch)."""
    prompt = FACTUALITY_PROMPT.format(path=format_path(triples))
    raw = await call_llm_async(client, messages=[{"role": "user", "content": prompt}],
                               model=model, temperature=0.0, max_tokens=1200)
    if raw is None:
        return None
    return parse_factuality(raw, len(triples))


CATEGORICAL_JUDGE_PROMPT = """Does the following path pass through at least one intermediate entity that is a kind of '{type_label}'?
The intermediate entities are those that are neither the first nor the last entity in the path.
Path (ordered triples): {path}
Return valid JSON only, exactly: {{ "explanation": "string", "satisfied": true or false }}"""


async def judge_categorical(client, model: str, type_label: str, triples: list) -> dict | None:
    """Judge whether an interior entity is of the required type (open-KG entities lack local types)."""
    prompt = CATEGORICAL_JUDGE_PROMPT.format(type_label=type_label, path=format_path(triples))
    raw = await call_llm_async(client, messages=[{"role": "user", "content": prompt}],
                               model=model, temperature=0.0, max_tokens=400)
    return _extract_json(raw) if raw else None


RELATION_CONSTRAINT_PROMPTS = {
    "exclusion": (
        "You are checking whether a multi-hop path AVOIDS a forbidden type of relationship.\n"
        "Forbidden relationship type: '{label}' -- also forbid any relationship with the SAME MEANING "
        "(synonyms/paraphrases; e.g. for 'instance of': 'is a', 'is a type of', 'is an example of').\n"
        "Path (ordered triples): {path}\n"
        "Does the path FULLY AVOID the forbidden relationship type (no triple expresses it)?\n"
        'Return valid JSON only, exactly: {{ "explanation": "string", "satisfied": true or false }}  '
        "(satisfied = the path avoids it)"
    ),
    "inclusion": (
        "You are checking whether a multi-hop path INCLUDES a required type of relationship.\n"
        "Required relationship type: '{label}' -- a relationship with this meaning counts, including "
        "synonyms/paraphrases.\n"
        "Path (ordered triples): {path}\n"
        "Does AT LEAST ONE triple express the required relationship type?\n"
        'Return valid JSON only, exactly: {{ "explanation": "string", "satisfied": true or false }}'
    ),
    "ordering": (
        "You are checking the ORDER of two relationship types in a multi-hop path.\n"
        "Path (ordered triples): {path}\n"
        "Requirement: a relationship meaning '{before}' must appear BEFORE any relationship meaning "
        "'{after}'. Both types (or synonyms/paraphrases) must be present, with the '{before}'-type first.\n"
        'Return valid JSON only, exactly: {{ "explanation": "string", "satisfied": true or false }}'
    ),
}


async def judge_relation_constraint(client, model: str, constraint: dict, triples: list) -> dict | None:
    """Judge whether a free-form path satisfies a relation-level constraint (open-vocab, semantic)."""
    t = constraint["type"]
    tmpl = RELATION_CONSTRAINT_PROMPTS.get(t)
    if tmpl is None:
        raise ValueError(f"not a relation-level constraint: {t}")
    if t == "ordering":
        prompt = tmpl.format(path=format_path(triples),
                             before=constraint["before_label"], after=constraint["after_label"])
    else:
        prompt = tmpl.format(path=format_path(triples), label=constraint["relation_label"])
    raw = await call_llm_async(client, messages=[{"role": "user", "content": prompt}],
                               model=model, temperature=0.0, max_tokens=500)
    return _extract_json(raw) if raw else None


async def judge_analogy(client, model: str, u: str, v: str, path1: list, path2: list) -> dict | None:
    prompt = ANALOGY_JUDGE_PROMPT.format(u=u, v=v, path1=format_path(path1), path2=format_path(path2))
    raw = await call_llm_async(client, messages=[{"role": "user", "content": prompt}],
                               model=model, temperature=0.0, max_tokens=800)
    return _extract_json(raw) if raw else None


async def judge_blending(client, model: str, u: str, v: str, path: list) -> dict | None:
    prompt = BLENDING_JUDGE_PROMPT.format(u=u, v=v, path=format_path(path))
    raw = await call_llm_async(client, messages=[{"role": "user", "content": prompt}],
                               model=model, temperature=0.0, max_tokens=800)
    return _extract_json(raw) if raw else None
