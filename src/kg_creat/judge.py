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

import asyncio
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

BLENDING_JUDGE_PROMPT = """You are judging whether two structures form a valid conceptual BLEND built
on a POLYSEMY of a single word.
Word: '{u}'
Reading 1 (ordered triples): {path1}
Reading 2 (ordered triples): {path2}
A valid blend requires that the two readings interpret the word '{u}' in two GENUINELY DIFFERENT
SENSES -- the same surface word meaning two distinct things (e.g. "Boxer" as a person vs. "Boxer" as
a dog breed; "Mercury" as the planet vs. the chemical element vs. the Roman god). It is NOT valid if
the two readings merely state two facts about the SAME sense of '{u}' (e.g. two different things one
band did, or two places one country borders). Judge three things:
1. Polysemy (the crux): do the two branches genuinely invoke two DISTINCT MEANINGS of the SAME
   spelled word, not one meaning developed twice? A homophone or near-pun with different spelling
   ("Beatles" vs "beetles") does NOT count -- it must be the identical word.
2. Shared frame: do both branches use the same relation sequence?
3. Factuality: within each sense, are the triples factually correct?
Only a path that passes all three is valid. Return valid JSON only, exactly:
{{ "explanation": "string", "sense_1": "string", "sense_2": "string", "valid": true or false }}"""


async def _ask(client, model: str, prompt: str, max_tokens: int = 800, attempts: int = 3) -> str | None:
    """One judge call, retried on transient provider failures, never raising.

    A single malformed response body from the provider used to propagate out of ``asyncio.gather``
    and abort an entire model's scoring mid-run (losing ~25 minutes of paid judging). A judge that
    cannot answer must degrade to one unjudged record, not take the run down with it.
    """
    for i in range(attempts):
        try:
            return await call_llm_async(client, messages=[{"role": "user", "content": prompt}],
                                        model=model, temperature=0.0, max_tokens=max_tokens)
        except Exception as e:  # noqa: BLE001 - provider/transport errors are all retry-or-skip
            if i == attempts - 1:
                print(f"    judge call failed after {attempts} attempts ({type(e).__name__}); "
                      f"marking unjudged")
                return None
            await asyncio.sleep(1.5 * (i + 1))
    return None


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
    raw = await _ask(client, model, prompt, max_tokens=1200)
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
    # 800, as everywhere else: a reasoning judge spends most of a small budget thinking and the
    # JSON never arrives, which silently turns satisfaction into an 'unjudged' hole in the cell.
    raw = await _ask(client, model, prompt, max_tokens=800)
    return _extract_json(raw) if raw else None


# Batched factuality: CREATE's K.2 criteria verbatim, only the I/O shape generalized to N paths.
_K2_CRITERIA = FACTUALITY_PROMPT.split("Output instructions:")[0].replace(
    "The path consists of an ordered list of triples.",
    "You will be given SEVERAL paths, each an ordered list of triples.")

FACTUALITY_BATCH_PROMPT = _K2_CRITERIA + """Output instructions:
- Return valid JSON only.
- Include exactly two keys: "explanation" and "paths".
- "paths" is a list with one entry per input path, in the SAME order, each an object with
  "path_id" (the given integer) and "judgments" (a list of "hallucinated" | "not hallucinated",
  one per triple of THAT path, in the same order as that path's triples).
- Judge every triple of every path independently, by the criteria above.
- Do not include any text outside the JSON object. No markdown, no extra keys.
The output must exactly match this schema:
{{ "explanation": "string", "paths": [ {{ "path_id": 1, "judgments": ["hallucinated" | "not hallucinated", ...] }} ] }}
Input:
{paths_block}"""


def parse_factuality_batch(response: str, triple_counts: list[int]) -> list[list[bool] | None]:
    """Map a batched response to per-path verdict lists. None for any path that doesn't line up."""
    out: list[list[bool] | None] = [None] * len(triple_counts)
    obj = _extract_json(response)
    if obj is None or not isinstance(obj.get("paths"), list):
        return out
    by_id = {}
    for entry in obj["paths"]:
        if isinstance(entry, dict) and "judgments" in entry:
            try:
                by_id[int(entry.get("path_id"))] = entry["judgments"]
            except (TypeError, ValueError):
                continue
    for i, n in enumerate(triple_counts):
        js = by_id.get(i + 1)
        if isinstance(js, list) and len(js) == n:
            out[i] = [str(j).strip().lower() == "not hallucinated" for j in js]
    return out


async def judge_factuality_batch(client, model: str, paths: list[list]) -> list[list[bool] | None]:
    """Factuality for several paths in ONE call (same criteria as the per-path judge)."""
    block = "\n".join(f"Path {i + 1}: {format_path(p)}" for i, p in enumerate(paths))
    prompt = FACTUALITY_BATCH_PROMPT.format(paths_block=block)
    raw = await _ask(client, model, prompt, max_tokens=3000)
    if raw is None:
        return [None] * len(paths)
    return parse_factuality_batch(raw, [len(p) for p in paths])


# Constraints are over relation CLASSES (data-derived), so the judge is given the class name plus
# its exemplars and asked about *kind* membership -- the call an LLM judge is reliable at.
RELATION_CONSTRAINT_PROMPTS = {
    "exclusion": (
        "You are checking whether a multi-hop path AVOIDS a forbidden KIND of relationship.\n"
        "Forbidden relationship class: {name}\n"
        "Relationships of this kind include: {exemplars} -- and any other relationship expressing "
        "that same kind of connection.\n"
        "Path (ordered triples): {path}\n"
        "Does the path FULLY AVOID relationships of this kind (no triple expresses one)?\n"
        'Return valid JSON only, exactly: {{ "explanation": "string", "satisfied": true or false }}  '
        "(satisfied = the path avoids that kind entirely)"
    ),
    "inclusion": (
        "You are checking whether a multi-hop path INCLUDES a required KIND of relationship.\n"
        "Required relationship class: {name}\n"
        "Relationships of this kind include: {exemplars} -- and any other relationship expressing "
        "that same kind of connection.\n"
        "Path (ordered triples): {path}\n"
        "Does AT LEAST ONE triple express a relationship of this kind?\n"
        'Return valid JSON only, exactly: {{ "explanation": "string", "satisfied": true or false }}'
    ),
    "ordering": (
        "You are checking the ORDER of two KINDS of relationship in a multi-hop path.\n"
        "Class A = {before_name} (e.g. {before_exemplars}).\n"
        "Class B = {after_name} (e.g. {after_exemplars}).\n"
        "Path (ordered triples): {path}\n"
        "Requirement: a relationship of class A must appear BEFORE any relationship of class B. "
        "Both kinds must be present, with the class-A one occurring first.\n"
        'Return valid JSON only, exactly: {{ "explanation": "string", "satisfied": true or false }}'
    ),
}
RELATION_CONSTRAINT_PROMPTS["inclusion_rare"] = RELATION_CONSTRAINT_PROMPTS["inclusion"]


def _fmt_ex(items, n=5):
    return ", ".join(f'"{e}"' for e in (items or [])[:n])


async def judge_relation_constraint(client, model: str, constraint: dict, triples: list) -> dict | None:
    """Judge whether a free-form path satisfies a relation-CLASS constraint (open-vocab, semantic)."""
    t = constraint["type"]
    tmpl = RELATION_CONSTRAINT_PROMPTS.get(t)
    if tmpl is None:
        raise ValueError(f"not a relation-level constraint: {t}")
    if t == "ordering":
        prompt = tmpl.format(path=format_path(triples),
                             before_name=constraint["before_name"],
                             before_exemplars=_fmt_ex(constraint.get("before_exemplars")),
                             after_name=constraint["after_name"],
                             after_exemplars=_fmt_ex(constraint.get("after_exemplars")))
    else:
        prompt = tmpl.format(path=format_path(triples), name=constraint["class_name"],
                             exemplars=_fmt_ex(constraint.get("exemplars")))
    raw = await _ask(client, model, prompt, max_tokens=800)  # see judge_categorical on small budgets
    return _extract_json(raw) if raw else None


async def judge_analogy(client, model: str, u: str, v: str, path1: list, path2: list) -> dict | None:
    prompt = ANALOGY_JUDGE_PROMPT.format(u=u, v=v, path1=format_path(path1), path2=format_path(path2))
    raw = await _ask(client, model, prompt, max_tokens=800)
    return _extract_json(raw) if raw else None


async def judge_blending(client, model: str, u: str, path1: list, path2: list) -> dict | None:
    prompt = BLENDING_JUDGE_PROMPT.format(u=u, path1=format_path(path1), path2=format_path(path2))
    raw = await _ask(client, model, prompt, max_tokens=800)
    return _extract_json(raw) if raw else None


# --- Fusion blending (docs/tracks/kg_creat/blending_fusion.md) -----------------------
# Utility for a fusion blend is judge-only: a blend is a NOVEL concept, so literal factuality of the
# blend is not the gate (F&T: elements "false or impossible in both inputs" are a feature). The judge
# checks it is a genuine, coherent fusion drawing on BOTH inputs, with a HARD gate on a real (non-
# vacuous) generic space.
BLEND_FUSION_JUDGE_PROMPT = """You are judging whether a proposed conceptual BLEND is a genuine fusion
of two input concepts, or merely a forced mashup.
Input concept 1: '{u}'
Input concept 2: '{v}'
Proposed blend concept: '{concept}'
Claimed generic space (shared schema): '{generic_space}'
Structure of the blend (ordered triples): {structure}

A valid blend requires ALL of:
1. GENERIC SPACE (hard gate): the claimed generic space is a real, SPECIFIC schema that both '{u}' and
   '{v}' genuinely fit. A vacuous or near-empty schema ("both exist", "both are things", "both involve
   change/energy") FAILS -- that is a mashup, not a blend.
2. DUAL PROJECTION: the structure draws on BOTH inputs -- it carries recognizable structure from '{u}'
   AND from '{v}', combined into one concept. If it is essentially just one of the inputs with the
   other's name attached, it FAILS.
3. COHERENCE: the blended concept holds together as a single intelligible concept (it need NOT be a
   real/existing thing, and may assert properties false of either input -- that is allowed).
Return valid JSON only, exactly:
{{ "explanation": "string", "generic_space_ok": true or false, "valid": true or false }}"""


async def judge_blend_fusion(client, model: str, u: str, v: str, concept: str,
                             generic_space: str, structure: list) -> dict | None:
    """Utility judge for a fusion blend: genuine coherent fusion of u,v (hard gate on generic space)."""
    prompt = BLEND_FUSION_JUDGE_PROMPT.format(
        u=u, v=v, concept=concept or "(unnamed)", generic_space=generic_space or "(none given)",
        structure=format_path(structure))
    raw = await _ask(client, model, prompt, max_tokens=800)
    return _extract_json(raw) if raw else None


# Emergent creativity for fusion blends: each candidate is EMERGENT STRUCTURE -- true of the blend but
# of NEITHER input alone. Unlike the generic emergent judge, "true in the real world" is NOT required
# (the blend may be fictional); we check coherence-in-the-blend + in-neither-input + non-triviality.
BLEND_EMERGENT_JUDGE_PROMPT = """You are judging the EMERGENT STRUCTURE of a conceptual blend.
The blend '{concept}' fuses two input concepts:
Input 1: '{u}'
Input 2: '{v}'
Blend structure (ordered triples): {structure}

You are given candidate EMERGENT properties of the blend. For each, decide independently:
- "emergent": is it a property that holds of the BLEND but of NEITHER '{u}' alone NOR '{v}' alone, AND
  is non-trivial (not merely restating that the two were combined, and not vacuous)? (true / false)
  If the property is already true of '{u}' by itself, or of '{v}' by itself, answer false.
  If it coherently follows only from fusing the two, answer true.

CANDIDATE EMERGENT PROPERTIES:
{inferences}

Return ONLY a JSON object mapping each number to its verdict, and nothing else. Example:
{{"1": {{"emergent": true}}, "2": {{"emergent": false}}}}"""


async def judge_blend_emergent(client, model: str, u: str, v: str, concept: str,
                               structure: list, inferences: list[str]) -> dict | None:
    """Verdict per candidate: is it emergent structure (of the blend, in neither input, non-trivial)?"""
    if not inferences:
        return {}
    inf_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(inferences))
    prompt = BLEND_EMERGENT_JUDGE_PROMPT.format(
        concept=concept or "(unnamed)", u=u, v=v, structure=format_path(structure), inferences=inf_block)
    raw = await _ask(client, model, prompt, max_tokens=800)
    return _extract_json(raw) if raw else None


def count_blend_emergent(verdict: dict | None, n_inferences: int) -> int:
    """Number of candidates the judge marked emergent (the blend emergent-creativity score)."""
    if not verdict:
        return 0
    return sum(1 for i in range(1, n_inferences + 1)
               if isinstance(verdict.get(str(i)), dict) and verdict[str(i)].get("emergent"))


EMERGENT_JUDGE_PROMPT = """You are a careful, factual judge assessing EMERGENT inferences licensed by a
combinatorial artifact. You are given an ARTIFACT (a whole), the PARTS it is built from, and a numbered
list of candidate INFERENCES the artifact is claimed to license.

For each inference, decide two things independently:
- "true": is the statement factually correct in the real world? (true / false)
- "emergent": does it follow from the ARTIFACT AS A WHOLE, yet NOT from any single PART on its own?
  (true / false). If any single part already licenses it, or if it does not actually follow from the
  whole, answer false.

ARTIFACT (the whole):
{artifact}

PARTS (each considered alone):
{parts}

CANDIDATE INFERENCES:
{inferences}

Return ONLY a JSON object mapping each inference number to its verdict, and nothing else. Example:
{{"1": {{"true": true, "emergent": true}}, "2": {{"true": true, "emergent": false}}}}"""


async def judge_emergent(client, model: str, artifact: str, parts: list[str],
                         inferences: list[str]) -> dict | None:
    """Verdict per candidate inference: is it true, and licensed by the whole but not any single part?"""
    if not inferences:
        return {}
    parts_block = "\n".join(f"- Part {i + 1}: {p}" for i, p in enumerate(parts))
    inf_block = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(inferences))
    prompt = EMERGENT_JUDGE_PROMPT.format(artifact=artifact, parts=parts_block, inferences=inf_block)
    raw = await _ask(client, model, prompt, max_tokens=800)
    return _extract_json(raw) if raw else None


def count_emergent(verdict: dict | None, n_inferences: int) -> int:
    """Number of inferences the judge marked BOTH true and emergent (the emergent-creativity score)."""
    if not verdict:
        return 0
    n = 0
    for i in range(1, n_inferences + 1):
        v = verdict.get(str(i))
        if isinstance(v, dict) and v.get("true") and v.get("emergent"):
            n += 1
    return n
