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


from collections import defaultdict as _defaultdict
_JUDGE_USAGE = _defaultdict(lambda: {"in": 0, "out": 0, "calls": 0})   # per-judge-model actual token usage


def reset_judge_usage() -> None:
    _JUDGE_USAGE.clear()


def get_judge_usage() -> dict:
    """Per-judge-model {in, out, calls} accumulated since the last reset (for the cost ledger)."""
    return {m: dict(u) for m, u in _JUDGE_USAGE.items()}


async def _ask(client, model: str, prompt: str, max_tokens: int = 800, attempts: int = 3) -> str | None:
    """One judge call, retried on transient provider failures, never raising. Accumulates actual token
    usage per model into ``_JUDGE_USAGE`` so scoring spend lands in the cost ledger.

    A single malformed response body from the provider used to propagate out of ``asyncio.gather``
    and abort an entire model's scoring mid-run (losing ~25 minutes of paid judging). A judge that
    cannot answer must degrade to one unjudged record, not take the run down with it.
    """
    for i in range(attempts):
        try:
            content, usage = await call_llm_async(client, messages=[{"role": "user", "content": prompt}],
                                                  model=model, temperature=0.0, max_tokens=max_tokens,
                                                  capture_usage=True)
            u = _JUDGE_USAGE[model]
            u["in"] += usage.get("in", 0); u["out"] += usage.get("out", 0); u["calls"] += 1
            return content
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
BLEND_FUSION_JUDGE_PROMPT = """You are judging whether a proposed BLEND is a genuine conceptual fusion
of two input concepts.

THE FORM OF A GENUINE BLEND (Fauconnier & Turner conceptual integration). A genuine blend fuses two
inputs into ONE new concept in which BOTH inputs contribute ORGANIZING STRUCTURE -- their relations and
roles combine, not merely their labels or a property each -- so the blend runs as a single coherent
concept and has emergent structure true of neither input alone. This is the "double-scope" form.
  Genuine example -- "computer virus" (biology + software): biology's frame (a self-replicating agent
  that infects a host and provokes an immune response) AND computing's frame (a program that spreads
  across machines and networks) BOTH organize the blend; emergent structure belonging to neither input:
  it can be quarantined; an antivirus behaves like an immune system.

It is NOT a genuine blend if it merely:
  (a) CONJUNCTION -- lists one property from each input joined by "and", a schema that fits neither
      cleanly (e.g. generic space "harnesses energy AND exhibits therapeutic effects" for
      photosynthesis + penicillin: energy from one, therapeutic from the other, stapled together); or
  (b) ADJECTIVE -- demotes one input to a modifier on the other, which stays fully itself (e.g. "a
      radioactive solar system" = a solar system that happens to be radioactive).

Now judge this blend:
Input 1 (u): '{u}'
Input 2 (v): '{v}'
Blend concept: '{concept}'
Generic space (claimed shared schema): '{generic_space}'
Structure (each triple tagged "u", "v", or "emergent"): {structure}

The blend need NOT be a real/existing thing and MAY assert properties false of either input -- that is allowed. Decide the following independently:
1. GENERIC SPACE (generic_ok): is '{generic_space}' a REAL, specific schema that BOTH '{u}' and '{v}' genuinely instantiate -- not vacuous ("both exist", "both involve change"), and not a one-from-each conjunction?
2. COHERENT (coherent): do the triples together describe a single, coherent new concept -- not two lists of properties placed side by side?
3. TAGS: verify each triple's tag. A "u" triple must be organizing structure genuinely from '{u}'; a "v" triple genuinely from '{v}'; an "emergent" triple must be true of the BLEND but of NEITHER '{u}' alone NOR '{v}' alone (a property already true of one input is inherited, not emergent). Ignore any triple whose tag is wrong.
4. SCOPE (scope): using only the correctly-tagged structure, classify the fusion: 1 = single-scope (only ONE input contributes organizing structure); 2 = double-scope (BOTH inputs contribute organizing structure); 3 = double-scope emergent (double-scope, AND at least one genuine "emergent" triple).
Return valid JSON only, exactly:
{{ "explanation": "string", "generic_ok": true or false, "coherent": true or false, "scope": 1 or 2 or 3 }}"""


def format_tagged_structure(triples: list, tags: list | None) -> str:
    """Render the blend structure with each triple's provenance tag, as the judge prompt expects
    (``(head, rel, tail)  [u|v|emergent]``). Without the tags the judge cannot verify them or assign
    the double-scope score Q_bl."""
    tags = tags or []
    lines = []
    for i, tr in enumerate(triples):
        tag = tags[i] if i < len(tags) and tags[i] else "?"
        h, r, t = (list(tr) + ["", "", ""])[:3]
        lines.append(f"  ({h}, {r}, {t})  [{tag}]")
    return "\n".join(lines) if lines else "  (none)"


async def judge_blend_fusion(client, model: str, u: str, v: str, concept: str,
                             generic_space: str, structure: list, tags: list | None = None) -> dict | None:
    """Task-specific blend judge J_Bl: genuine double-scope conceptual fusion of u,v.

    The genuine-blend form (both inputs project organizing structure; real, non-conjunction generic
    space) is baked in via description + one positive and two negative examples, so the judge shares
    the generator's definition of a blend. ``valid`` = generic_space_ok AND double_scope.
    """
    prompt = BLEND_FUSION_JUDGE_PROMPT.format(
        u=u, v=v, concept=concept or "(unnamed)", generic_space=generic_space or "(none given)",
        structure=format_tagged_structure(structure, tags))
    raw = await _ask(client, model, prompt, max_tokens=3000)   # reasoning judge (gpt-oss) needs room
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
    # Generous budget: a reasoning judge (gpt-oss) needs room to think AND emit a verdict per item;
    # 800 truncated the most generative models (e.g. gpt-5 emits up to 15) to a None verdict -> 0.
    raw = await _ask(client, model, prompt, max_tokens=3000)
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
    raw = await _ask(client, model, prompt, max_tokens=3000)  # reasoning judge + many items need room
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


# =====================================================================================
# 3-judge PANEL. Each judged criterion runs on several judge models (different vendors to
# limit same-vendor bias with the generators) and their verdicts are combined by MAJORITY
# vote. Panel functions are drop-in for the single-judge ones in score.py. Each returns the
# aggregated value plus, where useful, an "agreement" fraction (share of voting judges that
# matched the majority) for the judge-reliability report.
# =====================================================================================


def _majority(votes: list) -> tuple:
    """Majority of non-None boolean votes. Returns (verdict|None, agreement_fraction|None)."""
    v = [bool(x) for x in votes if x is not None]
    if not v:
        return None, None
    trues = sum(v)
    verdict = trues > len(v) / 2
    agree = (trues if verdict else len(v) - trues) / len(v)
    return verdict, agree


def _majority_val(votes: list) -> tuple:
    """Most-common non-None value (mode) + agreement fraction. For ORDINAL/categorical votes such as
    the Q_bl scope in {1,2,3}, where _majority's bool() coercion (any nonzero -> True) would be wrong."""
    from collections import Counter
    v = [x for x in votes if x is not None]
    if not v:
        return None, None
    val, cnt = Counter(v).most_common(1)[0]
    return val, cnt / len(v)


async def panel_factuality_batch(client, models: list, paths: list) -> list:
    """Per-path factuality by panel: each triple is majority-voted across judges. list[list[bool]|None]."""
    per_judge = await asyncio.gather(*[judge_factuality_batch(client, m, paths) for m in models])
    out = []
    for pi, path in enumerate(paths):
        judged = [per_judge[j][pi] for j in range(len(models))]  # each: list[bool]|None
        if all(x is None for x in judged):
            out.append(None)
            continue
        agg = []
        for ti in range(len(path)):
            votes = [j[ti] for j in judged if j is not None and ti < len(j)]
            agg.append(_majority(votes)[0])
        out.append(agg if all(x is not None for x in agg) else None)
    return out


async def panel_analogy(client, models: list, u: str, v: str, p1: list, p2: list) -> tuple:
    """(valid|None, agreement) for an analogy, majority-voted across judges."""
    outs = await asyncio.gather(*[judge_analogy(client, m, u, v, p1, p2) for m in models])
    return _majority([o.get("valid") if o else None for o in outs])


# --- Analogy INVENTION judge: emergent creativity (coherence + validity of the invented concept) -----
ANALOGY_INVENTION_JUDGE_PROMPT = """You are judging an ANALOGY that invents a new concept by projecting structure across a mapping.

You are given two aligned paths (position i of path_a corresponds to position i of path_b), a "projected" source concept, an "invention" name, and a "projection": a list of (source, image) triple pairs.
path_a: {path_a}
path_b: {path_b}
projected: {projected}
invention: {invention}
projection:
{projection}

Decide two things independently:
1. VALID (valid): is the invention a genuine projection through the mapping? All must hold: (a) each "source" triple is factually true of the "projected" concept in its own domain; (b) each "image" carries its "source" across the mapping -- same relationship, every entity replaced by its counterpart under the path alignment (path_a[i] corresponds to path_b[i]), with any new counterparts used consistently; (c) the invention is NOVEL to the target domain -- not already an established concept there.
2. COHERENT (coherent): do the "image" triples together describe a single, sensible new concept, not a disjointed list?

Return valid JSON only, exactly:
{{ "explanation": "string", "valid": true or false, "coherent": true or false }}"""


def _fmt_projection(projection) -> str:
    if not isinstance(projection, list):
        return "  (none)"
    lines = [f"  {p.get('source')} -> {p.get('image')}" for p in projection if isinstance(p, dict)]
    return "\n".join(lines) if lines else "  (none)"


async def judge_analogy_invention(client, model: str, path_a: list, path_b: list, projected: str,
                                  invention: str, projection: list) -> dict | None:
    """Emergent-creativity judge for the analogy invention: is it a genuine, novel projection (valid),
    and does it cohere into one concept (coherent)?"""
    prompt = ANALOGY_INVENTION_JUDGE_PROMPT.format(
        path_a=format_path(path_a), path_b=format_path(path_b),
        projected=projected or "(none)", invention=invention or "(unnamed)",
        projection=_fmt_projection(projection))
    raw = await _ask(client, model, prompt, max_tokens=3000)   # reasoning judge (gpt-oss) needs room
    return _extract_json(raw) if raw else None


async def panel_analogy_invention(client, models: list, path_a: list, path_b: list, projected: str,
                                  invention: str, projection: list) -> dict:
    """Majority-voted invention verdict: valid (J^val), coherent (J^coh), + agreement."""
    outs = await asyncio.gather(*[
        judge_analogy_invention(client, m, path_a, path_b, projected, invention, projection)
        for m in models])
    valid, agree = _majority([o.get("valid") if o else None for o in outs])
    coherent = _majority([o.get("coherent") if o else None for o in outs])[0]
    return {"valid": valid, "coherent": coherent, "agreement": agree,
            # per-judge raw output (explanation + verdicts), persisted so nothing is ever re-judged.
            "judges": [{"model": m, **(o or {})} for m, o in zip(models, outs)]}


async def panel_blend_fusion(client, models: list, u: str, v: str, concept: str,
                             generic_space: str, structure: list, tags: list | None = None) -> dict:
    """Majority-voted blend verdict: generic_ok (utility J^gen), coherent (J^coh), scope (Q_bl in {1,2,3}),
    plus agreement. Backward-compat keys (generic_space_ok / double_scope / valid) are derived from scope
    so existing score.py reads keep working until they adopt `scope` directly."""
    outs = await asyncio.gather(*[judge_blend_fusion(client, m, u, v, concept, generic_space, structure, tags)
                                  for m in models])
    generic_ok, agree = _majority([o.get("generic_ok") if o else None for o in outs])
    coherent = _majority([o.get("coherent") if o else None for o in outs])[0]
    scope = _majority_val([o.get("scope") if o else None for o in outs])[0]
    double = (scope in (2, 3)) if scope is not None else None
    return {"generic_ok": generic_ok, "coherent": coherent, "scope": scope, "agreement": agree,
            "generic_space_ok": generic_ok, "double_scope": double,
            "valid": (bool(generic_ok) and double) if double is not None else None,
            # per-judge raw output (explanation + verdicts), persisted so nothing is ever re-judged.
            "judges": [{"model": m, **(o or {})} for m, o in zip(models, outs)]}


async def panel_emergent_count(client, models: list, artifact: str, parts: list, inferences: list) -> int:
    """Emergent count by panel: an inference counts iff a majority of judges mark it true AND emergent."""
    if not inferences:
        return 0
    outs = await asyncio.gather(*[judge_emergent(client, m, artifact, parts, inferences) for m in models])
    n = 0
    for i in range(1, len(inferences) + 1):
        votes = [bool(o[str(i)].get("true") and o[str(i)].get("emergent"))
                 for o in outs if isinstance(o, dict) and isinstance(o.get(str(i)), dict)]
        if _majority(votes)[0]:
            n += 1
    return n


async def panel_blend_emergent_count(client, models: list, u: str, v: str, concept: str,
                                     structure: list, inferences: list) -> int:
    """Blend emergent-structure count by panel: majority of judges must mark each item emergent."""
    if not inferences:
        return 0
    outs = await asyncio.gather(*[judge_blend_emergent(client, m, u, v, concept, structure, inferences)
                                  for m in models])
    n = 0
    for i in range(1, len(inferences) + 1):
        votes = [bool(o[str(i)].get("emergent"))
                 for o in outs if isinstance(o, dict) and isinstance(o.get(str(i)), dict)]
        if _majority(votes)[0]:
            n += 1
    return n
