"""Render a sampled prompt spec into the model-facing prompt text.

Regime-A prompts are CREATE's base prompt (K.3, arXiv:2603.09970 p.34) adapted: we keep
its scaffolding (concrete-entity rule, disambiguation, dedup, strong/diverse guidance,
and the strict ``<answer>`` JSON output contract that ``parse.parse_paths`` expects) and
layer on our deliberate deviations -- a fixed hop count ``h``, a fixed path count ``k``,
and our constraint block replacing CREATE's terminal ``rel_b`` requirement. This keeps a
clean head-to-head with CREATE (the no-constraint baseline is a controlled CREATE variant).

Regime-B (analogy / blending) are our own ``(u, v)``-only open tasks; they reuse the shared
entity/dedup rules and the same output contract but not the many-paths diversity guidance.
"""

from __future__ import annotations

_OUTPUT_BLOCK = """Output requirements (strict):
- Return ONLY a JSON object wrapped in <answer> and </answer> tags. No other text.
- The JSON object's keys are integers starting from 1, one key per path (NOT a "paths" wrapper).
- Each path's value is a list of triples; each triple is [head entity, relationship, tail entity].
- Relationship strings must be 1-3 words. Use canonical, disambiguated entity names.
- If you cannot find a valid path, return an empty JSON object.

Required format (follow this shape exactly):
<answer>{"1": [["Entity A", "relation", "Entity B"], ["Entity B", "relation", "Entity C"]], "2": [["Entity A", "relation", "Entity D"], ["Entity D", "relation", "Entity C"]]}</answer>"""

# Blending's branches diverge instead of reconverging, so the shared example (both paths ending at
# 'Entity C') would demonstrate exactly the overlap the task forbids.
_OUTPUT_BLOCK_DIVERGENT = _OUTPUT_BLOCK.replace(
    '"2": [["Entity A", "relation", "Entity D"], ["Entity D", "relation", "Entity C"]]',
    '"2": [["Entity A", "relation", "Entity D"], ["Entity D", "relation", "Entity E"]]')

# CREATE's rules/dedup scaffolding (K.3), shared across modes.
_ENTITY_RULES = """Rules and quality constraints:
- Entities must be concrete, real-world entities only (people, organizations, works, places,
  genes, diseases, species, etc.). No abstract concepts.
- Do not ask follow-up questions; respond using the best available factual knowledge.
- Temporal connections are allowed (relationships may span different historical periods).
- Disambiguation is required: use canonical names and qualifiers where necessary
  (e.g., 'Michael Jordan (basketball)').

Deduplication:
- Do not repeat the same path, and do not repeat the same entity within a single path.
- Prefer paths that are meaningfully different (different intermediate nodes and/or
  relationships), not trivial rephrasings."""


def _ex(constraint: dict, key: str = "exemplars", n: int = 4) -> str:
    return ", ".join(f'"{e}"' for e in (constraint.get(key) or [])[:n])


def _a(word: str) -> str:
    """Article for a class name. Names are LLM-generated, so 'a affiliation-type' otherwise ships."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def _constraint_clause(constraint: dict | None) -> str:
    """The mode-specific hard constraint sentence.

    Constraints are over relation *classes* (derived from what models actually emit), not single
    labels: under an open vocabulary a specific label rarely appears verbatim, so we name the KIND
    of connection and show data-derived exemplars.
    """
    if constraint is None:
        return ""
    t = constraint["type"]
    if t == "exclusion":
        return (f"CONSTRAINT: none of your paths may use any {constraint['class_name']}-type "
                f"relationship — that is, relationships like {_ex(constraint)}, or any other "
                f"relationship expressing that same kind of connection. Avoid that kind of link entirely.")
    if t in ("inclusion", "inclusion_rare"):
        return (f"CONSTRAINT: every path must include at least one {constraint['class_name']}-type "
                f"relationship — that is, a relationship like {_ex(constraint)}, or another "
                f"expressing that same kind of connection.")
    if t == "ordering":
        return (f"CONSTRAINT: in every path, {_a(constraint['before_name'])} {constraint['before_name']}-type "
                f"relationship (like {_ex(constraint, 'before_exemplars', 3)}) must appear BEFORE any "
                f"{constraint['after_name']}-type relationship (like {_ex(constraint, 'after_exemplars', 3)}).")
    if t == "categorical":
        return (f"CONSTRAINT: every path must pass through at least one intermediate entity "
                f"that is a kind of '{constraint['type_label']}'.")
    raise ValueError(f"unknown Regime-A constraint type: {t}")


def _regime_a_prompt(spec: dict) -> str:
    u, v, k = spec["u_label"], spec["v_label"], spec["k"]
    clause = _constraint_clause(spec["constraint"])
    clause_block = f"\n{clause}\n" if clause else "\n"
    return f"""Query: What are different ways in which '{u}' is connected to '{v}'?

Task: Identify how these two real-world entities are connected by producing {k} DISTINCT
connection paths. A connection path is a sequence of factual triples (head, relationship,
tail) forming a continuous chain: consecutive triples share an entity, the first triple's
head is '{u}', and the last triple's tail is '{v}'.

BE CREATIVE: your goal is not merely to connect '{u}' and '{v}', but to connect them in the most
NOVEL, SURPRISING, and non-obvious ways you can imagine. Favour remote, unexpected intermediate
entities and imaginative routes over the first obvious link -- reach across distant domains. Every
triple must still be factually true.

Produce {k} DISTINCT paths that are as different from one another as possible (different intermediate
entities and relationships, spanning different domains; do not over-concentrate on the most obvious
domain). Prefer paths through intermediate entities rather than a single direct link, and prefer
specific, distinctive relationships over broad generic ones (e.g., 'attended', 'lives in').
{clause_block}
{_ENTITY_RULES}

{_OUTPUT_BLOCK}"""


def _analogy_prompt(spec: dict) -> str:
    u, v = spec["u_label"], spec["v_label"]
    return f"""Task: You are given two concepts: '{u}' and '{v}'. Find a deep analogy between them --
a shared relational structure in which '{u}' plays a role in its domain analogous to the role
'{v}' plays in its domain.

BE CREATIVE: the more SURPRISING and non-obvious the analogy -- the more distant and unexpected the
two domains -- the better, as long as the structural mapping is genuine and every triple is factual.

Produce exactly TWO paths:
- Path 1: factual triples describing '{u}' within its own domain.
- Path 2: factual triples describing '{v}' within its own domain.
CRITICAL: the two paths must use the EXACT SAME relationship word at every position -- if Path 1's
relations are [r1, r2, r3], Path 2's relations must be the identical words [r1, r2, r3], in the same
order. Do NOT paraphrase or substitute synonyms (e.g. if Path 1 uses 'awards', Path 2 must also use
'awards', not 'grants'). Only the ENTITIES differ between the two paths; the relations are identical.
The two paths must use disjoint, concrete, canonically-named entities that play corresponding roles.
Do not repeat an entity within a path.

{_OUTPUT_BLOCK}"""


def _blending_prompt(spec: dict) -> str:
    """Blending as antanaclasis: the anchor is fixed; the model must find a VALID POLYSEMY of it.

    A true blend hinges on one word carrying two genuinely different senses (the C6 'Boxer' figure:
    Boxer-the-athlete vs Boxer-the-dog). So the task is not "two facts about the anchor" but "two
    *meanings* of the anchor", each developed under a shared relational frame. Finding a second sense
    for an arbitrary given concept is the hard, creative act -- and for many anchors it is not
    possible, which is why baseline success is a measurement, not a precondition.
    """
    u = spec["u_label"]
    return f"""Task: You are given ONE concept: '{u}'. Your challenge is to find a VALID POLYSEMY for it
-- a second, genuinely different meaning that the word '{u}' can be read as -- and build a conceptual
BLEND that holds both meanings at once.

A polysemy reads the SAME word in two unrelated senses. For example, the word "Boxer":
  Path 1:  ["Boxer", "is a", "Athlete"], ["Athlete", "chases", "Records"]
  Path 2:  ["Boxer", "is a", "Dog"],     ["Dog", "chases", "Squirrels"]
"Boxer" means a person in one reading and a dog breed in the other. Both paths share the SAME frame
("is a ... chases ...") but land in completely different domains. That double meaning is the blend.

BE CREATIVE: the more SURPRISING and non-obvious the second meaning -- the more distant the two senses
-- the better the blend, as long as it is a genuine sense of the word and every triple is factual.

Produce exactly TWO paths:
- Both paths start at '{u}'.
- Path 1 develops '{u}' under one meaning; Path 2 develops '{u}' under a DIFFERENT meaning of the SAME word.
- The two readings must be genuinely distinct SENSES of the word -- NOT two facts about the same thing.
- The second meaning must be the SAME spelled word read differently -- not a homophone or near-pun
  (e.g. "Beatles" -> "beetles" does NOT count; it must be the identical word).
- Use the EXACT SAME relationship word at every position in both paths; only the entities differ.
- The two paths share no entity except '{u}'; the more unrelated the two meanings, the better the blend.

Use concrete, factual, canonically-named entities and do not repeat an entity within a path.
If '{u}' genuinely has no valid second meaning, return an empty JSON object.

{_OUTPUT_BLOCK_DIVERGENT}"""


def build_prompt(spec: dict) -> str:
    """Render a prompt spec (from sample.py) into model prompt text.

    Relations are free-form (open vocabulary, CREATE-style); the G_c-derived vocabulary is used
    only on the graph/sampling side, not imposed on the model.
    """
    mode = spec["mode"]
    if mode == "analogy":
        return _analogy_prompt(spec)
    if mode == "blending":
        return _blending_prompt(spec)
    return _regime_a_prompt(spec)
