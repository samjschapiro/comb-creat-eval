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

# All three tasks emit a SET of items (as many as the model can produce), each item a JSON object.
# Every item carries an "inferences" field: the true statements the WHOLE artifact licenses that its
# parts do not -- the emergent-creativity signal we score. Wording is deliberately explicit so the
# emitted structure is controllable and unambiguous to the parser.
_OUTPUT_BLOCK_ASSOC = """Output requirements (strict):
- Return ONLY a JSON array wrapped in <answer> and </answer> tags. No other text, before or after.
- Each element of the array is ONE connection: an object with exactly two keys, "path" and "inferences".
- "path" is a list of triples; each triple is [head entity, relationship, tail entity] forming a
  continuous chain from the first entity to the last.
- "inferences" is a list of short, factual statements that the WHOLE path reveals but that no single
  triple in it reveals on its own. Give only genuine ones; use [] if the path licenses none.
- Relationship strings must be 1-3 words. Use canonical, disambiguated entity names.
- List one object per connection. To stop, simply end the array -- do not pad with weak connections.
- If you can find no valid connection, return an empty array: <answer>[]</answer>.

Required format (follow this shape exactly):
<answer>[{"path": [["A", "r1", "B"], ["B", "r2", "C"]], "inferences": ["A short true statement the whole path reveals."]}, {"path": [["A", "s1", "D"], ["D", "s2", "C"]], "inferences": []}]</answer>"""

_OUTPUT_BLOCK_ANALOGY = """Output requirements (strict):
- Return ONLY a JSON array wrapped in <answer> and </answer> tags. No other text, before or after.
- Each element of the array is ONE analogy: an object with keys "path_a", "path_b", and "inferences".
- "path_a" and "path_b" are each a list of triples; each triple is [head entity, relationship, tail entity].
- Within one analogy, "path_a" and "path_b" MUST have the same number of triples and the IDENTICAL
  relationship word at every position (only the entities differ).
- "inferences" is a list of short, true statements the mapping licenses by transfer -- things it
  predicts about either concept from the other's structure, that you could not claim without the
  analogy. Use [] if none.
- Relationship strings must be 1-3 words. Use canonical, disambiguated entity names.
- List one object per analogy. To stop, simply end the array -- do not pad with weak analogies.
- If you can find no valid analogy, return an empty array: <answer>[]</answer>.

Required format (follow this shape exactly):
<answer>[{"path_a": [["A", "r1", "B"], ["B", "r2", "C"]], "path_b": [["D", "r1", "E"], ["E", "r2", "F"]], "inferences": ["A true statement the mapping predicts about A or D."]}]</answer>"""

_OUTPUT_BLOCK_BLENDING = """Output requirements (strict):
- Return ONLY a JSON array wrapped in <answer> and </answer> tags. No other text, before or after.
- Each element of the array is ONE blend (one polysemy): an object with keys "sense_1", "sense_2", and "inferences".
- "sense_1" and "sense_2" are each a list of triples; each triple is [head entity, relationship, tail entity].
  Both lists must start at the given anchor word.
- Within one blend, "sense_1" and "sense_2" MUST have the same number of triples and the IDENTICAL
  relationship word at every position (only the entities differ).
- "inferences" is a list of short, true statements that hold only when BOTH senses are read at once --
  things neither sense gives on its own. Use [] if none.
- Relationship strings must be 1-3 words. Use canonical, disambiguated entity names.
- List one object per distinct second meaning. To stop, simply end the array -- do not pad with non-genuine senses.
- If the word has no valid second meaning, return an empty array: <answer>[]</answer>.

Required format (follow this shape exactly):
<answer>[{"sense_1": [["Boxer", "is a", "Athlete"], ["Athlete", "chases", "Records"]], "sense_2": [["Boxer", "is a", "Dog"], ["Dog", "chases", "Squirrels"]], "inferences": ["A true statement that holds only under both readings."]}]</answer>"""

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
    u, v = spec["u_label"], spec["v_label"]
    clause = _constraint_clause(spec["constraint"])
    clause_block = f"\n{clause}\n" if clause else "\n"
    return f"""Query: What are different ways in which '{u}' is connected to '{v}'?

Task: Identify how these two real-world entities are connected by producing as MANY DISTINCT connection
paths as you can. A connection path is a sequence of factual triples (head, relationship, tail) forming
a continuous chain: consecutive triples share an entity, the first triple's head is '{u}', and the last
triple's tail is '{v}'.

We reward four things in every connection:
- TRUE: every triple is factually correct.
- REMOTE: the intermediate concepts sit in domains far from the two endpoints and from each other --
  reach across distant fields rather than taking the first obvious link.
- UNCOMMON: build the path from rare, specific concepts and relations, not the broad, generic ones most
  people would give.
- GENERATIVE: the path AS A WHOLE should reveal a true inference that no single link in it reveals on
  its own -- and you must state those inferences.

Produce as MANY DISTINCT connections as you can, most surprising first -- do not stop at a fixed number;
list every genuine connection you can find, and make them as different from one another as possible
(different intermediate entities and relations, spanning different domains).
{clause_block}
{_ENTITY_RULES}

{_OUTPUT_BLOCK_ASSOC}"""


def _analogy_prompt(spec: dict) -> str:
    u, v = spec["u_label"], spec["v_label"]
    return f"""Task: You are given two concepts: '{u}' and '{v}'. Find as MANY deep analogies between them
as you can. In each analogy, '{u}' plays a role in its domain analogous to the role '{v}' plays in its
domain, through a shared relational structure.

We reward four things in every analogy:
- TRUE: every triple is factually correct, and the mapping is a genuine structural correspondence.
- REMOTE: the two domains are as distant and unexpected as possible.
- UNCOMMON: use rare, specific roles and relations, not the obvious ones most people would give.
- GENERATIVE: state the true inferences the analogy licenses by transfer -- things it predicts about
  '{v}' from '{u}'s structure, and vice versa about '{u}', that you could not claim without the mapping.

Each analogy is exactly TWO paths that share an identical relation sequence:
- "path_a": factual triples describing '{u}' within its own domain, beginning at '{u}'.
- "path_b": factual triples describing '{v}' within its own domain, beginning at '{v}'.
CRITICAL (holds within each analogy): the "path_a" and "path_b" paths must use the EXACT SAME
relationship word at every position -- if "path_a" relations are [r1, r2, r3], "path_b" relations must
be the identical words [r1, r2, r3], in the same order. Do NOT paraphrase or substitute synonyms (e.g.
if "path_a" uses 'awards', "path_b" must also use 'awards', not 'grants'). Only the ENTITIES differ.
Use disjoint, concrete, canonically-named entities that play corresponding roles, and do not repeat an
entity within a path.

Produce as MANY DISTINCT analogies as you can, most surprising first -- do not stop at a fixed number;
give every genuine analogy you can find, and make them as different from one another as possible
(different relation sequences, different domains). Stop only when you can find no further genuine analogy.

{_OUTPUT_BLOCK_ANALOGY}"""


def _blending_prompt(spec: dict) -> str:
    """Blending as antanaclasis: the anchor is fixed; the model must find a VALID POLYSEMY of it.

    A true blend hinges on one word carrying two genuinely different senses (the C6 'Boxer' figure:
    Boxer-the-athlete vs Boxer-the-dog). So the task is not "two facts about the anchor" but "two
    *meanings* of the anchor", each developed under a shared relational frame. Finding a second sense
    for an arbitrary given concept is the hard, creative act -- and for many anchors it is not
    possible, which is why baseline success is a measurement, not a precondition.
    """
    u = spec["u_label"]
    return f"""Task: You are given ONE concept: '{u}'. Find as MANY VALID POLYSEMIES of it as you can --
each a second, genuinely different meaning that the word '{u}' can be read as -- and for each, build a
conceptual BLEND that holds both meanings at once.

A polysemy reads the SAME word in two unrelated senses. For example, the word "Boxer":
  "sense_1": [["Boxer", "is a", "Athlete"], ["Athlete", "chases", "Records"]]
  "sense_2": [["Boxer", "is a", "Dog"], ["Dog", "chases", "Squirrels"]]
"Boxer" means a person in one reading and a dog breed in the other. Both senses share the SAME frame
("is a ... chases ...") but land in completely different domains. That double meaning is one blend.

We reward four things in every blend:
- TRUE: every triple is factually correct, and both readings are genuine senses of the word.
- REMOTE: the two senses are as distant and unrelated as possible.
- UNCOMMON: pick rare, non-obvious second meanings, not the first one that comes to mind.
- GENERATIVE: state the true inferences that hold only when BOTH senses are read at once -- things
  neither sense gives on its own.

Each blend is exactly TWO paths, both beginning at '{u}':
- "sense_1": develops '{u}' under one meaning.
- "sense_2": develops '{u}' under a DIFFERENT meaning of the SAME word.
Rules that hold within each blend:
- The two readings must be genuinely distinct SENSES of the word -- NOT two facts about the same thing.
- The second meaning must be the SAME spelled word read differently -- not a homophone or near-pun
  (e.g. "Beatles" -> "beetles" does NOT count; it must be the identical word).
- "sense_1" and "sense_2" must use the EXACT SAME relationship word at every position; only the entities differ.
- The two senses share no entity except '{u}'; the more unrelated the two meanings, the better the blend.
Use concrete, factual, canonically-named entities, and do not repeat an entity within a path.

Produce as MANY DISTINCT blends (distinct second meanings) as you can, most surprising first -- do not
stop at a fixed number. Stop only when the word has no further genuine second meaning.

{_OUTPUT_BLOCK_BLENDING}"""


def _anagram_prompt(spec: dict) -> str:
    """Anagram (exploratory-creativity) prompt: rearrange the entity's letters into a DISTANT word.

    Single stimulus, string output (not triples). Novelty = semantic remoteness of the anagram from
    the source (same measure as the combinatorial tasks); utility is deterministic (exact letters +
    real word), so this task is judge-free.
    """
    u = spec["u_label"]
    return f"""Task: You are given ONE word or name: '{u}'. Rearrange ALL of its letters -- using every
letter exactly once, ignoring spaces and capitalization -- into a NEW, real word or phrase (an
ANAGRAM of '{u}') whose MEANING is as DIFFERENT and DISTANT from '{u}' as possible.

An anagram uses exactly the same letters as the original, in a different order. Examples (same
letters, unrelated meaning):
  "stressed" -> "desserts"   (a feeling -> a food)
  "Elvis"    -> "levis"      (a musician -> a jeans brand)
  "listen"   -> "tinsel"     (perception -> a decoration)

BE CREATIVE: the further the anagram's meaning is from '{u}' -- a completely different domain -- the
better. Every anagram must (1) be a real word or phrase, and (2) use EXACTLY the letters of '{u}':
the same letters, the same number of each, none added or dropped.

Produce as MANY DISTINCT valid anagrams as you can find, most semantically distant first -- do not stop
at a fixed number; list every one you can. If '{u}' genuinely has no valid anagram, return an empty JSON
object.

Output requirements (strict):
- Return ONLY a JSON object wrapped in <answer> and </answer> tags. No other text.
- Keys are integers starting from 1; each value is a single string (the anagram).
<answer>{{"1": "desserts", "2": "..."}}</answer>"""


def build_prompt(spec: dict) -> str:
    """Render a prompt spec (from sample.py) into model prompt text.

    Relations are free-form (open vocabulary, CREATE-style); the G_c-derived vocabulary is used
    only on the graph/sampling side, not imposed on the model.
    """
    mode = spec["mode"]
    if mode == "anagram":
        return _anagram_prompt(spec)
    if mode == "analogy":
        return _analogy_prompt(spec)
    if mode == "blending":
        return _blending_prompt(spec)
    return _regime_a_prompt(spec)
