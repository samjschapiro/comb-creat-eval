"""Render a sampled prompt spec into the model-facing prompt text.

Regime-A prompts are CREATE's base prompt (K.3, arXiv:2603.09970 p.34) adapted: we keep
its scaffolding (entity-recognizability rule, disambiguation, dedup, strong/diverse guidance,
and the strict ``<answer>`` JSON output contract that ``parse.parse_paths`` expects) and
layer on our deliberate deviations -- a fixed hop count ``h``, a fixed path count ``k``,
and our constraint block replacing CREATE's terminal ``rel_b`` requirement. This keeps a
clean head-to-head with CREATE (the no-constraint baseline is a controlled CREATE variant).

Regime-B (analogy / blending) are our own ``(u, v)``-only open tasks; they reuse the shared
entity/dedup rules and the same output contract but not the many-paths diversity guidance.

Prompt strings keep each paragraph and each bullet on a single physical line (no mid-sentence
line breaks): a newline in a triple-quoted string is a literal newline in the model-facing text.
"""

from __future__ import annotations

_OUTPUT_BLOCK = """Output requirements (strict):
- Return ONLY a JSON object wrapped in <answer> and </answer> tags. No other text.
- The JSON object's keys are integers starting from 1, one key per path (NOT a "paths" wrapper).
- Each path's value is a list of triples; each triple is [head entity, relationship, tail entity].
- Relationship strings must be 1-3 words. Keep every entity SHORT and CONCRETE: a few plain, recognizable words naming a real concept -- not a descriptive phrase, a clause, an invented CamelCase compound, or words joined by a dash into a coinage (write "petition threshold", never "PetitionThreshold" or "petition-threshold").
- If you cannot find a valid path, return an empty JSON object.

Required format (follow this shape exactly):
<answer>{"1": [["Entity A", "relation", "Entity B"], ["Entity B", "relation", "Entity C"]], "2": [["Entity A", "relation", "Entity D"], ["Entity D", "relation", "Entity C"]]}</answer>"""

# Blending's branches diverge instead of reconverging, so the shared example (both paths ending at
# 'Entity C') would demonstrate exactly the overlap the task forbids.
_OUTPUT_BLOCK_DIVERGENT = _OUTPUT_BLOCK.replace(
    '"2": [["Entity A", "relation", "Entity D"], ["Entity D", "relation", "Entity C"]]',
    '"2": [["Entity A", "relation", "Entity D"], ["Entity D", "relation", "Entity E"]]')

# Each item is a JSON object whose emergent-creativity signal is a NEW structured concept the model
# builds and annotates with its mechanism: analogy emits a "projected"/"invention"/"projection" (a
# concept invented by carrying source structure across the mapping), and blending emits a "structure"
# whose triples are each tagged "u"/"v"/"emergent" (which input each projects from, or the fused
# structure belonging to neither). Wording is deliberately explicit so the emitted structure is
# controllable and unambiguous to the parser.
_OUTPUT_BLOCK_ASSOC = """Output requirements (strict):
- Return ONLY a JSON array wrapped in <answer> and </answer> tags. No other text, before or after.
- Each element of the array is ONE connection: an object with a single key, "path".
- "path" is a list of triples; each triple is [head entity, relationship, tail entity] forming a continuous chain from the first entity to the last.
- Relationship strings must be 1-3 words. Keep every entity SHORT and CONCRETE: a few plain, recognizable words naming a real concept -- not a descriptive phrase, a clause, an invented CamelCase compound, or words joined by a dash into a coinage (write "petition threshold", never "PetitionThreshold" or "petition-threshold").
- List one object per connection. To stop, simply end the array -- do not pad with weak connections.
- If you can find no valid connection, return an empty array: <answer>[]</answer>.

Required format (follow this shape exactly):
<answer>[{"path": [["A", "r1", "B"], ["B", "r2", "C"]]}, {"path": [["A", "s1", "D"], ["D", "s2", "C"]]}]</answer>"""

_OUTPUT_BLOCK_ANALOGY = """Output requirements (strict):
- Return ONLY a JSON array wrapped in <answer> and </answer> tags, containing EXACTLY ONE analogy object. No other text, before or after.
- The single array element is the analogy: an object with keys "path_a", "path_b", "projected", "invention", and "projection".
- "path_a" and "path_b" are each a list of triples [head, relationship, tail] that establish the mapping. They MUST have the same number of triples and the IDENTICAL relationship word at every position (only the entities differ); the entity at position i of "path_a" corresponds to the entity at position i of "path_b".
- "projected" is a real concept from ONE domain (the source) that has NO counterpart in the other -- the thing you carry across the mapping to INVENT something new (e.g. "vaccine").
- "invention" is a short name for the NEW concept the projection creates in the other (target) domain.
- "projection" is a list of {"source": [triple], "image": [triple]} pairs. Each "source" is a true triple about the "projected" concept in its own domain; each "image" carries it across the mapping onto the invention -- same relationship, entities replaced by their counterparts. The invention need NOT already exist (that is the point), but every "image" must be a coherent projection of its "source".
- Relationship strings must be 1-3 words. Keep every entity SHORT and CONCRETE: a few plain, recognizable words naming a real concept -- not a descriptive phrase, a clause, an invented CamelCase compound, or words joined by a dash into a coinage (write "petition threshold", never "PetitionThreshold" or "petition-threshold").
- The array must contain exactly one analogy object.
- If you can find no valid analogy, return an empty array: <answer>[]</answer>.

Required format (follow this shape exactly):
<answer>[{"path_a": [["A", "r1", "B"], ["B", "r2", "C"]], "path_b": [["D", "r1", "E"], ["E", "r2", "F"]], "projected": "X", "invention": "Y", "projection": [{"source": ["X", "r", "B"], "image": ["Y", "r", "E"]}]}]</answer>"""

_OUTPUT_BLOCK_BLENDING = """Output requirements (strict):
- Return ONLY a SINGLE JSON object wrapped in <answer> and </answer> tags. No other text, before or after. (Produce ONE blend, not a list.)
- The object has exactly these three keys: "concept", "generic_space", "structure".
- "concept": a short name for the single blended concept you create by fusing the two inputs.
- "generic_space": ONE phrase naming the shared schema both inputs fit -- what makes them fusable. Be specific; a vacuous schema ("both exist", "both involve change") does not count as a blend.
- "structure": a list of objects {"triple": [head, relationship, tail], "from": TAG} describing the blend, where TAG is EXACTLY one of: "u" (this structure is projected from the FIRST input), "v" (this structure is projected from the SECOND input), or "emergent" (this structure is true of the BLEND but of NEITHER input alone; it arises only from running the fusion -- this is the point of the task, so elaborate the blend to find it). A genuine (double-scope) blend has structure tagged BOTH "u" AND "v", plus at least one "emergent".
- Relationship strings must be 1-3 words; the head is usually the blend. Keep every entity SHORT and CONCRETE: a few plain, recognizable words naming a real concept -- not a descriptive phrase, a clause, an invented CamelCase compound, or words joined by a dash into a coinage (write "voting claims", never "voting-claims"). State the generic space briefly and in plain language too, even when the underlying schema is abstract.

Required format (follow this shape exactly):
<answer>{"concept": "cyborg", "generic_space": "a functional system whose parts can fail and be replaced", "structure": [{"triple": ["cyborg", "can", "die"], "from": "u"}, {"triple": ["cyborg", "has", "components"], "from": "v"}, {"triple": ["cyborg", "replaces components", "without healing downtime"], "from": "emergent"}]}</answer>"""

# CREATE's rules/dedup scaffolding (K.3), shared across modes.
_ENTITY_RULES = """Rules and quality constraints:
- Entities may be concrete or abstract (people, works, places, species, events, ideas, phenomena, theories, etc.), but must be real and recognizable -- do not invent entities.
- Do not ask follow-up questions; respond using the best available factual knowledge.
- Temporal connections are allowed (relationships may span different historical periods).
- Disambiguation is required: use canonical names and qualifiers where necessary (e.g., 'Michael Jordan (basketball)').

Deduplication:
- Do not repeat the same path, and do not repeat the same entity within a single path.
- Prefer paths that are meaningfully different (different intermediate nodes and/or relationships), not trivial rephrasings."""


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

Task: Identify how these two entities are connected by producing as MANY DISTINCT connection paths as you can. A connection path is a sequence of factual triples (head, relationship, tail) forming a continuous chain: consecutive triples share an entity, the first triple's head is '{u}', and the last triple's tail is '{v}'.

We reward three things in every connection:
- TRUE: every triple is factually correct.
- REMOTE: the intermediate concepts sit in domains far from the two endpoints and from each other -- reach across distant fields rather than taking the first obvious link.
- UNCOMMON: build the path from rare, specific concepts and relations, not generic ones most people would give.

Produce as MANY DISTINCT connections as you can, most surprising first -- do not stop at a fixed number; list every genuine connection you can find, and make them as different from one another as possible (different intermediate entities and relations, spanning different domains).
{clause_block}
{_ENTITY_RULES}

{_OUTPUT_BLOCK_ASSOC}"""


def _analogy_prompt(spec: dict) -> str:
    u, v = spec["u_label"], spec["v_label"]
    return f"""Task: You are given two concepts: '{u}' and '{v}'. Build the single best analogy between them, and use it to INVENT a new concept in one of the two domains.

An analogy aligns the two domains through a shared relational structure -- '{u}' plays a role in its domain analogous to the role '{v}' plays in its. You then use that alignment to invent something new: carry a concept from one domain across the mapping into the other.

We reward four things, matching exactly how the analogy is scored:
- UTILITY: every triple in "path_a" and "path_b" is factually correct, and the two paths share an IDENTICAL relation sequence -- a genuine structural correspondence, not a loose resemblance.
- SURPRISE: the aligned entities in corresponding roles of the two paths are as semantically DISTANT as possible, so the mapping travels far across domains.
- ORIGINALITY: build the analogy from RARE, uncommon entities and relations, far from the obvious ones others would give.
- EMERGENT CREATIVITY (the invention): pick a real "projected" concept from ONE domain that has NO counterpart in the other, and carry its structure across the mapping to INVENT a new concept in the other domain -- rewarded for being novel, internally coherent, and a genuine projection through the mapping (the invention need NOT already exist -- inventing it is the point, as the solar system did for the atom).

Each analogy is TWO paths that share an identical relation sequence:
- "path_a": factual triples describing '{u}' within its own domain, beginning at '{u}'.
- "path_b": factual triples describing '{v}' within its own domain, beginning at '{v}'.
CRITICAL: "path_a" and "path_b" must use the EXACT SAME relationship word at every position (only the entities differ), so that position i of "path_a" corresponds to position i of "path_b". Do NOT paraphrase or substitute synonyms (e.g. if "path_a" uses 'awards', "path_b" must also use 'awards', not 'grants'). Use disjoint, recognizable, canonically-named entities, and do not repeat an entity within a path.

Produce exactly ONE analogy -- the single most surprising, genuine analogy you can build, together with its invention.

{_OUTPUT_BLOCK_ANALOGY}"""


def _blending_prompt(spec: dict) -> str:
    """Blending as conceptual FUSION (Fauconnier & Turner): fuse two concepts into ONE new concept.

    Given two concepts u, v, the model finds the shared generic space, projects selectively from both
    into a single blended concept, and reads off the concept's EMERGENT STRUCTURE -- properties true of
    the blend but of neither input alone. One blend per pair (fusion converges; generativity lives in
    the emergent structure, not in a count of blends). Supersedes the earlier polysemy framing, which
    degenerated into listing word-senses. See docs/tracks/kg_creat/blending_fusion.md.
    """
    u, v = spec["u_label"], spec["v_label"]
    return f"""Task: You are given TWO concepts: '{u}' and '{v}'. FUSE them into a SINGLE new concept, then describe the structure this fusion generates.

What a genuine blend is (the FORM). A real blend fuses two concepts into ONE new concept in which BOTH inputs contribute ORGANIZING STRUCTURE -- their relations and roles combine -- so the blend runs as a single coherent concept with emergent structure belonging to neither input.
Example -- "cyborg" (organism + machine): the organism frame (a living system of tissue that can die) AND the machine frame (a system of components that can be swapped and rebooted) BOTH organize it; emergent structure of neither input: its components can be surgically replaced without the downtime a body needs to let tissue heal -- neither a pure organism (which must heal) nor a pure machine (which has no tissue) works this way.
(Note what does NOT count as emergent here: "it can be rebooted" is already true of the machine input and "it can die" of the organism, so they are inherited, not emergent.)
Do NOT instead (a) list one property from each input joined by "and" ("harnesses energy AND is therapeutic"), or (b) treat one input as a mere adjective on the other ("a radioactive solar system"). Both inputs must do organizing work.

Build the blend in two moves:
1. GENERIC SPACE: name the shared schema both '{u}' and '{v}' instantiate -- the abstract structure that lets them fuse (for organism + machine: "a functional system whose parts can fail and be replaced"). Be specific; "both exist"/"both involve change" is vacuous, and a one-from-each conjunction does not count.
2. STRUCTURE: describe the single blended concept as triples, and TAG each triple by where its structure comes from -- "u" (projected from '{u}'), "v" (projected from '{v}'), or "emergent" (true of the BLEND but of NEITHER '{u}' nor '{v}' alone -- structure that appears only when you RUN the fusion forward). A genuine blend has triples tagged BOTH "u" and "v", and at least one "emergent".

We reward three things:
- UTILITY: the generic space is a REAL, specific schema that BOTH '{u}' and '{v}' genuinely instantiate -- not vacuous ("both exist") and not a one-from-each conjunction.
- ORIGINALITY: the generic space is UNCOMMON -- a shared schema few would think to name, not the first obvious one.
- EMERGENT CREATIVITY: a genuine DOUBLE-SCOPE fusion in which BOTH inputs contribute organizing structure, developed into a coherent, original blended concept with EMERGENT structure true of the BLEND but of NEITHER '{u}' alone NOR '{v}' alone (tag these "emergent"). Merely restating that the two were combined does not count.

Produce exactly ONE blend of '{u}' and '{v}'. Keep the structure tight: give the 4-6 triples that best showcase the blend's most important properties (at least one from each input, plus at least one emergent), not an exhaustive dump. Use recognizable, canonically-named entities in the structure.

{_OUTPUT_BLOCK_BLENDING}"""


def _anagram_prompt(spec: dict) -> str:
    """Anagram (exploratory-creativity) prompt: rearrange the entity's letters into a DISTANT word.

    Single stimulus, string output (not triples). Novelty = semantic remoteness of the anagram from
    the source (same measure as the combinatorial tasks); utility is deterministic (exact letters +
    real word), so this task is judge-free.
    """
    u = spec["u_label"]
    return f"""Task: You are given ONE word or name: '{u}'. Rearrange ALL of its letters -- using every letter exactly once, ignoring spaces and capitalization -- into a NEW, real word or phrase (an ANAGRAM of '{u}') whose MEANING is as DIFFERENT and DISTANT from '{u}' as possible.

An anagram uses exactly the same letters as the original, in a different order. Examples (same letters, unrelated meaning):
  "stressed" -> "desserts"   (a feeling -> a food)
  "Elvis"    -> "levis"      (a musician -> a jeans brand)
  "listen"   -> "tinsel"     (perception -> a decoration)

BE CREATIVE: the further the anagram's meaning is from '{u}' -- a completely different domain -- the better. Every anagram must (1) be a real word or phrase, and (2) use EXACTLY the letters of '{u}': the same letters, the same number of each, none added or dropped.

Produce as MANY DISTINCT valid anagrams as you can find, most semantically distant first -- do not stop at a fixed number; list every one you can. If '{u}' genuinely has no valid anagram, return an empty JSON object.

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
