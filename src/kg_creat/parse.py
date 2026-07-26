"""Bridge CREATE's robust path parser to kg_creat's :class:`EmittedPath`.

Raw model responses arrive in CREATE's ``<answer>``-wrapped JSON-of-triples format
(see the vendored ``prompt.py``). We reuse CREATE's battle-tested
``Path.parse_path_from_text`` (handles markdown fences, ``</think>`` tags, malformed
JSON, several triple encodings) and wrap each parsed connection path as an
:class:`EmittedPath` for constraint checking + novelty scoring.
"""

from __future__ import annotations

import json
import re

from src.kg_creat.scoring import EmittedPath
from src.kg_creat.vendor.create.path_evaluator import Path as _CreatePath

_TRIPLE_RE = re.compile(r'\[\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]')
_PATH_KEY_RE = re.compile(r'"(\d+)"\s*:\s*\[')


def _salvage(raw: str) -> list[list[tuple]]:
    """Recover complete paths from a response cut off mid-JSON by the token cap.

    A truncated answer fails whole-document JSON parsing, which would otherwise score as a
    total structural failure and confound verbosity with capability. We keep every path whose
    closing bracket arrived and drop the one that was mid-emission.
    """
    paths, spans = [], [(m.start(), m.end(), m.group(1)) for m in _PATH_KEY_RE.finditer(raw)]
    for i, (_, body_start, _key) in enumerate(spans):
        end = spans[i + 1][0] if i + 1 < len(spans) else len(raw)
        segment = raw[body_start:end]
        triples = [m for m in _TRIPLE_RE.finditer(segment)]
        if not triples:
            continue
        # Keep the path only if its array actually closed. A path cut off mid-emission is
        # missing its final hops, and scoring it would read as "never reached the target" —
        # a structural failure manufactured by the token cap rather than by the model.
        tail = segment[triples[-1].end():]
        closed = "]" in tail and (tail.index("]") < tail.index("[") if "[" in tail else True)
        if closed:
            paths.append([(m.group(1), m.group(2), m.group(3)) for m in triples])
    return paths


def parse_paths(raw_response: str | list | None) -> list[EmittedPath]:
    """Parse one raw model response into a list of :class:`EmittedPath` (one per path).

    Returns ``[]`` if nothing parses. Triples whose length is not 3 are dropped; a path
    left with no valid triples is skipped. Note CREATE's parser lowercases content, which
    is fine — our matching normalizes anyway.
    """
    if raw_response is None:
        return []
    parsed = _CreatePath(raw_prediction=raw_response).parse_path_from_text()
    if not parsed:
        parsed = _salvage(raw_response) if isinstance(raw_response, str) else []
    if not parsed:
        return []
    out: list[EmittedPath] = []
    for path in parsed:
        # Coerce every element to str: CREATE's parser can occasionally emit a non-string (e.g. a
        # set) for a malformed triple, which is both wrong to score and unserializable (it crashed
        # a whole run on json.dumps). str() keeps it serializable; a garbage triple then just fails
        # scoring naturally rather than taking the run down.
        triples = [tuple(str(x) for x in t) for t in path if t is not None and len(t) == 3]
        if triples:
            out.append(EmittedPath(triples=triples))
    return out
