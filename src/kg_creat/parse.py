"""Bridge CREATE's robust path parser to kg_creat's :class:`EmittedPath`.

Raw model responses arrive in CREATE's ``<answer>``-wrapped JSON-of-triples format
(see the vendored ``prompt.py``). We reuse CREATE's battle-tested
``Path.parse_path_from_text`` (handles markdown fences, ``</think>`` tags, malformed
JSON, several triple encodings) and wrap each parsed connection path as an
:class:`EmittedPath` for constraint checking + novelty scoring.
"""

from __future__ import annotations

from src.kg_creat.scoring import EmittedPath
from src.kg_creat.vendor.create.path_evaluator import Path as _CreatePath


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
        return []
    out: list[EmittedPath] = []
    for path in parsed:
        triples = [tuple(t) for t in path if t is not None and len(t) == 3]
        if triples:
            out.append(EmittedPath(triples=triples))
    return out
