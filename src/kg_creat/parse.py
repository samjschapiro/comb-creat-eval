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


_STR_VAL_RE = re.compile(r'"\d+"\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_anagrams(raw_response: str | None) -> list[str]:
    """Parse an anagram response (``<answer>``-wrapped JSON dict of index->string) into candidates.

    Order is preserved (the model is asked to list most-distant first). Deduplicates
    case-insensitively while keeping first occurrence. Falls back to a regex salvage if the JSON is
    truncated by the token cap, so a long list cut off mid-emission still yields its complete entries.
    """
    if not raw_response or not isinstance(raw_response, str):
        return []
    text = raw_response
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    cands: list[str] = []
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            cands = [str(v) for v in obj.values() if isinstance(v, (str, int, float))]
    except Exception:  # noqa: BLE001
        cands = [mo.group(1) for mo in _STR_VAL_RE.finditer(text)]
    out, seen = [], set()
    for c in cands:
        c = c.strip()
        key = c.lower()
        if c and key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _coerce_path(triples_raw) -> EmittedPath | None:
    """Coerce a raw list-of-triples into an :class:`EmittedPath`, or None if unusable."""
    if not isinstance(triples_raw, list):
        return None
    triples = [tuple(str(x) for x in t) for t in triples_raw
               if isinstance(t, list) and len(t) == 3]
    return EmittedPath(triples=triples) if triples else None


def _salvage_pairs(text: str) -> list[dict]:
    """Recover complete objects from a pair-array truncated mid-emission by the token cap.

    Trim to the last closed ``}`` and close the array, so a long list cut off partway still yields
    its complete items rather than scoring as a total structural failure.
    """
    t = text.strip()
    i = t.find("[")
    if i > 0:
        t = t[i:]
    last = t.rfind("}")
    if last == -1:
        return []
    try:
        obj = json.loads(t[:last + 1] + "]")
    except Exception:  # noqa: BLE001
        return []
    return [el for el in obj if isinstance(el, dict)] if isinstance(obj, list) else []


def parse_pairs(raw_response: str | None, keys: tuple[str, str]
                ) -> list[tuple[EmittedPath, EmittedPath]]:
    """Parse an ``<answer>``-wrapped JSON ARRAY of two-key objects into (path_a, path_b) pairs.

    Analogy uses ``keys=("source", "target")`` and blending ``("sense_1", "sense_2")``; each emits a
    SET of items, one pair per item. Returns ``[]`` if nothing parses; drops any element missing a key
    or with an unparseable path. Salvages complete items from a token-cap-truncated array.
    """
    if not raw_response or not isinstance(raw_response, str):
        return []
    text = raw_response
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    ka, kb = keys
    try:
        obj = json.loads(text.strip())
        items = [el for el in obj if isinstance(el, dict)] if isinstance(obj, list) else (
            [obj] if isinstance(obj, dict) else [])  # tolerate a single unwrapped object
    except Exception:  # noqa: BLE001
        items = _salvage_pairs(text)
    out: list[tuple[EmittedPath, EmittedPath]] = []
    for el in items:
        pa, pb = _coerce_path(el.get(ka)), _coerce_path(el.get(kb))
        if pa is not None and pb is not None:
            out.append((pa, pb))
    return out


def parse_items(raw_response: str | None, path_keys: tuple[str, ...],
                inference_key: str = "inferences") -> list[dict]:
    """Parse an ``<answer>``-wrapped JSON ARRAY of item objects into structured items.

    Each item object has one or two path fields (named by ``path_keys``: ``("path",)`` for association,
    ``("path_a", "path_b")`` for analogy, ``("sense_1", "sense_2")`` for blending) plus an optional
    ``"inferences"`` list (the emergent-creativity signal). Returns a list of
    ``{"paths": [EmittedPath, ...], "inferences": [str, ...]}``; drops any item missing a path.
    Salvages complete items from a token-cap-truncated array.
    """
    if not raw_response or not isinstance(raw_response, str):
        return []
    text = raw_response
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text.strip())
        raw_items = ([el for el in obj if isinstance(el, dict)] if isinstance(obj, list)
                     else [obj] if isinstance(obj, dict) else [])
    except Exception:  # noqa: BLE001
        raw_items = _salvage_pairs(text)
    out: list[dict] = []
    for el in raw_items:
        paths = [_coerce_path(el.get(k)) for k in path_keys]
        if any(p is None for p in paths):
            continue
        raw_inf = el.get(inference_key)
        inferences = ([str(x).strip() for x in raw_inf
                       if isinstance(x, (str, int, float)) and str(x).strip()]
                      if isinstance(raw_inf, list) else [])
        out.append({"paths": paths, "inferences": inferences})
    return out


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
