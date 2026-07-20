"""Build a common-knowledge Wikidata construction subgraph ``G_c`` via REST BFS.

We fetch entity claims from the Wikidata API (``wbgetentities``), BFS outward from
seed entities following ONLY a controlled **property whitelist** ``Sigma`` (the
"controlled relation vocabulary" the method calls for), and materialize a typed
:class:`~src.kg_creat.graph.KnowledgeGraph` via ``KnowledgeGraph.from_triples``.

The cached ``G_c`` is a derived artifact (``data/kg_creat/``): sourcing is a *builder*,
not a runtime dependency. CREATE does not disclose its sourcing mechanism, so we are
free to choose; the whitelist keeps ``G_c`` bounded and makes P31/P279 the natural
"taxonomic shortcut" for the exclusion constraint to bite on.

Fail-fast (repo policy): HTTP/parse errors raise. The one explicit, *counted* drop is
edges to entities without an English label -- they cannot appear in a common-knowledge
task -- and the count is printed, never silently swallowed.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

_MIN_INTERVAL = 0.34  # seconds between API calls (polite pacing, ~3 req/s)
_last_call = [0.0]

# Entity-valued but non-relational properties -- excluded from a frequency-derived vocabulary
# even when common. Two classes: (a) Wikimedia-administrative/reference metadata, and (b)
# ubiquitous identity/demographic attributes that don't form a meaningful conceptual bridge.
# Documented so the derived vocabulary stays reproducible ("top-N by frequency minus this list").
_PROPERTY_STOPLIST = frozenset({
    # (a) Wikimedia-administrative / reference
    "P143", "P1343", "P248", "P805", "P935", "P373", "P4390", "P1889",
    "P910",   # topic's main category
    "P5008",  # on focus list of Wikimedia project
    "P1424",  # topic has template
    "P1151",  # topic's main Wikimedia portal
    "P6104",  # maintained by WikiProject
    "P7867",  # category for maps
    "P3876",  # category for alumni
    # (b) ubiquitous identity/demographic attributes (not conceptual bridges)
    "P21",    # sex or gender
    "P735",   # given name
    "P734",   # family name
    "P1412",  # languages spoken, written or signed
})

from src.kg_creat.graph import KnowledgeGraph

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "kg_creat/0.1 (research; combinatorial-creativity benchmark)"
_MAX_IDS_PER_CALL = 50  # wbgetentities hard limit

# Controlled relation vocabulary Sigma (PID -> English label). BFS follows only these.
DEFAULT_PROPERTY_WHITELIST: dict[str, str] = {
    "P31": "instance of",
    "P279": "subclass of",
    "P361": "part of",
    "P527": "has part",
    "P171": "parent taxon",
    "P170": "creator",
    "P50": "author",
    "P57": "director",
    "P86": "composer",
    "P175": "performer",
    "P61": "discoverer or inventor",
    "P138": "named after",
    "P112": "founded by",
    "P127": "owned by",
    "P176": "manufacturer",
    "P186": "made from material",
    "P463": "member of",
    "P737": "influenced by",
    "P101": "field of work",
    "P106": "occupation",
    "P131": "located in",
    "P276": "location",
    "P159": "headquarters location",
    "P1542": "has effect",
    "P828": "has cause",
    "P366": "has use",
    "P921": "main subject",
}

_TYPE_PROPERTY = "P31"  # instance-of, mined for node types regardless of whitelist


def resolve_qid(label: str) -> str:
    """Resolve a plain entity name to its top Wikidata QID (so configs use names)."""
    data = _api_get(
        {
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "format": "json",
            "limit": 1,
            "type": "item",
        }
    )
    hits = data.get("search", [])
    if not hits:
        raise ValueError(f"FATAL: no Wikidata item found for '{label}'")
    return hits[0]["id"]


def _api_get(params: dict, retries: int = 6) -> dict:
    """GET the Wikidata API and return parsed JSON, paced and 429-aware.

    Enforces a minimum inter-call interval and, on HTTP 429, honors Retry-After
    (falling back to exponential backoff). Raises after exhausting retries.
    """
    url = f"{WIKIDATA_API}?{urllib.parse.urlencode(params)}"
    last_err: Exception | None = None
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                _last_call[0] = time.monotonic()
                return json.load(resp)
        except urllib.error.HTTPError as e:
            _last_call[0] = time.monotonic()
            last_err = e
            if e.code == 429:
                retry_after = e.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * (2 ** attempt)
                print(f"  [wikidata] 429 rate-limited; sleeping {delay:.0f}s (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                time.sleep(1.5 ** attempt)
        except Exception as e:  # noqa: BLE001 - retry any transport/parse error
            _last_call[0] = time.monotonic()
            last_err = e
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"Wikidata API failed after {retries} attempts: {url}\n  last error: {last_err}")


def _fetch_entities(qids: list[str]) -> dict[str, dict]:
    """Batch-fetch labels + claims for QIDs (<=50 per call handled internally)."""
    out: dict[str, dict] = {}
    for i in range(0, len(qids), _MAX_IDS_PER_CALL):
        batch = qids[i : i + _MAX_IDS_PER_CALL]
        data = _api_get(
            {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels|claims",
                "languages": "en",
                "format": "json",
            }
        )
        if "entities" not in data:
            raise RuntimeError(f"Wikidata response missing 'entities' for batch {batch}: {data.get('error')}")
        out.update(data["entities"])
    return out


def _en_label(entity: dict) -> str | None:
    return entity.get("labels", {}).get("en", {}).get("value")


def _entity_neighbors(entity: dict, whitelist: dict[str, str] | None) -> list[tuple[str, str]]:
    """Return [(pid, tail_qid), ...] for item-valued claims. If ``whitelist`` is None, all of them."""
    edges: list[tuple[str, str]] = []
    for pid, statements in entity.get("claims", {}).items():
        if whitelist is not None and pid not in whitelist:
            continue
        for st in statements:
            snak = st.get("mainsnak", {})
            if snak.get("snaktype") != "value" or snak.get("datatype") != "wikibase-item":
                continue
            val = snak.get("datavalue", {}).get("value", {})
            tail = val.get("id")
            if tail:
                edges.append((pid, tail))
    return edges


def _entity_types(entity: dict) -> list[str]:
    """P31 (instance-of) target QIDs -> node types."""
    types: list[str] = []
    for st in entity.get("claims", {}).get(_TYPE_PROPERTY, []):
        snak = st.get("mainsnak", {})
        if snak.get("snaktype") == "value":
            tid = snak.get("datavalue", {}).get("value", {}).get("id")
            if tid:
                types.append(tid)
    return types


def derive_vocabulary(
    seeds: list[str],
    top_n: int,
    sample_radius: int = 1,
    max_survey_entities: int = 600,
) -> dict[str, str]:
    """Data-derive the controlled relation vocabulary Sigma by property frequency.

    Surveys the seed entities and their ``sample_radius``-hop neighborhood (bounded by
    ``max_survey_entities``, sorted-QID sample), counts how many surveyed entities use each
    item-valued property (once per entity, so a high-fan-out entity can't dominate), drops the
    reference/metadata stoplist, and returns the ``top_n`` properties as ``{pid: english label}``.
    Fully reproducible: "the top-N most widely-used Wikidata relations among these entities."
    """
    prop_entity_count: Counter = Counter()
    visited: set[str] = set()
    frontier: list[str] = sorted(set(seeds))
    for depth in range(sample_radius + 1):
        to_fetch = [q for q in frontier if q not in visited][: max_survey_entities - len(visited)]
        if not to_fetch:
            break
        entities = _fetch_entities(to_fetch)
        next_frontier: set[str] = set()
        for q in sorted(to_fetch):
            visited.add(q)
            ent = entities.get(q)
            if ent is None or "missing" in ent:
                continue
            props_here: set[str] = set()
            for pid, tail in _entity_neighbors(ent, None):
                if pid in _PROPERTY_STOPLIST:
                    continue
                props_here.add(pid)
                if depth < sample_radius:
                    next_frontier.add(tail)
            for pid in props_here:
                prop_entity_count[pid] += 1
        frontier = sorted(next_frontier - visited)
        if len(visited) >= max_survey_entities:
            break

    ranked = [pid for pid, _ in prop_entity_count.most_common()][:top_n]
    prop_ents = _fetch_entities(ranked)
    vocab = {pid: (_en_label(prop_ents.get(pid, {})) or pid) for pid in ranked}
    print(f"[derive_vocabulary] surveyed {len(visited)} entities; top {top_n} relations "
          f"(count): " + ", ".join(f"{vocab[p]}({prop_entity_count[p]})" for p in ranked))
    return vocab


def build_gc(
    name: str,
    seeds: list[str],
    radius: int,
    whitelist: dict[str, str] | None = None,
    max_neighbors_per_relation: int = 40,
    seed_domains: dict[str, str] | None = None,
) -> tuple[KnowledgeGraph, dict[str, str]]:
    """BFS a Wikidata slice around ``seeds`` to depth ``radius`` and build a KnowledgeGraph.

    Args:
        name: name for the resulting G_c.
        seeds: seed entity QIDs (e.g. ["Q144", "Q18498"] for Dog, Wolf).
        radius: BFS depth (number of whitelisted hops explored from the seeds).
        whitelist: PID -> label controlled vocabulary (defaults to DEFAULT_PROPERTY_WHITELIST).
        max_neighbors_per_relation: cap tails kept per (entity, relation) for boundedness.
        seed_domains: QID -> domain label; entities inherit the domain of the seed whose BFS
            first reached them, giving a per-entity domain tag (domain is a study variable).

    Returns:
        (KnowledgeGraph over labeled entities, entity_domain map {qid: domain}).
    """
    whitelist = whitelist or DEFAULT_PROPERTY_WHITELIST
    for q in seeds:
        if not (q.startswith("Q") and q[1:].isdigit()):
            raise ValueError(f"FATAL: seed '{q}' is not a valid QID")

    labels: dict[str, str] = {}          # qid -> en label (only labeled entities kept)
    node_types: dict[str, list[str]] = {}  # qid -> [type_qid, ...]
    raw_edges: list[tuple[str, str, str]] = []  # (head, pid, tail) before label filtering
    type_qids: set[str] = set()
    entity_domain: dict[str, str] = dict(seed_domains) if seed_domains else {}

    visited: set[str] = set()
    frontier: list[str] = sorted(set(seeds))
    dropped_unlabeled = 0

    for depth in range(radius + 1):
        to_fetch = [q for q in frontier if q not in visited]
        if not to_fetch:
            break
        entities = _fetch_entities(to_fetch)
        next_frontier: set[str] = set()

        for q in sorted(to_fetch):
            visited.add(q)
            ent = entities.get(q)
            if ent is None or "missing" in ent:
                continue
            lbl = _en_label(ent)
            if lbl is None:
                dropped_unlabeled += 1
                continue
            labels[q] = lbl
            node_types[q] = _entity_types(ent)
            type_qids.update(node_types[q])

            if depth == radius:
                continue  # last layer: record the node, do not expand further

            # group neighbors by relation to apply a per-relation cap deterministically
            by_rel: dict[str, list[str]] = {}
            for pid, tail in _entity_neighbors(ent, whitelist):
                by_rel.setdefault(pid, []).append(tail)
            for pid, tails in by_rel.items():
                for tail in sorted(set(tails))[:max_neighbors_per_relation]:
                    raw_edges.append((q, pid, tail))
                    next_frontier.add(tail)
                    if tail not in entity_domain and q in entity_domain:
                        entity_domain[tail] = entity_domain[q]  # inherit reacher's domain

        frontier = sorted(next_frontier - visited)

    # Ensure every edge endpoint + type QID has a label (fetch any still-missing).
    missing = {t for _, _, t in raw_edges if t not in labels}
    missing |= {t for t in type_qids if t not in labels}
    if missing:
        for q, ent in _fetch_entities(sorted(missing)).items():
            lbl = _en_label(ent) if ent and "missing" not in ent else None
            if lbl is not None:
                labels[q] = lbl

    # Keep only edges whose BOTH endpoints are labeled (explicit, counted drop).
    triples: list[tuple[str, str, str]] = []
    for h, pid, t in raw_edges:
        if h in labels and t in labels:
            triples.append((h, pid, t))
        else:
            dropped_unlabeled += 1

    used_pids = {pid for _, pid, _ in triples}
    relation_labels = {pid: whitelist[pid] for pid in used_pids}
    type_labels = {t: labels[t] for t in type_qids if t in labels}
    # node_types must only reference entities that survived into the graph
    node_types = {q: [t for t in ts if t in labels] for q, ts in node_types.items() if q in labels}

    print(
        f"[build_gc] name={name} seeds={seeds} radius={radius}: "
        f"{len(labels)} entities, {len(triples)} triples, {len(used_pids)} relations, "
        f"{len(type_labels)} types; dropped {dropped_unlabeled} unlabeled endpoints"
    )
    if not triples:
        raise RuntimeError("FATAL: G_c has no triples -- check seeds/whitelist/radius")

    entity_domain = {q: d for q, d in entity_domain.items() if q in labels}
    if seed_domains:
        n_dom = len(set(entity_domain.values()))
        print(f"[build_gc] domain-tagged {len(entity_domain)}/{len(labels)} entities across {n_dom} domains")

    gc = KnowledgeGraph.from_triples(
        name=name,
        triples=triples,
        labels=labels,
        relation_labels=relation_labels,
        node_types=node_types,
        type_labels=type_labels,
        build_meta={
            "source": "wikidata",
            "seeds": seeds,
            "radius": radius,
            "whitelist": sorted(whitelist),
            "max_neighbors_per_relation": max_neighbors_per_relation,
            "dropped_unlabeled": dropped_unlabeled,
        },
    )
    return gc, entity_domain
