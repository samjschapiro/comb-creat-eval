"""SBV structural metric for plot twists (Exp 3): symbolic T_mod + preservation.

Operationalizes the Schapiro-Black-Varshney graphical theory of transformational
creativity (ICCC 2025, arXiv 2504.18687) on the READER'S story-DAG. An LLM extractor
reconstructs, from the story text alone, the reader's conceptual space just before the
reveal as a typed dependency DAG:

  - axioms 𝒜   : load-bearing reader assumptions ("the narrator is reliable", "X is
                 alive", "it is present day").
  - rules      : inferences derived from axioms.
  - artifacts  : narrated scenes/events, each with a SUPPORT SET (the axioms/rules that
                 make it coherent).
  - edge u->v  : v depends on u (v's support set names u).

The twist is an axiom modification a* -> a'. From the parsed DAG we compute, in Python
(networkx), the two STRUCTURAL twist metrics from the theory:

  - T_mod(a*)        = |depends(a*)| = number of downstream nodes that (transitively)
                       rely on the flipped axiom  -> the SURPRISE / magnitude term
                       (how much of the story must be re-evaluated). This is exactly
                       "number of dependents of the axiom node".
  - preservation(a') = fraction of prior artifacts whose support set stays satisfiable
                       after the flip  -> the COHERENCE / retro-fit term.

These are SYMBOLIC (graph-counted), in contrast to DSI's purely neural/embedding view
(Exp 2). The paper's claim: combining the two is needed to measure transformational
creativity. The metric is auditable -- the emitted DAG shows exactly which axiom flipped
and what depends on it.

Extraction is durable + resumable + ensemble-able (re-extract with a second model to
report inter-extractor reliability, the make-or-break number for §3).
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import networkx as nx

from src.plot_twist.llm import call_llm_async, get_async_client_openrouter

EXTRACTOR_VERSION = "v1"

EXTRACT_PROMPT = """You are a precise literary-structure analyst. Read the short story \
below and reconstruct the READER'S model of the story world, then identify the plot \
twist as a change to one load-bearing assumption.

Build a dependency graph with three node types:
- "axiom": a load-bearing assumption the reader holds before the reveal (e.g. "the \
narrator is a reliable adult", "the victim is a stranger", "the events happen in the \
present"). These are the deep, self-standing premises.
- "rule": an inference derived from axioms (e.g. "the protagonist will be believed").
- "artifact": a concrete narrated scene or event in the story, with a SUPPORT SET: the \
ids of the axioms/rules that make that scene make sense as written.

Then identify the TWIST as an axiom modification:
- "flipped_axiom": the id of the single axiom the reveal overturns (a*). If the story \
has NO genuine plot twist, set this to null.
- "a_prime": one sentence stating the new assumption that replaces it after the reveal.
- For EVERY artifact, set "survives_flip": true if that scene is still consistent (or \
makes even better sense) once the axiom is flipped to a_prime, false if the reveal \
contradicts it / makes it incoherent.

Give each node a short stable id (a1, a2 for axioms; r1 for rules; e1, e2 for artifacts). \
Every id in a support set or in flipped_axiom MUST be defined as a node. Aim for 3-7 \
axioms and 4-12 artifacts.

Story:
\"\"\"{story}\"\"\"

Output ONLY a single JSON object, nothing else, in EXACTLY this form:
{{"axioms": [{{"id": "a1", "statement": "..."}}],
 "rules": [{{"id": "r1", "statement": "...", "supports": ["a1"]}}],
 "artifacts": [{{"id": "e1", "statement": "...", "supports": ["a1"], "survives_flip": true}}],
 "flipped_axiom": "a1", "a_prime": "..."}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ExtractConfig:
    # Primary extractor first; extra models enable an inter-extractor reliability check.
    extractor_models: list[str] = field(
        default_factory=lambda: ["anthropic/claude-sonnet-4", "openai/gpt-4o"]
    )
    temperature: float = 0.0
    max_tokens: int = 1600
    concurrency: int = 12
    extractor_version: str = EXTRACTOR_VERSION


def _parse(raw: Optional[str]) -> Optional[dict]:
    """Extract + lightly validate the DAG JSON. Returns None if unparseable."""
    if not raw:
        return None
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "artifacts" not in data:
        return None
    data.setdefault("axioms", [])
    data.setdefault("rules", [])
    data.setdefault("flipped_axiom", None)
    return data


def compute_structural(dag: dict) -> dict:
    """Compute T_mod(a*) and preservation(a') from a parsed DAG via networkx.

    T_mod(a*)        = number of nodes that transitively DEPEND on the flipped axiom
                       (its descendants once edges point dependency -> dependent).
    t_mod_frac       = T_mod / (#rules + #artifacts), a length-robust version.
    preservation(a') = fraction of artifacts with survives_flip == true.
    Returns a dict of metrics; has_twist is False if no axiom was flagged as flipped."""
    axioms = {n["id"]: n for n in dag.get("axioms", []) if isinstance(n, dict) and "id" in n}
    rules = {n["id"]: n for n in dag.get("rules", []) if isinstance(n, dict) and "id" in n}
    arts = [n for n in dag.get("artifacts", []) if isinstance(n, dict) and "id" in n]
    art_ids = {n["id"] for n in arts}

    G = nx.DiGraph()
    for nid in list(axioms) + list(rules) + list(art_ids):
        G.add_node(nid)
    # edge u -> v means "v depends on u" (v's support set names u)
    for nid, node in {**rules, **{n["id"]: n for n in arts}}.items():
        for u in node.get("supports", []) or []:
            if u in G:
                G.add_edge(u, nid)

    a_star = dag.get("flipped_axiom")
    has_twist = a_star is not None and a_star in G
    n_downstream_universe = len(rules) + len(art_ids)

    if has_twist:
        depends = nx.descendants(G, a_star)  # everything relying on the flipped axiom
        t_mod = len(depends)
    else:
        t_mod = 0
    t_mod_frac = (t_mod / n_downstream_universe) if n_downstream_universe else float("nan")

    surv = [bool(n.get("survives_flip", True)) for n in arts]
    preservation = (sum(surv) / len(surv)) if surv else float("nan")

    return {
        "has_twist": bool(has_twist),
        "flipped_axiom": a_star,
        "t_mod": int(t_mod),
        "t_mod_frac": float(t_mod_frac),
        "preservation": float(preservation),
        "n_axioms": len(axioms),
        "n_rules": len(rules),
        "n_artifacts": len(art_ids),
        # structural analogue of the rubric product (surprise x coherence)
        "structural_ptc": float(t_mod_frac * preservation) if surv and n_downstream_universe else float("nan"),
    }


async def _extract_one(
    async_client,
    sem: asyncio.Semaphore,
    model: str,
    story: str,
    cfg: ExtractConfig,
) -> Optional[dict]:
    async with sem:
        try:
            raw = await call_llm_async(
                async_client=async_client,
                messages=[{"role": "user", "content": EXTRACT_PROMPT.format(story=story)}],
                model=model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            )
        except Exception:
            return None
    return _parse(raw)


async def extract_stories(
    cfg: ExtractConfig,
    items: Sequence[dict],  # each: {"id": str, "text": str}
    cache_dir: Path,
) -> list[dict]:
    """Extract a structural DAG + metrics for each story, with every extractor model.

    Per story, each extractor's parsed DAG and metrics are stored; metrics are also
    aggregated across extractors by the median (robust to one bad parse). Each result
    is written to cache_dir/<id>.json the instant it completes (durable + resumable),
    so a kill never loses paid extractor calls."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = get_async_client_openrouter()
    sem = asyncio.Semaphore(cfg.concurrency)

    async def _one(item: dict) -> dict:
        sid = str(item["id"])
        cpath = cache_dir / f"{sid}.json"
        if cpath.exists():
            prev = json.loads(cpath.read_text())
            if prev.get("metrics"):  # resume: already extracted
                return prev
        dags = await asyncio.gather(
            *(_extract_one(client, sem, m, item["text"], cfg) for m in cfg.extractor_models)
        )
        by_extractor = {}
        for m, dag in zip(cfg.extractor_models, dags):
            by_extractor[m] = {"dag": dag, "metrics": compute_structural(dag) if dag else None}
        ok = [v["metrics"] for v in by_extractor.values() if v["metrics"]]
        agg = _median_metrics(ok)
        rec = {"id": sid, "source": item.get("source"), "metrics": agg, "by_extractor": by_extractor}
        cpath.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        return rec

    return list(await asyncio.gather(*(_one(it) for it in items)))


_NUMERIC = ("t_mod", "t_mod_frac", "preservation", "structural_ptc", "n_axioms", "n_rules", "n_artifacts")


def _median_metrics(metrics_list: list[dict]) -> Optional[dict]:
    """Median per numeric metric across extractors; majority for has_twist."""
    import statistics
    if not metrics_list:
        return None
    agg: dict = {}
    for k in _NUMERIC:
        vals = [m[k] for m in metrics_list if m.get(k) is not None and m[k] == m[k]]  # drop NaN
        agg[k] = float(statistics.median(vals)) if vals else float("nan")
    tw = [m["has_twist"] for m in metrics_list]
    agg["has_twist"] = sum(tw) >= (len(tw) / 2)
    agg["n_extractors"] = len(metrics_list)
    return agg
