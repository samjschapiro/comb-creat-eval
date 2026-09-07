"""Resolve the curated Kombine entity pool to Wikidata QIDs and report recognizability stats.

The pool (``data/kg_creat/entities_curated.json``) is hand-written, so "recognizable" is a
curation judgment. This script grounds it: each label is resolved via ``wbsearchentities`` and
we report, per entity, the number of Wikipedia **sitelinks** (the standard recognizability proxy)
and the number of **statements** and **distinct properties** (a degree proxy in the full KG).

Resolution failures and suspicious resolutions (returned label != queried label) are COUNTED and
printed, never silently dropped -- a wrong QID would inflate the stats.

    python src/kg_creat/scripts/resolve_pool.py            # writes data/kg_creat/pool_wikidata.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.kg_creat.wikidata import _api_get, _MAX_IDS_PER_CALL  # noqa: E402

POOL = Path("data/kg_creat/entities_curated.json")
OUT = Path("data/kg_creat/pool_wikidata.json")


def load_pool() -> list[tuple[str, str]]:
    d = json.loads(POOL.read_text())
    return [(e, dom) for dom, ents in d.items()
            if not dom.startswith("_") and isinstance(ents, list) for e in ents]


# Description fragments marking a candidate as a creative work / publication rather than the concept.
_WORK_MARKERS = (
    "album", "song", "single by", "film", "movie", "video game", "episode", "novel", "book",
    "manga", "anime", "tv series", "television series", "sitcom", "musical", "band", "painting",
    "sculpture by", "journal", "scientific article", "publication", "magazine", "newspaper",
    "given name", "family name", "surname", "comics", "play by", "opera", "poem", "software",
    "genus of", "village", "settlement", "commune in", "municipality",
)


def _is_work(desc: str | None) -> bool:
    d = (desc or "").lower()
    return any(m in d for m in _WORK_MARKERS)


def _normalize(label: str) -> str:
    """Pool labels are written as display text ("The printing press"); Wikidata items are not."""
    return label.strip().removeprefix("The ").removeprefix("the ").strip()


def candidates(label: str) -> list[dict]:
    """Up to 10 wbsearchentities hits for a label. Taking only the top hit mis-resolves badly:
    the search API favors exact title matches, so "The internet" -> IMDb and "Zero" -> a video
    game. We disambiguate by sitelinks in resolve() instead."""
    data = _api_get({"action": "wbsearchentities", "search": _normalize(label), "language": "en",
                     "format": "json", "limit": 10, "type": "item"})
    return [{"qid": h["id"], "wd_label": h.get("label"), "description": h.get("description")}
            for h in (data.get("search") or [])]


def fetch_stats(qids: list[str]) -> dict[str, dict]:
    """Batch-fetch sitelinks + statement counts, split by whether the value is another entity.

    ``degree`` counts only ``wikibase-item``-valued statements -- the statements that are genuine
    edges to other entities, i.e. out-degree in the KG. ``statements`` counts everything, which is
    ~4x larger because roughly two thirds of statements on a well-known item are external database
    identifiers (VIAF, IMDb, library authority files) rather than edges.
    """
    out = {}
    for i in range(0, len(qids), _MAX_IDS_PER_CALL):
        batch = qids[i:i + _MAX_IDS_PER_CALL]
        data = _api_get({"action": "wbgetentities", "ids": "|".join(batch),
                         "props": "sitelinks|claims", "format": "json"})
        if "entities" not in data:
            raise RuntimeError(f"missing 'entities' for {batch}: {data.get('error')}")
        for qid, ent in data["entities"].items():
            claims = ent.get("claims") or {}
            degree = ext_id = 0
            for snaks in claims.values():
                for sn in snaks:
                    dt = (sn.get("mainsnak") or {}).get("datatype")
                    if dt == "wikibase-item":
                        degree += 1
                    elif dt == "external-id":
                        ext_id += 1
            out[qid] = {
                "sitelinks": len(ent.get("sitelinks") or {}),
                "degree": degree,
                "external_ids": ext_id,
                "statements": sum(len(v) for v in claims.values()),
                "properties": len(claims),
            }
    return out


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round((p / 100) * (len(xs) - 1)))))]


def update_stats_only():
    """Recompute stats for the QIDs already resolved in OUT (no re-searching)."""
    doc = json.loads(OUT.read_text())
    rows = doc["rows"]
    print(f"recomputing stats for {len(rows)} resolved entities...")
    stats = fetch_stats([r["qid"] for r in rows])
    for r in rows:
        r.update(stats.get(r["qid"], {}))
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"wrote {OUT}\n")
    report(rows, doc.get("unresolved") or [])


def main():
    pool = load_pool()
    print(f"pool: {len(pool)} entities across {len(set(d for _, d in pool))} domains\n")

    # Pass 1: gather candidate QIDs for every label.
    cands, unresolved = {}, []
    for i, (label, dom) in enumerate(pool, 1):
        try:
            cs = candidates(label)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(pool)}] {label!r}: API ERROR {e}")
            cs = []
        if not cs:
            unresolved.append((label, dom))
            continue
        cands[(label, dom)] = cs
        if i % 40 == 0:
            print(f"  searched {i}/{len(pool)}")

    # Pass 2: fetch stats for every candidate, then pick the most-linked item per label.
    all_qids = sorted({c["qid"] for cs in cands.values() for c in cs})
    print(f"\nsearched {len(pool)}: {len(cands)} with candidates, {len(unresolved)} without")
    print(f"fetching stats for {len(all_qids)} candidate items...")
    stats = fetch_stats(all_qids)

    rows = []
    for (label, dom), cs in cands.items():
        # Take the most-linked candidate. Filtering out "work-like" descriptions first was tried
        # and is WORSE: it strips the correct item in domains whose entities genuinely are works
        # (film, literature), e.g. "Film noir" and "Jane Austen". Residual mis-resolution is
        # ~6% and sits in the low-sitelink tail, so it depresses these stats rather than inflating.
        best = max(cs, key=lambda c: (stats.get(c["qid"], {}).get("sitelinks", 0),
                                      stats.get(c["qid"], {}).get("statements", 0)))
        rows.append({"label": label, "domain": dom, **best,
                     "n_candidates": len(cs),
                     "top_hit_qid": cs[0]["qid"],
                     "top_hit_was_wrong": cs[0]["qid"] != best["qid"],
                     **stats.get(best["qid"], {"sitelinks": 0, "statements": 0, "properties": 0})})

    n_fixed = sum(1 for r in rows if r["top_hit_was_wrong"])
    print(f"disambiguated by sitelinks: {n_fixed}/{len(rows)} labels where the top search hit "
          f"was NOT the most-linked item")

    OUT.write_text(json.dumps({"rows": rows, "unresolved": unresolved}, indent=1))
    print(f"wrote {OUT}\n")

    report(rows, unresolved)


def report(rows, unresolved):
    # --- distributions ---
    sl = [r["sitelinks"] for r in rows]
    st = [r["statements"] for r in rows]
    print("=== sitelinks (Wikipedia language editions) ===")
    print(f"  min {min(sl)}  p10 {pct(sl,10)}  median {pct(sl,50)}  p90 {pct(sl,90)}  max {max(sl)}")
    for thr in (5, 10, 25, 50, 100):
        n = sum(1 for x in sl if x >= thr)
        print(f"  >= {thr:3d} sitelinks: {n:3d}/{len(sl)}  ({100*n/len(sl):.0f}%)")
    if any("degree" in r for r in rows):
        dg = [r.get("degree", 0) for r in rows]
        print("\n=== degree (item-valued statements = TRUE out-degree in the KG) ===")
        print(f"  min {min(dg)}  p10 {pct(dg,10)}  median {pct(dg,50)}  p90 {pct(dg,90)}  max {max(dg)}")
        for thr in (10, 25, 50, 100):
            n = sum(1 for x in dg if x >= thr)
            print(f"  >= {thr:3d} edges: {n:3d}/{len(dg)}  ({100*n/len(dg):.0f}%)")
    print("\n=== statements (ALL, incl. external IDs -- NOT degree) ===")
    print(f"  min {min(st)}  p10 {pct(st,10)}  median {pct(st,50)}  p90 {pct(st,90)}  max {max(st)}")
    for thr in (10, 25, 50, 100):
        n = sum(1 for x in st if x >= thr)
        print(f"  >= {thr:3d} statements: {n:3d}/{len(st)}  ({100*n/len(st):.0f}%)")

    print("\n=== weakest 15 by sitelinks (check these for mis-resolution) ===")
    for r in sorted(rows, key=lambda r: r["sitelinks"])[:15]:
        print(f"  {r['sitelinks']:4d} sl {r['statements']:4d} st  {r['label']!r} -> "
              f"{r['qid']} {r['wd_label']!r} ({r['description']})")

    mismatch = [r for r in rows
                if (r["wd_label"] or "").lower().strip() != r["label"].lower().removeprefix("the ").strip()]
    print(f"\n=== resolved label != queried label: {len(mismatch)}/{len(rows)} (first 15) ===")
    for r in mismatch[:15]:
        print(f"  {r['label']!r} -> {r['wd_label']!r}  ({r['description']})")

    if unresolved:
        print(f"\n=== UNRESOLVED ({len(unresolved)}) ===")
        for label, dom in unresolved:
            print(f"  {label!r} [{dom}]")

    print("\n=== median sitelinks by domain ===")
    bydom = {}
    for r in rows:
        bydom.setdefault(r["domain"], []).append(r["sitelinks"])
    for dom, xs in sorted(bydom.items(), key=lambda kv: -pct(kv[1], 50)):
        print(f"  {dom:14s} n={len(xs):3d}  median {pct(xs,50):4d}  min {min(xs):4d}")


if __name__ == "__main__":
    if "--update-stats" in sys.argv:
        update_stats_only()
    else:
        main()
