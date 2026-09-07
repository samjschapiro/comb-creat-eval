"""Sample candidate Kombine anchor entities from Wikidata: draw -> sitelink gate -> TYPE FILTER.

Expanding the curated pool by hand does not scale, so this draws candidates systematically. Three
stages, cheapest first:

  1. DRAW      random QIDs in a low range (older = core items; ~17% clear 50 sitelinks there versus
               <0.2% drawing uniformly across all of Wikidata).
  2. SITELINK  keep items whose Wikimedia sitelink count clears --min-sitelinks. Random Wikidata sits
               at a median of 0 sitelinks, so this gate alone removes the esoteric bulk.
  3. TYPE      drop by ``P31`` class: people, places, taxa, creative works, organizations, calendar
               items, and Wikimedia internals. These are not esoteric -- they are simply not the
               concept/object/idea anchors the tasks need -- so a deterministic filter should reject
               them, NOT an LLM judge. Measured on a random sample, this stage is ~80% of all
               rejections; leaving it to a judge burns tokens on questions a class lookup answers.

What survives is the candidate set an LLM judge should then rate for common-knowledge recognizability
(the one question a type lookup cannot answer). This script stops before that judge.

    python src/kg_creat/scripts/sample_pool.py --n 3000 --min-sitelinks 78
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.kg_creat.wikidata import _api_get, _MAX_IDS_PER_CALL  # noqa: E402

OUT = Path("data/kg_creat/pool_candidates.json")

# P31 classes rejected outright, by QID.
_DROP_QIDS = {
    "Q5": "human",
    "Q4167836": "Wikimedia category", "Q4167410": "Wikimedia disambiguation page",
    "Q11266439": "Wikimedia template", "Q13406463": "Wikimedia list article",
    "Q4663903": "Wikimedia portal", "Q17442446": "Wikimedia internal item",
    "Q577": "year", "Q39911": "decade", "Q578": "century", "Q3186692": "calendar year",
    "Q47150325": "calendar day of a given year", "Q14795564": "point in time with respect to recurrent timeframe",
}

# Rejected when these appear in the P31 class's own English label. Keyword matching (rather than a
# hand-enumerated QID list) keeps this robust to Wikidata's very long tail of subclasses.
_DROP_CLASS_WORDS = (
    # places
    "comune", "commune", "city", "town", "village", "municipality", "settlement", "country",
    "province", "oblast", "prefecture", "district", "department", "county", "island", "river",
    "mountain", "lake", "sovereign state", "administrative", "territory", "region", "capital",
    # life
    "taxon", "species", "genus", "breed", "cultivar",
    # works & media
    "film", "album", "song", "single", "novel", "book", "manga", "anime", "television series",
    "video game", "painting", "sculpture", "magazine", "newspaper", "comic", "opera", "musical",
    "periodical", "journal", "article", "encyclopedia",
    # orgs & groups
    "band", "company", "business", "enterprise", "university", "school", "club", "team",
    "political party", "organization", "association", "agency", "airport", "station",
    # calendar / identifiers
    "year", "decade", "century", "date", "month", "day of", "wikimedia", "wikipedia",
    "disambiguation", "top-level domain",
    # administrative divisions the generic words above miss
    "state of", "canton", "krai", "federal subject", "autonomous community", "voivodeship",
    "governorate", "emirate", "borough", "parish", "ward", "constituency",
    # named works / products / figures
    "play by", "literary work", "written work", "work of art", "deity", "god", "goddess",
    "mythological", "brand", "product", "software", "operating system", "web browser",
)


def _is_proper_noun(label: str) -> bool:
    """Wikidata labels common nouns lowercase ("helium", "prayer") and proper nouns capitalized
    ("Arizona", "Macbeth", "Juno"). Acronyms (HIV, pH, DNA) are common-noun concepts, so an
    all-caps or mixed-short token is not treated as proper."""
    if not label:
        return False
    head = label.split()[0]
    if head.isupper() or (len(head) <= 3 and any(c.isupper() for c in head[1:])):
        return False        # HIV, pH, DNA, GPS
    return head[:1].isupper()


def _fetch(qids, props="sitelinks|claims|descriptions|labels"):
    out = {}
    for i in range(0, len(qids), _MAX_IDS_PER_CALL):
        data = _api_get({"action": "wbgetentities", "ids": "|".join(qids[i:i + _MAX_IDS_PER_CALL]),
                         "props": props, "languages": "en", "format": "json"})
        for qid, ent in (data.get("entities") or {}).items():
            if "missing" in ent:
                continue
            claims = ent.get("claims") or {}
            out[qid] = {
                "qid": qid,
                "sitelinks": len(ent.get("sitelinks") or {}),
                "label": (ent.get("labels", {}).get("en") or {}).get("value"),
                "desc": (ent.get("descriptions", {}).get("en") or {}).get("value"),
                "P31": [s["mainsnak"]["datavalue"]["value"]["id"] for s in claims.get("P31", [])
                        if s.get("mainsnak", {}).get("datavalue")],
            }
    return out


def _reject_reason(rec, class_labels, common_nouns_only=True) -> str | None:
    """Why this candidate is not a usable anchor, or None if it passes."""
    if not rec["label"] or not rec["desc"]:
        return "no en label/description"
    if common_nouns_only and _is_proper_noun(rec["label"]):
        return "proper noun (capitalized label)"
    if rec["label"].replace(".", "").replace("-", "").isdigit():
        return "bare numeral"
    if not rec["P31"]:
        return "no P31 class"
    for cls in rec["P31"]:
        if cls in _DROP_QIDS:
            return _DROP_QIDS[cls]
        lab = (class_labels.get(cls) or "").lower()
        for w in _DROP_CLASS_WORDS:
            if w in lab:
                return f"class:{lab[:34]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="random QIDs to draw")
    ap.add_argument("--min-sitelinks", type=int, default=78,
                    help="sitelink gate (78 = the curated pool's 10th percentile)")
    ap.add_argument("--qid-max", type=int, default=200_000, help="draw QIDs from [1, qid-max]")
    ap.add_argument("--allow-proper-nouns", action="store_true",
                    help="keep capitalized labels (Arizona, Macbeth, Juno); off by default")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    random.seed(a.seed)
    drawn = [f"Q{random.randint(1, a.qid_max)}" for _ in range(a.n)]
    print(f"drew {a.n} random QIDs from [1, {a.qid_max:,}]")
    recs = _fetch(drawn)
    print(f"  exist: {len(recs)}")

    gated = [r for r in recs.values() if r["sitelinks"] >= a.min_sitelinks]
    print(f"  >= {a.min_sitelinks} sitelinks: {len(gated)}  ({100*len(gated)/max(len(recs),1):.0f}%)")

    classes = sorted({c for r in gated for c in r["P31"]})
    class_labels = {q: v["label"] for q, v in _fetch(classes, props="labels").items()}

    kept, rejected = [], Counter()
    for r in gated:
        why = _reject_reason(r, class_labels, not a.allow_proper_nouns)
        if why:
            rejected[why] += 1
        else:
            r["class"] = class_labels.get(r["P31"][0]) if r["P31"] else None
            kept.append(r)

    print(f"  pass type filter: {len(kept)}  ({100*len(kept)/max(len(gated),1):.0f}% of gated, "
          f"{100*len(kept)/max(len(recs),1):.1f}% of all drawn)\n")

    print("top rejection reasons:")
    for why, n in rejected.most_common(12):
        print(f"  {n:4d}  {why}")

    kept.sort(key=lambda r: -r["sitelinks"])
    OUT.write_text(json.dumps({"params": vars(a), "kept": kept,
                               "rejected": dict(rejected)}, indent=1))
    print(f"\nwrote {OUT}  ({len(kept)} candidates)\n")
    print("SURVIVORS (these go to the recognizability judge):")
    for r in kept:
        print(f"  {r['sitelinks']:4d} sl  {str(r['label'])[:32]:32s} | {str(r['desc'])[:52]}")


if __name__ == "__main__":
    main()
