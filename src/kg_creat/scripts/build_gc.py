"""Build and cache a Wikidata construction subgraph G_c.

Reads seed entity names, BFS-builds a typed G_c over the controlled property
whitelist, and caches it to ``output_dir/gc.json``. Free (network only). The cached
G_c is consumed downstream by the matched-bundle sampler.

    python src/kg_creat/scripts/build_gc.py configs/kg_creat/build_gc.yaml --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory, load_config, save_config  # noqa: E402
from src.kg_creat.wikidata import build_gc, derive_vocabulary, resolve_qid, DEFAULT_PROPERTY_WHITELIST  # noqa: E402


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)  # requires output_dir
    for field in ("seeds", "radius"):
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    seeds = config["seeds"]  # {entity name: domain}
    if not isinstance(seeds, dict):
        raise ValueError("FATAL: 'seeds' must be a mapping {entity name: domain}")
    if debug:
        seeds = dict(list(seeds.items())[:3])
        print(f"DEBUG: truncated to {len(seeds)} seeds")

    # Resolve seed names -> QIDs (fail loud on any unresolved name); build the domain map.
    print(f"Resolving {len(seeds)} domain-tagged seeds -> QIDs ...")
    seed_qids, seed_domains = [], {}
    for name, domain in seeds.items():
        qid = resolve_qid(name)
        print(f"  {name!r} [{domain}] -> {qid}")
        seed_qids.append(qid)
        seed_domains[qid] = domain

    # Relation vocabulary: frequency-derived from the seeds (default) or an explicit override.
    if config.get("top_n_relations"):
        print(f"Deriving controlled vocabulary: top {config['top_n_relations']} relations by frequency ...")
        vocab = derive_vocabulary(
            seed_qids,
            top_n=config["top_n_relations"],
            sample_radius=config.get("vocab_sample_radius", 1),
            max_survey_entities=config.get("vocab_survey_entities", 600),
        )
    else:
        vocab = config.get("whitelist") or DEFAULT_PROPERTY_WHITELIST
        print(f"Using {'explicit' if config.get('whitelist') else 'default'} vocabulary ({len(vocab)} relations)")

    gc, entity_domains = build_gc(
        name=config.get("name", Path(config["output_dir"]).name),
        seeds=seed_qids,
        radius=config["radius"],
        whitelist=vocab,
        max_neighbors_per_relation=config.get("max_neighbors_per_relation", 40),
        seed_domains=seed_domains,
    )

    gc_path = output_dir / "gc.json"
    gc.save(gc_path)
    (output_dir / "entity_domains.json").write_text(json.dumps(entity_domains, indent=2))
    print(f"Saved G_c -> {gc_path}  (+ entity_domains.json)")
    print(f"stats: {gc.stats()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to config file")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output directory")
    parser.add_argument("--debug", action="store_true", help="Debug mode (truncate seeds)")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
