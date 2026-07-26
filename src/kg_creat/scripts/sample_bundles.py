"""Sample the round-1 prompt set from a cached G_c.

Downstream of ``build_gc.py``: loads ``upstream_dir/gc.json``, samples Regime-A matched
bundles (baseline + biting A-D constraints per hub-sharing endpoint pair) and Regime-B
curated analogy/blend items, and writes a flat ``prompts.json`` the elicitation runner
consumes (plus ``bundles.json`` with the nested Regime-A structure for reference).

Free (no API).

    python src/kg_creat/scripts/sample_bundles.py configs/kg_creat/sample_bundles.yaml --overwrite
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory, load_config, save_config  # noqa: E402
from src.kg_creat.graph import KnowledgeGraph  # noqa: E402
from src.kg_creat.sample import sample_matched_bundles, sample_random_bundles, sample_regime_b  # noqa: E402


def _flatten_regime_a(bundles: list[dict]) -> list[dict]:
    """One prompt spec per bundle cell (baseline + each constraint)."""
    specs = []
    for b in bundles:
        for mode, cell in b["cells"].items():
            specs.append({
                "prompt_id": f"{b['bundle_id']}.{mode}",
                "bundle_id": b["bundle_id"], "regime": "A", "mode": mode,
                "u": b["u"], "v": b["v"], "u_label": b["u_label"], "v_label": b["v_label"],
                "h": b["h"], "k": b["k"], "constraint": cell["constraint"],
                # domain is a study variable; carried through for random-pair endpoints
                "domain_u": b.get("domain_u"), "domain_v": b.get("domain_v"),
                "cross_domain": b.get("cross_domain"),
            })
    return specs


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    if "upstream_dir" not in config:
        raise ValueError("FATAL: 'upstream_dir' (the G_c run dir) is required")
    upstream_dir = Path(config["upstream_dir"])
    gc_path = upstream_dir / "gc.json"
    if not gc_path.exists():
        raise FileNotFoundError(f"FATAL: no G_c at {gc_path} -- run build_gc.py first")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    gc = KnowledgeGraph.load(gc_path)
    print(f"Loaded G_c: {gc.stats()}")

    # The controlled vocabulary the Regime-A prompts must use = the relations G_c is built from.
    allowed_relations = sorted({gc.relation_label(p) for p in gc.relation_vocabulary()})
    (output_dir / "allowed_relations.json").write_text(json.dumps(allowed_relations, indent=2))
    print(f"Controlled vocabulary ({len(allowed_relations)} relations): {allowed_relations}")

    s = config.get("sampler", {})
    if debug:
        s = {**s, "max_bundles": 2, "n_bundles": 2}
        print("DEBUG: 2 bundles")
    ed_path = upstream_dir / "entity_domains.json"
    entity_domains = json.loads(ed_path.read_text()) if ed_path.exists() else {}

    # 'random' (default): arbitrary entity pairs, like the analogy task -- no path/biting
    # verification (that selects for unsurprising pairs and defeats combinatorial creativity;
    # biting is handled post hoc by make_pass2's baseline-derived targets). 'matched' is the
    # legacy connectivity-filtered sampler.
    strategy = s.get("strategy", "random")
    if strategy == "random":
        bundles = sample_random_bundles(
            gc, n_bundles=s.get("n_bundles", 40), k=s.get("k", 5), seed=s.get("seed", 0),
            min_degree=s.get("min_degree", 3), max_per_source=s.get("max_per_source", 3),
            entity_domains=entity_domains)
    elif strategy == "matched":
        bundles = sample_matched_bundles(
            gc, h=s.get("h", 3), k=s.get("k", 5),
            min_routes=s.get("min_routes", 6), min_degree=s.get("min_degree", 4),
            max_bundles=s.get("max_bundles", 10), max_paths=s.get("max_paths", 64),
            require_constraints=s.get("require_constraints", 3),
            max_per_source=s.get("max_per_source", 2))
    else:
        raise ValueError(f"FATAL: unknown sampler.strategy {strategy!r} (use 'random' or 'matched')")
    rb = config.get("regime_b", {})
    regime_b = sample_regime_b(gc, n_analogy=rb.get("n_analogy", 8), n_blend=rb.get("n_blend", 6),
                               seed=rb.get("seed", 0), min_degree=rb.get("min_degree", 3),
                               entity_domains=entity_domains)

    prompts = _flatten_regime_a(bundles)
    for i, spec in enumerate(regime_b):
        spec["prompt_id"] = spec["bundle_id"]
        spec["h"] = None
        spec["k"] = s.get("k", 5)
        prompts.append(spec)

    (output_dir / "bundles.json").write_text(json.dumps(bundles, indent=2))
    (output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))

    n_a = sum(1 for p in prompts if p["regime"] == "A")
    n_e = sum(1 for p in prompts if p["mode"] == "analogy")
    n_f = sum(1 for p in prompts if p["mode"] == "blending")
    print(f"Wrote {len(prompts)} prompt specs ({n_a} Regime-A cells over {len(bundles)} bundles, "
          f"{n_e} analogy, {n_f} blending) -> {output_dir / 'prompts.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
