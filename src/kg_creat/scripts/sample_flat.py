"""Sample Kombine prompts from a FLAT curated entity pool -- no graph, no BFS.

The model supplies the connections from its own knowledge (open KG); this script only draws the
anchor entities we hand it. Entities come from a domain-tagged JSON pool (see
``data/kg_creat/entities_curated.json``). Association and analogy draw CROSS-DOMAIN pairs (two
distinct domains -> one entity each) so the anchors are genuinely remote; blending REUSES the
analogy pairs (fusion of the same two concepts). Output is a flat ``prompts.json`` in the same
spec shape ``run_elicit``/``score`` expect.

    python src/kg_creat/scripts/sample_flat.py configs/kg_creat/sample_flat.yaml --overwrite
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.utils import load_config, init_directory, save_config  # noqa: E402


def _load_pool(path: Path) -> list[tuple[str, str]]:
    """Flatten the {domain: [entities]} pool to [(entity, domain), ...]; drop _note keys."""
    data = json.loads(path.read_text())
    pool = [(e, dom) for dom, ents in data.items()
            if not dom.startswith("_") and isinstance(ents, list) for e in ents]
    if len(pool) < 10:
        raise ValueError(f"FATAL: entity pool too small ({len(pool)}) at {path}")
    return pool


def _cross_domain_pairs(pool, n, rng):
    """n distinct cross-domain (u,v) pairs: pick two different domains, one entity from each."""
    by_dom: dict[str, list[str]] = {}
    for e, d in pool:
        by_dom.setdefault(d, []).append(e)
    domains = list(by_dom)
    pairs, seen = [], set()
    tries = 0
    while len(pairs) < n and tries < n * 200:
        tries += 1
        du, dv = rng.sample(domains, 2)
        u, v = rng.choice(by_dom[du]), rng.choice(by_dom[dv])
        key = frozenset((u, v))
        if u != v and key not in seen:
            seen.add(key)
            pairs.append((u, du, v, dv))
    return pairs


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    pool = _load_pool(Path(config["entities"]))
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    n_assoc = config.get("n_association", 30)
    n_ana = config.get("n_analogy", 30)
    n_blend = config.get("n_blend", 30)
    k = config.get("k", 5)
    if debug:
        n_assoc = n_ana = n_blend = 2

    rng = random.Random(config.get("seed", 0))
    prompts = []

    def spec(pid, mode, regime, u, du, v, dv):
        return {"prompt_id": pid, "bundle_id": pid, "regime": regime, "mode": mode,
                "u": u, "v": v, "u_label": u, "v_label": v, "h": None, "k": k, "constraint": None,
                "domain_u": du, "domain_v": dv, "cross_domain": (dv is not None and du != dv)}

    for i, (u, du, v, dv) in enumerate(_cross_domain_pairs(pool, n_assoc, rng)):
        prompts.append(spec(f"A{i}.baseline", "baseline", "A", u, du, v, dv))
    analogy_pairs = _cross_domain_pairs(pool, n_ana, rng)
    for i, (u, du, v, dv) in enumerate(analogy_pairs):
        prompts.append(spec(f"E{i}", "analogy", "B", u, du, v, dv))
    # Blending reuses the analogy pairs verbatim: analogy and fusion run on the IDENTICAL (u,v) so the
    # map-between vs. fuse-into distinction is isolated (docs/tracks/kg_creat/blending_fusion.md).
    for i, (u, du, v, dv) in enumerate(analogy_pairs[:n_blend]):
        prompts.append(spec(f"F{i}", "blending", "B", u, du, v, dv))

    (output_dir / "prompts.json").write_text(json.dumps(prompts, indent=2))
    n_a = sum(1 for p in prompts if p["mode"] == "baseline")
    n_e = sum(1 for p in prompts if p["mode"] == "analogy")
    n_f = sum(1 for p in prompts if p["mode"] == "blending")
    print(f"Wrote {len(prompts)} prompt specs ({n_a} association, {n_e} analogy, {n_f} blending) "
          f"from a flat pool of {len(pool)} entities -> {output_dir / 'prompts.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
