"""Generate the graph and evaluation prompt set.

Step 1 of the pipeline: builds a random labeled graph and generates
path-finding prompts at varying hop counts and difficulty levels.

Usage:
    uv run python src/comb_eval/scripts/generate_eval.py configs/comb_eval/generate_eval.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import load_config, init_directory, save_config
from src.comb_eval.graph import build_graph, graph_to_adjacency_text, save_graph, graph_stats
from src.comb_eval.prompts import generate_eval_set


def main(config_path: str, overwrite: bool = False, debug: bool = False):
    config = load_config(config_path)
    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, output_dir)

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    # Build graph
    graph_cfg = config["graph"]
    print(f"Building graph: {graph_cfg['n_nodes']} nodes, avg_degree={graph_cfg['avg_degree']}")
    G = build_graph(
        n_nodes=graph_cfg["n_nodes"],
        avg_degree=graph_cfg["avg_degree"],
        seed=graph_cfg["seed"],
    )

    stats = graph_stats(G)
    print(f"Graph stats: {json.dumps(stats, indent=2)}")

    # Save graph
    save_graph(G, output_dir / "graph.json")
    adjacency_text = graph_to_adjacency_text(G)
    (output_dir / "graph_adjacency.txt").write_text(adjacency_text)
    (output_dir / "results" / "graph_stats.json").write_text(
        json.dumps(stats, indent=2)
    )

    print(f"Adjacency text length: {len(adjacency_text)} chars, {len(adjacency_text.splitlines())} lines")

    # Generate eval prompts
    eval_cfg = config["eval"]
    print(
        f"Generating eval set: hops={eval_cfg['hop_counts']}, "
        f"max_level={eval_cfg['max_level']}, prompts_per_hop={eval_cfg['prompts_per_hop']}"
    )
    prompts = generate_eval_set(
        G,
        hop_counts=eval_cfg["hop_counts"],
        max_level=eval_cfg["max_level"],
        prompts_per_hop=eval_cfg["prompts_per_hop"],
        p_include=eval_cfg["p_include"],
        min_valid_paths=eval_cfg["k_paths"],
        seed=eval_cfg["seed"],
    )

    print(f"Generated {len(prompts)} prompts")

    # Summary by hop count and difficulty
    from collections import Counter
    by_hop = Counter()
    by_level = Counter()
    for p in prompts:
        by_hop[p.hop_count] += 1
        by_level[p.difficulty_level] += 1

    print(f"By hop count: {dict(sorted(by_hop.items()))}")
    print(f"By difficulty: {dict(sorted(by_level.items()))}")

    # Save prompts
    prompts_data = [p.to_dict() for p in prompts]
    prompts_path = output_dir / "prompts.json"
    with open(prompts_path, "w") as f:
        json.dump(prompts_data, f, indent=2)

    summary = {
        "n_prompts": len(prompts),
        "by_hop_count": dict(sorted(by_hop.items())),
        "by_difficulty_level": dict(sorted(by_level.items())),
        "graph_stats": stats,
        "k_paths": eval_cfg["k_paths"],
    }
    with open(output_dir / "results" / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved to {output_dir}")

    if debug:
        print("\n--- Debug: first 3 prompts ---")
        for p in prompts[:3]:
            print(json.dumps(p.to_dict(), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
