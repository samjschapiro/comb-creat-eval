"""Compute set-level diversity D over an M-sampled elicitation run (free; no judge).

Reads each model's responses.json (tagged with temperature/sample_idx by run_elicit), computes
per-(prompt, temperature) D_all and D_valid, and aggregates to (model, mode, temperature). Diversity
is a pure embedding measure, so this runs off the raw responses without any utility judging.

    .venv_mlx/bin/python src/kg_creat/scripts/compute_diversity.py data/kg_creat/responses_<run>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.kg_creat.embed import get_embedder  # noqa: E402
from src.kg_creat.diversity import per_prompt_diversity, aggregate_diversity  # noqa: E402


def main(responses_dir):
    responses_dir = Path(responses_dir)
    embed = get_embedder()
    all_agg = {}
    for md in sorted(d for d in responses_dir.iterdir() if (d / "responses.json").exists()):
        recs = json.loads((md / "responses.json").read_text())
        rows = per_prompt_diversity(recs, embed)
        (md / "diversity.json").write_text(json.dumps(rows, indent=2))
        agg = aggregate_diversity(rows)
        (md / "diversity_agg.json").write_text(json.dumps(agg, indent=2))
        all_agg[md.name] = agg
        print(f"{md.name}:")
        for k, v in sorted(agg.items()):
            print(f"  {k:22s} D_all={v['mean_D_all']}  D_valid={v['mean_D_valid']}  "
                  f"(n_prompts={v['n_prompts_all']})")
    (responses_dir / "diversity_summary.json").write_text(json.dumps(all_agg, indent=2))
    print(f"\nsaved diversity to {responses_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/responses_rand_stage1")
