"""Run NoveltyBench on one model.

Loads prompts from HF (yimingzhang/novelty-bench), generates k responses
per prompt via OpenRouter, scores them with utility_k (LLM-judge stand-ins
for the canonical DeBERTa-classifier and Skywork-RM scorers — see
src/new_tests/noveltybench.py for the deviations), and writes summary +
per-prompt outputs to output_dir.

Usage:
    uv run python src/new_tests/scripts/run_noveltybench.py \\
        configs/new_tests/noveltybench.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.new_tests.noveltybench import (  # noqa: E402
    NoveltyBenchConfig,
    save_result,
    score_model,
)
from src.utils import init_directory  # noqa: E402


SUBSET_ALIASES = {
    "NB-Curated": "curated",
    "nb-curated": "curated",
    "curated": "curated",
    "NB-WildChat": "wildchat",
    "nb-wildchat": "wildchat",
    "wildchat": "wildchat",
}


def load_prompts(subset: str, max_n: int | None) -> list[dict]:
    """Load NoveltyBench prompts. subset accepts paper notation ('NB-Curated',
    'NB-WildChat') or HF split names ('curated', 'wildchat')."""
    ds = load_dataset("yimingzhang/novelty-bench")
    canonical = SUBSET_ALIASES.get(subset)
    if canonical is None or canonical not in ds:
        raise ValueError(
            f"Unknown subset {subset!r}. Available splits: {list(ds.keys())}"
        )
    split = ds[canonical]
    items = []
    for i, row in enumerate(split):
        # The dataset has 'id' and 'prompt' fields — but verify both exist.
        prompt_text = row.get("prompt") or row.get("query") or row.get("text")
        if prompt_text is None:
            raise ValueError(
                f"Row {i} has no prompt field. Available keys: {list(row.keys())}"
            )
        items.append(
            {
                "id": str(row.get("id", f"{subset.lower()}-{i}")),
                "prompt": prompt_text,
            }
        )
        if max_n is not None and len(items) >= max_n:
            break
    return items


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if "test_model" not in config:
        raise ValueError("FATAL: 'test_model' is required in config")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    # Save the config alongside outputs for reproducibility.
    (output_dir / "config.yaml").write_text(Path(config_path).read_text())

    cfg = NoveltyBenchConfig(
        k=config.get("k", 10),
        patience=config.get("patience", 0.8),
        temperature=config.get("temperature", 1.0),
        top_p=config.get("top_p", 1.0),
        max_tokens=config.get("max_tokens", 512),
        distinctness_method=config.get("distinctness_method", "deberta"),
        distinctness_judge_model=config.get(
            "distinctness_judge_model", "anthropic/claude-haiku-4-5"
        ),
        distinctness_threshold=config.get("distinctness_threshold", 0.5),
        distinctness_device=config.get("distinctness_device", "cpu"),
        quality_judge_model=config.get(
            "quality_judge_model", "openai/gpt-4o-mini"
        ),
        judge_concurrency=config.get("judge_concurrency", 16),
        generation_concurrency=config.get("generation_concurrency", 16),
    )

    subset = config.get("subset", "NB-Curated")
    max_n = config.get("max_prompts")  # None = all
    if debug and max_n is None:
        max_n = 3
        print(f"[DEBUG] capping prompts at {max_n}")

    prompts = load_prompts(subset, max_n=max_n)
    print(
        f"Loaded {len(prompts)} prompts from {subset}; "
        f"running {cfg.k} generations / prompt on {config['test_model']!r}"
    )

    result = asyncio.run(score_model(cfg, config["test_model"], prompts))
    summary_path = save_result(result, output_dir)
    print(
        f"Done. mean_utility_k = {result.mean_utility_k:.3f}, "
        f"frac_distinct = {result.fraction_distinct:.3f}, "
        f"mean_quality = {result.mean_quality:.2f}\n"
        f"Summary written to {summary_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, overwrite=args.overwrite, debug=args.debug)
