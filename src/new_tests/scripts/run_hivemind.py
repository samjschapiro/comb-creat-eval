"""Run Hivemind intra-model homogeneity scoring on one model.

Loads open-ended prompts (default: NB-WildChat from yimingzhang/novelty-
bench, as a stand-in for the unreleased Infinity-Chat dataset),
generates k responses per prompt at temperature 1.0, embeds with
text-embedding-3-small, computes mean pairwise cosine similarity within
each prompt, and reports the cross-prompt average plus the diversity
complement (1 - sim).

Usage:
    uv run python src/new_tests/scripts/run_hivemind.py \\
        configs/new_tests/hivemind.yaml [--overwrite] [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.new_tests.hivemind import (  # noqa: E402
    HivemindConfig,
    save_result,
    score_model,
)
from src.utils import init_directory  # noqa: E402


def load_prompts(prompt_source: str, max_n: int | None) -> list[dict]:
    """Load open-ended prompts. prompt_source can be:
    - 'infinite-chats-eval' (default): the 100-prompt evaluation subset of
      Infinity-Chat — liweijiang/infinite-chats-eval. Single 'query' column.
    - 'infinite-chats-taxonomy': the full 26K Infinity-Chat dataset —
      liweijiang/infinite-chats-taxonomy. Pulls user prompts out of the
      first message of each conversation.
    """
    src = prompt_source.lower()
    if src == "infinite-chats-eval":
        ds = load_dataset("liweijiang/infinite-chats-eval")["train"]
        items = []
        for i, row in enumerate(ds):
            items.append({"id": f"ice-{i}", "prompt": row["query"]})
            if max_n is not None and len(items) >= max_n:
                break
        return items
    if src == "infinite-chats-taxonomy":
        ds = load_dataset("liweijiang/infinite-chats-taxonomy")["train"]
        items = []
        for i, row in enumerate(ds):
            msgs = row["messages"]
            user_msg = next((m for m in msgs if m.get("role") == "user"), None)
            if user_msg is None:
                continue
            items.append(
                {
                    "id": str(row.get("conversation_id", f"ict-{i}")),
                    "prompt": user_msg["content"],
                }
            )
            if max_n is not None and len(items) >= max_n:
                break
        return items
    raise ValueError(f"Unknown prompt_source {prompt_source!r}")


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if "output_dir" not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    if "test_model" not in config:
        raise ValueError("FATAL: 'test_model' is required in config")

    output_dir = init_directory(config["output_dir"], overwrite=overwrite)
    (output_dir / "config.yaml").write_text(Path(config_path).read_text())

    cfg = HivemindConfig(
        k=config.get("k", 8),
        temperature=config.get("temperature", 1.0),
        top_p=config.get("top_p", 1.0),
        max_tokens=config.get("max_tokens", 512),
        embedding_model=config.get("embedding_model", "text-embedding-3-small"),
        generation_concurrency=config.get("generation_concurrency", 8),
    )

    prompt_source = config.get("prompt_source", "nb-wildchat")
    max_n = config.get("max_prompts")
    if debug and max_n is None:
        max_n = 3
        print(f"[DEBUG] capping prompts at {max_n}")

    prompts = load_prompts(prompt_source, max_n=max_n)
    print(
        f"Loaded {len(prompts)} prompts from {prompt_source}; "
        f"k={cfg.k} responses/prompt at T={cfg.temperature} on "
        f"{config['test_model']!r}"
    )

    result = asyncio.run(score_model(cfg, config["test_model"], prompts))
    summary_path = save_result(result, output_dir)
    print(
        f"Done. intra_sim={result.intra_model_mean_similarity:.4f}, "
        f"intra_div={result.intra_model_mean_diversity:.4f}, "
        f"pct>=0.8={result.pct_pairs_similarity_above_0_8:.3f}\n"
        f"Summary written to {summary_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, overwrite=args.overwrite, debug=args.debug)
