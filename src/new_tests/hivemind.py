"""Hivemind scoring implementation (intra-model homogeneity).

Reference: Jiang et al. 2025, "Artificial Hivemind: The Open-Ended
Homogeneity of Language Models (and Beyond)" (arXiv 2510.22954, NeurIPS
2025 Best Paper). The published Hivemind repo ships analysis code only,
not a runnable benchmark scorer; this module reimplements the
intra-model homogeneity metric from the paper so we can score a new
model.

Metric (paper §4): for each prompt, generate k responses, compute the
mean pairwise cosine similarity between the OpenAI text-embedding-3-small
embeddings of those responses, then average across prompts. Higher =
more homogeneous (= more "hivemind-like"); lower = more diverse.

The dat_eval paper reports the *complement* of this similarity (1 - sim)
to align with the divergent thinking construct (higher = more divergent).
We expose both.

Prompt source: the Hivemind paper uses Infinity-Chat. Two HF datasets
from the paper authors are useful here:

- liweijiang/infinite-chats-eval: 100-prompt curated evaluation subset.
  Default for population-comparable scoring.
- liweijiang/infinite-chats-taxonomy: full 26K Infinity-Chat with
  per-prompt taxonomy categories. Use when you want the full distribution.
"""

from __future__ import annotations

import asyncio
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.new_tests.llm import (
    embed_texts,
    get_async_client,
)


@dataclass
class HivemindConfig:
    k: int = 8  # responses per prompt; paper uses up to 16, scales linearly
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 512
    embedding_model: str = "text-embedding-3-small"  # paper canonical
    generation_concurrency: int = 8


@dataclass
class PerPromptResult:
    prompt_id: str
    prompt: str
    generations: list[str]
    pairwise_mean_similarity: float


@dataclass
class HivemindResult:
    test_model: str
    config: dict
    per_prompt: list[PerPromptResult]
    intra_model_mean_similarity: float
    intra_model_mean_diversity: float  # = 1 - intra_model_mean_similarity
    pct_pairs_similarity_above_0_8: float


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def _generate_for_prompt(
    cfg: HivemindConfig,
    async_client,
    test_model: str,
    prompt: str,
    sem: asyncio.Semaphore,
) -> list[str]:
    async def _one() -> str:
        async with sem:
            r = await async_client.chat.completions.create(
                model=test_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
                n=1,
            )
        return r.choices[0].message.content or ""

    return list(await asyncio.gather(*(_one() for _ in range(cfg.k))))


async def score_model(
    cfg: HivemindConfig,
    test_model: str,
    prompts: list[dict],
) -> HivemindResult:
    async_client = get_async_client()
    gen_sem = asyncio.Semaphore(cfg.generation_concurrency)

    per_prompt: list[PerPromptResult] = []
    all_pair_sims: list[float] = []

    for item in prompts:
        pid = str(item["id"])
        ptxt = item["prompt"]
        gens = await _generate_for_prompt(cfg, async_client, test_model, ptxt, gen_sem)
        # Embed all k generations in one batch.
        # Note: embed_texts is sync — fine here; embedding API is fast.
        embs = embed_texts(gens, model=cfg.embedding_model)
        emb_arr = np.array(embs, dtype=np.float64)

        # Pairwise cosine similarities (upper triangle only).
        n = len(gens)
        sims: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                sims.append(_cosine(emb_arr[i], emb_arr[j]))
        mean_sim = float(np.mean(sims)) if sims else float("nan")

        per_prompt.append(
            PerPromptResult(
                prompt_id=pid,
                prompt=ptxt,
                generations=gens,
                pairwise_mean_similarity=mean_sim,
            )
        )
        all_pair_sims.extend(sims)

    intra_sim = float(np.mean(all_pair_sims)) if all_pair_sims else float("nan")
    intra_div = 1.0 - intra_sim if not math.isnan(intra_sim) else float("nan")
    above = float(np.mean([1.0 if s >= 0.8 else 0.0 for s in all_pair_sims])) if all_pair_sims else 0.0

    return HivemindResult(
        test_model=test_model,
        config={
            "k": cfg.k,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "embedding_model": cfg.embedding_model,
            "implementation_notes": (
                "Intra-model mean pairwise cosine similarity over k responses "
                "per prompt under text-embedding-3-small. Prompt source is "
                "configurable (infinite-chats-eval default; "
                "infinite-chats-taxonomy for full 26K)."
            ),
        },
        per_prompt=per_prompt,
        intra_model_mean_similarity=intra_sim,
        intra_model_mean_diversity=intra_div,
        pct_pairs_similarity_above_0_8=above,
    )


def save_result(result: HivemindResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    detail_path = output_dir / "per_prompt.json"

    summary = {
        "test_model": result.test_model,
        "config": result.config,
        "n_prompts": len(result.per_prompt),
        "intra_model_mean_similarity": result.intra_model_mean_similarity,
        "intra_model_mean_diversity": result.intra_model_mean_diversity,
        "pct_pairs_similarity_above_0_8": result.pct_pairs_similarity_above_0_8,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    detail = [
        {
            "prompt_id": r.prompt_id,
            "prompt": r.prompt,
            "generations": r.generations,
            "pairwise_mean_similarity": r.pairwise_mean_similarity,
        }
        for r in result.per_prompt
    ]
    detail_path.write_text(json.dumps(detail, indent=2, ensure_ascii=False))
    return summary_path
