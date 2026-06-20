"""Exp 2 generation: prompt-method intervention for plot-twist stories.

Generates, on a small reasoning-capable model subset, two conditions per model at a
fixed temperature: an INDEPENDENT baseline and IN-CONTEXT REGENERATION (the model is
shown its prior stories and asked for a different twist). Output (generations.json) is
scored by the same generic rubric/realism judges downstream
(run_rubric_stimuli.py, run_realism.py), then analyzed by run_prompt_analysis.py.

Generators MUST be disjoint from the judge ensemble (anti-circularity).

Usage:
    python src/plot_twist/scripts/run_prompt_methods.py configs/plot_twist/prompt_methods.yaml --overwrite [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.utils import init_directory, load_config, save_config
from src.plot_twist.incontext import (
    PromptMethodsConfig, generate_prompt_methods, model_id_to_key, rec_id_for,
)


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg_dict = load_config(config_path)
    for field_name in ("output_dir", "generator_models", "n_samples"):
        if field_name not in cfg_dict:
            raise ValueError(f"FATAL: '{field_name}' required in config")

    # Anti-circularity: a generator may not also be a judge.
    judges = set(cfg_dict.get("judge_models_excluded", []))
    overlap = judges & set(cfg_dict["generator_models"])
    if overlap:
        raise ValueError(
            f"FATAL: generator(s) overlap the judge ensemble {sorted(overlap)}; "
            "keep generators disjoint from judges (anti-circularity)."
        )

    if overwrite:
        out = init_directory(cfg_dict["output_dir"], overwrite=True)
    else:
        out = Path(cfg_dict["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
    save_config(cfg_dict, out)
    stories_dir = out / "stories"

    models = cfg_dict["generator_models"]
    methods = cfg_dict.get("methods", ["baseline", "incontext_regen"])
    n_samples = cfg_dict["n_samples"]
    temperatures = cfg_dict.get("temperatures", [1.0])
    if debug:
        models = models[:2]
        n_samples = min(n_samples, 2)
        temperatures = temperatures[:1]

    cfg = PromptMethodsConfig(
        generator_models=models,
        methods=methods,
        n_samples=n_samples,
        temperatures=temperatures,
        max_tokens=cfg_dict.get("max_tokens", 4500),
        concurrency=cfg_dict.get("concurrency", 12),
        target_words=tuple(cfg_dict.get("target_words", (2000, 3000))),
    )

    total = len(models) * len(temperatures) * len(methods) * n_samples
    already = 0
    if stories_dir.exists():
        for m in models:
            for t in temperatures:
                for meth in methods:
                    for i in range(n_samples):
                        p = stories_dir / model_id_to_key(m) / f"{rec_id_for(m, t, meth, i)}.json"
                        if p.exists() and json.loads(p.read_text()).get("story"):
                            already += 1
    print(
        f"generating {total} stories = {len(models)} models x "
        f"{len(temperatures)} temps {temperatures} x {len(methods)} methods {methods} x {n_samples} samples"
    )
    print(f"target length: {cfg.target_words[0]}-{cfg.target_words[1]} words (human-gold median band)")
    print(f"resume: {already}/{total} already on disk (will not re-spend); {total - already} new")
    print(f"models: {', '.join(models)}\n")

    records = asyncio.run(generate_prompt_methods(cfg, stories_dir))

    gen_path = out / "generations.json"
    gen_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    n_ok = sum(1 for r in records if r["story"])
    print(f"saved {n_ok}/{len(records)} stories: {gen_path}\n")

    # Per-(model, method) coverage; a method that yields 0 stories for a model voids
    # that model's within-model contrast, so report it loudly.
    print("coverage (non-empty stories / attempts) per (model, method):")
    dead = []
    for m in models:
        for meth in methods:
            rows = [r for r in records if r["model"] == m and r["prompt_method"] == meth]
            ok = sum(1 for r in rows if r["story"])
            flag = "  <-- ALL FAILED" if rows and ok == 0 else ("  <-- some failures" if ok < len(rows) else "")
            print(f"  {m:<38} {meth:<16} {ok}/{len(rows)}{flag}")
            if rows and ok == 0:
                dead.append((m, meth, next((r["error"] for r in rows if r["error"]), "unknown")))
    if dead:
        raise RuntimeError(
            "FATAL: (model, method) cell(s) produced nothing (generations.json still saved): "
            + "; ".join(f"{m}/{meth} [{err[:80]}]" for m, meth, err in dead)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
