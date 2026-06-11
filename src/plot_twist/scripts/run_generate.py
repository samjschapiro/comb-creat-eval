"""Generate the LLM twist set: N open-ended plot-twist stories per model.
Output (generations.json) is scored by the same fixed-rubric judge downstream
(run_rubric_stimuli.py).

Generators MUST be disjoint from the judge ensemble (anti-circularity); this
script errors loudly if a configured generator also appears as a judge.

Usage:
    python src/plot_twist/scripts/run_generate.py configs/plot_twist/generate_llm_twists.yaml --overwrite [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.utils import init_directory, load_config, save_config
from src.plot_twist.generate import GenerateConfig, generate_stories, model_id_to_key


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

    # --overwrite wipes and starts fresh; otherwise RESUME from whatever per-story
    # files already exist (do not wipe), so a re-run never re-spends on stories we
    # already have on disk.
    if overwrite:
        out = init_directory(cfg_dict["output_dir"], overwrite=True)
    else:
        out = Path(cfg_dict["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
    save_config(cfg_dict, out)
    stories_dir = out / "stories"

    models = cfg_dict["generator_models"]
    n_samples = cfg_dict["n_samples"]
    temperatures = cfg_dict.get("temperatures", [0.9, 1.0, 1.2])
    if debug:
        models = models[:2]
        n_samples = min(n_samples, 2)
        temperatures = temperatures[:1]

    cfg = GenerateConfig(
        generator_models=models,
        n_samples=n_samples,
        temperatures=temperatures,
        max_tokens=cfg_dict.get("max_tokens", 4500),
        concurrency=cfg_dict.get("concurrency", 16),
        target_words=tuple(cfg_dict.get("target_words", (2000, 3000))),
    )

    total = len(models) * len(temperatures) * n_samples
    # Count how many stories are already on disk (resume: these cost no new API).
    already = 0
    if stories_dir.exists():
        for m in models:
            for t in temperatures:
                for i in range(n_samples):
                    mk = model_id_to_key(m)
                    p = stories_dir / mk / f"{mk}__t{int(round(t * 10)):02d}__s{i:02d}.json"
                    if p.exists() and json.loads(p.read_text()).get("story"):
                        already += 1
    print(
        f"generating {total} stories = {len(models)} models x "
        f"{len(temperatures)} temps {temperatures} x {n_samples} samples"
    )
    print(f"target length: {cfg.target_words[0]}-{cfg.target_words[1]} words (human-gold median band)")
    print(f"resume: {already}/{total} already on disk (will not re-spend); {total - already} new")
    print(f"models: {', '.join(models)}\n")

    records = asyncio.run(generate_stories(cfg, stories_dir))

    # Save FIRST, so a single dead model can't discard the good generations.
    gen_path = out / "generations.json"
    gen_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    n_ok = sum(1 for r in records if r["story"])
    print(f"saved {n_ok}/{len(records)} stories: {gen_path}\n")

    # Loud failure report: a generator that errors on everything is a bad ID/auth.
    print("generation reliability (non-empty stories / attempts):")
    dead = []
    for m in models:
        rows = [r for r in records if r["model"] == m]
        ok = sum(1 for r in rows if r["story"])
        flag = "  <-- ALL FAILED" if ok == 0 else ("  <-- some failures" if ok < len(rows) else "")
        print(f"  {m:<40} {ok}/{len(rows)}{flag}")
        if ok == 0:
            dead.append((m, next((r["error"] for r in rows if r["error"]), "unknown")))
    if dead:
        raise RuntimeError(
            "FATAL: generator(s) produced nothing (generations.json still saved for the rest): "
            + "; ".join(f"{m} [{err[:80]}]" for m, err in dead)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
