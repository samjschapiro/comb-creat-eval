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
from src.plot_twist.generate import GenerateConfig, generate_stories, model_id_to_key, rec_id_for


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
    # Exp 1 thinking intervention: optional list of {"name","reasoning"} levels.
    reasoning_levels = cfg_dict.get("reasoning_levels")
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
        reasoning_levels=reasoning_levels,
    )
    levels = cfg.levels()

    total = len(models) * len(temperatures) * len(levels) * n_samples
    # Count how many stories are already on disk (resume: these cost no new API).
    already = 0
    if stories_dir.exists():
        for m in models:
            for t in temperatures:
                for lvl in levels:
                    for i in range(n_samples):
                        p = stories_dir / model_id_to_key(m) / f"{rec_id_for(m, t, lvl['name'], i)}.json"
                        if p.exists() and json.loads(p.read_text()).get("story"):
                            already += 1
    lvl_names = [l["name"] for l in levels]
    print(
        f"generating {total} stories = {len(models)} models x "
        f"{len(temperatures)} temps {temperatures} x {len(levels)} levels {lvl_names} x {n_samples} samples"
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
    # Exp 1: per-(model, level) thinking coverage. Confirms the reasoning intervention
    # actually took effect -- a level that returns 0 reasoning tokens did NOT think
    # (provider ignored the spec), which would silently void the causal contrast.
    # Printed BEFORE the dead-model raise so a smoke across many models always reports.
    if reasoning_levels:
        print("\nthinking coverage (ok stories | median reasoning_tokens) per (model, level):")
        import statistics
        for m in models:
            for lvl in levels:
                rows = [r for r in records
                        if r["model"] == m and r.get("reasoning_level") == lvl["name"]]
                ok = [r for r in rows if r["story"]]
                rtoks = [r["reasoning_tokens"] for r in ok if r.get("reasoning_tokens")]
                med = int(statistics.median(rtoks)) if rtoks else 0
                flag = "  <-- NO reasoning tokens returned" if ok and not rtoks else ""
                print(f"  {m:<38} {str(lvl['name']):<8} {len(ok):>2}/{len(rows):<2}  rtok~{med}{flag}")

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
