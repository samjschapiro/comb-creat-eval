"""Exp 2 analysis: does a PROMPTING METHOD improve transformational creativity?

Downstream of the prompt-method generation + rubric scoring + realism scoring. For each
model we have stories under two conditions at fixed temperature: an INDEPENDENT baseline
and IN-CONTEXT REGENERATION. We:
  1. annotate each story's `reveal` (for the diversity facet) and load realism scores;
  2. build per-(model, method) cells with the four TC facets -- mean surprise, mean
     coherence, reveal diversity, mean realism -- plus mean overall;
  3. report the WITHIN-MODEL effect of the method: Δ(treatment - baseline) per facet +
     the 4-facet TC composite, with a sign test across models.

Claim tested: prompt-based interventions weakly/inconsistently move transformational
creativity, and in-context regeneration (the one consistent lever in NoveltyBench /
CREATE) lifts the DIVERSITY facet largely by construction, not surprise/coherence.

Usage:
    python src/plot_twist/scripts/run_prompt_analysis.py configs/plot_twist/prompt_methods_analysis.yaml --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import init_directory, load_config, save_config
from src.plot_twist.annotate import AnnotateConfig, annotate_stories
from src.plot_twist.join import mean_pairwise_distance

DIMS = ("surprise", "coherence", "overall")
FACETS = [("mean_surprise", "Surprise"), ("mean_coherence", "Coherence"),
          ("div", "Diversity"), ("mean_realism", "Realism")]
FACET_KEYS = [k for k, _ in FACETS]


def _load_scores(csv_path: Path) -> dict[str, dict]:
    out = {}
    for r in csv.DictReader(csv_path.open()):
        def f(k):
            try:
                return float(r[k])
            except (TypeError, ValueError, KeyError):
                return None
        out[r["slug"]] = {d: f(d) for d in DIMS}
    return out


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "generations_json", "rubric_scores_csv"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    baseline = cfg.get("baseline_method", "baseline")
    # one or more treatment methods (e.g. be_creative, incontext_regen), each compared
    # against the reused baseline.
    treatments = cfg.get("treatment_methods") or [cfg.get("treatment_method", "incontext_regen")]
    order = cfg.get("method_order", [baseline, *treatments])

    # --- treatments (the newly generated stories) + their scores ---
    gens = json.loads(Path(cfg["generations_json"]).read_text())
    gens = [r for r in gens if r.get("story") and r.get("prompt_method") in set(treatments)]
    scores = _load_scores(Path(cfg["rubric_scores_csv"]))
    realism = {}
    if cfg.get("realism_scores") and Path(cfg["realism_scores"]).exists():
        realism = json.loads(Path(cfg["realism_scores"]).read_text())

    # --- baseline: REUSE the main-run scored stories for the same models (no re-spend) ---
    if cfg.get("baseline_generations_json"):
        subset = set(cfg["subset_models"])
        btemp = cfg.get("baseline_temperature", 1.0)
        bn = cfg.get("baseline_n_samples", cfg.get("n_samples", 8))
        ball = json.loads(Path(cfg["baseline_generations_json"]).read_text())
        per_model: dict[str, list] = {}
        for r in ball:
            if (r.get("model") in subset and r.get("story")
                    and float(r.get("temperature", -1)) == float(btemp)
                    and not r.get("reasoning_level") and not r.get("prompt_method")):
                per_model.setdefault(r["model"], []).append(r)
        base_rows = []
        for m, rs in per_model.items():
            rs = sorted(rs, key=lambda r: r.get("sample", 0))[:bn]
            for r in rs:
                r = {**r, "prompt_method": baseline}
                base_rows.append(r)
        gens += base_rows
        # merge baseline rubric + realism scores into the lookup tables
        scores.update(_load_scores(Path(cfg["baseline_rubric_scores_csv"])))
        if cfg.get("baseline_realism_scores") and Path(cfg["baseline_realism_scores"]).exists():
            realism.update(json.loads(Path(cfg["baseline_realism_scores"]).read_text()))
        print(f"reused baseline: {len(base_rows)} stories from {cfg['baseline_generations_json']}")

    print(f"loaded realism for {len(realism)} stories")
    if debug:
        gens = gens[:20]
    print(f"{len(gens)} stories across {len(set(r['model'] for r in gens))} models x "
          f"{len(set(r['prompt_method'] for r in gens))} methods")

    # 1) annotate reveals (cached) for the diversity facet
    items = [{"id": r["id"], "source": r["model"], "story": r["story"],
              "scores": scores.get(r["id"], {})} for r in gens]
    acfg = AnnotateConfig(model=cfg["annotator_model"], concurrency=cfg.get("concurrency", 16))
    annos = asyncio.run(annotate_stories(acfg, items, cache_dir=out / "annotate_cache"))
    reveal_by_id = {a["id"]: a.get("reveal") for a in annos}

    from sentence_transformers import SentenceTransformer
    embed = SentenceTransformer(cfg.get("embed_model", "sentence-transformers/all-mpnet-base-v2"))
    have = [r for r in gens if reveal_by_id.get(r["id"])]
    embs = embed.encode([reveal_by_id[r["id"]] for r in have],
                        normalize_embeddings=True, show_progress_bar=False)
    emb_by_id = {r["id"]: e for r, e in zip(have, np.asarray(embs, dtype=np.float32))}

    # 2) per-(model, method) cells with all four facets + overall
    cells: dict[tuple, list] = {}
    for r in gens:
        cells.setdefault((r["model"], r["prompt_method"]), []).append(r)
    rows = []
    for (mdl, meth), rs in cells.items():
        def dim_mean(d):
            vs = [scores.get(r["id"], {}).get(d) for r in rs]
            vs = [v for v in vs if v is not None]
            return float(np.mean(vs)) if vs else float("nan")
        E = np.array([emb_by_id[r["id"]] for r in rs if r["id"] in emb_by_id])
        rv = [realism[r["id"]] for r in rs if r["id"] in realism]
        rows.append({
            "model": mdl, "method": meth, "n": len(rs),
            "mean_surprise": dim_mean("surprise"), "mean_coherence": dim_mean("coherence"),
            "mean_overall": dim_mean("overall"), "div": mean_pairwise_distance(E),
            "mean_realism": float(np.mean(rv)) if rv else float("nan"),
        })

    # 3) 4-facet TC composite, z-scored WITHIN model (isolates the method effect)
    by_model: dict[str, list] = {}
    for d in rows:
        by_model.setdefault(d["model"], []).append(d)
    for mdl, drs in by_model.items():
        for k in FACET_KEYS:
            v = np.array([d[k] for d in drs], float)
            mu, sd = np.nanmean(v), (np.nanstd(v) or 1.0)
            for d in drs:
                d[f"z_{k}"] = (d[k] - mu) / sd
        for d in drs:
            d["tc_within"] = float(np.nanmean([d[f"z_{k}"] for k in FACET_KEYS]))
    (out / "prompt_cells.json").write_text(json.dumps(rows, indent=2))

    # 4) Δ(treatment - baseline) per facet + composite, with sign test across models,
    #    for EACH treatment method against the reused baseline.
    metrics = [("mean_surprise", "surprise"), ("mean_coherence", "coherence"),
               ("div", "diversity"), ("mean_realism", "realism"),
               ("mean_overall", "overall"), ("tc_within", "TC composite")]
    summ = {}
    for treatment in treatments:
        print(f"\nWithin-model Δ({treatment} - {baseline}):")
        print(f"  {'metric':<14}{'mean Δ':>10}{'# improving':>14}{'base→treat (raw means)':>28}")
        summ[treatment] = {}
        for key, label in metrics:
            deltas, b_all, t_all = [], [], []
            for mdl, drs in by_model.items():
                lut = {d["method"]: d for d in drs}
                if (baseline in lut and treatment in lut
                        and np.isfinite(lut[baseline][key]) and np.isfinite(lut[treatment][key])):
                    deltas.append(lut[treatment][key] - lut[baseline][key])
                    b_all.append(lut[baseline][key]); t_all.append(lut[treatment][key])
            deltas = np.array(deltas)
            npos = int((deltas > 0).sum())
            print(f"  {label:<14}{deltas.mean():>+10.3f}{f'{npos}/{len(deltas)}':>14}"
                  f"{f'{np.mean(b_all):.2f} → {np.mean(t_all):.2f}':>28}")
            summ[treatment][label] = {"mean_delta": float(deltas.mean()), "n_improving": npos,
                                      "n_models": len(deltas), "baseline_mean": float(np.mean(b_all)),
                                      "treatment_mean": float(np.mean(t_all))}
    (out / "prompt_summary.json").write_text(json.dumps(summ, indent=2))

    # --- figure: facets + composite across methods, per-model faint + bold pooled mean ---
    panels = [("mean_surprise", "Surprise (1–5)"), ("mean_coherence", "Coherence (1–5)"),
              ("div", "Diversity"), ("mean_realism", "Realism (1–5)"),
              ("tc_within", "TC composite (within-model z)")]
    xpos = {m: i for i, m in enumerate(order)}
    fig, axes = plt.subplots(1, 5, figsize=(23, 5.2))
    for ax, (key, label) in zip(axes, panels):
        ys_by_model = []
        for mdl, drs in by_model.items():
            lut = {d["method"]: d for d in drs}
            xs = [xpos[m] for m in order if m in lut and np.isfinite(lut[m][key])]
            ys = [lut[m][key] for m in order if m in lut and np.isfinite(lut[m][key])]
            if len(xs) == len(order):
                ax.plot(xs, ys, color="#BBBBBB", lw=1.2, marker="o", ms=4, zorder=1)
                ys_by_model.append([lut[m][key] for m in order])
        if ys_by_model:
            M = np.array(ys_by_model, float)
            mean = np.nanmean(M, axis=0)
            se = np.nanstd(M, axis=0) / np.sqrt(M.shape[0])
            xs = list(xpos.values())
            ax.errorbar(xs, mean, yerr=se, color="#103D5F", lw=3, marker="o", ms=10,
                        capsize=5, zorder=3, label="pooled mean ± SE")
        ax.set_xticks(list(xpos.values())); ax.set_xticklabels(order, fontsize=12)
        ax.set_title(label, fontsize=15, fontweight="bold", pad=8)
        ax.set_xlabel("Prompt method", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
    axes[0].legend(loc="lower left", fontsize=11)
    fig.suptitle("Effect of prompting method on transformational creativity "
                 "(within-model; grey = each model, blue = pooled mean)",
                 fontsize=17, fontweight="bold", y=1.04)
    fig.tight_layout()
    p = out / "prompt_facets.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved: {out/'prompt_cells.json'}\n       {out/'prompt_summary.json'}\n       {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
