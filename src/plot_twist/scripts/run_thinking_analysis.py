"""Exp 1 analysis: does THINKING improve transformational creativity?

Downstream of the thinking-intervention generation + rubric scoring + realism scoring.
For each model we have stories at low/medium/high reasoning EFFORT (temperature fixed).
The number of reasoning tokens is NOT treated as the mediator (it conflates verbosity
with reasoning strength); the nominal effort level is the intervention.

We:
  1. annotate each story's `reveal` (for the diversity facet) and load realism scores;
  2. build per-(model, effort level) cells with the four TC facets -- mean surprise,
     mean coherence, reveal diversity, mean realism -- plus mean overall;
  3. score each cell with the headline 4-facet TC composite (overall_eq: equal-weight z
     of surprise/coherence/diversity/realism), z-scored WITHIN model so the composite
     measures the within-model thinking effect, not cross-model capability;
  4. report the within-model effect of effort: Δ(high - low) per facet + composite, with
     a sign test across models.

Claim tested (workshop §4): more thinking does NOT buy transformational creativity --
a controlled, within-model version of the observational "thinking doesn't help" result.

Usage:
    python src/plot_twist/scripts/run_thinking_analysis.py configs/plot_twist/thinking_analysis.yaml --overwrite
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
from src.plot_twist.join import mean_pairwise_distance, gated_means

DIMS = ("surprise", "coherence", "overall")
# Display facets (raw key -> label) shown in the per-facet panels, matching make_tc_barplot.
FACETS = [("mean_surprise", "Surprise"), ("mean_coherence", "Coherence"),
          ("div", "Diversity"), ("mean_realism", "Realism")]
FACET_KEYS = [k for k, _ in FACETS]
# Composite facets = realism-GATED surprise/coherence + diversity (the headline metric).
COMPOSITE_FACETS = ["mean_surprise_g", "mean_coherence_g", "div"]


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

    gens = json.loads(Path(cfg["generations_json"]).read_text())
    gens = [r for r in gens if r.get("story") and r.get("reasoning_level")]
    scores = _load_scores(Path(cfg["rubric_scores_csv"]))
    realism = {}
    if cfg.get("realism_scores") and Path(cfg["realism_scores"]).exists():
        realism = json.loads(Path(cfg["realism_scores"]).read_text())
        print(f"loaded realism for {len(realism)} stories")
    if debug:
        gens = gens[:20]
    print(f"{len(gens)} thinking stories across "
          f"{len(set(r['model'] for r in gens))} models x effort levels")

    # 1) annotate reveals (cached) for the diversity facet
    items = [{"id": r["id"], "source": r["model"], "story": r["story"],
              "scores": scores.get(r["id"], {})} for r in gens]
    acfg = AnnotateConfig(model=cfg["annotator_model"], concurrency=cfg.get("concurrency", 16))
    # Cache lives OUTSIDE output_dir (which --overwrite wipes) so reveal annotations -- a
    # paid step -- are never discarded on re-runs. See memory "never-waste-api-spend".
    annos = asyncio.run(annotate_stories(acfg, items, cache_dir=out.parent / "annotate_cache"))
    reveal_by_id = {a["id"]: a.get("reveal") for a in annos}

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg.get("embed_model", "sentence-transformers/all-mpnet-base-v2"))
    have = [r for r in gens if reveal_by_id.get(r["id"])]
    embs = model.encode([reveal_by_id[r["id"]] for r in have],
                        normalize_embeddings=True, show_progress_bar=False)
    emb_by_id = {r["id"]: e for r, e in zip(have, np.asarray(embs, dtype=np.float32))}

    # 2) per-(model, level) cells with all four facets + overall
    cells: dict[tuple, list] = {}
    for r in gens:
        cells.setdefault((r["model"], r["reasoning_level"]), []).append(r)
    rows = []
    for (mdl, lvl), rs in cells.items():
        def dim_mean(d):
            vs = [scores.get(r["id"], {}).get(d) for r in rs]
            vs = [v for v in vs if v is not None]
            return float(np.mean(vs)) if vs else float("nan")
        E = np.array([emb_by_id[r["id"]] for r in rs if r["id"] in emb_by_id])
        gm = gated_means([{"id": r["id"], "scores": scores.get(r["id"], {})} for r in rs], realism)
        rows.append({
            "model": mdl, "level": lvl, "n": len(rs),
            "mean_surprise": dim_mean("surprise"), "mean_coherence": dim_mean("coherence"),
            "mean_overall": dim_mean("overall"), "div": mean_pairwise_distance(E),
            "mean_realism": gm["mean_realism"],
            "mean_surprise_g": gm["mean_surprise_g"], "mean_coherence_g": gm["mean_coherence_g"],
        })

    # 3) realism-gated TC composite, z-scored WITHIN model (isolates the thinking effect)
    by_model: dict[str, list] = {}
    for d in rows:
        by_model.setdefault(d["model"], []).append(d)
    for mdl, drs in by_model.items():
        for k in COMPOSITE_FACETS:
            v = np.array([d[k] for d in drs], float)
            mu, sd = np.nanmean(v), (np.nanstd(v) or 1.0)
            for d in drs:
                d[f"z_{k}"] = (d[k] - mu) / sd
        for d in drs:
            d["tc_within"] = float(np.nanmean([d[f"z_{k}"] for k in COMPOSITE_FACETS]))
    (out / "thinking_cells.json").write_text(json.dumps(rows, indent=2))

    # 4) Δ(high - low) per facet + composite, with sign test across models
    order = cfg.get("level_order", ["low", "medium", "high"])
    metrics = [("mean_surprise", "surprise"), ("mean_coherence", "coherence"),
               ("div", "diversity"), ("mean_realism", "realism"),
               ("mean_overall", "overall"), ("tc_within", "TC composite")]
    print(f"\nWithin-model Δ(high - low) by effort level:")
    print(f"  {'metric':<14}{'mean Δ':>10}{'# improving':>14}{'low→high (raw means)':>26}")
    summ = {}
    for key, label in metrics:
        deltas, lo_all, hi_all = [], [], []
        for mdl, drs in by_model.items():
            lut = {d["level"]: d for d in drs}
            if "low" in lut and "high" in lut and np.isfinite(lut["low"][key]) and np.isfinite(lut["high"][key]):
                deltas.append(lut["high"][key] - lut["low"][key])
                lo_all.append(lut["low"][key]); hi_all.append(lut["high"][key])
        deltas = np.array(deltas)
        npos = int((deltas > 0).sum())
        print(f"  {label:<14}{deltas.mean():>+10.3f}{f'{npos}/{len(deltas)}':>14}"
              f"{f'{np.mean(lo_all):.2f} → {np.mean(hi_all):.2f}':>26}")
        summ[label] = {"mean_delta": float(deltas.mean()), "n_improving": npos,
                       "n_models": len(deltas), "low_mean": float(np.mean(lo_all)),
                       "high_mean": float(np.mean(hi_all))}
    (out / "thinking_summary.json").write_text(json.dumps(summ, indent=2))

    # --- NEW FIGURE: facets + composite vs effort, per-model faint + bold pooled mean ---
    panels = [("mean_surprise", "Surprise (1–5)"), ("mean_coherence", "Coherence (1–5)"),
              ("div", "Diversity"), ("mean_realism", "Realism (1–5)"),
              ("tc_within", "TC composite (within-model z)")]
    xpos = {l: i for i, l in enumerate(order)}
    fig, axes = plt.subplots(1, 5, figsize=(23, 5.2))
    for ax, (key, label) in zip(axes, panels):
        ys_by_model = []
        for mdl, drs in by_model.items():
            lut = {d["level"]: d for d in drs}
            xs = [xpos[l] for l in order if l in lut and np.isfinite(lut[l][key])]
            ys = [lut[l][key] for l in order if l in lut and np.isfinite(lut[l][key])]
            if len(xs) == len(order):
                ax.plot(xs, ys, color="#BBBBBB", lw=1.2, marker="o", ms=4, zorder=1)
                ys_by_model.append([lut[l][key] for l in order])
        if ys_by_model:
            M = np.array(ys_by_model, float)
            mean = np.nanmean(M, axis=0)
            se = np.nanstd(M, axis=0) / np.sqrt(M.shape[0])
            xs = list(xpos.values())
            ax.errorbar(xs, mean, yerr=se, color="#103D5F", lw=3, marker="o", ms=10,
                        capsize=5, zorder=3, label="pooled mean ± SE")
        ax.set_xticks(list(xpos.values())); ax.set_xticklabels(order, fontsize=13)
        ax.set_title(label, fontsize=15, fontweight="bold", pad=8)
        ax.set_xlabel("Reasoning effort", fontsize=13)
        ax.tick_params(axis="y", labelsize=12)
    axes[0].legend(loc="lower left", fontsize=11)
    fig.suptitle("Effect of reasoning effort on transformational creativity "
                 "(within-model; grey = each model, blue = pooled mean)",
                 fontsize=17, fontweight="bold", y=1.04)
    fig.tight_layout()
    p = out / "thinking_facets.png"
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"\nsaved: {out/'thinking_cells.json'}\n       {out/'thinking_summary.json'}\n       {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
