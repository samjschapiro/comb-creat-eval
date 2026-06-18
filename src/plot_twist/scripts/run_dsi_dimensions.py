"""Exp 2 (workshop): does DSI predict the DIMENSIONS of transformational creativity?

DSI (Divergent Semantic Integration; Johnson et al. 2022) is a purely EMBEDDING-based
creativity measure: mean pairwise BERT distance among a text's words. We test, on the
full 74-source plot_twist pool, whether DSI predicts the transformational-creativity
dimensions (surprise, coherence, overall plot-twist quality) -- and contrast that with
a PROSE-QUALITY positive control (DSI should track lexical/semantic richness). If DSI
is blind to surprise/coherence yet still tracks prose quality, the null is twist-
specific, not a broken DSI: the narrative is the limit of embedding-only measures for
transformational creativity (motivating the symbolic structural metric in Exp 3).

Per-story DSI is cached by id (local compute, no API) so re-runs are instant.

Usage:
    python src/plot_twist/scripts/run_dsi_dimensions.py configs/plot_twist/dsi_dimensions.yaml --overwrite [--debug]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.utils import init_directory, load_config, save_config
from src.plot_twist.dsi import DSIScorer, DSIConfig
from src.plot_twist.join import (
    load_annotations, attach_story_text, score_num,
)

# Transformational-creativity dimensions (the metrics under test) + the prose control.
TC_DIMS = ("surprise", "coherence", "overall")
CONTROL_DIM = "prose_quality"


def _bootstrap_ci(x: np.ndarray, y: np.ndarray, iters: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for Pearson r."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rs = np.empty(iters)
    for b in range(iters):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if xb.std() == 0 or yb.std() == 0:
            rs[b] = 0.0
        else:
            rs[b] = np.corrcoef(xb, yb)[0, 1]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "annotations_json"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    # Join annotations -> story text (drop any story whose text isn't on disk).
    recs = load_annotations(cfg["annotations_json"])
    n_before = len(recs)
    recs = attach_story_text(recs, cfg["llm_stories_dir"], cfg["human_texts_dir"])
    print(f"joined {len(recs)}/{n_before} stories to text "
          f"({sum(r['source']=='human' for r in recs)} human, "
          f"{sum(r['source']!='human' for r in recs)} LLM)")
    if debug:
        recs = recs[:12]

    # Per-story DSI, cached by id (BERT is slow; never recompute).
    cache_dir = out / "dsi_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_words = cfg.get("dsi_chunk_words", 600)
    scorer = DSIScorer(DSIConfig())
    items = []
    for k, r in enumerate(recs):
        cpath = cache_dir / f"{r['id']}.json"
        if cpath.exists():
            dsi = json.loads(cpath.read_text())["dsi"]
        else:
            dsi = float(scorer.score_chunked(r["text"], chunk_words=chunk_words)["mean"])
            cpath.write_text(json.dumps({"id": r["id"], "dsi": dsi}))
        items.append({
            "id": r["id"], "source": r["source"], "dsi": dsi,
            "len": len(r["text"].split()),
            **{d: score_num(r, d) for d in (*TC_DIMS, CONTROL_DIM)},
        })
        if (k + 1) % 100 == 0:
            print(f"  DSI {k+1}/{len(recs)}")

    (out / "dsi_dimensions.json").write_text(json.dumps(items, indent=2))

    # Correlate DSI vs each dimension, with bootstrap CIs + length covariate.
    iters = 200 if debug else cfg.get("bootstrap_iters", 5000)
    seed = cfg.get("seed", 0)

    def report(sel: list[dict], label: str) -> dict:
        print(f"\n[{label}]  n={len(sel)}")
        dsi = np.array([i["dsi"] for i in sel])
        ln = np.array([i["len"] for i in sel], dtype=float)
        block = {"n": len(sel), "dims": {}}
        print(f"  {'dimension':<16}{'Pearson r':>12}{'95% CI':>20}{'Spearman':>12}{'p':>10}")
        for d in (*TC_DIMS, CONTROL_DIM):
            y = np.array([i[d] for i in sel], dtype=float)
            keep = ~np.isnan(y) & ~np.isnan(dsi)
            xx, yy = dsi[keep], y[keep]
            if len(xx) < 4 or xx.std() == 0 or yy.std() == 0:
                print(f"  {d:<16}{'(degenerate)':>12}")
                continue
            rp, pp = pearsonr(xx, yy)
            rs, _ = spearmanr(xx, yy)
            lo, hi = _bootstrap_ci(xx, yy, iters, seed)
            tag = "  <-- control" if d == CONTROL_DIM else ""
            print(f"  {d:<16}{rp:>+12.3f}{f'[{lo:+.2f},{hi:+.2f}]':>20}{rs:>+12.3f}{pp:>10.3f}{tag}")
            block["dims"][d] = {"pearson_r": rp, "ci95": [lo, hi], "spearman_r": rs, "p": pp, "n": int(len(xx))}
        # DSI-length confound (DSI is known to scale with length)
        keep = ~np.isnan(ln) & ~np.isnan(dsi)
        if keep.sum() >= 4 and dsi[keep].std() and ln[keep].std():
            rl = pearsonr(dsi[keep], ln[keep])[0]
            print(f"  {'(DSI vs length)':<16}{rl:>+12.3f}")
            block["dsi_vs_length_r"] = rl
        return block

    summary = {
        "all": report(items, "ALL (human + LLM)"),
        "llm": report([i for i in items if i["source"] != "human"], "LLM only"),
        "human": report([i for i in items if i["source"] == "human"], "Human gold only"),
    }
    (out / "dsi_dimensions_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved: {out/'dsi_dimensions.json'}\n       {out/'dsi_dimensions_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
