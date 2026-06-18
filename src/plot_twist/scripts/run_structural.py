"""Exp 3 (workshop): SBV structural metric (T_mod, preservation) vs DSI.

An LLM extractor reconstructs each story's reader story-DAG and we compute the symbolic
twist metrics from the SBV theory: T_mod(a*) = number of nodes depending on the flipped
axiom (surprise term) and preservation(a') = fraction of artifacts that survive the flip
(coherence term). We then test, on a stratified subset, whether these SYMBOLIC metrics
predict the rubric transformational-creativity dimensions -- and whether they add signal
OVER DSI (the neural/embedding metric from Exp 2). If the structural metric tracks
surprise/coherence where DSI is blind, the paper's claim follows: measuring
transformational creativity needs BOTH symbolic structure and neural semantics.

Pre/post OpenRouter /key cost is logged (paid extractor calls).

Usage:
    python src/plot_twist/scripts/run_structural.py configs/plot_twist/structural.yaml --overwrite [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.utils import init_directory, load_config, save_config
from src.plot_twist.join import load_annotations, attach_story_text, score_num
from src.plot_twist.sets import twist_types
from src.plot_twist.structural import ExtractConfig, extract_stories
from src.plot_twist.scripts.cost_log import key_usage

TC_DIMS = ("surprise", "coherence", "overall")


def _stratified_subset(recs, n_llm, strata, seed):
    """All human STRONG stories + an LLM sample spread across overall-score strata."""
    human = [r for r in recs if r["source"] == "human"]
    llm = [r for r in recs if r["source"] != "human" and score_num(r, "overall") is not None]
    rng = np.random.default_rng(seed)
    # bin LLM stories by overall score, sample proportionally across bins
    overalls = np.array([score_num(r, "overall") for r in llm])
    edges = np.quantile(overalls, np.linspace(0, 1, strata + 1))
    picked = []
    per = max(1, n_llm // strata)
    for b in range(strata):
        lo, hi = edges[b], edges[b + 1]
        inb = [r for r, o in zip(llm, overalls) if (o >= lo and o < hi) or (b == strata - 1 and o == hi)]
        if inb:
            idx = rng.choice(len(inb), size=min(per, len(inb)), replace=False)
            picked.extend(inb[i] for i in idx)
    return human + picked


def _ols_r2(X, y):
    """R^2 of an OLS fit of y on columns X (with intercept)."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "annotations_json", "extractor_models"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    # Build stratified subset, then join to story text.
    recs = load_annotations(cfg["annotations_json"])
    # keep only human STRONG (genuine reinterpretation) for the human leg
    types = twist_types(cfg["manifest"])
    recs = [r for r in recs if r["source"] != "human" or types.get(r["id"]) == "STRONG"]
    subset = _stratified_subset(recs, cfg.get("n_llm_subset", 200), cfg.get("strata", 10), cfg.get("seed", 0))
    subset = attach_story_text(subset, cfg["llm_stories_dir"], cfg["human_texts_dir"])
    if debug:
        subset = subset[:6]
    print(f"extracting DAGs for {len(subset)} stories "
          f"({sum(r['source']=='human' for r in subset)} human STRONG, "
          f"{sum(r['source']!='human' for r in subset)} LLM) with {len(cfg['extractor_models'])} extractors")

    # Cost guard: ground-truth /key usage before any paid call.
    budget = cfg.get("budget_usd")
    start_usage = key_usage()
    print(f"OpenRouter /key usage before extraction: ${start_usage:.4f}"
          + (f"  (budget cap ${budget})" if budget else ""))

    ecfg = ExtractConfig(
        extractor_models=cfg["extractor_models"],
        max_tokens=cfg.get("max_tokens", 1600),
        concurrency=cfg.get("concurrency", 12),
    )
    results = asyncio.run(extract_stories(ecfg, subset, out / "cache"))

    end_usage = key_usage()
    print(f"OpenRouter /key usage after extraction:  ${end_usage:.4f}  (Δ ${end_usage-start_usage:.4f})")
    if budget and (end_usage - start_usage) > budget:
        print(f"WARNING: extraction Δ ${end_usage-start_usage:.4f} exceeded budget ${budget}")

    # Merge structural metrics + rubric dims + DSI (Exp 2) per story.
    by_id = {r["id"]: r for r in subset}
    dsi_by_id = {}
    if cfg.get("dsi_json") and Path(cfg["dsi_json"]).exists():
        dsi_by_id = {d["id"]: d["dsi"] for d in json.loads(Path(cfg["dsi_json"]).read_text())}
    rows = []
    for res in results:
        m = res.get("metrics")
        if not m:
            continue
        src = by_id.get(res["id"], {})
        rows.append({
            "id": res["id"], "source": res["source"],
            "t_mod": m["t_mod"], "t_mod_frac": m["t_mod_frac"],
            "preservation": m["preservation"], "structural_ptc": m["structural_ptc"],
            "has_twist": m["has_twist"], "n_extractors": m.get("n_extractors"),
            "dsi": dsi_by_id.get(res["id"]),
            **{d: score_num(src, d) for d in TC_DIMS},
        })
    (out / "structural.json").write_text(json.dumps(rows, indent=2))
    print(f"\n{len(rows)} stories with valid structural metrics")

    # --- correlations: structural metric vs each TC dimension, head-to-head with DSI ---
    def col(key):
        return np.array([r[key] if r[key] is not None else np.nan for r in rows], dtype=float)

    def corr(xkey, ykey, label):
        x, y = col(xkey), col(ykey)
        keep = ~np.isnan(x) & ~np.isnan(y)
        if keep.sum() < 4 or x[keep].std() == 0 or y[keep].std() == 0:
            print(f"  {label:<34} (degenerate, n={int(keep.sum())})"); return None
        rp, pp = pearsonr(x[keep], y[keep]); rs, _ = spearmanr(x[keep], y[keep])
        print(f"  {label:<34} n={int(keep.sum()):>3}  r={rp:+.3f} (p={pp:.3f})  rho={rs:+.3f}")
        return rp

    print("\nStructural metric vs rubric dimensions (the theory's pairing):")
    corr("t_mod_frac", "surprise", "T_mod_frac  vs surprise")
    corr("preservation", "coherence", "preservation vs coherence")
    corr("structural_ptc", "overall", "structural_ptc vs overall")
    print("\nDSI (neural) vs the same dimensions (Exp-2 metric, head-to-head):")
    corr("dsi", "surprise", "DSI vs surprise")
    corr("dsi", "coherence", "DSI vs coherence")
    corr("dsi", "overall", "DSI vs overall")

    # --- incremental R^2: does structure add over DSI (and vice versa)? ---
    print("\nIncremental variance explained (OLS R^2):")
    inc = {}
    for ykey, structkey in (("surprise", "t_mod_frac"), ("coherence", "preservation"), ("overall", "structural_ptc")):
        y = col(ykey); d = col("dsi"); s = col(structkey)
        keep = ~np.isnan(y) & ~np.isnan(d) & ~np.isnan(s)
        if keep.sum() < 8:
            print(f"  {ykey:<10} (too few complete rows: {int(keep.sum())})"); continue
        yy, dd, ss = y[keep], d[keep], s[keep]
        r2_dsi = _ols_r2(dd.reshape(-1, 1), yy)
        r2_struct = _ols_r2(ss.reshape(-1, 1), yy)
        r2_both = _ols_r2(np.column_stack([dd, ss]), yy)
        print(f"  {ykey:<10} DSI={r2_dsi:.3f}  struct={r2_struct:.3f}  both={r2_both:.3f}  "
              f"(struct adds {r2_both-r2_dsi:+.3f} over DSI; DSI adds {r2_both-r2_struct:+.3f} over struct)")
        inc[ykey] = {"r2_dsi": r2_dsi, "r2_struct": r2_struct, "r2_both": r2_both, "n": int(keep.sum())}

    # --- inter-extractor reliability (the make-or-break number for the metric) ---
    print("\nInter-extractor reliability (T_mod_frac, preservation across the 2 extractors):")
    rel = _extractor_reliability(results, cfg["extractor_models"])
    for k, v in rel.items():
        print(f"  {k:<16} r={v['r']:+.3f}  n={v['n']}" if v else f"  {k:<16} (n/a)")

    summary = {"n": len(rows), "incremental_r2": inc, "reliability": rel,
               "cost_delta_usd": end_usage - start_usage}
    (out / "structural_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved: {out/'structural.json'}\n       {out/'structural_summary.json'}")


def _extractor_reliability(results, extractor_models):
    """Pearson r between the two extractors' per-story t_mod_frac and preservation."""
    out = {}
    if len(extractor_models) < 2:
        return out
    m0, m1 = extractor_models[0], extractor_models[1]
    for key in ("t_mod_frac", "preservation"):
        a, b = [], []
        for res in results:
            be = res.get("by_extractor", {})
            ma = (be.get(m0) or {}).get("metrics")
            mb = (be.get(m1) or {}).get("metrics")
            if ma and mb and ma.get(key) == ma.get(key) and mb.get(key) == mb.get(key):
                a.append(ma[key]); b.append(mb[key])
        if len(a) >= 4 and np.std(a) and np.std(b):
            r, _ = pearsonr(a, b)
            out[key] = {"r": float(r), "n": len(a)}
        else:
            out[key] = None
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
