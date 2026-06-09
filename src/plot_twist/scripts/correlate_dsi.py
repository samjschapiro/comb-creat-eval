"""H4 test: does DSI (Divergent Semantic Integration) track plot-twist quality?

Computes DSI per story for the full pool (human gold + LLM-generated), joins each
story's surprise*coherence (rubric judge, median over judges), and correlates DSI
vs S*Coh per story. H4 predicts DSI is BLIND to twist quality (low correlation),
even though it tracks general/exploratory creativity. Also reports DSI-vs-length
(the known DSI length confound) and within-group correlations.

Usage:
    python src/plot_twist/scripts/correlate_dsi.py configs/plot_twist/dsi_quality.yaml --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.utils import init_directory, load_config, save_config
from src.plot_twist.dsi import DSIScorer, DSIConfig
from src.plot_twist.sets import twist_types


def _scores(csv_path):
    out = {}
    for r in csv.DictReader(Path(csv_path).open()):
        try:
            out[r["slug"]] = (float(r["surprise"]), float(r["coherence"]))
        except (TypeError, ValueError):
            pass
    return out


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    llm_scores = _scores(cfg["llm_scores_csv"])
    human_scores = _scores(cfg["human_scores_csv"])
    types = twist_types(cfg["manifest"])

    # assemble (id, source, text, S, Coh)
    items = []
    for r in json.loads(Path(cfg["llm_generations"]).read_text()):
        if r.get("story") and r["id"] in llm_scores:
            s, c = llm_scores[r["id"]]
            items.append({"id": r["id"], "src": "llm", "text": r["story"], "S": s, "Coh": c})
    for txt in sorted(Path(cfg["human_texts_dir"]).glob("*.txt")):
        slug = txt.stem
        if slug in human_scores:
            s, c = human_scores[slug]
            items.append({"id": slug, "src": "human", "text": txt.read_text(encoding="utf-8"),
                          "S": s, "Coh": c, "strong": types.get(slug) == "STRONG"})
    if debug:
        items = items[:8]
    print(f"scoring DSI for {len(items)} stories "
          f"({sum(i['src']=='llm' for i in items)} LLM, {sum(i['src']=='human' for i in items)} human)")

    chunk_words = cfg.get("dsi_chunk_words", 600)
    scorer = DSIScorer(DSIConfig())
    for k, it in enumerate(items):
        res = scorer.score_chunked(it["text"], chunk_words=chunk_words)  # DSI per 600-word chunk, averaged
        it["dsi"] = float(res["mean"])
        it["n_chunks"] = int(res["n_chunks"])
        it["len"] = len(it["text"].split())
        if (k + 1) % 40 == 0:
            print(f"  {k+1}/{len(items)}")

    json.dump(items, open(out / "dsi_quality.json", "w"), indent=2)

    def corr(sel, label):
        d = np.array([i["dsi"] for i in sel])
        q = np.array([i["S"] * i["Coh"] for i in sel])
        if len(sel) < 3:
            print(f"  {label:<28} n={len(sel)} (too few)"); return
        rp, pp = pearsonr(d, q)
        rs, ps = spearmanr(d, q)
        print(f"  {label:<28} n={len(sel):>3}  Pearson r={rp:+.3f} (p={pp:.3f})  Spearman rho={rs:+.3f} (p={ps:.3f})")

    print("\nDSI vs surprise*coherence:")
    corr(items, "ALL (human + LLM)")
    corr([i for i in items if i["src"] == "llm"], "LLM only")
    corr([i for i in items if i["src"] == "human"], "Human (all)")
    corr([i for i in items if i["src"] == "human" and i.get("strong")], "Human (STRONG only)")

    # length confound + components
    d = np.array([i["dsi"] for i in items]); ln = np.array([i["len"] for i in items])
    S = np.array([i["S"] for i in items]); C = np.array([i["Coh"] for i in items])
    print("\ncontext:")
    print(f"  DSI vs length (words)        Pearson r={pearsonr(d, ln)[0]:+.3f}")
    print(f"  DSI vs surprise              Pearson r={pearsonr(d, S)[0]:+.3f}")
    print(f"  DSI vs coherence             Pearson r={pearsonr(d, C)[0]:+.3f}")
    print(f"\nsaved: {out/'dsi_quality.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
