"""Compute DSI for the fetched public-domain gold-set stories.

Reads data/plot_twist/human_twists/{texts/*.txt, fetched_manifest.json}, scores
each with src/plot_twist/dsi.py, writes data/plot_twist/dsi/dsi_scores.json, and
prints a table sorted by DSI. First run downloads bert-base-uncased (~440 MB).

Usage: uv run python src/plot_twist/scripts/run_dsi.py   (or .venv/bin/python)
"""

from __future__ import annotations

import json
from pathlib import Path

from src.plot_twist.dsi import DSIConfig, DSIScorer

ROOT = Path("data/plot_twist/human_twists")
OUT = Path("data/plot_twist/dsi")


def main() -> None:
    fm = json.loads((ROOT / "fetched_manifest.json").read_text())["stories"]
    cfg = DSIConfig()
    scorer = DSIScorer(cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    CHUNK = 100  # words per chunk (within DSI's validated short-narrative range)
    rows = []
    for s in fm:
        txt = (ROOT / "texts" / f"{s['slug']}.txt").read_text(encoding="utf-8")
        dsi, n = scorer.score(txt)
        ch = scorer.score_chunked(txt, chunk_words=CHUNK)
        rows.append({
            "slug": s["slug"], "title": s["title"], "author": s["author"],
            "word_count": s["word_count"],
            "dsi_whole": round(dsi, 4), "words_used": n,
            "dsi_chunked": round(ch["mean"], 4), "dsi_chunked_std": round(ch["std"], 4),
            "n_chunks": ch["n_chunks"],
        })
        print(f"  {s['slug']:<22} whole={dsi:.4f}  chunked={ch['mean']:.4f}"
              f"(±{ch['std']:.3f}, {ch['n_chunks']} chunks)  words={s['word_count']}")

    rows.sort(key=lambda r: r["dsi_chunked"])
    (OUT / "dsi_scores.json").write_text(json.dumps(
        {"config": {**cfg.__dict__, "chunk_words": CHUNK}, "scores": rows},
        indent=2, ensure_ascii=False))

    import numpy as np
    wc = np.array([r["word_count"] for r in rows], float)
    pear = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    spear = lambda a, b: float(np.corrcoef(np.argsort(np.argsort(a)),
                                           np.argsort(np.argsort(b)))[0, 1])
    print("\n=== chunked DSI, sorted low -> high ===")
    for r in rows:
        print(f"  {r['dsi_chunked']:.4f}  {r['title'][:40]:<40} ({r['author']})")
    for tag, key in [("whole-text DSI (capped @600)", "dsi_whole"),
                     ("chunked DSI (mean of 100-word chunks)", "dsi_chunked")]:
        v = np.array([r[key] for r in rows], float)
        print(f"\n{tag}: mean={v.mean():.4f} range={v.min():.4f}-{v.max():.4f}")
        print(f"   r(word_count, {key}) Pearson={pear(wc, v):+.3f}  Spearman={spear(wc, v):+.3f}")
    print(f"\nsaved: {OUT/'dsi_scores.json'}")


if __name__ == "__main__":
    main()
