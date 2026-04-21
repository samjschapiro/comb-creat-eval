"""Rescore greedy and LLM DAT responses under all 3 embeddings (GloVe,
FastText, SBERT), then average across embeddings to produce a single
per-trial score. Feeds the 1-panel greedy-vs-LLM figure.

Inputs:
  data/dat_eval/greedy_baseline_v1/trials.json        (120 greedy trials)
  data/dat_eval/run_v1/<model>/dat_responses_t1-0.json (LLM trials @ T=1.0)

Output:
  data/dat_eval/greedy_baseline_v1/trials_multi_embed.json
  data/dat_eval/greedy_baseline_v1/llm_trials_multi_embed.json
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.dat_eval.scripts.multi_embed_appendix import (  # noqa: E402
    GloVeEmbedder,
    FastTextEmbedder,
    SBERTEmbedder,
    score_dat_trial,
)
from src.dat_eval.dat import GloVeEmbeddings  # noqa: E402

GREEDY_PATH = ROOT / "data" / "dat_eval" / "greedy_baseline_v1" / "trials.json"
RUN_DIR = ROOT / "data" / "dat_eval" / "run_v1"
OUT_DIR = ROOT / "data" / "dat_eval" / "greedy_baseline_v1"
GLOVE_PATH = ROOT / "resources" / "glove.840B.300d.txt"


def load_llm_trials_t10() -> list[tuple[str, list[str]]]:
    """Yield (model_key, words) tuples for every DAT trial at T=1.0."""
    out: list[tuple[str, list[str]]] = []
    for model_dir in sorted(RUN_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        f = model_dir / "dat_responses_t1-0.json"
        if not f.exists():
            continue
        for trial in json.load(open(f)):
            if trial.get("words"):
                out.append((model_dir.name, trial["words"]))
    return out


def main():
    print("Loading GloVe (for vocab validation + scoring)...", flush=True)
    glove_vocab = GloVeEmbeddings(GLOVE_PATH)
    glove = GloVeEmbedder()
    fasttext = FastTextEmbedder()
    sbert = SBERTEmbedder()

    embedders = [glove, fasttext, sbert]
    names = ["glove", "fasttext", "sbert"]

    # -- Greedy --
    greedy_trials = json.load(open(GREEDY_PATH))
    print(f"Rescoring {len(greedy_trials)} greedy trials...", flush=True)
    greedy_out = []
    for t in greedy_trials:
        words = t["words"]
        scores = {
            name: score_dat_trial(words, emb, glove_vocab)
            for name, emb in zip(names, embedders)
        }
        avg = float(np.nanmean(list(scores.values())))
        greedy_out.append({
            "trial": t["trial"],
            "words": words,
            "scores_per_embed": scores,
            "mean_across_embeds": avg,
        })
    with open(OUT_DIR / "trials_multi_embed.json", "w") as f:
        json.dump(greedy_out, f, indent=2)

    g_avgs = np.array([x["mean_across_embeds"] for x in greedy_out])
    print(f"  greedy mean-across-embeds: M={g_avgs.mean():.2f}, SD={g_avgs.std(ddof=1):.2f}, n={len(g_avgs)}")
    for name in names:
        per = np.array([x["scores_per_embed"][name] for x in greedy_out])
        print(f"    {name}: M={np.nanmean(per):.2f}, SD={np.nanstd(per, ddof=1):.2f}")

    # -- LLMs --
    llm = load_llm_trials_t10()
    print(f"Rescoring {len(llm)} LLM trials @ T=1.0...", flush=True)
    llm_out = []
    for i, (model, words) in enumerate(llm):
        scores = {
            name: score_dat_trial(words, emb, glove_vocab)
            for name, emb in zip(names, embedders)
        }
        vals = [v for v in scores.values() if not math.isnan(v)]
        if not vals:
            continue
        avg = float(np.mean(vals))
        llm_out.append({
            "model": model,
            "words": words,
            "scores_per_embed": scores,
            "mean_across_embeds": avg,
        })
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(llm)}", flush=True)
    with open(OUT_DIR / "llm_trials_multi_embed.json", "w") as f:
        json.dump(llm_out, f, indent=2)

    l_avgs = np.array([x["mean_across_embeds"] for x in llm_out])
    print(f"  LLM mean-across-embeds: M={l_avgs.mean():.2f}, SD={l_avgs.std(ddof=1):.2f}, n={len(l_avgs)}")
    for name in names:
        per = np.array([x["scores_per_embed"][name] for x in llm_out])
        print(f"    {name}: M={np.nanmean(per):.2f}, SD={np.nanstd(per, ddof=1):.2f}")


if __name__ == "__main__":
    main()
