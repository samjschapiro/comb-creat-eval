"""Rescore DAT/CDAT/PACE under three embedding models for appendix robustness.

Each model's existing raw responses (already on disk under
`data/dat_eval/run_v1/`) are rescored under three embeddings:

  - GloVe 840B 300d (DAT original)
  - FastText crawl-300d-2M (PACE original)
  - SBERT all-mpnet-base-v2 (CDAT original)

Output:
  data/dat_eval/run_v1/downstream/scores_v1/results/multi_embed_scores.json
  data/dat_eval/run_v1/downstream/scores_v1/results/multi_embed_correlations.json

Each (task, embedder) pair yields a per-model score. We then correlate each
column with Arena CW and compute the joint partial controlling for both
Arena Overall Elo and MMLU-Pro. Result is a compact 9-row table for the
appendix: {task x embedder} x {simple r, partial r | O+M}.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr, spearmanr, rankdata

# ----- Repo paths -----
ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "data" / "dat_eval" / "run_v1"
BENCH_PATH = ROOT / "configs" / "comb_eval" / "benchmarks.json"
OUT_DIR = RUN_DIR / "downstream" / "scores_v1" / "results"
GLOVE_PATH = ROOT / "resources" / "glove.840B.300d.txt"
FASTTEXT_PATH = ROOT / "resources" / "crawl-300d-2M.vec"

sys.path.insert(0, str(ROOT))
from src.dat_eval.dat import GloVeEmbeddings, validate_words  # noqa: E402


# ---------------------------------------------------------------------------
# Unified embedder interface
# ---------------------------------------------------------------------------
class Embedder:
    """Simple protocol: encode(word) -> np.ndarray; returns zero vector on OOV.
    has(word) -> bool tells you whether this word is natively in-vocab.
    """
    name: str

    def has(self, word: str) -> bool:  # pragma: no cover
        raise NotImplementedError

    def encode(self, word: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class GloVeEmbedder(Embedder):
    name = "glove"

    def __init__(self):
        self._g = GloVeEmbeddings(GLOVE_PATH)

    def has(self, word: str) -> bool:
        return word in self._g

    def encode(self, word: str) -> np.ndarray:
        if word in self._g:
            return self._g[word]
        return np.zeros(300, dtype=np.float32)


class FastTextEmbedder(Embedder):
    name = "fasttext"

    def __init__(self):
        self._v: dict[str, np.ndarray] = {}
        print(f"Loading FastText from {FASTTEXT_PATH}...", flush=True)
        with open(FASTTEXT_PATH, "r", encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.rstrip().split(" ")
                self._v[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
        print(f"  FastText loaded: {len(self._v)} vectors", flush=True)

    def has(self, word: str) -> bool:
        return word in self._v or word.lower() in self._v

    def encode(self, word: str) -> np.ndarray:
        if word in self._v:
            return self._v[word]
        if word.lower() in self._v:
            return self._v[word.lower()]
        return np.zeros(300, dtype=np.float32)


class SBERTEmbedder(Embedder):
    name = "sbert"

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"Loading SBERT ({model_name})...", flush=True)
        self._m = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def has(self, word: str) -> bool:
        return True

    def encode(self, word: str) -> np.ndarray:
        if word not in self._cache:
            self._cache[word] = self._m.encode(word, normalize_embeddings=True)
        return self._cache[word]


# ---------------------------------------------------------------------------
# Generic scoring functions (embedder-agnostic)
# ---------------------------------------------------------------------------
def cosine_d(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return float("nan")
    d = cosine_distance(a, b)
    return float(d) if np.isfinite(d) else float("nan")


def score_dat_trial(words: list[str], emb: Embedder, glove_vocab: GloVeEmbeddings) -> float:
    """Original DAT: first 7 valid GloVe words, mean pairwise cosine * 100.
    We keep GloVe's vocab check so the word set is identical across embedders;
    only the distance computation changes."""
    valid = validate_words(words, glove_vocab)
    if len(valid) < 7:
        return math.nan
    vecs = [emb.encode(w) for w in valid[:7]]
    ds = []
    for i in range(7):
        for j in range(i + 1, 7):
            d = cosine_d(vecs[i], vecs[j])
            if not math.isnan(d):
                ds.append(d)
    return float(np.mean(ds)) * 100 if ds else math.nan


def score_cdat_trial(cue: str, words: list[str], emb: Embedder) -> tuple[float, float]:
    """CDAT under arbitrary embedder: novelty = pairwise distance among the
    first 7 valid words (matching the paper and cdat.py); appropriateness =
    mean cos-sim to the cue on the shifted [0, 200] scale (matching
    cdat.py and cdat_gate.py)."""
    if not words:
        return math.nan, math.nan
    # Keep only valid words (non-empty embedding) and truncate to first 7
    valid = []
    for w in words:
        v = emb.encode(w)
        if np.linalg.norm(v) > 0:
            valid.append((w, v))
        if len(valid) == 7:
            break
    if len(valid) < 2:
        return math.nan, math.nan
    vecs = [v for _, v in valid]
    # Novelty: mean pairwise cosine distance of first 7 valid words × 100
    ds = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            d = cosine_d(vecs[i], vecs[j])
            if not math.isnan(d):
                ds.append(d)
    nov = float(np.mean(ds)) * 100 if ds else math.nan
    # Appropriateness: mean (1 + cos_sim) × 100, shifted to [0, 200]
    cue_v = emb.encode(cue)
    if np.linalg.norm(cue_v) == 0:
        app = math.nan
    else:
        sims = []
        for v in vecs:
            d = cosine_d(cue_v, v)
            if not math.isnan(d):
                sims.append(1.0 - d)
        app = float(np.mean([100.0 * (1.0 + s) for s in sims])) if sims else math.nan
    return nov, app


def score_pace_chain(chain: list[str], emb: Embedder) -> float:
    """PACE chain score = mean-over-positions of mean-cosine-distance to all
    earlier positions. Matches Qiu & Hu; embedder pluggable."""
    vecs = [emb.encode(w.lower()) for w in chain]
    vecs = [v for v in vecs if np.linalg.norm(v) > 0]
    if len(vecs) < 2:
        return math.nan
    per_pos = []
    for i in range(1, len(vecs)):
        ds = []
        for j in range(i):
            d = cosine_d(vecs[i], vecs[j])
            if not math.isnan(d):
                ds.append(d)
        if ds:
            per_pos.append(float(np.mean(ds)))
    return float(np.mean(per_pos)) if per_pos else math.nan


# ---------------------------------------------------------------------------
# Per-model rescore driver
# ---------------------------------------------------------------------------
def rescore_model(model_dir: Path, emb: Embedder, glove_vocab: GloVeEmbeddings) -> dict:
    out = {"dat": math.nan, "cdat_novelty": math.nan,
           "cdat_appropriateness": math.nan, "pace": math.nan}

    # --- DAT: pooled across temperatures ---
    dat_trials = []
    for tf in sorted(model_dir.glob("dat_responses_t*.json")):
        try:
            data = json.loads(tf.read_text())
        except Exception:
            continue
        for tr in data:
            words = tr.get("words") or []
            if not words:
                continue
            s = score_dat_trial(words, emb, glove_vocab)
            if not math.isnan(s):
                dat_trials.append(s)
    if dat_trials:
        out["dat"] = float(np.mean(dat_trials))

    # --- CDAT: pooled across temperatures and cues ---
    cdat_nov, cdat_app = [], []
    for tf in sorted(model_dir.glob("cdat_responses_t*.json")):
        try:
            data = json.loads(tf.read_text())
        except Exception:
            continue
        # data is dict cue -> {words, ...}
        for cue, entry in data.items():
            words = entry.get("words") or []
            if not words:
                continue
            nov, app = score_cdat_trial(cue, words, emb)
            if not math.isnan(nov):
                cdat_nov.append(nov)
            if not math.isnan(app):
                cdat_app.append(app)
    if cdat_nov:
        out["cdat_novelty"] = float(np.mean(cdat_nov))
    if cdat_app:
        out["cdat_appropriateness"] = float(np.mean(cdat_app))

    # --- PACE: pooled across seeds and chains ---
    pace_file = model_dir / "pace_responses_t0-0.json"
    if pace_file.exists():
        try:
            data = json.loads(pace_file.read_text())
        except Exception:
            data = {}
        # Match the original PACE pipeline's filtering: only chains with >=3
        # words, then drop chain scores <= 0 and seed scores <= 0 before
        # taking means. This keeps multi-embed numbers comparable to the
        # main-pipeline PACE values in all_scores.json (within ~1e-4).
        seed_scores = []
        for seed, entry in data.items():
            chains = entry.get("chains") or []
            chain_scores = []
            for chain_obj in chains:
                words = chain_obj.get("chain") if isinstance(chain_obj, dict) else None
                if not words or len(words) < 3:
                    continue
                s = score_pace_chain(list(words), emb)
                if not math.isnan(s) and s > 0:
                    chain_scores.append(s)
            if chain_scores:
                seed_score = float(np.mean(chain_scores))
                if seed_score > 0:
                    seed_scores.append(seed_score)
        if seed_scores:
            out["pace"] = float(np.mean(seed_scores))

    return out


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------
def joint_partial_r(target, predictor, controls):
    """Joint partial Pearson r controlling for a stack of controls.

    Returns (r, p) where p uses df = n - 2 - k for k = len(controls), the
    correct partial-correlation degrees of freedom. (scipy.stats.pearsonr
    on residuals would use df = n - 2 and slightly overstate significance.)
    """
    X = np.column_stack([np.ones_like(target)] + [np.asarray(c) for c in controls])
    bt, *_ = np.linalg.lstsq(X, target, rcond=None)
    bp, *_ = np.linalg.lstsq(X, predictor, rcond=None)
    rt, rp = target - X @ bt, predictor - X @ bp
    r = float(pearsonr(rt, rp).statistic)
    n, k = len(target), len(controls)
    df = n - 2 - k
    if df <= 0 or not (-1.0 < r < 1.0):
        return r, float("nan")
    from scipy.stats import t as _t
    tval = r * np.sqrt(df / (1.0 - r * r))
    return r, float(2.0 * _t.sf(abs(tval), df))


def joint_partial_rho(target, predictor, controls):
    """Joint partial Spearman rho: rank-transform everything, then partial Pearson."""
    tr = rankdata(target)
    pr = rankdata(predictor)
    cr = [rankdata(c) for c in controls]
    return joint_partial_r(tr, pr, cr)


def correlate(scores_by_model: dict[str, dict[str, float]], benchmarks: dict) -> dict:
    """Returns {bench_key: {task: {simple_r, partial_r, ..., n}}}.

    Computes correlations against each of the four creative-writing benchmarks
    (arena_cw, eq_bench_cw, mazur_cw_v2, hivemind_diversity). Joint partial
    controls for Arena Overall and MMLU-Pro simultaneously.
    """
    tasks = ["dat", "cdat", "pace"]
    bench_keys = ["arena_cw", "eq_bench_cw", "mazur_cw_v2",
                  "hivemind_diversity", "noveltybench_utility",
                  "arc_agi_v2"]
    result: dict[str, dict] = {bk: {} for bk in bench_keys}

    for bk in bench_keys:
        for task in tasks:
            xs, ys, ao, mp = [], [], [], []
            for mk, row in scores_by_model.items():
                v = row.get(task)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue
                b = benchmarks.get(mk, {})
                if bk not in b or "arena_overall" not in b or "mmlu_pro" not in b:
                    continue
                xs.append(v)
                ys.append(b[bk])
                ao.append(b["arena_overall"])
                mp.append(b["mmlu_pro"])
            if len(xs) < 5:
                result[bk][task] = None
                continue
            xs_a = np.asarray(xs, float)
            ys_a = np.asarray(ys, float)
            ao_a = np.asarray(ao, float)
            mp_a = np.asarray(mp, float)

            simple_r, simple_p = pearsonr(xs_a, ys_a)
            partial_r, partial_p = joint_partial_r(ys_a, xs_a, [ao_a, mp_a])
            simple_rho, simple_rho_p = spearmanr(xs_a, ys_a)
            partial_rho, partial_rho_p = joint_partial_rho(ys_a, xs_a, [ao_a, mp_a])

            result[bk][task] = {
                "simple_r": float(simple_r),
                "simple_p": float(simple_p),
                "partial_r": float(partial_r),
                "partial_p": float(partial_p),
                "simple_rho": float(simple_rho),
                "simple_rho_p": float(simple_rho_p),
                "partial_rho": float(partial_rho),
                "partial_rho_p": float(partial_rho_p),
                "n": len(xs),
            }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    benchmarks = json.loads(BENCH_PATH.read_text())
    model_dirs = sorted(d for d in RUN_DIR.iterdir() if d.is_dir() and d.name != "downstream")
    print(f"{len(model_dirs)} model directories to rescore.")

    # Shared GloVe vocab used for DAT validation across all embedders.
    glove_vocab = GloVeEmbeddings(GLOVE_PATH)

    embedders = [
        ("glove", lambda: GloVeEmbedder()),
        ("fasttext", lambda: FastTextEmbedder()),
        ("sbert", lambda: SBERTEmbedder("all-mpnet-base-v2")),
    ]

    all_scores: dict[str, dict[str, dict[str, float]]] = {}
    # Structure: all_scores[emb_name][model_key][task] = score

    for emb_name, factory in embedders:
        print(f"\n=== Embedder: {emb_name} ===", flush=True)
        emb = factory()
        # Reuse the same GloVeEmbeddings instance for DAT validation
        by_model = {}
        for mdir in model_dirs:
            scores = rescore_model(mdir, emb, glove_vocab)
            by_model[mdir.name] = scores
            nonan = {k: v for k, v in scores.items() if not math.isnan(v)}
            if nonan:
                print(f"  {mdir.name}: {nonan}", flush=True)
        all_scores[emb_name] = by_model

    # Inject gated CDAT scores per embedding from cdat_gated_scores.json
    # (computed by src/dat_eval/scripts/cdat_gate.py, lives outside the
    # output_dir so it survives --overwrite runs of score_evals.py).
    gated_path = RUN_DIR / "cdat_gated_scores.json"
    if gated_path.exists():
        gated = json.loads(gated_path.read_text())
        for emb_name in all_scores:
            for model_key, scores in all_scores[emb_name].items():
                g = gated.get(emb_name, {}).get(model_key, {}).get("cdat")
                scores["cdat"] = g if g is not None else float("nan")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "multi_embed_scores.json").write_text(json.dumps(all_scores, indent=2))
    print(f"\nSaved scores to {OUT_DIR / 'multi_embed_scores.json'}")

    # Correlations
    corr_out = {}
    for emb_name, by_model in all_scores.items():
        corr_out[emb_name] = correlate(by_model, benchmarks)
    (OUT_DIR / "multi_embed_correlations.json").write_text(json.dumps(corr_out, indent=2))

    # Pretty print: a compact joint-partial table per benchmark.
    bench_labels = {
        "arena_cw": "Arena CW",
        "eq_bench_cw": "EQ-Bench CW",
        "mazur_cw_v2": "Mazur V2",
        "hivemind_diversity": "Hivemind diversity",
        "noveltybench_utility": "NoveltyBench Utility",
        "arc_agi_v2": "ARC-AGI v2",
    }
    for bk, bench_label in bench_labels.items():
        print(f"\n=== Joint partial correlations vs {bench_label} (r / rho) ===")
        for emb_name in ("glove", "fasttext", "sbert"):
            tasks = corr_out[emb_name][bk]
            print(f"\n[{emb_name}]")
            for task in ("dat", "cdat", "pace"):
                c = tasks.get(task)
                if c is None:
                    print(f"  {task:25s}: n<5")
                    continue
                print(f"  {task:25s}: r={c['partial_r']:+.3f} (p={c['partial_p']:.4f}), "
                      f"rho={c['partial_rho']:+.3f} (p={c['partial_rho_p']:.4f}), n={c['n']}")


if __name__ == "__main__":
    main()
