"""Rescore DRAT pilot responses under multiple embeddings and benchmarks.

Loads existing raw_results.json from pilot_v1 and pilot_v2, recomputes
DRAT scores under GloVe, FastText, and SBERT, and correlates against
the full benchmark stack (LIB, Arena CW, EQ-Bench CW, Mazur, Hivemind,
NoveltyBench).

Uses no API calls — operates entirely on the cached responses.
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.dat_eval.cdat import SBERTEmbeddings, validate_words_sbert
from src.dat_eval.dat import GloVeEmbeddings
from src.dat_eval.pace import FastTextEmbeddings


def ormap(m: str) -> str:
    return m.replace("/", "_").replace(".", "-")


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    return num / (dx2 * dy2) ** 0.5 if dx2 * dy2 > 0 else float("nan")


def semi_partial(xs, ys, gs):
    """Semi-partial r(X, Y - Y_hat | g): OLS Y on g, residuals correlate with X."""
    n = len(xs)
    G = np.array([[1.0] + [g[i] for g in gs] for i in range(n)])
    Y = np.array(ys)
    beta, *_ = np.linalg.lstsq(G, Y, rcond=None)
    Y_hat = G @ beta
    resid = Y - Y_hat
    return pearson(xs, resid.tolist())


# --- Embedding adapters: each returns a vector for a given string. ---

def glove_vec(emb: GloVeEmbeddings, s: str) -> np.ndarray | None:
    """Mean of token vectors for multi-word strings; None if no token in vocab."""
    emb._load()
    parts = s.lower().split()
    vecs = [emb._vectors[t] for t in parts if t in emb._vectors]
    return np.mean(vecs, axis=0) if vecs else None


def fasttext_vec(emb: FastTextEmbeddings, s: str) -> np.ndarray | None:
    emb._load()
    parts = s.lower().split()
    vecs = [emb._vectors[t] for t in parts if t in emb._vectors]
    return np.mean(vecs, axis=0) if vecs else None


def sbert_vec(emb: SBERTEmbeddings, s: str) -> np.ndarray | None:
    return emb.encode_batch([s])[0]


# --- Scoring under a single embedding. ---

def score_response(words: list[str], anchor_a: str, anchor_b: str,
                   vec_fn, tau_pool: list[str], n_min: int = 5):
    """Recompute DRAT for a response under a given embedding (via vec_fn).

    Returns dict with mean_util, novelty, gated_n5, gated_n3, composite, tau, n_valid.
    """
    valid = validate_words_sbert(words)
    if len(valid) < 2:
        return None
    a_vec = vec_fn(anchor_a)
    b_vec = vec_fn(anchor_b)
    if a_vec is None or b_vec is None:
        return None

    # Calibrate tau for this pair using the noun pool (90th percentile)
    pool_utils = []
    for n in tau_pool:
        v = vec_fn(n)
        if v is None:
            continue
        cos_a = float(1.0 - cosine_distance(v, a_vec))
        cos_b = float(1.0 - cosine_distance(v, b_vec))
        pool_utils.append(max(cos_a, cos_b))
    if len(pool_utils) < 10:
        return None
    tau = float(np.percentile(pool_utils, 90.0))

    # Encode the words
    word_vecs = []
    word_labels = []
    for w in valid[:10]:
        v = vec_fn(w)
        if v is not None:
            word_vecs.append(v)
            word_labels.append(w)
    if len(word_vecs) < 2:
        return None

    # Per-word utilities
    utilities = []
    for v in word_vecs:
        cos_a = float(1.0 - cosine_distance(v, a_vec))
        cos_b = float(1.0 - cosine_distance(v, b_vec))
        utilities.append(max(cos_a, cos_b))

    # Pairwise distances over ALL words (for multiplicative)
    n_words = len(word_vecs)
    all_dists = [float(cosine_distance(word_vecs[i], word_vecs[j]))
                 for i in range(n_words) for j in range(i + 1, n_words)]
    all_novelty = float(np.mean(all_dists)) if all_dists else 0.0
    mean_util = float(np.mean(utilities))
    composite = mean_util * all_novelty * 100.0

    # Gated scores
    def gated(n_min_val):
        survivor_idx = [i for i, u in enumerate(utilities) if u > tau]
        if len(survivor_idx) < n_min_val:
            return 0.0
        sv = [word_vecs[i] for i in survivor_idx]
        k = len(sv)
        d = [float(cosine_distance(sv[i], sv[j]))
             for i in range(k) for j in range(i + 1, k)]
        return 100.0 * float(np.mean(d))

    return {
        "tau": tau,
        "n_valid_in_emb": len(word_vecs),
        "mean_util": mean_util,
        "all_novelty": all_novelty,
        "gated_n5": gated(5),
        "gated_n3": gated(3),
        "composite": composite,
    }


def main():
    # Config
    glove_path = "resources/glove.840B.300d.txt"
    fasttext_path = "resources/crawl-300d-2M.vec"

    # Load raw responses
    with open("data/new_tests/drat/pilot_v1/raw_results.json") as f:
        raw = json.load(f)
    with open("data/new_tests/drat/pilot_v2/raw_results.json") as f:
        raw += json.load(f)
    print(f"Total responses: {len(raw)}")

    # Load CDAT cue pool for tau calibration
    from src.dat_eval.cdat import DEFAULT_CUES
    tau_pool = list(DEFAULT_CUES)

    # Load benchmarks
    with open("configs/comb_eval/benchmarks.json") as f:
        bench = json.load(f)

    # Load embeddings
    print("Loading embeddings (this takes a couple minutes for GloVe)...")
    sbert = SBERTEmbeddings()
    glove = GloVeEmbeddings(glove_path)
    fasttext = FastTextEmbeddings(fasttext_path)

    embedders = [
        ("sbert", lambda s: sbert_vec(sbert, s)),
        ("glove", lambda s: glove_vec(glove, s)),
        ("fasttext", lambda s: fasttext_vec(fasttext, s)),
    ]

    # Rescore everything under each embedding
    per_model: dict[str, dict[str, dict[str, list[float]]]] = {}
    n_skip: dict[str, int] = {name: 0 for name, _ in embedders}
    for r in raw:
        if "error" in r:
            continue
        model = r["model"]
        words = r.get("extracted_words", [])
        a, b = r["anchor_a"], r["anchor_b"]
        for name, vec_fn in embedders:
            scores = score_response(words, a, b, vec_fn, tau_pool, n_min=5)
            if scores is None:
                n_skip[name] += 1
                continue
            d = per_model.setdefault(model, {}).setdefault(
                name, {"gated_n5": [], "gated_n3": [], "composite": []}
            )
            d["gated_n5"].append(scores["gated_n5"])
            d["gated_n3"].append(scores["gated_n3"])
            d["composite"].append(scores["composite"])
    for name, n in n_skip.items():
        print(f"  {name}: skipped {n} responses (insufficient embedding coverage)")

    # Aggregate per-model means and assemble benchmark vectors
    benchmarks_to_test = [
        ("liveideabench", "LIB"),
        ("arena_cw", "Arena CW"),
        ("eq_bench_cw", "EQ-Bench CW"),
        ("mazur_cw_v2", "Mazur CW"),
        ("hivemind_diversity", "Hivemind div"),
        ("noveltybench_utility", "NoveltyBench U"),
    ]

    capability_keys = ["arena_overall", "mmlu_pro"]

    # For each variant × embedding, build (DRAT scores, benchmark scores) pairs
    # and compute r and r|g for each benchmark.
    print("\n" + "=" * 96)
    print("Per-model DRAT means under each embedding (gated n_min=5)")
    print("=" * 96)
    print(f"{'model':<48}  {'sbert':>6}  {'glove':>6}  {'fasttext':>8}  {'LIB':>5}")
    for m in sorted(per_model):
        row = [m.split("/")[-1][:46]]
        for name, _ in embedders:
            vals = per_model[m].get(name, {}).get("gated_n5", [])
            row.append(f"{statistics.mean(vals):>6.2f}" if vals else "  N/A")
        b = bench.get(ormap(m), {})
        lib = b.get("liveideabench", "N/A")
        print(f"{row[0]:<48}  {row[1]:>6}  {row[2]:>6}  {row[3]:>8}  {str(lib):>5}")

    # Correlations: for each variant × embedding × benchmark, compute r and r|g
    def get_models_with(*keys):
        out = []
        for m in per_model:
            b = bench.get(ormap(m), {})
            if all(k in b for k in keys + tuple(capability_keys)):
                out.append(m)
        return out

    variants = ["gated_n5", "gated_n3", "composite"]
    print("\n" + "=" * 96)
    print("Pearson r and semi-partial r|g (gated n_min=5)")
    print("=" * 96)

    for variant in variants:
        print(f"\n--- variant: {variant} ---")
        print(f"{'benchmark':<18}  ", end="")
        for name, _ in embedders:
            print(f"{name+' r':>10}  {name+' r|g':>10}  ", end="")
        print(f"{'n':>3}")
        for bk, blabel in benchmarks_to_test:
            models_with = get_models_with(bk)
            n = len(models_with)
            if n < 4:
                print(f"{blabel:<18}  (n={n}, skipping)")
                continue
            arenas = [bench[ormap(m)]["arena_overall"] for m in models_with]
            mmlus = [bench[ormap(m)]["mmlu_pro"] for m in models_with]
            ys = [bench[ormap(m)][bk] for m in models_with]
            print(f"{blabel:<18}  ", end="")
            for name, _ in embedders:
                xs = [statistics.mean(per_model[m][name][variant]) for m in models_with]
                r = pearson(xs, ys)
                r_sp = semi_partial(xs, ys, [arenas, mmlus])
                print(f"{r:>+10.3f}  {r_sp:>+10.3f}  ", end="")
            print(f"{n:>3}")

    # Save all per-model scores
    out = {
        "per_model": {
            m: {
                name: {
                    v: statistics.mean(per_model[m][name][v])
                    for v in variants if per_model[m].get(name, {}).get(v)
                } for name in [n for n, _ in embedders]
            }
            for m in per_model
        },
        "embeddings": [n for n, _ in embedders],
        "skip_counts": n_skip,
    }
    out_path = Path("data/new_tests/drat/multi_embed_scores.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
