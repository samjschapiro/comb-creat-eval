"""Exp 2 capability-control leg: is DSI's prediction of twist quality SPECIFIC?

Exp 2 found DSI correlates ~0.5 with surprise/coherence/overall -- about the same as
with prose quality. That is the signature of a CAPABILITY proxy: DSI rides the weak->
strong model axis, not a twist-specific signal. This script tests specificity two ways,
reusing the dat_eval validity/specificity recipe (semi-partial r(X, Y - Y_hat_g)):

(A) Per-story, controlling prose_quality (the rubric's explicitly twist-INDEPENDENT
    quality measure) as a 1-proxy capability stack g. Full n. If the semi-partial
    r(DSI, dim | prose) collapses toward 0, DSI's twist prediction is non-specific.

(B) Per-model (dat_eval style): aggregate to model means, control an external 2-proxy
    capability stack (z(arena_overall) + z(mmlu_pro)). Only ~19-25 models have coverage,
    so this is EXPLORATORY corroboration, flagged as such.

Validity = raw r(DSI, dim); Specificity = semi-partial r(DSI, dim residualized on g).

Usage:
    python src/plot_twist/scripts/dsi_specificity.py configs/plot_twist/dsi_specificity.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

from src.utils import init_directory, load_config, save_config
from src.plot_twist.join import EQ_FACETS  # noqa: F401  (kept for parity / future use)

TC_DIMS = ("surprise", "coherence", "overall")


def _semipartial(x: np.ndarray, y: np.ndarray, g: np.ndarray) -> float:
    """r(x, y - y_hat_g): residualize y on capability stack g (cols, w/ intercept),
    correlate the residual with x. The dat_eval specificity estimator."""
    if g.ndim == 1:
        g = g.reshape(-1, 1)
    G = np.column_stack([np.ones(len(y)), g])
    beta, *_ = np.linalg.lstsq(G, y, rcond=None)
    resid = y - G @ beta
    if x.std() == 0 or resid.std() == 0:
        return float("nan")
    return float(pearsonr(x, resid)[0])


def _key(source: str) -> str:
    return source.replace("/", "_").replace(".", "-")


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    if "output_dir" not in cfg:
        raise ValueError("FATAL: 'output_dir' required")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    rows = json.loads(Path(cfg["dsi_json"]).read_text())
    # keep complete per-story rows with FINITE values (some short stories -> NaN DSI)
    def _finite(r, k):
        v = r.get(k)
        return v is not None and np.isfinite(v)
    rows = [r for r in rows if all(_finite(r, k) for k in ("dsi", *TC_DIMS, "prose_quality"))]
    print(f"per-story rows with finite DSI + all dims + prose: {len(rows)}")

    dsi = np.array([r["dsi"] for r in rows], float)
    prose = np.array([r["prose_quality"] for r in rows], float)

    # (A) per-story: validity vs specificity controlling prose_quality
    print("\n(A) Per-story  [capability proxy g = prose_quality]")
    print(f"  {'dimension':<12}{'validity r':>12}{'specificity r|prose':>22}{'shrinkage':>12}")
    perstory = {}
    for d in TC_DIMS:
        y = np.array([r[d] for r in rows], float)
        val = float(pearsonr(dsi, y)[0])
        spec = _semipartial(dsi, y, prose)
        print(f"  {d:<12}{val:>+12.3f}{spec:>+22.3f}{val - spec:>+12.3f}")
        perstory[d] = {"validity": val, "specificity_prose": spec, "n": len(rows)}

    # (B) per-model dat_eval-style with external arena+mmlu capability stack
    print("\n(B) Per-model  [capability stack g = z(arena_overall) + z(mmlu_pro)]  (EXPLORATORY)")
    bm = json.loads(Path(cfg["benchmarks_json"]).read_text())
    # aggregate per source
    by_src: dict[str, list] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    msrc, mdsi, mdim = [], [], {d: [] for d in TC_DIMS}
    arena, mmlu = [], []
    for s, rs in by_src.items():
        if s == "human":
            continue
        rec = bm.get(_key(s), {})
        a, m = rec.get("arena_overall"), rec.get("mmlu_pro")
        if a is None or m is None:
            continue
        msrc.append(s)
        mdsi.append(np.mean([x["dsi"] for x in rs]))
        for d in TC_DIMS:
            mdim[d].append(np.mean([x[d] for x in rs]))
        arena.append(float(a)); mmlu.append(float(m))
    n_models = len(msrc)
    permodel = {"n_models": n_models}
    if n_models >= 6:
        arena = np.array(arena); mmlu = np.array(mmlu)
        g = np.column_stack([(arena - arena.mean()) / (arena.std() or 1),
                             (mmlu - mmlu.mean()) / (mmlu.std() or 1)])
        mdsi = np.array(mdsi)
        print(f"  models with arena+mmlu coverage: {n_models}")
        print(f"  {'dimension':<12}{'validity r':>12}{'specificity r|cap':>20}{'shrinkage':>12}")
        for d in TC_DIMS:
            y = np.array(mdim[d])
            val = float(pearsonr(mdsi, y)[0])
            spec = _semipartial(mdsi, y, g)
            print(f"  {d:<12}{val:>+12.3f}{spec:>+20.3f}{val - spec:>+12.3f}")
            permodel[d] = {"validity": val, "specificity_cap": spec, "n_models": n_models}
    else:
        print(f"  too few models with coverage ({n_models}); skipped")

    summary = {"per_story": perstory, "per_model": permodel}
    (out / "dsi_specificity.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved: {out/'dsi_specificity.json'}")
    print("\nInterpretation: if specificity << validity (large shrinkage toward 0), DSI's "
          "apparent twist prediction is mostly a capability artifact, not twist-specific.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
