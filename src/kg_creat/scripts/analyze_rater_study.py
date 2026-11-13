"""Analyse the co-author rater study (see build_rater_study.py).

Reads every rater file in data/kg_creat/rater_study/ratings/*.json plus the hidden key, and reports
for each judged dimension:

  * human-human reliability: Krippendorff's alpha (nominal for yes/no, ordinal for scope, interval
    for 1-5 originality) and mean pairwise percent agreement;
  * panel-human agreement: the LLM panel's verdict against each rater, against the rater majority,
    and the decisive comparison -- is the panel as close to a held-out human as the other humans
    are? For each rater h: agreement(panel, h) vs agreement(majority of the OTHER raters, h);
  * originality validity: Spearman between mean human originality and the benchmark's rho
    (pooled over answers, and within entity pair, the comparison rho is built for).

"Unsure" is treated as missing. Writes data/kg_creat/rater_study/analysis.json.

    .venv/bin/python -m src.kg_creat.scripts.analyze_rater_study
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path("data/kg_creat/rater_study")

# (task, rating field) -> (key field accessor, level of measurement)
BINARY = {
    ("association", "factual"): lambda k: k["sat"],
    ("analogy", "valid_analogy"): lambda k: k["pair_sat"],
    ("analogy", "mapping_used"): lambda k: k["panel"]["valid"],
    ("analogy", "coherent"): lambda k: k["panel"]["coherent"],
    ("blending", "generic_ok"): lambda k: k["panel"]["generic_ok"],
    ("blending", "coherent"): lambda k: k["panel"]["coherent"],
}
ORIG = {
    ("association", "orig"): "originality",
    ("analogy", "orig_analogy"): "originality",
    ("analogy", "orig_invention"): "em_originality",
    ("blending", "orig_generic"): "originality",
    ("blending", "orig_invention"): "em_originality",
}


def human_value(task, field, r):
    """Rater's value for a dimension; None if missing/unsure. Blend scope is derived."""
    if field == "scope":
        s, e = r.get("shared_slot"), r.get("emergent")
        if s in (None, "unsure") or (s is True and e in (None, "unsure")):
            return None
        return 1 if s is False else (3 if e is True else 2)
    v = r.get(field)
    return None if v in (None, "unsure") else v


# ------------------------------------------------------------------ Krippendorff's alpha
def kalpha(units, metric):
    """units: list of lists of values (one list per unit, missing dropped). metric: nominal|ordinal|interval."""
    units = [[v for v in u if v is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return None
    vals = sorted({v for u in units for v in u})
    if len(vals) < 2:
        return None
    idx = {v: i for i, v in enumerate(vals)}
    o = np.zeros((len(vals), len(vals)))                  # coincidence matrix
    for u in units:
        m = len(u)
        for a, b in itertools.permutations(range(m), 2):
            o[idx[u[a]], idx[u[b]]] += 1 / (m - 1)
    n_c = o.sum(1)
    n = n_c.sum()
    if metric == "nominal":
        d = 1 - np.eye(len(vals))
    elif metric == "interval":
        x = np.array(vals, float)
        d = (x[:, None] - x[None, :]) ** 2
    else:                                                   # ordinal
        d = np.zeros((len(vals), len(vals)))
        for i in range(len(vals)):
            for j in range(len(vals)):
                lo, hi = min(i, j), max(i, j)
                d[i, j] = (n_c[lo:hi + 1].sum() - (n_c[lo] + n_c[hi]) / 2) ** 2
    d_o = (o * d).sum() / n
    d_e = (np.outer(n_c, n_c) * d).sum() / (n * (n - 1))
    return None if d_e == 0 else float(1 - d_o / d_e)


def majority(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    c = defaultdict(int)
    for v in vals:
        c[v] += 1
    best = max(c.values())
    tops = [v for v, k in c.items() if k == best]
    return tops[0] if len(tops) == 1 else None             # ties -> undecided


def agree(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    return (sum(x == y for x, y in pairs) / len(pairs), len(pairs)) if pairs else (None, 0)


def main(ratings_dir):
    key = json.loads((ROOT / "key.json").read_text())
    files = sorted(Path(ratings_dir).glob("*.json"))
    raters = {}
    for f in files:
        d = json.loads(f.read_text())
        raters[d["rater"]] = d["ratings"]
    names = sorted(raters)
    print(f"{len(names)} raters: {', '.join(names)}")
    if len(names) < 2:
        print("need >= 2 raters for reliability")
    out = {"raters": names, "binary": {}, "scope": {}, "originality": {}}

    def column(task, field):
        aids = [a for a, k in key.items() if k["task"] == task]
        return aids, [[human_value(task, field, raters[h].get(a, {})) for h in names] for a in aids]

    print("\n== judged dimensions (Unsure = missing)")
    print(f"{'dimension':28s} {'alpha':>6s} {'h-h %':>6s} {'panel-maj %':>11s} "
          f"{'panel-h %':>9s} {'others-h %':>10s} {'n':>4s}")
    dims = [(t, f, "nominal", fn) for (t, f), fn in BINARY.items()] + \
           [("blending", "scope", "ordinal", lambda k: k["panel"]["scope"])]
    for task, field, metric, panel_fn in dims:
        aids, M = column(task, field)
        panel = [panel_fn(key[a]) for a in aids]
        if metric == "nominal":
            panel = [None if p is None else bool(p) for p in panel]
        alpha = kalpha(M, metric)
        hh = [agree([row[i] for row in M], [row[j] for row in M])[0]
              for i, j in itertools.combinations(range(len(names)), 2)]
        hh = [x for x in hh if x is not None]
        maj = [majority(row) for row in M]
        pm, n_pm = agree(panel, maj)
        # held-out comparison: panel vs rater h, and majority-of-others vs rater h
        ph, oh = [], []
        for i in range(len(names)):
            hcol = [row[i] for row in M]
            others = [majority([v for j, v in enumerate(row) if j != i]) for row in M]
            a1, _ = agree(panel, hcol)
            a2, _ = agree(others, hcol)
            if a1 is not None and a2 is not None:
                ph.append(a1)
                oh.append(a2)
        rec = {"alpha": alpha, "human_human_pct": float(np.mean(hh)) if hh else None,
               "panel_vs_majority_pct": pm, "n_majority": n_pm,
               "panel_vs_heldout_human_pct": float(np.mean(ph)) if ph else None,
               "others_vs_heldout_human_pct": float(np.mean(oh)) if oh else None, "n_items": len(aids)}
        (out["scope"] if field == "scope" else out["binary"])[f"{task}.{field}"] = rec
        f_ = lambda x: "  —  " if x is None else f"{100 * x:5.1f}"
        print(f"{task + '.' + field:28s} {('  —  ' if alpha is None else f'{alpha:6.2f}')} {f_(rec['human_human_pct'])} "
              f"{f_(pm):>11s} {f_(rec['panel_vs_heldout_human_pct']):>9s} "
              f"{f_(rec['others_vs_heldout_human_pct']):>10s} {len(aids):4d}")

    print("\n== originality: human 1-5 vs benchmark rho")
    print(f"{'dimension':28s} {'alpha':>6s} {'rho_pooled':>10s} {'p':>7s} {'rho_within':>10s} {'n':>4s}")
    for (task, field), kf in ORIG.items():
        aids, M = column(task, field)
        alpha = kalpha(M, "interval")
        mean_h = [np.mean([v for v in row if v is not None]) if any(v is not None for v in row) else None
                  for row in M]
        xs, ys, grp = [], [], []
        for a, h in zip(aids, mean_h):
            r = key[a].get(kf)
            if h is not None and r is not None:
                xs.append(h)
                ys.append(r)
                grp.append(key[a]["prompt_id"])
        rho, p = spearmanr(xs, ys) if len(xs) >= 4 else (None, None)
        # within-pair: rank-correlate inside each entity pair, then average (the comparison rho makes)
        within = []
        for g in set(grp):
            ix = [i for i, x in enumerate(grp) if x == g]
            if len(ix) >= 3 and len({xs[i] for i in ix}) > 1 and len({ys[i] for i in ix}) > 1:
                within.append(spearmanr([xs[i] for i in ix], [ys[i] for i in ix])[0])
        rec = {"alpha": alpha, "spearman_pooled": None if rho is None else float(rho),
               "p": None if p is None else float(p),
               "spearman_within_pair_mean": float(np.mean(within)) if within else None,
               "n": len(xs), "n_pairs_within": len(within)}
        out["originality"][f"{task}.{field}"] = rec
        g_ = lambda x, w=6, d=2: (" " * (w - 1) + "—") if x is None else f"{x:{w}.{d}f}"
        print(f"{task + '.' + field:28s} {g_(alpha)} {g_(rec['spearman_pooled'], 10)} {g_(rec['p'], 7, 3)} "
              f"{g_(rec['spearman_within_pair_mean'], 10)} {len(xs):4d}")

    (ROOT / "analysis.json").write_text(json.dumps(out, indent=1))
    print(f"\n-> {ROOT / 'analysis.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=str(ROOT / "ratings"))
    main(ap.parse_args().ratings)
