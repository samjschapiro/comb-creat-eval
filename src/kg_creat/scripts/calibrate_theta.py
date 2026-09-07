"""Choose theta by a stated false-positive rate against a cross-item null.

theta (the cosine at which two "relation object" texts count as the same property) has no derivation
-- 0.58 is a number that was typed. This picks it the way a detection threshold should be picked.

NULL. Two inventions answering DIFFERENT anchor pairs cannot share an item-specific property. Match
them with the same greedy one-to-one rule and the resulting cosines are the null: matches that arise
from generic phrasing rather than from convergence on the item. theta is then set so that the null
match rate is at most some alpha, and the criterion inherits a stated meaning -- "at alpha = 1%,
fewer than one in a hundred property pairs from unrelated inventions would clear the bar."

VALIDATION. The same-name label is held out of the criterion, so it can be used to report what each
candidate theta recovers (TPR) at its alpha, without ever being fitted to.

    .venv_mlx/bin/python -m src.kg_creat.scripts.calibrate_theta
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import src.kg_creat.scripts.analyze_inventive_multiples as M
from src.kg_creat.embed import get_embedder

ALPHAS = [0.10, 0.05, 0.02, 0.01, 0.005, 0.001]
BAND = [round(x, 3) for x in np.arange(0.495, 0.5851, 0.01)]  # the band under consideration
N_NULL = 60000


def load():
    d = np.load(M.NPZ, allow_pickle=True)
    names, tk, us, vs, mo = d["names"], d["tasks"], d["u"], d["v"], d["models"]
    _, _, struct, _ = M.load_records()
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    slot_vec, SLOTS = {}, {}
    for i in range(len(names)):
        st = M.slot_texts(str(tk[i]), struct.get((str(tk[i]), str(us[i]), str(vs[i]), str(mo[i])), []))
        for t, _ in st:
            if t not in slot_vec:
                slot_vec[t] = un(np.asarray(embed(t), float))
        SLOTS[i] = st
    SMAT = {i: (np.vstack([slot_vec[t] for t, _ in SLOTS[i]]) if SLOTS[i] else np.zeros((0, 384)))
            for i in range(len(names))}
    return names, tk, us, vs, mo, SLOTS, SMAT


def best_matches(SMAT, a, b):
    """Greedy one-to-one matched cosines, unthresholded, descending."""
    A, B = SMAT[a], SMAT[b]
    if not len(A) or not len(B):
        return []
    Mx = A @ B.T
    used, out = set(), []
    for ai in np.argsort(-Mx.max(axis=1)):
        cand = [(Mx[ai, j], j) for j in range(Mx.shape[1]) if j not in used]
        if not cand:
            break
        s, bj = max(cand)
        out.append(float(s)); used.add(bj)
    return sorted(out, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/kg_creat/kombine_test30/analysis/theta_calibration.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    names, tk, us, vs, mo, SLOTS, SMAT = load()
    rng = np.random.default_rng(a.seed)

    groups = defaultdict(list)
    for i in range(len(names)):
        groups[(str(tk[i]), str(us[i]), str(vs[i]))].append(i)
    item_of = {i: (str(us[i]), str(vs[i])) for i in range(len(names))}

    # --- null: same task, DIFFERENT anchor pair ------------------------------------------------
    by_task = defaultdict(list)
    for i in range(len(names)):
        by_task[str(tk[i])].append(i)
    null = []
    for _ in range(N_NULL):
        t = "blending" if rng.random() < 0.5 else "analogy"
        pool = by_task[t]
        i, j = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
        if i == j or item_of[i] == item_of[j]:
            continue
        null.extend(best_matches(SMAT, i, j))
    null = np.asarray(null)

    # --- real co-response pairs ----------------------------------------------------------------
    meta = [(t, x, y) for (t, _, _), idx in groups.items() for x, y in itertools.combinations(idx, 2)]
    task = np.array([t for t, _, _ in meta])
    same_prov = np.array([M._provider(mo[x]) == M._provider(mo[y]) for _, x, y in meta])
    same_name = np.array([bool(M._nn(names[x]) == M._nn(names[y]) and M._nn(names[x]))
                          for _, x, y in meta])
    matches = [best_matches(SMAT, x, y) for _, x, y in meta]

    print(f"null: {len(null):,} matched property pairs from inventions on DIFFERENT anchors")
    print(f"real: {len(meta):,} co-response pairs\n")
    print(f"{'alpha':>7}{'theta':>8}{'tau=2 rate':>12}{'blend/an':>10}{'same/cross':>12}"
          f"{'same-name TPR':>15}{'lift':>7}")
    print("-" * 71)
    rows = []
    for al in ALPHAS:
        th = float(np.quantile(null, 1 - al))
        sh = np.array([sum(1 for s in m if s >= th) for m in matches])
        hit = sh >= 2
        r_bl, r_an = hit[task == "blending"].mean(), hit[task == "analogy"].mean()
        r_s, r_c = hit[same_prov].mean(), hit[~same_prov].mean()
        tpr, fpr = hit[same_name].mean(), hit[~same_name].mean()
        rows.append({"alpha": al, "theta": th, "rate_pct": 100 * hit.mean(),
                     "blend_over_analogy": (r_bl / r_an) if r_an else None,
                     "same_over_cross": (r_s / r_c) if r_c else None,
                     "same_name_tpr": tpr, "other_fpr": fpr})
        ba = f"{r_bl / r_an:.1f}x" if r_an else "--"
        print(f"{al:>7.3f}{th:>8.3f}{100 * hit.mean():>11.2f}%{ba:>10}"
              f"{(r_s / r_c if r_c else float('nan')):>11.1f}x{100 * tpr:>14.1f}%"
              f"{(tpr / fpr if fpr else float('nan')):>6.1f}x")

    # --- fine sweep of a chosen band, reported by theta rather than by alpha ------------------
    print(f"\nBAND SWEEP  {BAND[0]:.3f} -- {BAND[-1]:.3f}")
    print(f"{'theta':>7}{'alpha':>9}{'tau=2 rate':>12}{'blend/an':>10}{'same/cross':>12}"
          f"{'same-name TPR':>15}{'lift':>7}")
    print("-" * 72)
    band_rows = []
    for th in BAND:
        al = float((null >= th).mean())
        sh = np.array([sum(1 for s in m if s >= th) for m in matches])
        hit = sh >= 2
        r_bl, r_an = hit[task == "blending"].mean(), hit[task == "analogy"].mean()
        r_s, r_c = hit[same_prov].mean(), hit[~same_prov].mean()
        tpr, fpr = hit[same_name].mean(), hit[~same_name].mean()
        band_rows.append({"theta": th, "alpha": al, "rate_pct": 100 * hit.mean(),
                          "blend_over_analogy": (r_bl / r_an) if r_an else None,
                          "same_over_cross": (r_s / r_c) if r_c else None,
                          "same_name_tpr": tpr})
        ba = f"{r_bl / r_an:.1f}x" if r_an else "--"
        print(f"{th:>7.3f}{100 * al:>8.2f}%{100 * hit.mean():>11.2f}%{ba:>10}"
              f"{(r_s / r_c if r_c else float('nan')):>11.1f}x{100 * tpr:>14.1f}%"
              f"{(tpr / fpr if fpr else float('nan')):>6.1f}x")

    cur = 0.58
    sh = np.array([sum(1 for s in m if s >= cur) for m in matches])
    al_cur = float((null >= cur).mean())
    print(f"\ncurrent theta = {cur}: implied alpha = {100 * al_cur:.2f}% "
          f"(that share of unrelated-invention property pairs clears it)")
    Path(a.out).write_text(json.dumps(
        {"n_null": int(len(null)), "n_pairs": len(meta), "alphas": rows, "band": band_rows,
         "current_theta": cur, "current_alpha": al_cur}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
