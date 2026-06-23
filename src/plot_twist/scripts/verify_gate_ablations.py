"""Verify the V1 realism gate demotes the intervention cells below the human line.

Recomputes, under the gated composite eq-z{S·1[R=5], Coh·1[R=5], Div}:
  - the main-pool z-frame (72 sources),
  - the human best-8 reference line,
  - every reasoning-effort cell and every prompting-strategy cell,
and prints the max cell per condition vs the human line (the cells that USED to beat humans).

Usage:
    PYTHONPATH=. .venv/bin/python src/plot_twist/scripts/verify_gate_ablations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.plot_twist.join import gated_means, mean_pairwise_distance
from src.plot_twist.sets import twist_types, keep_story

ANN = "data/plot_twist/annotations/annotations.json"
REAL = "data/plot_twist/realism/realism_scores.json"
TC = "data/plot_twist/tc/tc.json"
MANIFEST = "configs/plot_twist/pd_manifest.json"
GATED = ["mean_surprise_g", "mean_coherence_g", "div"]


def key_of(model):  # full model name -> story-id key (matches generation rec ids)
    return model.replace("/", "_").replace(".", "-")


def zparams(rows, facets=GATED):
    return {k: (float(np.nanmean([d[k] for d in rows])),
                float(np.nanstd([d[k] for d in rows])) or 1.0) for k in facets}


def overall(d, zs, facets=GATED):
    return float(np.nanmean([(d[k] - zs[k][0]) / zs[k][1] for k in facets]))


def rubric_map(path):
    """story_id -> {surprise, coherence} from a rubric_scores.json."""
    o = json.loads(Path(path).read_text())
    return {s["story_id"]: s for s in o["scores"]}


def cell_gated(story_ids, scores, realism, div):
    """Gated facet dict for a set of stories (div supplied from precomputed cell)."""
    recs = [{"id": sid, "scores": scores[sid]} for sid in story_ids if sid in scores]
    gm = gated_means(recs, realism)
    gm["div"] = div
    return gm


def main():
    ann = json.loads(Path(ANN).read_text())
    main_real = json.loads(Path(REAL).read_text())
    tc = json.loads(Path(TC).read_text())
    div_by_src = {d["source"]: d["div"] for d in tc}
    types = twist_types(MANIFEST)

    # ---- gated main-pool z-frame (72 sources) ----
    groups = {}
    for r in ann:
        if not r.get("reveal") or r["source"] not in div_by_src:
            continue
        if not keep_story(r["id"], types, strong_only=True):
            continue
        groups.setdefault(r["source"], []).append(r)
    pool = []
    for src, rs in groups.items():
        gm = gated_means(rs, main_real)
        gm["div"] = div_by_src[src]
        pool.append(gm)
    zs = zparams(pool)
    print("gated pool z-params:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in zs.items()})

    # ---- human best-8 line (gated) ----
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def num(a, k):
        try:
            return float(a["scores"][k])
        except (TypeError, ValueError, KeyError):
            return None

    hrows = []
    for a in (a for a in ann if a["source"] == "human"):
        s, c, rv = num(a, "surprise"), num(a, "coherence"), main_real.get(a["id"])
        if None in (s, c) or rv is None:
            continue
        hrows.append((s + c + rv, a, s, c, rv))
    hrows.sort(key=lambda x: -x[0])
    top = hrows[:8]
    E = st.encode([a.get("reveal") or "" for _, a, *_ in top], normalize_embeddings=True, show_progress_bar=False)
    g = lambda s, c, rv: (s if rv >= 5 else 0.0, c if rv >= 5 else 0.0)
    Sg = [g(x[2], x[3], x[4])[0] for x in top]
    Cg = [g(x[2], x[3], x[4])[1] for x in top]
    human = {"mean_surprise_g": float(np.mean(Sg)), "mean_coherence_g": float(np.mean(Cg)),
             "div": mean_pairwise_distance(np.array(E))}
    human_z = overall(human, zs)
    print(f"\nHUMAN best-8 gated composite z = {human_z:+.3f}")
    print("=" * 78)

    # ---- effort cells ----
    th_scores = rubric_map("data/plot_twist/thinking/downstream/rubric/rubric_scores.json")
    th_real = json.loads(Path("data/plot_twist/thinking/downstream/realism/realism_scores.json").read_text())
    th_cells = json.loads(Path("data/plot_twist/thinking/downstream/analysis/thinking_cells.json").read_text())
    print("\n[REASONING EFFORT]  cell vs human line")
    for level in ["low", "medium", "high"]:
        vals = []
        for c in th_cells:
            if c["level"] != level:
                continue
            k = key_of(c["model"])
            sids = [sid for sid in th_scores if sid.startswith(k + "__") and f"__r{level}__" in sid]
            gm = cell_gated(sids, th_scores, th_real, c["div"])
            z = overall(gm, zs)
            vals.append((z, c["model"].split("/")[-1], c["n"]))
        vals.sort(reverse=True)
        top3 = "  ".join(f"{m}({n}) z={z:+.2f}{'*ABOVE' if z > human_z else ''}" for z, m, n in vals[:3])
        nab = sum(z > human_z for z, _, _ in vals)
        print(f"  {level:<7} max={vals[0][0]:+.2f} ({vals[0][1]})  | {nab}/{len(vals)} above human |  {top3}")

    # ---- prompting-strategy cells ----
    pr_scores = rubric_map("data/plot_twist/prompt_methods/downstream/rubric/rubric_scores.json")
    pr_real = json.loads(Path("data/plot_twist/prompt_methods/downstream/realism/realism_scores.json").read_text())
    pr_cells = json.loads(Path("data/plot_twist/prompt_methods/downstream/analysis/prompt_cells.json").read_text())
    # baseline stories live in the MAIN run; reuse main scores + realism
    main_scores = {a["id"]: a["scores"] for a in ann}
    print("\n[PROMPTING STRATEGY]  cell vs human line")
    for method in ["baseline", "be_creative", "incontext_regen"]:
        vals = []
        for c in pr_cells:
            if c["method"] != method:
                continue
            k = key_of(c["model"])
            if method == "baseline":
                sids = [sid for sid in main_scores if sid.startswith(k + "__t10__") and "__p" not in sid]
                gm = cell_gated(sids, main_scores, main_real, c["div"])
            else:
                sids = [sid for sid in pr_scores if sid.startswith(k + "__") and f"__p{method}__" in sid]
                gm = cell_gated(sids, pr_scores, pr_real, c["div"])
            z = overall(gm, zs)
            vals.append((z, c["model"].split("/")[-1], c["n"]))
        vals.sort(reverse=True)
        top3 = "  ".join(f"{m}({n}) z={z:+.2f}{'*ABOVE' if z > human_z else ''}" for z, m, n in vals[:3])
        nab = sum(z > human_z for z, _, _ in vals)
        print(f"  {method:<16} max={vals[0][0]:+.2f} ({vals[0][1]})  | {nab}/{len(vals)} above human |  {top3}")


if __name__ == "__main__":
    main()
