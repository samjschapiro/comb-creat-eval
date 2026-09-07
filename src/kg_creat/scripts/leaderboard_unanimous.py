"""Leaderboard restricted to CONFIDENTLY-judged artifacts: analogy/blend artifacts are kept only where
the 3-judge panel was UNANIMOUS on the subjective verdicts (blend: generic_ok, coherent, scope;
analogy invention: valid, coherent). Association is factuality-only (single judge), so unaffected.
Files untouched; computed in memory. Prints the filtered leaderboard, per-model retained-n, and the
delta vs the full-panel leaderboard.

    .venv/bin/python -m src.kg_creat.scripts.leaderboard_unanimous
"""
import json
from pathlib import Path

from src.kg_creat.scripts.compute_composite import artifact_dims, TASK_DIMS, _mean, _ok
from src.kg_creat.scripts.leaderboard_single_judge import per_task_overall, DISP

SCORES = Path("data/kg_creat/kombine_test30/scores")


def _unan(judges, keys):
    for k in keys:
        vs = [j.get(k) for j in judges if j.get(k) is not None]
        if len(vs) < 3:
            return False
        norm = [int(v) if k == "scope" else int(bool(v)) for v in vs]
        if len(set(norm)) != 1:
            return False
    return True


def filter_unanimous(recs):
    out = []
    for r in recs:
        if r.get("mode") == "blending" and r.get("triples"):
            if r.get("blend_judges") and _unan(r["blend_judges"], ["generic_ok", "coherent", "scope"]):
                out.append(r)
        elif r.get("mode") == "analogy" and "pair_sat" in r:
            if r.get("invention_judges") and _unan(r["invention_judges"], ["valid", "coherent"]):
                out.append(r)
        elif r.get("mode") == "analogy":
            out.append(r)                # non-head analogy path records (kept; harmless)
        else:
            out.append(r)                # association + anything else
    return out


def main():
    panel = json.loads((SCORES / "composite.json").read_text())["per_model"]
    prank = {m: i + 1 for i, m in enumerate(
        sorted(panel, key=lambda m: panel[m]["overall"] or -1, reverse=True))}
    rows = {}
    for md in sorted(SCORES.iterdir()):
        ps = md / "path_scores.json"
        if not ps.exists():
            continue
        recs = json.loads(ps.read_text())
        dims = artifact_dims(filter_unanimous(recs))
        rows[md.name] = (per_task_overall(dims), dims)
    rank = sorted(rows, key=lambda m: rows[m][0]["overall"] or -1, reverse=True)
    print("Leaderboard on PANEL-UNANIMOUS analogy/blend artifacts (association unchanged)\n")
    print(f"{'#':>2} {'model':16s} {'overall':>8} {'(full)':>7} {'Δ':>6} "
          f"{'ana_n':>6} {'bl_n':>5}  panelRank")
    for i, m in enumerate(rank, 1):
        pt, dims = rows[m]
        full = panel[m]["overall"]
        an = dims["analogy"]["_n_artifacts"]; bn = dims["blending"]["_n_artifacts"]
        mv = prank[m] - i
        tag = "" if mv == 0 else f" ({'+' if mv > 0 else ''}{mv})"
        print(f"{i:>2} {DISP.get(m, m):16s} {pt['overall']:>8.1f} {full:>7.1f} "
              f"{pt['overall']-full:>+6.1f} {an:>6} {bn:>5}  #{prank[m]}{tag}")


if __name__ == "__main__":
    main()
