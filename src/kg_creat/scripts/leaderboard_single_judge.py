"""What would the leaderboard be if a SINGLE judge scored the subjective dimensions instead of the
3-judge panel majority? Re-derives, in memory (files untouched), the blend gate/coherent/scope and the
analogy invention valid/coherent from ONE judge's stored per-judge vote; factuality, surprise, and
originality are unchanged. Prints the resulting per-task % leaderboard and the delta vs the panel.

    .venv/bin/python -m src.kg_creat.scripts.leaderboard_single_judge --judge openai/o3
"""
import argparse
import json
from pathlib import Path

from src.kg_creat.scripts.compute_composite import artifact_dims, TASK_DIMS, _mean, _ok
from src.kg_creat.scripts.score import finalize_sat

SCORES = Path("data/kg_creat/kombine_test30/scores")


def override(recs, judge):
    for r in recs:
        if r.get("mode") == "blending" and r.get("blend_judges"):
            jv = next((j for j in r["blend_judges"] if j.get("model") == judge), None)
            if jv:
                r["semantic_sat"] = bool(jv.get("generic_ok"))   # blend utility gate = generic space
                r["blend_utility"] = bool(jv.get("coherent"))
                r["blend_integration"] = jv.get("scope")
                finalize_sat(r)                                  # sat <- semantic_sat
        elif r.get("mode") == "analogy" and r.get("invention_judges") and "pair_sat" in r:
            jv = next((j for j in r["invention_judges"] if j.get("model") == judge), None)
            if jv:                                               # pair_sat (utility) is factual+structural, unchanged
                r["invention_utility"] = bool(jv.get("coherent"))
                r["invention_integration"] = bool(jv.get("valid"))
    return recs


def per_task_overall(raw_m):
    per = {}
    for task, dims in TASK_DIMS.items():
        vals = [raw_m[task][k] for k in dims if _ok(raw_m[task].get(k))]
        per[task] = 100.0 * _mean(vals) if vals else None
    per["overall"] = _mean([v for v in per.values() if _ok(v)])
    return per


def build(judge):
    out = {}
    for md in sorted(SCORES.iterdir()):
        ps = md / "path_scores.json"
        if not ps.exists():
            continue
        recs = override(json.loads(ps.read_text()), judge)
        out[md.name] = per_task_overall(artifact_dims(recs))
    return out


DISP = {"openai_gpt-5-6-sol": "gpt-5.6-sol", "x-ai_grok-4-6": "grok-4.6", "openai_gpt-5": "gpt-5",
        "anthropic_claude-opus-4-6": "opus-4.6", "x-ai_grok-4-5": "grok-4.5",
        "anthropic_claude-opus-4-5": "opus-4.5", "google_gemini-3-1-pro-preview": "gemini-3.1-pro",
        "anthropic_claude-fable-5": "fable-5", "openai_gpt-5-2": "gpt-5.2",
        "google_gemini-3-7-flash": "gemini-3.7-flash", "anthropic_claude-opus-5": "opus-5",
        "z-ai_glm-4-6": "glm-4.6", "anthropic_claude-sonnet-4-5": "sonnet-4.5",
        "meta-llama_llama-3-3-70b-instruct": "llama-3.3-70b", "openai_gpt-5-mini": "gpt-5-mini",
        "anthropic_claude-sonnet-5": "sonnet-5", "google_gemini-2-5-pro": "gemini-2.5-pro",
        "deepseek_deepseek-r1": "deepseek-r1", "deepseek_deepseek-chat": "deepseek-chat",
        "qwen_qwen3-max": "qwen3-max", "google_gemini-3-flash-preview": "gemini-3-flash"}


def main(judge):
    single = build(judge)
    panel = json.loads((SCORES / "composite.json").read_text())["per_model"]
    rank = sorted(single, key=lambda m: single[m]["overall"] or -1, reverse=True)
    prank = {m: i + 1 for i, m in enumerate(
        sorted(panel, key=lambda m: panel[m]["overall"] or -1, reverse=True))}
    print(f"Leaderboard with SINGLE judge = {judge}  (subjective dims only; vs panel)\n")
    print(f"{'#':>2} {'model':16s} {'overall':>8} {'(panel)':>8} {'Δ':>6}  {'blend':>6} {'(pan)':>6}  panelRank")
    for i, m in enumerate(rank, 1):
        s = single[m]; p = panel[m]["per_task"]; po = panel[m]["overall"]
        d = s["overall"] - po
        arrow = "" if prank[m] == i else f" ({'+' if prank[m]-i>0 else ''}{prank[m]-i})"
        print(f"{i:>2} {DISP.get(m,m):16s} {s['overall']:>8.1f} {po:>8.1f} {d:>+6.1f}  "
              f"{s['blending']:>6.1f} {p['blending']:>6.1f}  #{prank[m]}{arrow}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="openai/o3")
    main(ap.parse_args().judge)
