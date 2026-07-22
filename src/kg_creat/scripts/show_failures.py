"""Paired failure examples: same model, same endpoints, unconstrained success vs constrained failure.

The aggregate numbers say each constraint type costs creativity by a different amount; they cannot
say what the model actually did wrong. This pulls matched pairs -- one model on one endpoint bundle,
its satisfying baseline path beside its constraint-violating path for the same pair -- so the failure
mode is legible rather than inferred.

Only failures on the CONSTRAINT channel are shown. A path that fails because it hallucinated an edge
tells us nothing about the constraint, so mixing those in would misrepresent what the constraint did.

    .venv/bin/python src/kg_creat/scripts/show_failures.py [--markdown] [--per-type N]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

SCORES = "data/kg_creat/scores_regimeA_all"
PASS2 = "data/kg_creat/prompts_regimeA_pass2"
MODES = ["exclusion", "inclusion", "inclusion_rare", "ordering", "categorical"]
PRETTY = {"anthropic_claude-sonnet-4-6": "Claude Sonnet 4.6", "anthropic_claude-haiku-4-5": "Claude Haiku 4.5",
          "openai_gpt-4-1-mini": "GPT-4.1-mini", "openai_gpt-4o-mini": "GPT-4o-mini",
          "google_gemini-2-5-flash": "Gemini 2.5 Flash", "google_gemini-2-5-flash-lite": "Gemini 2.5 Flash-Lite",
          "meta-llama_llama-3-3-70b-instruct": "Llama 3.3 70B", "meta-llama_llama-3-1-8b-instruct": "Llama 3.1 8B"}


def load():
    specs = {s["prompt_id"]: s for s in json.loads((Path(PASS2) / "prompts.json").read_text())}
    classes = {c["name"]: {m.strip().lower() for m in c["members"]}
               for c in json.loads((Path(PASS2) / "classes_targets.json").read_text())["classes"]}
    per_model = {}
    for d in sorted(Path(SCORES).iterdir()):
        p = d / "path_scores.json"
        if p.exists():
            per_model[d.name] = json.loads(p.read_text())
    return specs, classes, per_model


def rels(rec):
    return [str(t[1]).strip().lower() for t in rec["triples"]]


def render(rec):
    """A path as 'A --rel--> B --rel--> C'."""
    if not rec["triples"]:
        return "(no path)"
    out = str(rec["triples"][0][0])
    for h, r, t in rec["triples"]:
        out += f"  --[{r}]->  {t}"
    return out


def diagnose(rec, spec, classes):
    """Name the specific violation, so the example is evidence rather than an anecdote."""
    c = spec.get("constraint") or {}
    t = c.get("type")
    rs = rels(rec)
    if t == "exclusion":
        members = classes.get(c["class_name"], set())
        hits = [r for r in rs if r in members]
        return (f"used the forbidden {c['class_name']} class"
                + (f" via {', '.join(repr(h) for h in dict.fromkeys(hits))}" if hits else
                   " (judge found a member the exact-match list misses)"))
    if t in ("inclusion", "inclusion_rare"):
        return f"never used any {c['class_name']} relation (emitted: {', '.join(repr(r) for r in rs)})"
    if t == "ordering":
        B, A = classes.get(c["before_name"], set()), classes.get(c["after_name"], set())
        iB = [i for i, r in enumerate(rs) if r in B]
        iA = [i for i, r in enumerate(rs) if r in A]
        if not iB and not iA:
            return f"path contains neither a {c['before_name']} nor a {c['after_name']} relation"
        if not iB:
            return f"has {c['after_name']} but never a {c['before_name']} relation to precede it"
        if not iA:
            return f"has {c['before_name']} but never a {c['after_name']} relation to follow it"
        return (f"order inverted: {c['after_name']} at hop {min(iA)+1} precedes "
                f"{c['before_name']} at hop {min(iB)+1}")
    if t == "categorical":
        mids = [str(x[2]) for x in rec["triples"][:-1]]
        return (f"no intermediate entity is a kind of '{c['type_label']}' "
                f"(intermediates: {', '.join(repr(m) for m in mids) or 'none'})")
    return "?"


def constraint_text(spec):
    c = spec.get("constraint") or {}
    t = c.get("type")
    if t == "exclusion":
        return f"avoid ALL {c['class_name']} relations (e.g. {', '.join(c['exemplars'][:3])})"
    if t in ("inclusion", "inclusion_rare"):
        return f"use at least one {c['class_name']} relation (e.g. {', '.join(c['exemplars'][:3])})"
    if t == "ordering":
        return (f"a {c['before_name']} relation must come BEFORE any {c['after_name']} relation")
    if t == "categorical":
        return f"pass through an intermediate entity that is a kind of '{c['type_label']}'"
    return "?"


def find(specs, classes, per_model, per_type=2):
    """Matched pairs, preferring short readable paths and a spread of models."""
    out = {}
    for mode in MODES:
        cands = []
        for model, recs in per_model.items():
            by_bundle = {}
            for r in recs:
                by_bundle.setdefault(r["bundle_id"], {}).setdefault(r["mode"], []).append(r)
            for bundle, cells in by_bundle.items():
                good = [x for x in cells.get("baseline", []) if x.get("sat") is True]
                bad = [x for x in cells.get(mode, []) if x.get("channel") == "constraint"]
                if not good or not bad:
                    continue
                g = min(good, key=lambda x: len(x["triples"]))
                # For exclusion, prefer a violation the member list can actually name: an example
                # whose diagnosis is "the judge saw something we can't point to" is not evidence.
                spec = specs.get((bad[0])["prompt_id"], {})
                c = spec.get("constraint") or {}
                if c.get("type") == "exclusion":
                    members = classes.get(c.get("class_name", ""), set())
                    named = [x for x in bad if any(r in members for r in rels(x))]
                    bad = named or bad
                    if not named:
                        continue
                b = min(bad, key=lambda x: len(x["triples"]))
                if not (2 <= len(b["triples"]) <= 3 and 2 <= len(g["triples"]) <= 3):
                    continue
                cands.append((model, bundle, g, b))
        picked, seen_models, seen_bundles = [], set(), set()
        for model, bundle, g, b in cands:                      # spread across models/bundles
            if model in seen_models or bundle in seen_bundles:
                continue
            picked.append((model, bundle, g, b))
            seen_models.add(model)
            seen_bundles.add(bundle)
            if len(picked) == per_type:
                break
        out[mode] = picked
    return out


def main(per_type, markdown):
    specs, classes, per_model = load()
    found = find(specs, classes, per_model, per_type)
    for mode in MODES:
        title = mode.replace("_", " ").upper()
        print(f"\n{'='*96}\n{title}\n{'='*96}" if not markdown else f"\n### {title}\n")
        for model, bundle, g, b in found[mode]:
            spec = specs[b["prompt_id"]]
            head = (f"{PRETTY.get(model, model)}   ·   {g['u_label']} → {g['v_label']}   ·   {bundle}")
            if markdown:
                print(f"**{head}**\n")
                print(f"- **Constraint:** {constraint_text(spec)}")
                print(f"- **Unconstrained (satisfied):** `{render(g)}`")
                print(f"- **Constrained (failed):** `{render(b)}`")
                print(f"- **Why it failed:** {diagnose(b, spec, classes)}\n")
            else:
                print(f"\n{head}")
                print(f"  constraint : {constraint_text(spec)}")
                print(f"  baseline ✓ : {render(g)}")
                print(f"  constrained ✗: {render(b)}")
                print(f"  violation  : {diagnose(b, spec, classes)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=2)
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()
    main(a.per_type, a.markdown)
