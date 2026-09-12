"""Collect the inventions every judge rated well, as candidates for a showcase figure.

"Unanimously good" is a strict, judge-only bar (no similarity thresholds):
  blend    -- all three panel judges say the generic space is valid, the concept is coherent, and the
              blend reaches scope 3 (double-scope with emergent properties);
  analogy  -- all three panel judges say the projection is valid and the invention coherent, and every
              triple of the path it was projected along was judged factual.
Originality and surprise (R) are attached for reference but play no part in the bar or the ranking.

Candidates are ranked by PLAINNESS, not originality: a showcase should be readable by a layperson,
so each invention's text (name, generic space or projected source, every property) is scored by word
commonness (wordfreq Zipf frequency). Primary key: number of rare words (Zipf < RARE_ZIPF, roughly
"not in an ordinary reader's vocabulary"); tie-break: mean Zipf of the content words, higher first.

Writes analysis/showcase_inventions.json (every qualifying invention, with the tagged structure or
the source->image projection pulled from the response file, plus the plainness scores) and a markdown
listing of the top candidates per task, one per anchor pair and at most two per model, for choosing
by eye.

    .venv/bin/python -m src.kg_creat.scripts.compile_showcase_inventions
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from wordfreq import zipf_frequency

from src.kg_creat.model_names import DISPLAY

RUN = Path("data/kg_creat/kombine_test30")
OUT = RUN / "analysis" / "showcase_inventions.json"
MD = Path("scratch/showcase_inventions/candidates.md")
TOP = 15
MIN_PROJECTION = 2     # analogy candidates must carry at least this many projected properties
RARE_ZIPF = 3.3        # ~1 per 5 million words; "dormancy" 2.5, "ledger" 3.5, "conscience" 4.0
STOP = set("a an the of to in on by for with as at from into via and or is are be its it their his her this that than".split())


def plainness(text: str):
    """(rare-word count, mean Zipf of content words, the rare words) for one invention's text."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", text) if w.lower() not in STOP]
    z = [(w, zipf_frequency(w.lower(), "en")) for w in words]
    rare = [w for w, f in z if f < RARE_ZIPF]
    return len(rare), (sum(f for _, f in z) / len(z) if z else 0.0), rare


def invention_text(x) -> str:
    if x["task"] == "blending":
        return " ".join([x["name"], x["generic_space"] or ""] + [f"{p['relation']} {p['object']}" for p in x["properties"]])
    return " ".join([x["name"], x["projected"] or ""] + [" ".join(p["source"]) + " " + " ".join(p["image"]) for p in x["projection"]])


def _responses(model):
    return {(r["prompt_id"], r["sample_idx"]): r for r in json.loads((RUN / "responses" / model / "responses.json").read_text())}


def collect():
    rows = []
    for sd in sorted((RUN / "scores").iterdir()):
        if not (sd / "path_scores.json").exists():
            continue
        model = sd.name
        resp = _responses(model)
        for r in json.loads((sd / "path_scores.json").read_text()):
            base = dict(model=model, display=DISPLAY.get(model, model), u=r["u_label"], v=r["v_label"],
                        originality=r.get("originality"), surprise=r.get("R"), prompt_id=r["prompt_id"], sample_idx=r["sample_idx"])
            if r["mode"] == "blending":
                js = r.get("blend_judges") or []
                if not (len(js) == 3 and all(j.get("generic_ok") and j.get("coherent") and j.get("scope") == 3 for j in js)):
                    continue
                it = (resp[(r["prompt_id"], r["sample_idx"])].get("items") or [{}])[0]
                rows.append(dict(base, task="blending", name=it.get("concept") or r["triples"][0][0], generic_space=it.get("generic_space"),
                                 properties=[dict(relation=t[1], object=t[2], tag=g) for t, g in zip(r["triples"], it.get("tags") or [])],
                                 emergent_count=r.get("emergent_count", 0)))
            elif r["mode"] == "analogy" and r.get("invention"):
                js = r.get("invention_judges") or []
                if not (len(js) == 3 and all(j.get("valid") and j.get("coherent") for j in js) and r.get("factual") and all(r["factual"])):
                    continue
                it = next((i for i in resp[(r["prompt_id"], r["sample_idx"])]["items"] if i.get("invention") == r["invention"]), {})
                rows.append(dict(base, task="analogy", name=r["invention"], projected=it.get("projected"),
                                 path=[r["triples"][0][0]] + [t[2] for t in r["triples"]],
                                 projection=[dict(source=p["source"], image=p["image"]) for p in it.get("projection", [])]))
    for x in rows:
        n_rare, mean_z, rare = plainness(invention_text(x))
        x.update(rare_words=rare, n_rare=n_rare, mean_zipf=round(mean_z, 3))
    return rows


def shortlist(rows, task):
    """Plainest candidates first (fewest rare words, then most common vocabulary), one per anchor
    pair and at most two per model, for diversity."""
    seen_pair, per_model, out = set(), Counter(), []
    for x in sorted((x for x in rows if x["task"] == task), key=lambda x: (x["n_rare"], -x["mean_zipf"])):
        if task == "analogy" and len(x["projection"]) < MIN_PROJECTION:      # a one-line projection has nothing to show
            continue
        if (x["u"], x["v"]) in seen_pair or per_model[x["model"]] >= 2:
            continue
        seen_pair.add((x["u"], x["v"])); per_model[x["model"]] += 1; out.append(x)
        if len(out) == TOP:
            break
    return out


def md(rows):
    L = ["# Unanimously good inventions -- showcase candidates", "",
         "Bar: every panel judge (Haiku 4.5, GPT-5.4, o3) passes the invention; blends must reach scope 3; analogy paths must be fully factual.",
         f"Ranked by plainness: fewest rare words (Zipf < {RARE_ZIPF}), then most common vocabulary. One entry per anchor pair and at most two per model. Tags: u / v = from one input, uv = fused, em = emergent.", ""]
    for task, op in (("blending", "+"), ("analogy", "::")):
        n = sum(x["task"] == task for x in rows)
        L += [f"## {task}  ({n} qualify of {'1,033' if task == 'blending' else '1,037'}; {len({x['model'] for x in rows if x['task'] == task})} models, {len({(x['u'], x['v']) for x in rows if x['task'] == task})} anchor pairs)", ""]
        for x in shortlist(rows, task):
            rare = f"; rare: {', '.join(x['rare_words'])}" if x["rare_words"] else ""
            L.append(f"**{x['u']} {op} {x['v']} -> \"{x['name']}\"** -- {x['display']} (rare words {x['n_rare']}, mean Zipf {x['mean_zipf']:.2f}{rare}; originality {x['originality']:.2f})")
            if task == "blending":
                L.append(f"- generic space: *{x['generic_space']}*")
                L += [f"- [{p['tag'][:2]}] {p['relation']} {p['object']}" for p in x["properties"]]
            else:
                L.append(f"- projected from \"{x['projected']}\" along {' -> '.join(x['path'])}")
                L += [f"- {' '.join(p['source'])}  =>  {' '.join(p['image'])}" for p in x["projection"]]
            L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=Path, default=MD)
    a = ap.parse_args()
    rows = collect()
    OUT.write_text(json.dumps(rows, indent=1))
    a.md.parent.mkdir(parents=True, exist_ok=True)
    a.md.write_text(md(rows))
    for task in ("blending", "analogy"):
        n = sum(x["task"] == task for x in rows)
        print(f"{task}: {n} unanimously good; top models: {Counter(x['display'] for x in rows if x['task'] == task).most_common(5)}")
    print(f"wrote {OUT} and {a.md}")


if __name__ == "__main__":
    main()
