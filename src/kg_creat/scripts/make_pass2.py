"""Build Pass-2 (constrained) prompt specs from Pass-1 baseline responses.

Derives relation CLASSES from the baseline corpus (what models actually emitted), names them,
picks per-bundle constraint targets *against each bundle's own default behaviour*, and writes a
prompt set whose endpoints are IDENTICAL to Pass 1 — only the constraint changes, so within-bundle
deltas are causal in constraint type.

Cells: exclusion · inclusion (common class) · inclusion_rare (niche class) · categorical.
(Ordering was piloted and dropped — see assessment.md §7c.)

    .venv_mlx/bin/python src/kg_creat/scripts/make_pass2.py
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from collections import Counter, defaultdict  # noqa: E402

from src.utils import init_directory  # noqa: E402
from src.kg_creat.embed import get_embedder  # noqa: E402
from src.kg_creat.graph import KnowledgeGraph  # noqa: E402
from src.kg_creat.relation_classes import collect, derive_classes, derive_targets, name_classes  # noqa: E402


# Over-generic types make the categorical constraint trivially satisfiable ("pass through a human /
# country") -- almost every path already does. Excluding them forces a SPECIFIC, biting type target
# (island country, music genre, academy of sciences, ...).
_GENERIC_TYPES = {
    "human", "person", "country", "sovereign state", "state", "nation", "republic",
    "city", "big city", "human settlement", "member state of the united nations",
    "country bordering the mediterranean sea",
}


def derive_categorical(pass1_dir, gc):
    """Per-bundle categorical target from the interior entities models actually used in baseline.

    On arbitrary endpoints there are no pre-enumerated routes, so (unlike the old matched-bundle
    version) we type the interior entities the models themselves produced, via G_c's types, drop
    over-generic types (`_GENERIC_TYPES`), and pick the MOST CONTRASTIVE remaining type -- one
    present in ~half the typed interiors -- so the constraint bites by construction (some baseline
    paths already pass through it, some do not). Free: G_c-local typing only, no Wikidata calls.
    """
    lab2node = {gc.label(n).strip().lower(): n for n in gc.nodes()}
    interior = defaultdict(list)
    for md in Path(pass1_dir).iterdir():
        p = md / "responses.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r.get("mode") != "baseline":
                continue
            for path in r["paths"]:
                if len(path) < 2:
                    continue
                for tr in path[:-1]:            # interior = every tail except the final target
                    interior[r["bundle_id"]].append(str(tr[2]).strip().lower())

    out = {}
    for bid, ents in interior.items():
        tcount, seen = Counter(), 0
        for e in set(ents):
            node = lab2node.get(e)
            if not node:
                continue
            ts = [t for t in gc.types(node) if gc.type_label(t).strip().lower() not in _GENERIC_TYPES]
            if ts:
                seen += 1
            tcount.update(ts)
        cands = [(t, c) for t, c in tcount.items() if 0 < c < seen] if seen >= 2 else []
        if not cands:
            continue
        t, _ = min(cands, key=lambda tc: abs(tc[1] - seen / 2))   # most contrastive => most biting
        out[bid] = {"type": "categorical", "entity_type": t, "type_label": gc.type_label(t)}
    return out


async def main(pass1_dir, bundles_dir, out_dir, gc_dir, k=8, top_n=150, min_share=0.08,
               name_model="openai/gpt-oss-120b", overwrite=False):
    from src.dat_eval.llm import get_async_client

    embed = get_embedder()
    counts, per_bundle, seqs = collect(pass1_dir)
    classes = derive_classes(counts, embed, k=k, top_n=top_n)
    classes = await name_classes(classes, get_async_client(), name_model)
    targets = derive_targets(per_bundle, seqs, classes, min_share=min_share)
    by_id = {c["id"]: c for c in classes}
    gc = KnowledgeGraph.load(Path(gc_dir) / "gc.json")
    categorical = derive_categorical(pass1_dir, gc)

    base = {s["bundle_id"]: s for s in json.loads((Path(bundles_dir) / "prompts.json").read_text())
            if s["mode"] == "baseline"}

    def spec(b, mode, constraint):
        return {**{k: b[k] for k in ("bundle_id", "regime", "u", "v", "u_label", "v_label", "h", "k")},
                "prompt_id": f"{b['bundle_id']}.{mode}", "mode": mode, "constraint": constraint}

    out = []
    for bid, t in targets.items():
        b = base.get(bid)
        if b is None:
            continue
        excl, incl, rare, order = t["exclusion"], t["inclusion"], t["inclusion_rare"], t["ordering"]
        if excl is not None:
            c = by_id[excl]
            out.append(spec(b, "exclusion", {"type": "exclusion", "class_id": c["id"],
                                             "class_name": c["name"], "exemplars": c["members"][:6]}))
        if incl is not None:
            c = by_id[incl]
            out.append(spec(b, "inclusion", {"type": "inclusion", "class_id": c["id"],
                                             "class_name": c["name"], "exemplars": c["members"][:6]}))
        if rare is not None:
            c = by_id[rare]
            out.append(spec(b, "inclusion_rare", {"type": "inclusion_rare", "class_id": c["id"],
                                                  "class_name": c["name"], "exemplars": c["members"][:6]}))
        # Ordering is intentionally NOT emitted. As derived (target = reverse of the natural class
        # order) it measured an anti-natural double-inclusion, not sequencing, and was dropped from
        # the constraint set (assessment.md §7c). derive_targets still returns `order`; a future
        # re-derivation would use the NATURAL order plus a "both classes, any order" control.
        _ = order
        # categorical: baseline-derived type target (interior entities models actually used, typed
        # via G_c and picked to bite) -- works on arbitrary endpoints, unlike the old route-based one.
        if bid in categorical:
            out.append(spec(b, "categorical", categorical[bid]))

    output_dir = init_directory(out_dir, overwrite=True) if overwrite else Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts.json").write_text(json.dumps(out, indent=2))
    (output_dir / "classes_targets.json").write_text(
        json.dumps({"classes": classes, "targets": targets, "categorical": categorical}, indent=2))
    print(f"Wrote {len(out)} Pass-2 specs over {len(targets)} bundles -> {output_dir/'prompts.json'}")
    print(f"  cells: {dict(Counter(s['mode'] for s in out))}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pass1", default="data/kg_creat/responses_regimeA_pass1")
    p.add_argument("--bundles", default="data/kg_creat/prompts_regimeA_v1")
    p.add_argument("--gc", default="data/kg_creat/gc_domains_v2")
    p.add_argument("--out", default="data/kg_creat/prompts_regimeA_pass2")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    asyncio.run(main(a.pass1, a.bundles, a.out, a.gc, overwrite=a.overwrite))
