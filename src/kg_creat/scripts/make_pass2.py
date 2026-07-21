"""Build Pass-2 (constrained) prompt specs from Pass-1 baseline responses.

Derives relation CLASSES from the baseline corpus (what models actually emitted), names them,
picks per-bundle constraint targets *against each bundle's own default behaviour*, and writes a
prompt set whose endpoints are IDENTICAL to Pass 1 — only the constraint changes, so within-bundle
deltas are causal in constraint type.

Cells: exclusion · inclusion (common class) · inclusion_rare (niche class) · ordering · categorical.

    .venv_mlx/bin/python src/kg_creat/scripts/make_pass2.py
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils import init_directory  # noqa: E402
from src.kg_creat.embed import get_embedder  # noqa: E402
from src.kg_creat.relation_classes import collect, derive_classes, derive_targets, name_classes  # noqa: E402


async def main(pass1_dir, bundles_dir, out_dir, k=8, top_n=150, min_share=0.08,
               name_model="openai/gpt-oss-120b", overwrite=False):
    from src.dat_eval.llm import get_async_client

    embed = get_embedder()
    counts, per_bundle, seqs = collect(pass1_dir)
    classes = derive_classes(counts, embed, k=k, top_n=top_n)
    classes = await name_classes(classes, get_async_client(), name_model)
    targets = derive_targets(per_bundle, seqs, classes, min_share=min_share)
    by_id = {c["id"]: c for c in classes}

    base = {s["bundle_id"]: s for s in json.loads((Path(bundles_dir) / "prompts.json").read_text())
            if s["mode"] == "baseline"}

    def spec(b, mode, constraint):
        return {**{k: b[k] for k in ("bundle_id", "regime", "u", "v", "u_label", "v_label", "h", "k")},
                "prompt_id": f"{b['bundle_id']}.{mode}", "mode": mode, "constraint": constraint}

    out, skipped = [], 0
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
        if order is not None:
            a, bb = by_id[order[0]], by_id[order[1]]
            out.append(spec(b, "ordering", {"type": "ordering",
                                            "before_name": a["name"], "before_exemplars": a["members"][:4],
                                            "after_name": bb["name"], "after_exemplars": bb["members"][:4]}))
        else:
            skipped += 1
        # categorical stays G_c-derived (entity typing isn't recoverable from baseline text)
        cat = next((s for s in json.loads((Path(bundles_dir) / "prompts.json").read_text())
                    if s["bundle_id"] == bid and s["mode"] == "categorical"), None)
        if cat and cat.get("constraint"):
            out.append(spec(b, "categorical", cat["constraint"]))

    output_dir = init_directory(out_dir, overwrite=True) if overwrite else Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompts.json").write_text(json.dumps(out, indent=2))
    (output_dir / "classes_targets.json").write_text(
        json.dumps({"classes": classes, "targets": targets}, indent=2))
    from collections import Counter
    print(f"Wrote {len(out)} Pass-2 specs over {len(targets)} bundles -> {output_dir/'prompts.json'}")
    print(f"  cells: {dict(Counter(s['mode'] for s in out))}"
          + (f"  ({skipped} bundles had no orderable class pair)" if skipped else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pass1", default="data/kg_creat/responses_regimeA_pass1")
    p.add_argument("--bundles", default="data/kg_creat/prompts_regimeA_v1")
    p.add_argument("--out", default="data/kg_creat/prompts_regimeA_pass2")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    asyncio.run(main(a.pass1, a.bundles, a.out, overwrite=a.overwrite))
