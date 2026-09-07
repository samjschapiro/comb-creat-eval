"""Sample anchor entities for the anagram (exploratory-creativity) task.

One stimulus per spec: a prominent G_c entity whose label is short enough to be anagram-tractable
(4-13 alphabetic letters). Domain is carried for reference-only analysis (never shown to the model).
Output is a prompts.json of anagram specs, consumed by run_anagram.py.

    python src/kg_creat/scripts/sample_anagram.py --gc data/kg_creat/gc_domains_v2 \
        --n 150 --min-letters 4 --max-letters 13 --seed 5 --out data/kg_creat/prompts_anagram
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.kg_creat.graph import KnowledgeGraph  # noqa: E402
from src.kg_creat.sample import _prominent_nodes  # noqa: E402


def _alpha_len(label: str) -> int:
    return sum(1 for c in label if c.isalpha())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc", type=str, required=True, help="dir holding gc.json (+ entity_domains.json)")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--min-letters", type=int, default=4)
    ap.add_argument("--max-letters", type=int, default=13)
    ap.add_argument("--min-degree", type=int, default=3)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    gc_dir = Path(args.gc)
    gc = KnowledgeGraph.load(gc_dir / "gc.json")
    domains = {}
    dpath = gc_dir / "entity_domains.json"
    if dpath.exists():
        domains = json.loads(dpath.read_text())

    pool = _prominent_nodes(gc, args.min_degree)
    # tractable-length labels, unique by lowercased letter-multiset (so we don't sample two anchors
    # that share the exact same anagram search space)
    seen_letters, cands = set(), []
    for n in pool:
        lab = gc.label(n)
        if not (args.min_letters <= _alpha_len(lab) <= args.max_letters):
            continue
        key = "".join(sorted(c for c in lab.lower() if c.isalpha()))
        if key in seen_letters:
            continue
        seen_letters.add(key)
        cands.append(n)

    rng = random.Random(args.seed)
    rng.shuffle(cands)
    chosen = cands[: args.n]

    specs = []
    for i, n in enumerate(chosen):
        lab = gc.label(n)
        specs.append({
            "prompt_id": f"AN{i}.anagram",
            "bundle_id": f"AN{i}",
            "regime": "C",
            "mode": "anagram",
            "u": n,
            "v": None,
            "u_label": lab,
            "v_label": None,
            "h": None,
            "k": None,
            "constraint": None,
            "domain_u": domains.get(n),   # reference-only; never rendered into the prompt
            "domain_v": None,
            "cross_domain": None,
        })

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.json").write_text(json.dumps(specs, indent=2))
    lens = [_alpha_len(s["u_label"]) for s in specs]
    print(f"pool={len(pool)} tractable-unique={len(cands)} sampled={len(specs)} -> {out/'prompts.json'}")
    print(f"letters: min={min(lens)} max={max(lens)} mean={sum(lens)/len(lens):.1f}")
    print("sample:", [s["u_label"] for s in specs[:15]])


if __name__ == "__main__":
    main()
