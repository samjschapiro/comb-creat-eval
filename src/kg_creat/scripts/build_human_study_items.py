"""Build the Kombine human study's item bundles and slot plan from the benchmark items the LLMs answered.

Reads data/kg_creat/kombine_test30/prompts/prompts.json (30 association pairs; 30 pairs shared by analogy and
blending) and writes the study's js/stimuli-data.js items block:

  * ASSOC_BUNDLES: the 30 association pairs cut into 6 bundles of 5 (prompt order).
  * PAIR_BUNDLES:  the analogy/blending pairs minus EXCLUDE_PAIRS, cut into 6 bundles as evenly as possible
    (28 pairs -> 5,5,5,5,4,4). EXCLUDE_PAIRS are the pairs the study's worked examples show (Democracy + Banking,
    The blue whale :: The mattress): a participant given one would already have seen a model's answer to it.
  * SLOTS: 120 slots (K = 20 responses per item). Slot s = (association bundle a, analogy bundle i, blending
    bundle j) with j != i, so no participant sees the same pair in both analogy and blending. Every ordered
    (i, j) pair with i != j appears 4 times (30 x 4 = 120), so each analogy bundle and each blending bundle is
    used exactly 20 times; a = s mod 6, so each association bundle is used exactly 20 times. The slots are
    ordered so any prefix is close to balanced (the server fills the lowest open slot first).

    python src/kg_creat/scripts/build_human_study_items.py \
        ~/Desktop/Experiments/llm_creativity_mech_interp/src/experiments/kombine_generation/js/stimuli-data.js
"""
import json, sys
from pathlib import Path

PROMPTS = Path("data/kg_creat/kombine_test30/prompts/prompts.json")
N_BUNDLES, PER_BUNDLE, K = 6, 5, 20
EXCLUDE_PAIRS = {"E0", "E26"}   # analogy prompt ids; their blending twins (F0, F26) go with them
BEGIN, END = "// === BEGIN GENERATED ITEMS (build_human_study_items.py) ===", "// === END GENERATED ITEMS ==="


def main(out_path: str):
    pr = json.loads(PROMPTS.read_text()); pr = pr if isinstance(pr, list) else pr["prompts"]
    assoc = [p for p in pr if p["mode"] == "baseline"]
    an = [p for p in pr if p["mode"] == "analogy"]
    bl = [p for p in pr if p["mode"] == "blending"]
    if len(assoc) != 30 or len(an) != 30 or [(p["u_label"], p["v_label"]) for p in an] != [(p["u_label"], p["v_label"]) for p in bl]:
        sys.exit("FATAL: expected 30 association pairs and 30 identical analogy/blending pairs")
    item = lambda p, pid: {"prompt_id": pid, "u": p["u_label"], "v": p["v_label"]}
    assoc_items = [item(p, p["prompt_id"]) for p in assoc]
    pair_items = [{"prompt_id_analogy": a["prompt_id"], "prompt_id_blending": b["prompt_id"], "u": a["u_label"], "v": a["v_label"]}
                  for a, b in zip(an, bl) if a["prompt_id"] not in EXCLUDE_PAIRS]
    if len(pair_items) != 30 - len(EXCLUDE_PAIRS):
        sys.exit(f"FATAL: EXCLUDE_PAIRS {EXCLUDE_PAIRS} did not all match analogy prompt ids")

    def chunks(xs):   # as even as possible, larger bundles first
        base, extra = divmod(len(xs), N_BUNDLES); out, k = [], 0
        for b in range(N_BUNDLES):
            n = base + (1 if b < extra else 0); out.append(xs[k:k + n]); k += n
        return out
    # ordered (i, j), i != j, interleaved round-robin so early slots already cover every bundle
    ij = [(i, (i + d) % N_BUNDLES) for d in range(1, N_BUNDLES) for i in range(N_BUNDLES)]   # 30 pairs
    slots = [{"slot": s, "assoc": s % N_BUNDLES, "analogy": ij[s % 30][0], "blending": ij[s % 30][1]}
             for s in range(K * N_BUNDLES)]
    for key in ("assoc", "analogy", "blending"):
        counts = [sum(1 for x in slots if x[key] == b) for b in range(N_BUNDLES)]
        assert counts == [K] * N_BUNDLES, (key, counts)
    assert all(x["analogy"] != x["blending"] for x in slots)
    block = "\n".join([BEGIN,
        f"window.ASSOC_BUNDLES = {json.dumps(chunks(assoc_items), indent=1)};",
        f"window.PAIR_BUNDLES = {json.dumps(chunks(pair_items), indent=1)};",
        f"window.SLOTS = {json.dumps(slots)};",
        END])
    out = Path(out_path).expanduser(); s = out.read_text()
    if BEGIN in s:
        a, b = s.index(BEGIN), s.index(END) + len(END); s = s[:a] + block + s[b:]
    else:
        s = block + "\n\n" + s
    out.write_text(s)
    print(f"wrote {len(assoc_items)} association + {len(pair_items)} analogy/blending pairs "
          f"(bundle sizes {[len(b) for b in chunks(pair_items)]}), {len(slots)} slots -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
