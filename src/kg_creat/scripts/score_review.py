"""Compute human-vs-LLM-judge agreement from a filled blind-review set (CREATE-style reliability).

Reads the human-filled CSVs + ``_judge_key.json`` and reports, per task:
  - % agreement and Cohen's kappa (chance-corrected);
  - for factuality, precision/recall/F1 on the *hallucinated* class and balanced accuracy
    (the numbers CREATE reports; their judge was 0.94 recall / 0.52 precision on bad relations).

    .venv_mlx/bin/python src/kg_creat/scripts/score_review.py data/kg_creat/scores_analogy_v2
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path


def _kappa(pairs):
    """Cohen's kappa over (human, judge) label pairs."""
    n = len(pairs)
    po = sum(1 for h, j in pairs if h == j) / n
    hc, jc = Counter(h for h, _ in pairs), Counter(j for _, j in pairs)
    pe = sum((hc[l] / n) * (jc[l] / n) for l in set(hc) | set(jc))
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def _load_log(review_dir, prefix):
    """Read the web UI's responses.jsonl (preferred); fall back to CSV columns if absent."""
    log = review_dir / "responses.jsonl"
    if log.exists():
        latest = {}
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["id"]] = r["v"].strip().lower()  # last write wins
        return [(iid, v) for iid, v in latest.items() if iid.startswith(prefix)]
    # CSV fallback
    csv_path = review_dir / ("review_factuality.csv" if prefix == "F" else "review_analogy.csv")
    col = "YOUR_VERDICT (true / hallucinated)" if prefix == "F" else "YOUR_VERDICT (valid / invalid)"
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            v = (row.get(col) or "").strip().lower()
            if v:
                rows.append((row["item_id"], v))
    return rows


def _report(name, human_rows, key, positive=None):
    pairs = [(h, key[iid]) for iid, h in human_rows if iid in key]
    if not pairs:
        print(f"\n{name}: no filled rows yet.")
        return
    po, k = _kappa(pairs)
    print(f"\n{name}  (n={len(pairs)} reviewed)")
    print(f"  agreement = {po*100:.1f}%   Cohen's kappa = {k:.2f}")
    if positive:
        tp = sum(1 for h, j in pairs if h == positive and j == positive)
        fp = sum(1 for h, j in pairs if h != positive and j == positive)
        fn = sum(1 for h, j in pairs if h == positive and j != positive)
        tn = sum(1 for h, j in pairs if h != positive and j != positive)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
        tpr = tp / (tp + fn) if tp + fn else float("nan")
        tnr = tn / (tn + fp) if tn + fp else float("nan")
        print(f"  on '{positive}' class (vs human as ground truth):")
        print(f"    precision={prec:.2f}  recall={rec:.2f}  F1={f1:.2f}  balanced_acc={(tpr+tnr)/2:.2f}")


def main(scores_dir):
    rev = Path(scores_dir) / "human_review"
    key = json.loads((rev / "_judge_key.json").read_text())
    _report("FACTUALITY judge", _load_log(rev, "F"), key, positive="hallucinated")
    _report("SEMANTIC-ANALOGY judge", _load_log(rev, "A"), key, positive="invalid")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_analogy_v2")
