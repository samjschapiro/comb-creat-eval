"""Augment papers/iccc-2026/tables_neurips/per_test_scores.tex with RAT and
DRAT columns.

Reads the existing 5-column tex (DAT/CDAT/CDAT-N/CDAT-A/PACE), parses each
model row, and appends two columns:
  - RAT: zero-shot strict accuracy (\\%) with no SEM (single proportion
    across 30 normed items).
  - DRAT: mean +/- SEM of SBERT-scored DRAT(k=4 expert) across the 30
    anchor groups.

The existing OpenAI/Anthropic/... provider sectioning is preserved.
"""

import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine as cosine_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.dat_eval.cdat import SBERTEmbeddings, validate_words_sbert, DEFAULT_CUES


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "papers" / "iccc-2026" / "tables_jmlr" / "per_test_scores.tex"
DST = SRC

PROVIDER_PREFIX = {
    "OpenAI":    "openai",
    "Anthropic": "anthropic",
    "Google":    "google",
    "Meta":      "meta-llama",
    "Mistral":   "mistralai",
    "Qwen":      "qwen",
    "DeepSeek":  "deepseek",
    "Cohere":    "cohere",
    "NVIDIA":    "nvidia",
    "Microsoft": "microsoft",
    "Amazon":    "amazon",
    "xAI":       "x-ai",
}


# --- RAT lookup (pilot + expansion, pilot wins on overlap) ---
def load_rat() -> dict[str, float]:
    out: dict = {}
    for path in [
        ROOT / "data/new_tests/rat/expansion_v1/summary.json",
        ROOT / "data/new_tests/rat/pilot_v1/summary.json",  # pilot overrides
    ]:
        s = json.loads(path.read_text())
        for m, v in s.items():
            if v["n_total"] > 0 and v["n_errors"] < v["n_total"]:
                out[m] = v["zs_accuracy_strict"]
    return out


# --- DRAT lookup (k=4 expert: base + extension), SBERT-scored ---
def load_drat_sbert() -> dict[str, tuple[float, float]]:
    """Return {or_id: (mean, sem)} of SBERT-scored DRAT k=4 across 30 groups."""
    raw = []
    for path in [
        ROOT / "data/new_tests/drat/ablation_k4_expert/raw_results.json",
        ROOT / "data/new_tests/drat/ablation_k4_expert_ext/raw_results.json",
    ]:
        if path.exists():
            raw += json.loads(path.read_text())

    print(f"Loading SBERT to score {len(raw)} DRAT items...")
    sbert = SBERTEmbeddings()
    def sv(s): return sbert.encode_batch([s])[0]

    def score_response(words, anchors, n_min=3):
        valid = validate_words_sbert(words)
        a_vecs = [sv(a) for a in anchors]
        if len(valid) < 2: return 0.0
        pool_us = []
        for n in DEFAULT_CUES:
            v = sv(n)
            pool_us.append(max(float(1.0 - cosine_distance(v, av)) for av in a_vecs))
        if len(pool_us) < 10: return 0.0
        tau = float(np.percentile(pool_us, 90.0))
        word_vecs = [sv(w) for w in valid[:10]]
        if len(word_vecs) < 2: return 0.0
        utils = [max(float(1.0 - cosine_distance(v, av)) for av in a_vecs)
                 for v in word_vecs]
        survivors = [word_vecs[i] for i, u in enumerate(utils) if u > tau]
        if len(survivors) < n_min: return 0.0
        k = len(survivors)
        d = [float(cosine_distance(survivors[i], survivors[j]))
             for i in range(k) for j in range(i + 1, k)]
        return 100.0 * float(np.mean(d))

    by_model: dict[str, list[float]] = {}
    for r in raw:
        if "error" in r or r.get("score", {}).get("drat") is None:
            continue
        m = r["model"]
        s = score_response(r.get("extracted_words", []), r["anchors"])
        by_model.setdefault(m, []).append(s)

    out = {}
    for m, scores in by_model.items():
        if len(scores) >= 2:
            out[m] = (float(np.mean(scores)),
                      float(np.std(scores, ddof=1) / np.sqrt(len(scores))))
        elif len(scores) == 1:
            out[m] = (float(scores[0]), 0.0)
    return out


def main():
    rat = load_rat()
    drat = load_drat_sbert()
    print(f"  RAT pool:  {len(rat)} models")
    print(f"  DRAT pool: {len(drat)} models")

    src_text = SRC.read_text()

    # --- Patch column count: 6 -> 8 (1 model col + 5 old + 2 new = 8) ---
    src_text = src_text.replace(
        r"\begin{longtable}{@{}lrrrrr@{}}",
        r"\begin{longtable}{@{}lrrrrrrr@{}}",
    )
    src_text = src_text.replace(
        r"\multicolumn{6}{@{}l}",
        r"\multicolumn{8}{@{}l}",
    )
    src_text = src_text.replace(
        r"\multicolumn{6}{r@{}}",
        r"\multicolumn{8}{r@{}}",
    )

    # Update header rows (both first-head and continuation header)
    src_text = src_text.replace(
        r"Model & DAT & CDAT & CDAT-N  & CDAT-A  & PACE \\",
        r"Model & DAT & CDAT & CDAT-N  & CDAT-A  & PACE & RAT & DRAT \\",
    )
    src_text = src_text.replace(
        r"Model & DAT  & CDAT  & CDAT-N  & CDAT-A  & PACE \\",
        r"Model & DAT  & CDAT  & CDAT-N  & CDAT-A  & PACE & RAT & DRAT \\",
    )

    # Update caption — append note on new columns
    src_text = src_text.replace(
        r"``---'' indicates cells where no valid scores were collected (no temperature passed the appropriateness gate) on the CDAT, or responses were otherwise invalid.",
        r"``---'' indicates cells where no valid scores were collected (no temperature passed the appropriateness gate) on the CDAT, or responses were otherwise invalid. RAT is reported as zero-shot strict accuracy (\\%) on the 30-item normed bank; DRAT (k=4 expert, SBERT) is reported as mean$\\pm$SEM across the 30 anchor groups.",
    )

    # --- Append RAT/DRAT cells to each model row ---
    cur_provider = None
    out_lines = []
    section_re = re.compile(r"\\multicolumn\{8\}\{@\{\}l\}\{\\textit\{([^}]+)\}\}")
    row_re = re.compile(r"^(\\texttt\{([^}]+)\})\s*&(.*?)\\\\\s*$")

    for line in src_text.split("\n"):
        m_sec = section_re.search(line)
        if m_sec:
            cur_provider = m_sec.group(1).strip()
            out_lines.append(line)
            continue
        m_row = row_re.match(line)
        if m_row and cur_provider in PROVIDER_PREFIX:
            prefix = PROVIDER_PREFIX[cur_provider]
            tex_name = m_row.group(2)
            # tex_name uses "-" instead of "." in versions; reverse-mangle.
            # Actual OR-id is usually prefix/<tex-with-dots>. We try a few
            # candidate keys.
            candidates = [
                f"{prefix}/{tex_name}",
                f"{prefix}/{tex_name.replace('-', '.', 100)}",
            ]
            # version-style: gpt-5-4 -> gpt-5.4
            def restore_versions(s: str) -> str:
                # Replace digit-hyphen-digit with digit-dot-digit
                return re.sub(r"(\d)-(\d)", r"\1.\2", s)
            candidates.append(f"{prefix}/{restore_versions(tex_name)}")
            # claude-3-5-haiku -> claude-3.5-haiku
            candidates.append(f"{prefix}/{restore_versions(tex_name)}")
            # Also try with .v1 forms used in OR ids: nova-pro-v1
            candidates = list(dict.fromkeys(candidates))  # dedup, preserve order

            rat_v = next((rat[c] for c in candidates if c in rat), None)
            drat_v = next((drat[c] for c in candidates if c in drat), None)

            rat_cell = f"{rat_v*100:.0f}" if rat_v is not None else "---"
            if drat_v is not None:
                m_, s_ = drat_v
                drat_cell = f"{m_:.2f}$\\pm${s_:.2f}"
            else:
                drat_cell = "---"

            new_line = line.replace(r"\\", f"& {rat_cell} & {drat_cell} \\\\", 1)
            out_lines.append(new_line)
            continue
        out_lines.append(line)

    DST.write_text("\n".join(out_lines))
    print(f"Wrote {DST}")


if __name__ == "__main__":
    main()
