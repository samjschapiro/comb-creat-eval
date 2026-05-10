"""Regenerate papers/iccc-2026/tables/per_benchmark_scores.tex from
configs/comb_eval/benchmarks.json. Re-run whenever benchmarks.json changes
so the appendix table stays in sync.

Usage:
    uv run python src/comb_eval/scripts/build_per_benchmark_table.py
"""

import json
from collections import defaultdict
from pathlib import Path


PROVIDERS = [
    ("OpenAI",    "openai"),
    ("Anthropic", "anthropic"),
    ("Google",    "google"),
    ("Meta",      "meta-llama"),
    ("Mistral",   "mistralai"),
    ("Qwen",      "qwen"),
    ("DeepSeek",  "deepseek"),
    ("Cohere",    "cohere"),
    ("NVIDIA",    "nvidia"),
    ("Microsoft", "microsoft"),
]

# (benchmark key, formatter)
COLUMNS = [
    ("arena_overall",        lambda v: f"{int(round(v))}"),
    ("mmlu_pro",             lambda v: f"{v:.2f}"),
    ("arena_cw",             lambda v: f"{int(round(v))}"),
    ("eq_bench_cw",          lambda v: f"{int(round(v))}"),
    ("mazur_cw_v2",          lambda v: f"{v:.2f}"),
    ("hivemind_diversity",   lambda v: f"{v:.2f}"),
    ("noveltybench_utility", lambda v: f"{v:.2f}"),
    ("liveideabench",        lambda v: f"{v:.2f}"),
]


def main():
    root = Path(__file__).resolve().parents[3]
    bench_path = root / "configs" / "comb_eval" / "benchmarks.json"
    out_path = root / "papers" / "iccc-2026" / "tables_jmlr" / "per_benchmark_scores.tex"
    bench = json.loads(bench_path.read_text())

    groups: dict[str, list] = defaultdict(list)
    for or_key, fields in bench.items():
        if not any(col in fields for col, _ in COLUMNS):
            continue
        prefix = or_key.split("_", 1)[0]
        name = or_key.split("_", 1)[1] if "_" in or_key else or_key
        groups[prefix].append((name, fields))
    for k in groups:
        groups[k].sort(key=lambda x: x[0])

    def fmt_row(name, fields):
        cells = [r"\texttt{" + name.replace("_", r"\_") + r"}"]
        for col, formatter in COLUMNS:
            cells.append(formatter(fields[col]) if col in fields else "---")
        return " & ".join(cells) + r" \\"

    head = r"""
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{@{}lcccccccc@{}}
\caption{\textbf{Per-model benchmark scores.} Columns: Arena Overall (Elo), MMLU-Pro (accuracy), Arena CW (Elo), EQ-Bench CW v3 (Elo), Mazur CW, Hivemind diversity, NoveltyBench Utility, LiveIdeaBench (5-dim Average). ``---'' indicates the corresponding benchmark does not score that model.}
\label{tab:per-model-benchmarks} \\
\toprule
Model & Arena Ovr & MMLU-Pro & Arena CW & EQ-B. CW & Mazur & Hive. & NovB. & LiveIdea \\
\midrule
\endfirsthead
\multicolumn{9}{@{}l}{\textit{(continued from previous page)}} \\
\toprule
Model & Arena Ovr & MMLU-Pro & Arena CW & EQ-B. CW & Mazur & Hive. & NovB. & LiveIdea \\
\midrule
\endhead
\midrule
\multicolumn{9}{r@{}}{\textit{(continued on next page)}} \\
\endfoot
\bottomrule
\endlastfoot
""".strip("\n")

    body_lines = []
    first = True
    for label, prefix in PROVIDERS:
        rows = groups.get(prefix, [])
        if not rows:
            continue
        if not first:
            body_lines.append(r"\midrule")
        first = False
        body_lines.append(r"\multicolumn{9}{@{}l}{\textit{" + label + r"}} \\")
        for name, fields in rows:
            body_lines.append(fmt_row(name, fields))

    out = "\n" + head + "\n" + "\n".join(body_lines) + "\n\\end{longtable}\n\\normalsize\n"
    out_path.write_text(out)

    n = sum(len(v) for v in groups.values())
    print(f"Wrote {out_path} ({n} model rows)")


if __name__ == "__main__":
    main()
