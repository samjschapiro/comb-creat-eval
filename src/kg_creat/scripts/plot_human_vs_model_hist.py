"""Composite score distributions, humans vs. models: Overall and each task, with each group's mean.

Models: one composite per model (35). Humans: one composite per kept participant, the paper's composite
(compute_composite.artifact_dims) computed on that participant's own answers. All scores come from the shared
human + model originality pool on the 86 items both answered (data/kg_creat/human_study/combined_scores).
Skipped items count as failed answers (0), as in the group composite (build_skip0_scores.py). A person's score rests on ~14 answers against ~90 per
model, so the human distribution is wider partly from noise.

    .venv/bin/python -m src.kg_creat.scripts.plot_human_vs_model_hist
"""
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.kg_creat.scripts.compute_composite import TASK_DIMS, _mean, _ok, artifact_dims

SCORES = Path("data/kg_creat/human_study/combined_scores_skip0")   # skipped items count as failed (0); see build_skip0_scores.py
OUT = Path("data/kg_creat/human_study/figures/composite_hist.png")
MOCK = Path("scratch/human_vs_model_space/composite_hist.png")
HUMAN, MODEL = "#d95f02", "#1b6ca8"
VIEWS = [("Overall", "overall"), ("Association", "association"), ("Analogy", "analogy"), ("Blending", "blending")]


def composite(recs):
    raw = artifact_dims(recs)
    per_task = {}
    for task, dims in TASK_DIMS.items():
        if not raw[task]["_n_artifacts"]:
            continue
        vals = [raw[task][k] for k in dims if _ok(raw[task].get(k))]
        if vals:
            per_task[task] = 100 * _mean(vals)
    return {**per_task, "overall": _mean(list(per_task.values()))} if per_task else {}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if "Nimbus Roman" not in {f.name for f in font_manager.fontManager.ttflist}:
        raise SystemExit("FATAL: Nimbus Roman is not installed")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"], "mathtext.fontset": "custom",
                         "mathtext.rm": "Nimbus Roman", "font.size": 15, "axes.titlesize": 17,
                         "axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13})

    comp = json.loads((SCORES / "composite.json").read_text())["per_model"]
    models = {m: {"overall": v["overall"], **v["per_task"]} for m, v in comp.items() if not m.startswith("human")}
    by_person = defaultdict(list)
    for r in json.loads((SCORES / "human_kept" / "path_scores.json").read_text()):
        by_person[r.get("sample_idx")].append(r)
    people = [c for c in (composite(recs) for recs in by_person.values()) if c]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.2), sharey=False)
    bins = np.arange(0, 102, 4)
    for ax, letter, (title, key) in zip(axes, "abcd", VIEWS):
        h = np.array([p[key] for p in people if key in p and _ok(p[key])])
        m = np.array([v[key] for v in models.values() if _ok(v.get(key))])
        ax.hist(m, bins=bins, color=MODEL, alpha=0.6, label=f"models (n = {len(m)})", density=True)
        ax.hist(h, bins=bins, color=HUMAN, alpha=0.6, label=f"humans (n = {len(h)})", density=True)
        ax.axvline(m.mean(), color=MODEL, lw=2.5, ls="--", label=f"model mean {m.mean():.1f}")
        ax.axvline(h.mean(), color=HUMAN, lw=2.5, ls="--", label=f"human mean {h.mean():.1f}")
        ax.set_title(f"({letter}) {title}")
        ax.set_xlabel("composite (% of max)")
        if letter == "a":
            ax.set_ylabel("density")
        ax.set_xlim(0, 100)
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180)
    MOCK.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(OUT, MOCK)
    for title, key in VIEWS:
        h = [p[key] for p in people if key in p and _ok(p[key])]
        m = [v[key] for v in models.values() if _ok(v.get(key))]
        print(f"{title:12s} humans n={len(h)} mean {np.mean(h):.1f} median {np.median(h):.1f} | "
              f"models n={len(m)} mean {np.mean(m):.1f} | humans above model mean: {sum(x > np.mean(m) for x in h)}")
    print(f"wrote {OUT} (copy {MOCK})")


if __name__ == "__main__":
    main()
