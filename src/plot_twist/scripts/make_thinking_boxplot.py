"""Vertical boxplot of the Overall TC composite vs reasoning effort (Exp 1).

x-axis: reasoning effort (low / medium / high). y-axis: the Overall composite (within-model
$z$-score of surprise/coherence/diversity/realism) -- one value per model per effort level.
Per-model points are overlaid (and faintly connected) so the within-model effect is visible.
If thinking helped, the boxes would rise from low to high; they do not.

Reads thinking_cells.json (from run_thinking_analysis.py) -- no API, no re-annotation.

Usage:
    python src/plot_twist/scripts/make_thinking_boxplot.py configs/plot_twist/thinking_boxplot.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.utils import init_directory, load_config, save_config

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "font.size": 11, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 11,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
})

LEVELS = ["low", "medium", "high"]


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "cells_json"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    cells = json.loads(Path(cfg["cells_json"]).read_text())
    metric = cfg.get("metric", "tc_within")

    # group metric values by level; also keep per-model so we can connect the dots
    by_level = {lv: [] for lv in LEVELS}
    by_model: dict[str, dict] = {}
    for c in cells:
        v = c.get(metric)
        if v is None or not np.isfinite(v):
            continue
        by_level[c["level"]].append(v)
        by_model.setdefault(c["model"], {})[c["level"]] = v
    data = [by_level[lv] for lv in LEVELS]
    print(f"boxplot of '{metric}' by effort level:")
    for lv in LEVELS:
        a = np.array(by_level[lv])
        print(f"  {lv:<7} n={len(a)}  median={np.median(a):+.3f}  mean={a.mean():+.3f}")

    # batlow gradient (blue->green->orange) for the three levels
    from cmcrameri import cm as cmc
    box_cols = [cmc.batlow(x) for x in (0.12, 0.5, 0.86)]

    # HORIZONTAL boxes: levels stacked on the y-axis, Overall on the x-axis.
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    bp = ax.boxplot(data, positions=range(len(LEVELS)), widths=0.55, patch_artist=True,
                    vert=False,
                    medianprops=dict(color="black", lw=1.6),
                    whiskerprops=dict(color="#444"), capprops=dict(color="#444"),
                    flierprops=dict(marker="", alpha=0))  # outliers hidden (points overlaid instead)
    for patch, col in zip(bp["boxes"], box_cols):
        patch.set_facecolor((*col[:3], 0.35)); patch.set_edgecolor("#333"); patch.set_linewidth(0.9)

    # jittered points per level (jitter along the category/y direction); no connecting lines
    rng = np.random.default_rng(0)
    for i, lv in enumerate(LEVELS):
        xs = by_level[lv]
        jy = i + (rng.random(len(xs)) - 0.5) * 0.18
        ax.scatter(xs, jy, s=22, color=box_cols[i], edgecolor="#333", linewidth=0.4, zorder=3)

    ax.set_yticks(range(len(LEVELS)))
    ax.set_yticklabels(["low", "medium", "high"])
    ax.invert_yaxis()  # low at top, high at bottom (matches the requested order)
    ax.set_ylabel("Reasoning effort")
    ax.set_xlabel("Overall (within-model $z$)")
    ax.axvline(0, color="#bbb", lw=0.8, ls=":", zorder=0)
    fig.tight_layout()
    p = out / "thinking_overall_boxplot.pdf"
    fig.savefig(p)
    fig.savefig(out / "thinking_overall_boxplot.png")
    plt.close(fig)
    print(f"saved: {p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
