"""Point vs distributional creativity, humans and LLMs side by side (downstream of analyze_sctt_diversity.py).

Point creativity: DAT score (one response, semantic distance; same scorer for both populations).
Distributional creativity: SCTT population distinctiveness, the mean SBERT distance from an individual's
responses to every other individual's in the same population (the one distributional measure both
populations have). Reads the saved unit tables; nothing is rescored.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.drat_human import save, set_camera_ready_style
from src.utils import init_directory, load_config, save_config

COLORS = {"human": "#2a78d6", "llm": "#eb6834"}  # dataviz reference slots 1-2, as in fig_sctt_diversity


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "point", "distributional", "label_units", "output_dir"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    up = Path(config["upstream_dir"])
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()
    x, y = config["point"]["column"], config["distributional"]["column"]

    pops = {"human": pd.read_csv(up / "human_units.csv", index_col="unit"),
            "llm": pd.read_csv(up / "llm_units.csv", index_col="unit")}
    for name, t in pops.items():
        missing = {x, y} - set(t.columns)
        if missing:
            raise ValueError(f"FATAL: {missing} not in {name}_units.csv")
    data = {k: t[[x, y]].dropna() for k, t in pops.items()}
    res = {}
    for k, d in data.items():
        rho, p = stats.spearmanr(d[x], d[y])
        res[k] = {"n": len(d), "rho": round(float(rho), 3), "p": float(p),
                  "point_mean": round(float(d[x].mean()), 3), "distributional_mean": round(float(d[y].mean()), 4)}
    (out / "results.json").write_text(json.dumps(res, indent=2))

    set_camera_ready_style()
    allx = pd.concat([d[x] for d in data.values()])
    ally = pd.concat([d[y] for d in data.values()])
    padx, pady = 0.04 * (allx.max() - allx.min()), 0.04 * (ally.max() - ally.min())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True, gridspec_kw={"wspace": 0.08})
    titles = {"human": f"(a) Humans (n = {res['human']['n']})", "llm": f"(b) LLMs (n = {res['llm']['n']})"}
    for ax, (k, d) in zip(axes, data.items()):
        ax.scatter(d[x], d[y], s=26 if len(d) > 100 else 60, color=COLORS[k], alpha=0.6 if len(d) > 100 else 0.85,
                   edgecolors="white", linewidths=0.6, zorder=2)
        r = res[k]
        ax.text(0.03, 0.97, f"ρ = {r['rho']:+.2f}".replace("-", "−") + (" *" if r["p"] < 0.05 else "") + f"\np = {r['p']:.2g}",
                transform=ax.transAxes, va="top", ha="left", fontsize=13, color="#333333")
        ax.set_title(titles[k])
        ax.set_xlabel(config["point"]["label"])
        ax.set_xlim(allx.min() - padx, allx.max() + padx)
        ax.set_ylim(ally.min() - pady, ally.max() + pady)
        for unit in config["label_units"]:
            if unit in d.index:
                ax.annotate(unit.split("/")[-1], (d.at[unit, x], d.at[unit, y]), xytext=(-6, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=11, color="#555555")
    axes[0].set_ylabel(config["distributional"]["label"])
    save(fig, out / "figures" / "fig_point_vs_distributional")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
