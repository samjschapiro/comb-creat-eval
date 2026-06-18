"""Plot transformational creativity (TC composite) vs model RELEASE DATE.

Per-model TC = the headline composite `overall_eq` (equal-weight z of surprise/
coherence/diversity/realism) from the main benchmark (tc.json). Release date = the
OpenRouter /models `created` timestamp (ground truth, fetched once and cached to the
output dir for reproducibility -- no fabricated dates).

The question: is transformational creativity improving as newer models ship? An OLS
trend (TC per year) + a running frontier (best TC released so far) summarize it; the
human gold ceiling is drawn as a reference line.

Usage:
    python src/plot_twist/scripts/tc_over_time.py configs/plot_twist/tc_over_time.yaml --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from scipy.stats import pearsonr

from src.utils import init_directory, load_config, save_config

# --- Camera-ready styling (Times serif; sized for a SINGLE-column figure with large,
#     legible fonts -- rendered ~1:1 at \columnwidth, so these are the on-page point sizes). ---
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "font.size": 9,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# EXACT provider->colour scheme from make_tc_barplot.py (the tc_scorecard figure), so
# the two figures are colour-consistent: a fixed long-tail bucket is grey "Other"; every
# other provider gets the categorical batlow (batlowS) colour at its index in the sorted
# provider list, with the same near-white darkening (_adj).
OTHER_PROVIDERS = {"minimax", "morph", "z-ai", "deepcogito", "nousresearch", "tencent", "ai21", "baidu"}
OTHER_COLOR = "#8c8c8c"


def _adj(c):  # identical to make_tc_barplot._adj
    r, g, b = c[0], c[1], c[2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    if lum > 0.74:
        s = 0.62 / lum
        r, g, b = r * s, g * s, b * s
    return (r, g, b, 1.0)


def provider_colors(providers):
    """Replicate tc_scorecard's PROV_COLOR exactly: batlowS(i) per sorted-provider index,
    Other-bucket greyed. `providers` MUST be the sorted full non-human provider list."""
    from cmcrameri import cm as _cmc
    return {p: (OTHER_COLOR if p in OTHER_PROVIDERS else _adj(_cmc.batlowS(i)))
            for i, p in enumerate(providers)}


def _fetch_created(cache_path: Path) -> dict[str, int]:
    """OpenRouter model -> `created` unix ts. Cached to disk for reproducibility."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    load_dotenv()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
    )
    data = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    created = {m["id"]: m.get("created") for m in data if m.get("created")}
    cache_path.write_text(json.dumps(created, indent=2))
    return created


# provider -> colour + display name (matches make_tc_barplot's palette family)
PROV_NAME = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google",
             "meta-llama": "Meta", "deepseek": "DeepSeek", "qwen": "Qwen",
             "mistralai": "Mistral", "amazon": "Amazon", "nvidia": "NVIDIA",
             "z-ai": "Z-AI", "moonshotai": "Moonshot", "deepcogito": "DeepCogito",
             "nousresearch": "Nous", "ai21": "AI21", "baidu": "Baidu",
             "minimax": "MiniMax", "ibm-granite": "IBM", "tencent": "Tencent", "morph": "Morph"}


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg = load_config(config_path)
    for f in ("output_dir", "tc_json"):
        if f not in cfg:
            raise ValueError(f"FATAL: '{f}' required in config")
    out = init_directory(cfg["output_dir"], overwrite=overwrite)
    save_config(cfg, out)

    tc = json.loads(Path(cfg["tc_json"]).read_text())
    created = _fetch_created(out / "model_created.json")

    # Per-model TC = mean PERCENTILE rank (over the pool) of the four facets -- the same
    # percentile scale as the radar figure and the leaderboard table.
    FACETS = ["mean_surprise", "mean_coherence", "div", "mean_realism"]
    pool = {k: np.array([d[k] for d in tc], dtype=float) for k in FACETS}

    def pctrank(k, x):
        v = pool[k]
        return 100.0 * (np.sum(v < x) + 0.5 * np.sum(v == x)) / len(v)

    def overall_pctl(d):
        return float(np.mean([pctrank(k, d[k]) for k in FACETS]))

    # join models to (date, TC); humans have no release date -> kept only as the ceiling
    pts = []
    for d in tc:
        s = d["source"]
        if s == "human":
            continue
        ts = created.get(s)
        if not ts:
            continue
        pts.append({"model": s, "date": datetime.fromtimestamp(ts, timezone.utc),
                    "tc": overall_pctl(d), "provider": s.split("/")[0]})
    pts.sort(key=lambda p: p["date"])
    human_tc = next((overall_pctl(d) for d in tc if d["source"] == "human"), None)
    n_drop = sum(1 for d in tc if d["source"] != "human") - len(pts)
    print(f"{len(pts)} models with release dates ({n_drop} unmatched, dropped); "
          f"human ceiling overall_eq = {human_tc:.3f}")

    x = np.array([mdates.date2num(p["date"]) for p in pts])
    y = np.array([p["tc"] for p in pts])

    # OLS trend (TC per year) + correlation
    slope, intercept = np.polyfit(x, y, 1)
    r, p = pearsonr(x, y)
    tc_per_year = slope * 365.25
    print(f"trend: TC per year = {tc_per_year:+.3f}  (Pearson r={r:+.3f}, p={p:.4f}, n={len(pts)})")

    # running frontier: best TC among models released up to each date, with the model
    # that SET each new frontier record (these are temporally spread -> labels don't collide)
    frontier = np.maximum.accumulate(y)
    record_idx = [i for i in range(len(y)) if y[i] == frontier[i] and (i == 0 or y[i] > frontier[i - 1])]

    # provider colour: EXACT batlow map from tc_scorecard (sorted full non-human provider
    # list -> batlowS index), so a provider has the same colour in both figures.
    PROVIDERS = sorted({d["source"].split("/")[0] for d in tc if d["source"] != "human"})
    PROV_COLOR = provider_colors(PROVIDERS)

    def color_of(pr):
        return PROV_COLOR.get(pr, OTHER_COLOR)

    fig, ax = plt.subplots(figsize=(3.5, 3.4))   # SINGLE column (rendered ~1:1 at \columnwidth)
    # grey "Other" models first (muted, behind), then batlow-coloured providers on top
    for p_ in pts:
        hi = p_["provider"] not in OTHER_PROVIDERS
        ax.scatter(p_["date"], p_["tc"], s=20 if hi else 12, color=color_of(p_["provider"]),
                   edgecolor="#333" if hi else "none", linewidth=0.35,
                   alpha=0.95 if hi else 0.55, zorder=3 if hi else 2)

    ax.plot([pts[i]["date"] for i in range(len(pts))], frontier, color="#000", lw=1.3,
            drawstyle="steps-post", zorder=4, label="Frontier (best released so far)")
    if human_tc is not None:
        ax.axhline(human_tc, color="#000", ls=":", lw=1.2, zorder=1)
        ax.text(x.min(), human_tc, "Expert-human ceiling", va="bottom", ha="left",
                fontsize=11, style="italic")

    # In one column there is no room for per-model frontier labels; we label only the two
    # most recent releases (the headline), ringed with a leader line to the open lower band.
    hl_models = cfg.get("annotate_models", [])
    hl_offsets = {
        "anthropic/claude-opus-4.8": (-6, -34),
        "openai/gpt-5.5": (10, 16),
        "anthropic/claude-sonnet-4.5": (-95, -16),
    }
    for p_ in pts:
        if p_["model"] in hl_models:
            ax.scatter(p_["date"], p_["tc"], s=70, facecolors="none", edgecolors="#000",
                       linewidths=1.1, zorder=6)
            ax.annotate(p_["model"].split("/")[-1], (p_["date"], p_["tc"]),
                        xytext=hl_offsets.get(p_["model"], (-30, -28)), textcoords="offset points",
                        fontsize=11, fontweight="bold", zorder=7,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color="#000", shrinkA=0, shrinkB=3))


    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_xlabel("Model release date")
    ax.set_ylabel("Transformational creativity")

    from matplotlib.patches import Patch
    # legend: every non-Other provider present (batlow colour) + the grey Other bucket,
    # mirroring the tc_scorecard legend items/colours.
    colored_present = [pr for pr in PROVIDERS if pr not in OTHER_PROVIDERS
                       and any(p_["provider"] == pr for p_ in pts)]
    prov_handles = [Patch(color=PROV_COLOR[pr], label=PROV_NAME.get(pr, pr)) for pr in colored_present]
    if any(p_["provider"] in OTHER_PROVIDERS for p_ in pts):
        prov_handles.append(Patch(color=OTHER_COLOR, label="Other"))
    # provider legend BELOW the axis (compact strip); the frontier line is left unlabelled.
    ax.legend(handles=prov_handles, loc="upper center", bbox_to_anchor=(0.5, -0.40),
              ncol=4, fontsize=7.5, handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    pfig = out / "tc_over_time.pdf"
    fig.savefig(pfig)
    fig.savefig(out / "tc_over_time.png")   # PNG for quick viewing
    plt.close(fig)

    json.dump({"trend_tc_per_year": tc_per_year, "pearson_r": r, "p": p, "n": len(pts),
               "human_tc": human_tc,
               "points": [{"model": p_["model"], "date": p_["date"].strftime("%Y-%m-%d"), "tc": p_["tc"]}
                          for p_ in pts]},
              open(out / "tc_over_time.json", "w"), indent=2)
    print(f"saved: {pfig}\n       {out/'tc_over_time.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
