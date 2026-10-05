"""Figures for the human vs. LLM language contrast (report finding 5).

  word clouds  typical concepts and relations of each group, pooled over the three tasks. Units are whole
               phrases (lower-cased), sized by how much more one group uses the phrase than the other: the
               log-odds z with an informative Dirichlet prior (Monroe et al. 2008, as in
               analyze_human_llm_language.py); only phrases with z > 0 for that group appear.
  bars         share of concepts (noun supersense) and relations (verb supersense) in each WordNet category,
               humans vs. LLMs, per task, from analyze_supersenses.py's output; 95% bootstrap CIs on the
               difference that exclude zero are starred.

Kept humans and the 35 scored models, on the 86 shared items.

    .venv/bin/python -m src.kg_creat.scripts.plot_human_llm_language
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch
from wordcloud import WordCloud

from src.kg_creat.scripts.analyze_human_llm_language import elements, log_odds

SUPERSENSES = Path("data/kg_creat/human_study/analysis/supersenses.json")
OUT = Path("data/kg_creat/human_study/figures")
MOCK = Path("scratch/human_llm_language")
NIMBUS = Path.home() / "Library/Fonts/NimbusRoman-Regular.otf"
HUMAN, MODEL = "#DA7678", "#71A8C4"
SHADES = {"human": ["#DA7678", "#B5484B", "#8E2F33", "#C95F62"], "llm": ["#71A8C4", "#3F7FA0", "#2A5D78", "#5B93B1"]}
TOP_CATS, MAX_WORDS = 8, 120
LABEL = {"not in WordNet": "other"}   # phrases WordNet does not know (mostly names and technical terms)


def style():
    if "Nimbus Roman" not in {f.name for f in font_manager.fontManager.ttflist}:
        raise SystemExit("FATAL: Nimbus Roman is not installed")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"], "font.size": 16,
                         "axes.titlesize": 19, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 16,
                         "legend.fontsize": 17})


def norm(s):
    return " ".join(str(s).lower().split())


def wordclouds(rows):
    z = {}
    for kind in ("c", "r"):
        ch = Counter(norm(r[4]) for r in rows if r[0] == "human" and r[3] == kind)
        cl = Counter(norm(r[4]) for r in rows if r[0] == "llm" and r[3] == kind)
        zz = log_odds(ch, cl)
        z[kind] = {"human": {w: v for w, v in zz.items() if v > 0}, "llm": {w: -v for w, v in zz.items() if v < 0}}
    panels = [("c", "human", "Human concepts"), ("c", "llm", "LLM concepts"),
              ("r", "human", "Human relations"), ("r", "llm", "LLM relations")]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5))
    for ax, letter, (kind, g, title) in zip(axes.flat, "abcd", panels):
        cols = SHADES[g]
        wc = WordCloud(width=1400, height=900, background_color="white", max_words=MAX_WORDS, prefer_horizontal=0.95,
                       min_font_size=10, max_font_size=160, relative_scaling=0.6, font_path=str(NIMBUS), random_state=3,
                       color_func=lambda *a, **k: cols[hash(a[0]) % len(cols)], collocations=False)
        wc.generate_from_frequencies(z[kind][g])
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"({letter}) {title}", loc="left", fontsize=24)
        top = sorted(z[kind][g].items(), key=lambda x: -x[1])[:10]
        print(f"{title:16s} {len(z[kind][g])} phrases; top: " + ", ".join(f"{w} ({v:.1f})" for w, v in top))
    fig.tight_layout(h_pad=1.0, w_pad=1.5)
    save(fig, "language_wordclouds")
    for ax, letter in zip(axes.flat, "abcd"):           # one file per panel, no title, for LaTeX subfigures
        f2, a2 = plt.subplots(figsize=(7.5, 4.8))
        a2.imshow(ax.get_images()[0].get_array(), interpolation="bilinear"); a2.axis("off")
        f2.savefig(OUT / f"language_wordclouds_{letter}.pdf", dpi=200, bbox_inches="tight", pad_inches=0.02)
        plt.close(f2)


def bars():
    res = json.loads(SUPERSENSES.read_text())
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    for col, task in enumerate(("association", "analogy", "blending")):
        for row, (name, label) in enumerate((("concepts", "concepts"), ("relations", "relations"))):
            ax = axes[row, col]
            cats = sorted(res[task][name], key=lambda r: -(r["human"] + r["llm"]))[:TOP_CATS]
            y = np.arange(len(cats))[::-1]
            h = 0.38
            ax.barh(y + h / 2, [100 * c["human"] for c in cats], h, color=HUMAN)
            ax.barh(y - h / 2, [100 * c["llm"] for c in cats], h, color=MODEL)
            sig = lambda c: c["ci95"][0] > 0 or c["ci95"][1] < 0
            ax.set_yticks(y, [LABEL.get(c["category"], c["category"]) + (" *" if sig(c) else "") for c in cats])
            ax.set_title(f"({'abcdef'[row * 3 + col]}) {task.capitalize()} {label}", loc="left")
            ax.set_xlabel("share (%)")
            ax.spines[["top", "right"]].set_visible(False)
    fig.legend([Patch(color=HUMAN), Patch(color=MODEL)], ["Humans", "LLMs"], loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=2.0, w_pad=2.0)
    save(fig, "supersense_bars")


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    MOCK.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
        shutil.copy(OUT / f"{stem}.{ext}", MOCK / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {OUT / stem}.{{png,pdf}} (copy in {MOCK})")


def main():
    style()
    rows, _ = elements()
    wordclouds(rows)
    bars()


if __name__ == "__main__":
    main()
