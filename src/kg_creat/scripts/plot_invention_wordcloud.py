"""Word cloud of the names models gave their inventions, one panel per task.

Each entry is a coined name (the blend's `concept`, or the analogy's `invention`), lower-cased and
kept as a whole phrase; its size is the number of inventions across all models that carry exactly
that name. So the big words are the names many models converged on (Finding #6: naming vs
inventing), and the long tail of small words is the variety.

    .venv/bin/python -m src.kg_creat.scripts.plot_invention_wordcloud
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

RUN = Path("data/kg_creat/kombine_test30")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
PALETTE = {"blending": ["#1B4F72", "#2E86C1", "#154360", "#21618C", "#2874A6"],
           "analogy": ["#7B241C", "#C0392B", "#641E16", "#A93226", "#922B21"]}
MAX_WORDS = 250
NIMBUS = Path.home() / "Library/Fonts/NimbusRoman-Regular.otf"     # the paper's face, for the words and the titles

plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"]})


def names():
    out = {"blending": Counter(), "analogy": Counter()}
    for sd in sorted((RUN / "scores").iterdir()):
        if not (sd / "path_scores.json").exists():
            continue
        resp = {(r["prompt_id"], r["sample_idx"]): r for r in json.loads((RUN / "responses" / sd.name / "responses.json").read_text())}
        for r in json.loads((sd / "path_scores.json").read_text()):
            if r["mode"] == "blending":
                it = (resp[(r["prompt_id"], r["sample_idx"])].get("items") or [{}])[0]
                n = it.get("concept") or r["triples"][0][0]
            elif r["mode"] == "analogy" and r.get("invention"):
                n = r["invention"]
            else:
                continue
            out[r["mode"]][" ".join(n.lower().split())] += 1
    return out


def main():
    freqs = names()
    font = str(NIMBUS) if NIMBUS.exists() else next((f for f in font_manager.findSystemFonts() if f.endswith(("Times New Roman.ttf", "DejaVuSerif.ttf"))), None)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))
    for ax, task, letter, op in zip(axes, ("blending", "analogy"), "ab", ("+", "::")):
        f = freqs[task]
        colors = PALETTE[task]
        wc = WordCloud(width=1500, height=1000, background_color="white", max_words=MAX_WORDS, prefer_horizontal=0.95,
                       min_font_size=9, max_font_size=150, relative_scaling=0.6, font_path=font, random_state=3,
                       color_func=lambda *a, **k: colors[hash(a[0]) % len(colors)], collocations=False)
        wc.generate_from_frequencies(f)
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        ax.set_title(f"({letter}) {task}", fontsize=22, loc="left", pad=8)
        print(f"{task}: {sum(f.values()):,} inventions, {len(f):,} distinct names; top: {f.most_common(6)}")
    fig.tight_layout(w_pad=1.5)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_invention_wordcloud.{ext}", dpi=200, bbox_inches="tight")
    print("saved", OUT / "fig_invention_wordcloud.{png,pdf}")


if __name__ == "__main__":
    main()
