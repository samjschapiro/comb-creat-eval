"""Qualitative showcase: the highest-scoring analogy and blend INVENTIONS across models.

Selects the top invention per task (utility-passed, judged valid+useful, ranked by originality; one per
model for diversity) and renders them as colour-coded cards -- the reader-facing "what does good
combinatorial creativity look like" figure.

    .venv/bin/python -m src.kg_creat.scripts.plot_creativity_gallery
"""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "text.color": "#222222",
})

SCORES = Path("data/kg_creat/kombine_test30/scores")
RESP = Path("data/kg_creat/kombine_test30/responses")
OUT = Path("docs/reports/2026-08-31_kg_creat_invention_homogeneity/figures")

COL = {"u": "#3B6EA5", "v": "#C2703D", "emergent": "#7B5EA7", "new": "#B8860B", "muted": "#8A8A8A"}
DISP = lambda k: k.split("_", 1)[1] if "_" in k else k  # noqa: E731
N_PER_TASK = 3


def resp_item(m, pid):
    for r in json.loads((RESP / m / "responses.json").read_text()):
        if r["prompt_id"] == pid and r.get("items"):
            return r["items"][0]
    return None


def top_per_task():
    best_an, best_bl = {}, {}
    for f in SCORES.glob("*/path_scores.json"):
        m = f.parent.name
        for r in json.loads(f.read_text()):
            o = r.get("originality") or 0
            if (r["mode"] == "analogy" and r.get("pair_sat") and r.get("invention_integration")
                    and r.get("invention_utility")):
                if m not in best_an or o > best_an[m][0]:
                    best_an[m] = (o, r["prompt_id"], r.get("u_label"), r.get("v_label"))
            if (r["mode"] == "blending" and r.get("generic_ok") and r.get("blend_utility")
                    and r.get("blend_integration") == 3):
                if m not in best_bl or o > best_bl[m][0]:
                    best_bl[m] = (o, r["prompt_id"], r.get("u_label"), r.get("v_label"))
    an = sorted(best_an.items(), key=lambda kv: -kv[1][0])[:N_PER_TASK]
    bl = sorted(best_bl.items(), key=lambda kv: -kv[1][0])[:N_PER_TASK]
    return an, bl


def wrap(s, w):
    return textwrap.fill(s, w)


def card(ax, x, y, w, h, header, model, title, body_lines, accent):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.1, edgecolor="#D8D8D8", facecolor="#FCFCFC", zorder=1))
    ax.add_patch(plt.Rectangle((x, y + h - 0.004), w, 0.004, color=accent, zorder=2))  # accent bar
    pad = 0.018
    cy = y + h - 0.03
    ax.text(x + pad, cy, header, fontsize=12.5, color=COL["muted"], style="italic", va="top")
    ax.text(x + w - pad, cy, model, fontsize=11, color=COL["muted"], ha="right", va="top")
    cy -= 0.052
    ax.text(x + pad, cy, title, fontsize=17, color=COL["new"], fontweight="bold", va="top")
    cy -= 0.055
    for txt, color, style in body_lines:
        ax.text(x + pad, cy, txt, fontsize=12, color=color, style=style, va="top", family="serif")
        cy -= 0.030 * (txt.count("\n") + 1) + 0.012


def main():
    an, bl = top_per_task()
    fig, ax = plt.subplots(figsize=(15, 9.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # column headers
    ax.text(0.25, 0.985, "ANALOGY  —  invent by projecting structure", fontsize=17,
            fontweight="bold", color=COL["u"], ha="center", va="top")
    ax.text(0.75, 0.985, "BLENDING  —  fuse into a new concept", fontsize=17,
            fontweight="bold", color=COL["v"], ha="center", va="top")

    cw, ch = 0.46, 0.285
    x_an, x_bl = 0.02, 0.52
    y0, gap = 0.63, 0.31

    for i, (m, (o, pid, u, v)) in enumerate(an):
        it = resp_item(m, pid)
        proj = (it.get("projection") or [])[:3]
        body = []
        for p in proj:
            src = ", ".join(str(x) for x in (p.get("source") or []))
            img = ", ".join(str(x) for x in (p.get("image") or []))
            body.append((f"({src})", COL["u"], "normal"))
            body.append((f"     ⇒  ({img})", COL["new"], "normal"))
        card(ax, x_an, y0 - i * gap, cw, ch, f"{u}  ::  {v}", DISP(m),
             it.get("invention", ""), body, COL["u"])

    for i, (m, (o, pid, u, v)) in enumerate(bl):
        it = resp_item(m, pid)
        body = [(wrap(it.get("generic_space", ""), 52), COL["muted"], "italic")]
        tags = it.get("tags") or []
        for j, t in enumerate(it["paths"][0][:4]):
            tag = str(tags[j]) if j < len(tags) else "?"
            c = COL.get(tag, "#333333")
            trip = ", ".join(str(x) for x in t)
            body.append((f"({trip})  [{tag}]", c, "normal"))
        card(ax, x_bl, y0 - i * gap, cw, ch, f"{u}  +  {v}", DISP(m),
             it.get("concept", ""), body, COL["v"])

    fig.savefig(OUT / "fig_creativity_gallery.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_creativity_gallery.pdf", bbox_inches="tight")
    print("saved fig_creativity_gallery ->", OUT)


if __name__ == "__main__":
    main()
