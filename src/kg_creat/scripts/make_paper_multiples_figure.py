"""Stack the two multiples figures into the single image the paper includes.

`media/inventive_multiples.png` is one figure in the paper (Fig. \\ref{fig:profiles}) but two figures
on disk: the model x property matrix on top, the MDS invention landscape below. That stacking used to
be a manual step outside the repo, so the paper's copy silently went stale whenever either half was
regenerated. This does it reproducibly: both halves are scaled to a common width and stacked.

    .venv/bin/python -m src.kg_creat.scripts.make_paper_multiples_figure
"""
from pathlib import Path

from PIL import Image

FIGS = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
TOP, BOTTOM = FIGS / "fig_multiples_matrix.png", FIGS / "fig_invention_landscape.png"
OUT = Path("papers/kg_creat-iclr/media/inventive_multiples.png")
GAP = 40          # white gutter between the halves, in px at the common width
BG = (255, 255, 255)


def main():
    for f in (TOP, BOTTOM):
        if not f.exists():
            raise FileNotFoundError(f"FATAL: {f} is missing -- regenerate it before stacking")
    ims = [Image.open(f).convert("RGB") for f in (TOP, BOTTOM)]
    w = max(im.width for im in ims)
    ims = [im if im.width == w else
           im.resize((w, round(im.height * w / im.width)), Image.LANCZOS) for im in ims]
    out = Image.new("RGB", (w, sum(im.height for im in ims) + GAP), BG)
    y = 0
    for im in ims:
        out.paste(im, (0, y)); y += im.height + GAP
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT, dpi=(300, 300))
    print(f"wrote {OUT}  ({out.width} x {out.height}) "
          f"= {TOP.name} over {BOTTOM.name}")


if __name__ == "__main__":
    main()
