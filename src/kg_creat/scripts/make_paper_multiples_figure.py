"""Assemble the paper's inventive-multiples figure assets from the report figures.

The paper keeps one flat image folder, media/figures/ (the old/new figure switch was removed on
2026-09-11). This writes the inventive-multiples assets there:

  media/figures/inventive_multiples.png          one stacked image: the model x property matrix
                                                 over the two-panel MDS landscape
  media/figures/inventive_multiples_matrix.pdf   panel (a) the matrix
  media/figures/inventive_multiples_mds_a.pdf    panel (b) the first landscape item
  media/figures/inventive_multiples_mds_b.pdf    panel (c) the second landscape item
  media/figures/multiples_examples.pdf           the one-row examples figure

None of these is referenced by the paper at the moment (the multiples figure was replaced by the
examples tables and the by-item failures grid); they are kept current so they can be re-included.

    .venv/bin/python -m src.kg_creat.scripts.make_paper_multiples_figure
"""
import shutil
from pathlib import Path

from PIL import Image

FIGS = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples/figures")
MEDIA = Path("papers/kg_creat-iclr/media/figures")
MATRIX, LAND = FIGS / "fig_multiples_matrix", FIGS / "fig_invention_landscape"
GAP = 40          # white gutter between the stacked halves, in px at the common width
BG = (255, 255, 255)


def stack(tops, out):
    ims = [Image.open(f).convert("RGB") for f in tops]
    w = max(im.width for im in ims)
    ims = [im if im.width == w else im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
           for im in ims]
    canvas = Image.new("RGB", (w, sum(im.height for im in ims) + GAP * (len(ims) - 1)), BG)
    y = 0
    for im in ims:
        canvas.paste(im, (0, y)); y += im.height + GAP
    canvas.save(out, dpi=(300, 300))
    return canvas.size


def main():
    need = [MATRIX.with_suffix(".png"), LAND.with_suffix(".png"), MATRIX.with_suffix(".pdf"),
            Path(f"{LAND}_a.pdf"), Path(f"{LAND}_b.pdf"), FIGS / "fig_multiples_examples_row.pdf"]
    for f in need:
        if not f.exists():
            raise FileNotFoundError(f"FATAL: {f} is missing -- regenerate it before assembling")
    MEDIA.mkdir(parents=True, exist_ok=True)
    stacked = MEDIA / "inventive_multiples.png"
    size = stack([MATRIX.with_suffix(".png"), LAND.with_suffix(".png")], stacked)
    print(f"wrote {stacked}  ({size[0]} x {size[1]})")
    for src, dst in ((MATRIX.with_suffix(".pdf"), "inventive_multiples_matrix.pdf"),
                     (Path(f"{LAND}_a.pdf"), "inventive_multiples_mds_a.pdf"),
                     (Path(f"{LAND}_b.pdf"), "inventive_multiples_mds_b.pdf"),
                     (Path(FIGS / "fig_multiples_examples_row.pdf"), "multiples_examples.pdf")):
        shutil.copyfile(src, MEDIA / dst)
        print(f"wrote {MEDIA / dst}")


if __name__ == "__main__":
    main()
