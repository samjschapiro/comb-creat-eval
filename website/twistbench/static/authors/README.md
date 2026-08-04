# Author headshots

Drop one image per author here and redeploy — the page picks them up automatically and
crops them to a circle. Until a file exists, that author shows their initials instead.

Expected filenames (any of `.jpg`, `.jpeg`, `.png`, `.webp`):

    samuel-schapiro.jpg
    alexi-gladstone.jpg
    heng-ji.jpg
    roger-beaty.jpg

Square images work best — they are cropped with `object-fit: cover` and displayed at 84px,
so roughly 256x256 or larger keeps them sharp on retina screens.

The slugs come from `data-photo` on each `.avatar` in `index.html`; changing the author
list means changing both.

**Only add photos you have the right to publish and that the person has agreed to.** This
page is public, so a headshot here is a photo of a real person republished on the open web.
