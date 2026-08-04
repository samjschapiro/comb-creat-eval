# Author headshots

Displayed as circular avatars in the page header, cropped with `object-fit: cover` at 84px.
The page probes for `static/authors/<slug>.{jpg,jpeg,png,webp}` and falls back to the
author's initials if no file is there, so a missing photo never renders as a broken image.
Slugs come from `data-photo` on each `.avatar` in `index.html`.

## Provenance

Each image was taken from the author's own site or their official university directory —
not from a third-party aggregator — then centre-cropped square (biased toward the top so
heads are not cut off) and resized to 320x320.

| File | Source | Retrieved |
|---|---|---|
| `samuel-schapiro.jpg` | <https://schapiro.ai/files/images/new_headshot.jpg> (personal site) | 2026-08-04 |
| `alexi-gladstone.jpg` | <https://alexiglad.github.io/assets/img/prof_pic.png> (personal site) | 2026-08-04 |
| `heng-ji.jpg` | Siebel School faculty directory, <https://siebelschool.illinois.edu/about/people/faculty/hengji> | 2026-08-04 |
| `roger-beaty.jpg` | Penn State Psychology faculty page, <https://psych.la.psu.edu/people/rub736/> | 2026-08-04 |

These are photographs of real people on a public page. The two university headshots are
their institutions' copyright and appear here to identify the paper's authors. Replace or
remove any image on request from the person pictured.

## Replacing one

Drop a new file in with the same slug and redeploy (`bash scripts/plot_twist/deploy_site.sh`).
Square images around 256px or larger stay sharp on retina screens.
