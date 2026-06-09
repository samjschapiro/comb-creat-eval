# PD gold set — public-domain short stories with plot twists

Curated human-twist corpus for **H5** (human vs LLM) and the appendix table. All
entries are **US public domain** (published before 1930, or author died well
before the life+70 cutoff). These are the *ceiling / upper-bound* condition: they
are strong human twists **but heavily present in pretraining**, so an LLM judge may
reward familiarity — the matched-prompt WritingPrompts arm is the fair comparison
(see [experiments.md](experiments.md), Exp 2).

Candidate titles were drawn in part from curated lists of twist-ending short stories,
including the [Short Story Guide](https://www.shortstoryguide.com/short-stories-with-twists-endings-that-might-surprise-you/)
(cited in the paper as `shortstoryguide`), then filtered to genuine reinterpretation twists
that are US public domain.

Twist-type tags: **R** = reinterpretation (the reveal re-reads earlier events — the
transformational case we care about most); **I** = ironic reversal; **H** = horror/
supernatural sting. Source = Project Gutenberg (mostly within the author's collected
works; exact eBook IDs to be pinned by the fetch script).

| # | Title | Author | Year | The twist (one line) | Type |
|---|-------|--------|------|----------------------|------|
| 1 | The Necklace | Maupassant | 1884 | The diamond necklace she lost and spent ten years' poverty repaying was a worthless paste fake. | R |
| 2 | The Jewelry (The False Gems) | Maupassant | 1883 | A widower finds his dead wife's "paste" jewels are real — proof she had been kept by another man. | R |
| 3 | A Piece of String | Maupassant | 1883 | Seen pocketing a bit of string, a peasant is accused of stealing a wallet; cleared but never believed, he dies protesting. | I |
| 4 | The Gift of the Magi | O. Henry | 1905 | She sold her hair to buy his watch-chain; he sold his watch to buy her hair-combs. | R |
| 5 | The Last Leaf | O. Henry | 1907 | The ivy leaf that never falls — keeping the sick girl alive — was painted on the wall by old Behrman, who died of pneumonia painting it. | R |
| 6 | The Ransom of Red Chief | O. Henry | 1907 | The kidnapped boy is so unbearable the kidnappers end up paying his father to take him back. | I |
| 7 | The Cop and the Anthem | O. Henry | 1904 | Every attempt to get jailed for winter shelter fails; the instant he resolves to reform, he is arrested for loitering. | I |
| 8 | After Twenty Years | O. Henry | 1906 | The man waiting for his old friend is met by a cop; the friend *is* the cop, who recognizes him as wanted and has a colleague arrest him. | R |
| 9 | Hearts and Hands | O. Henry | 1917 | A handcuffed man is taken for a marshal escorting a convict; in fact he is the convict, and the real marshal posed as prisoner to spare a lady's feelings. | R |
| 10 | A Retrieved Reformation | O. Henry | 1903 | A reformed safecracker betrays himself by cracking a vault to save a trapped child; the watching detective pretends not to know him. | R |
| 11 | Witches' Loaves | O. Henry | 1911 | A baker secretly slips butter into a poor man's stale bread as a kindness — ruining the architectural drawing he used the bread to erase. | I |
| 12 | The Open Window | Saki | 1914 | A girl's tale of ghostly hunters who never returned is pure invention; the "ghosts" are just the family back from shooting, and the guest bolts. | R |
| 13 | The Interlopers | Saki | 1919 | Two enemies pinned under a fallen tree finally reconcile and call for help — the shapes loping toward them are wolves. | I |
| 14 | Sredni Vashtar | Saki | 1911 | A neglected boy prays to a ferret kept as a secret god; it kills his tyrannical guardian. | H |
| 15 | Dusk | Saki | 1924 | A cynic who saw through a young con-man's hard-luck story is conned anyway — the soap that "disproved" it was the con-man's after all. | I |
| 16 | An Occurrence at Owl Creek Bridge | Bierce | 1890 | A condemned man's vivid escape and journey home is a hallucination in the instant before the noose breaks his neck. | R |
| 17 | The Boarded Window | Bierce | 1891 | A frontiersman boards up a window for years; the reveal is that the wife he "buried" had only been comatose and woke as a panther dragged her off. | H |
| 18 | A Horseman in the Sky | Bierce | 1889 | A Union sentry shoots a Confederate scout off a cliff — and it was his own father. | R |
| 19 | Chickamauga | Bierce | 1889 | A child plays at soldiers among real corpses; the burning house is his own and the dead woman his mother. He is a deaf-mute who heard none of it. | R |
| 20 | The Story of an Hour | Chopin | 1894 | A wife quietly exults in freedom at news of her husband's death; he walks in alive and she drops dead — "of joy," the doctors say. | R |
| 21 | Désirée's Baby | Chopin | 1893 | A husband casts out wife and child over the baby's mixed race; a letter reveals it is *he* who is of black descent. | R |
| 22 | The Bet | Chekhov | 1889 | A lawyer endures 15 years' solitary confinement to win two million — then renounces the money and walks out hours before the term ends. | R |
| 23 | The Lottery Ticket | Chekhov | 1887 | A couple's fantasy of winning curdles into mutual loathing as each imagines spending it alone; the number loses. | I |
| 24 | How Much Land Does a Man Need? | Tolstoy | 1886 | Promised all the land he can walk around in a day, a peasant overreaches, drops dead, and is buried in a grave six feet long. | I |
| 25 | The Country of the Blind | H. G. Wells | 1904 | A sighted man enters a valley of the blind expecting to rule; they deem sight a disease and prepare to remove his eyes. | R |
| 26 | The Monkey's Paw | W. W. Jacobs | 1902 | The wished-for £200 arrives as death-compensation for their son; the next wish brings him back as a horror, and the last dismisses whatever knocks at the door. | H |
| 27 | Haircut | Ring Lardner | 1925 | A chatty barber recounts a dead prankster's "jokes"; the reader sees the cruel man was murdered in a staged "accident" — the narrator suspects nothing. | R |
| 28 | A Jury of Her Peers | Susan Glaspell | 1917 | As men hunt for a motive, two women find a strangled canary, grasp the abused wife's reason to kill, and quietly hide the evidence. | R |
| 29 | The Most Dangerous Game | Richard Connell | 1924 | A big-game hunter becomes the quarry of a man who hunts people; the last line reveals the hunted has turned and killed his hunter. | I |
| 30 | The Man That Corrupted Hadleyburg | Mark Twain | 1899 | A town renowned for honesty is exposed as venal when a stranger's sack of "gold" tempts its leading citizens into public fraud. | R |
| 31 | Luck | Mark Twain | 1891 | A celebrated military hero is revealed to be a blundering fool whose every catastrophic mistake happened, by sheer luck, to succeed. | R |
| 32 | Mateo Falcone | Mérimée | 1829 | A Corsican father, learning his young son took a bribe to betray a fugitive, executes the boy himself to keep the family's honor. | R |
| 33 | Markheim | R. L. Stevenson | 1885 | After a murder, a stranger (the devil, or his conscience) tempts Markheim; the twist is his sudden choice to confess, and the visitor's face transfigures. | R |
| 34 | The Signal-Man | Dickens | 1866 | A railway signalman is haunted by a spectre whose cries foretell disaster; the reveal is that the phantom's words were the warning of the signalman's *own* death. | R |
| 35 | A Terribly Strange Bed | Wilkie Collins | 1852 | A gambler who wins big at a shady house wakes to find the canopy of his four-poster bed silently descending to smother him in his sleep. | H |
| 36 | The Three Strangers | Thomas Hardy | 1883 | At a christening party, the casual stranger sharing the hearth with the hangman turns out to be the very man condemned to hang — escaping in plain sight. | R |
| 37 | The Cask of Amontillado | Poe | 1846 | The genial narrator lures a "friend" into the vaults and walls him up alive; the sting is the fifty years of unpunished, premeditated calm. | R |
| 38 | War | Pirandello | 1918 | A father loudly insists sons who die for country are not to be mourned; asked if his boy is "really gone," he collapses into sobs. | R |

**Excluded but worth noting (why):**
- *The Lady, or the Tiger?* (Stockton, 1882) — famously *no* resolution; an open question, not a twist.
- *Roman Fever* (Wharton, 1934) — superb twist, but **not** PD until 2030.
- *Lamb to the Slaughter* (Dahl), *The Lottery* (Jackson), *La noche boca arriba* (Cortázar) — canonical twists but **in copyright**; use only for illustration, never the dataset.

## Fetch methodology

Reproducible pipeline in [src/plot_twist/scripts/fetch_pd_stories.py](../../../src/plot_twist/scripts/fetch_pd_stories.py),
driven by [configs/plot_twist/pd_manifest.json](../../../configs/plot_twist/pd_manifest.json).

1. **Source.** Project Gutenberg plain-text cache, `https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt`
   (never the HTML ebook pages — PG blocks scraping those). Each story maps to a
   collection-volume eBook id, pinned by probing the title line of candidate ids
   directly on PG (Gutendex was unreachable from the build environment).
2. **Strip** the PG header/footer between the `*** START OF ... ***` / `*** END OF ... ***` markers.
3. **Extract** the single story: find its title as a heading (exact normalized match,
   else a heading-like line whose tokens superset the title — handles "The False Gems"
   for *The Jewelry*, "No. 1 Branch Line: The Signal-Man" for *The Signal-Man*); capture
   to the next boundary = next table-of-contents/sibling title **or** a generic ALL-CAPS
   isolated heading. A heading may sit under a section numeral (`IX`), not only a blank
   line; intra-story section markers (`I`/`II`/`III`) are ignored as boundaries.
   `start_heading`/`end_heading` in the manifest override the heuristic (used for
   *The Jewelry* and to stop *Hadleyburg* running past an internal all-caps inscription).
4. **Clean** whitespace; record `word_count` and `sha256` of the cleaned text so any
   later change is detectable.
5. **Output.** `data/plot_twist/human_twists/texts/<slug>.txt` (cleaned text) +
   `fetched_manifest.json` (per-story provenance: id, source_url, translation,
   extract_method, word_count, sha256). Raw collections cached under `_raw/`.

**Verification.** The set is small — eyeball every row once. A wrong id or bad heading
shows as a wildly off `word_count` or a preview that starts mid-story. Run
`... fetch_pd_stories.py <manifest> --dry-run` to review word counts/previews before writing.

**Status (2026-06-08).** **35 / 38 fetched and word-count-verified** (range 906--21,718;
all sane). eBook ids resolved via Gutenberg's OPDS search feed (`/ebooks/search.opds/`;
Gutendex was blocked). The anthology cases (*Luck* #60900 + `end_heading`; *Haircut* #77328;
*Jury* #20872; *Hearts and Hands* #2295 — not Sixes-and-Sevens) were resolved by extractor
fixes: strip `[n]` footnote markers, prefer ALL-CAPS body headings over Title-Case TOC/index
lines, allow trailing-period titles, and end at the first boundary. **3 remain unavailable**
on Project Gutenberg in clean PD text and need another source (or drop): Chopin *The Story of
an Hour* (uncollected in her PD-era books), Connell *The Most Dangerous Game* (1924, not on
PG), Pirandello *War* (the 1918 story; the PG Italian anthology #70578 predates it). Note:
*A Terribly Strange Bed* fetched at ~13.9k (likely captures the next After-Dark frame — add
an `end_heading` to trim). Translations: Maupassant (PG complete ed.) and Tolstoy (Maude) recorded;
Chekhov/Mérimée need a *PD* translation pinned.

**Caveats for use.**
- **Contamination:** every story here is in pretraining; treat as a ceiling, report alongside the matched-prompt arm.
- **Translations** (Maupassant, Chekhov, Tolstoy, Mérimée, Pirandello): the *work* is PD, but a specific modern English translation may be in copyright. Use a PD translation (e.g., Constance Garnett for Chekhov/Tolstoy; the Gutenberg translations) and record which one.
- **Length varies** widely (Hadleyburg is long; The Story of an Hour is ~1k words) — record word counts; length-match where the analysis needs it.
