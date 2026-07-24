# Constraints and creativity: what each constraint type does to a model's paths

**2026-07-22 · kg_creat track · Regime A · 8 models × 30 fixed endpoint bundles × 6 cells**

Creativity is novelty *and* utility — a remote artifact that is also fit for the requirement. In
this benchmark the constraints *are* the utility requirement: to be useful, a path from `u` to `v`
must be factual, well-formed, and satisfy a semantic/structural constraint. So the study has one
dependent variable, creativity `E[R·U]`, and one independent variable, constraint type. This report
walks the result almost entirely through concrete paths the models actually produced.

Every cell is administered on the **same 30 endpoint pairs**, so within a pair only the constraint
changes and every comparison below is a model differenced against itself.

---

## 1. One endpoint pair, all six cells

Here is Claude Sonnet 4.6 on the single pair **Japan → United Nations**, one path per cell. This is
the whole experiment in miniature — the endpoints never move; only the demand does.

| cell | constraint | path | verdict |
|---|---|---|---|
| baseline | *(none)* | japan —[represented by]→ Kōichirō Matsuura —[director-general of]→ UNESCO —[specialized agency of]→ UN | ✓ |
| categorical | through a kind of *international organization* | japan —[member of]→ WHO —[specialized agency of]→ UN | ✓ |
| exclusion | avoid *membership* relations | japan —[hosts headquarters of]→ UN University —[chartered by]→ UN | ✓ |
| inclusion (rare) | use an *international-relations* relation | japan —[hosts headquarters of]→ UN University —[chartered by]→ UN | ✓ |
| inclusion (common) | use a *location-or-origin* relation | japan —[joined]→ UN —[headquartered in]→ New York City | ✗ structural |
| ordering | *affiliation* before *membership* | japan —[hosts headquarters of]→ UN University —[subsidiary organ of]→ UN | ✗ constraint |

Two failures already visible in one pair: the inclusion path bolts the required relation on at the
end and in doing so **runs off the target** (`… → New York City`, never reaching the UN); the
ordering path is factual and reaches the UN but contains **neither** a affiliation nor a membership
relation to order. Both foreshadow the aggregate findings.

---

## 2. Creativity falls under every constraint, by very different amounts

![creativity by constraint type](figures/fig_creativity_by_constraint.png)

Each box is the 8 per-model paired effects (constrained − unconstrained, same endpoints); stars are
a one-sample t-test on those 8 values, Holm-corrected.

| cell | creativity `E[R·U]` | vs baseline | utility `U` | novelty of useful paths `R_valid` | sig. |
|---|---|---|---|---|---|
| baseline | 0.201 | — | 0.484 | 0.420 | |
| categorical | 0.168 | −16 % | 0.360 | 0.465 | n.s. |
| exclusion | 0.147 | −27 % | 0.326 | 0.458 | ** |
| inclusion (rare) | 0.121 | −40 % | 0.266 | 0.463 | ** |
| inclusion (common) | 0.109 | −46 % | 0.254 | 0.424 | ** |
| ordering | 0.028 | **−86 %** | 0.056 | **0.496** | *** |

The ordering column is a different regime: an 86 % collapse, significant at `***`, consistent across
all eight models. Categorical, at the other end, is the only cell whose effect the 8 models do not
agree on — mean negative but not distinguishable from zero.

---

## 3. Two ways a model fails a constraint

Restricting to **constraint-channel** failures (factual, well-formed paths that just miss the
requirement), two behaviours recur across every constraint type.

### (a) The minimal edit — change a word, not the path

The model keeps its default path and tweaks a single surface token, which does not move the
underlying relation into (or out of) the target class.

**Haiku 4.5 · UK → Paul McCartney · *avoid all participation relations***
```
✓ unconstrained:  united kingdom —[is home to]→ the beatles —[member]→ paul mccartney
✗ constrained:    united kingdom —[is home to]→ the beatles —[included member]→ paul mccartney
```
It edited the *second* hop (`member` → `included member`) and left the forbidden `is home to`
untouched — changing the one relation that was already fine.

**Sonnet 4.6 · Japan → UN · *use an international-relations relation***
```
✓ unconstrained:  japan —[contributes to]→ un peacekeeping ops —[operated by]→ united nations
✗ constrained:    japan —[contributes to]→ un peacekeeping ops —[administered by]→ united nations
```
`operated by` → `administered by`. One synonym swapped; the required class never appears.

**Haiku 4.5 · Germany → Australia Group · *use a location-or-origin relation***
```
✓ unconstrained:  germany —[is member of]→ nuclear suppliers group —[coordinates with]→ australia group
✗ constrained:    germany —[member of]→ nuclear suppliers group —[coordinates with]→ australia group
```
`is member of` → `member of`. Functionally the identical path; the constraint is simply ignored.

### (b) The clean rebuild — a different path that still doesn't encode the demand

The model *does* produce a new path, perfectly factual, that nonetheless never satisfies the
constraint. The sharpest version:

**Llama 3.3 70B · UK → Australia Group · *membership before participation***
```
✓ unconstrained:  uk —[member of]→ commonwealth —[includes]→ australia —[founding member]→ australia group
✗ constrained:    uk —[signed]→ wassenaar arrangement —[part of]→ export control regimes —[related to]→ australia group
```
The **baseline already satisfied the ordering**: `member of` (membership) precedes `includes`
(participation). Naming the requirement made the model throw the path away and build one whose three
relations are `signed` / `part of` (both membership) / `related to` (collaboration) — no participation
relation anywhere, so nothing to order. It abandoned a satisfying path for a failing one.

**Sonnet 4.6 · Brazil → Golden Globe Awards · *avoid all participation relations***
```
✓ unconstrained:  brazil (film) —[directed by]→ terry gilliam —[directed]→ the fisher king —[nominated for]→ golden globe awards
✗ constrained:    brazil (film) —[starred]→ jonathan pryce —[starred in]→ evita —[won]→ golden globe awards
```
A completely rebuilt cast-based path — and its final hop `won` is itself a participation relation, so
the rebuild reintroduced exactly the class it was told to avoid.

---

## 4. Constraints do not make models hallucinate — they defeat compliance

If constraints pushed models past the edge of their knowledge, we'd see the **factual** failure
channel swell under constraint. It does not: factual failures sit at a flat ~32–37 % in every cell,
*including the unconstrained baseline* (32.3 %). Hallucination is a roughly constant background tax, not a
constraint effect. The entire *added* cost of a constraint lands in the **constraint** channel.

![failure channels](figures/fig_regimeA_channels.png)

That baseline factuality floor is real and worth seeing — these are unconstrained paths the judge
rejected as hallucinated:

**Haiku 4.5, baseline:**
```
united states —[is home to]→ princeton university —[employed]→ albert einstein
   ✗ flagged: 'princeton university —employed→ albert einstein'
   (Einstein was at the Institute for Advanced Study in Princeton, not employed by the university)
```
```
united states —[produced]→ the beatles —[originated from]→ liverpool —[located in]→ united kingdom
   ✗ flagged: 'united states —produced→ the beatles'
```
**Llama 3.1 8B, baseline:**
```
united states —[founded]→ princeton university —[alumnus]→ albert einstein
   ✗ flagged both hops: the US did not found Princeton; Einstein was not an alumnus
```

None of these is a constraint failure — they are ordinary factual errors that occur at the same rate
whether or not a constraint is present. Under ordering, by contrast, **40.6 %** of all paths fail on
the constraint channel (vs 8–16 % for the other constraints), and those paths are typically
*factually fine*.

---

## 5. Ordering is not hard as *sequencing* — it is hard as *double inclusion*

Ordering costs ~2× any other constraint. But decomposing its 495 constraint-failures shows the
reason is not that models get the order backwards:

| what went wrong | share |
|---|---|
| the *after*-class relation is present but the *before*-class one is missing | 39.6 % |
| neither required class is present | 30.1 % |
| the *before*-class is present but the *after*-class is missing | 18.2 % |
| **both present, order genuinely inverted** | **11.5 %** |

Only ~1 in 9 failures is a real sequencing error. Overwhelmingly, models fail because they never get
**both** required relation classes into a single path — ordering is really a conjunction of two
inclusion constraints, and *that* is what's near-infeasible on a fixed pair of endpoints.

**Haiku 4.5 · US → UK · *participation before membership***
```
✓ unconstrained:  us —[home to]→ harvard —[academic partnership with]→ cambridge —[located in]→ uk
✗ constrained:    us —[signatory to]→ treaty of paris (1783) —[recognizes independence from]→ uk
   → contains neither a participation nor a membership relation; there is nothing to order
```

**Sonnet 4.6 · Japan → UN · *affiliation before membership***
```
✗ constrained:    japan —[hosts headquarters of]→ un university —[subsidiary organ of]→ united nations
   → factual and reaches the target, but has neither required class
```

When a model *does* thread both classes, the result is often the most novel path in the whole study —
which is why ordering has the highest `R_valid` (0.496) despite the lowest utility (0.056):

```
Haiku 4.5 · Colombia → Nuclear Suppliers Group  (R = 0.74, satisfied)
   colombia —[maintains diplomatic relations with]→ united states —[is member of]→ nuclear suppliers group
Haiku 4.5 · Japan → Australia Group  (R = 0.68, satisfied)
   japan —[cooperates with]→ united states —[founding member of]→ australia group
```
These succeed by routing through a hub (the US) that lets an international-relations hop precede a
membership hop. Remote and valid — but models find this arrangement only ~6 % of the time.

---

## 6. Categorical is the one constraint that can *raise* creativity

Categorical is the only cell where naming the constraint sometimes produces a *more* creative path
than the model's own unconstrained answer — and the effect is real enough to flip 2 of 8 models
(Sonnet 4.6, GPT-4.1-mini) above their baseline. Being told to route through a *type of entity*
redirects the search without restricting the relations used to build the path.

**Sonnet 4.6 · US → UK · *pass through a kind of 'human'***
```
baseline (R 0.38):   us —[founded by]→ george washington —[fought against]→ british army —[serves]→ uk
categorical (R 0.58): us —[birthplace of]→ sylvia plath —[married]→ ted hughes —[poet laureate of]→ uk
```
The type requirement pulled the model off the obvious founding-war route and onto a literary one —
Plath (American) married Hughes (UK poet laureate). More novel *and* satisfying: creativity up.

**Sonnet 4.6 · US → Albert Einstein · *through a kind of 'social state'***
```
baseline:     us —[hosted institution]→ institute for advanced study —[employed]→ albert einstein
categorical:  us —[recognized state]→ israel —[offered presidency to]→ albert einstein
```
The unusual fact that Israel offered Einstein its presidency surfaces *because* the type constraint
forced an intermediate that the default path had no reason to visit.

**GPT-4.1-mini · Japan → UN · *through an international organization***
```
baseline (R 0.27):     japan —[hosted]→ un university —[operates under]→ un
categorical (R 0.48):  japan —[member of]→ OECD —[observer at]→ un general assembly —[main organ of]→ un
```

Contrast this with what the *relation-class* constraints do (§3): those restrict the vocabulary the
model builds with, and models respond by minimally editing or abandoning good paths. Categorical
constrains the *waypoint* and leaves the machinery free — which is why it is the one constraint that
occasionally helps rather than hurts.

---

## 7. Caveats, with examples

**Judge borderline cases concentrate in the categorical cell** — which is exactly why the owed
human reliability pass matters most there:
```
Sonnet 4.6 · Brazil → Hitchcock · through a 'superpower'
   brazil —[diplomatic relations with]→ france —[awarded légion d'honneur to]→ alfred hitchcock
   → Is France a "superpower"? The verdict decides the cell.
Gemini 2.5 Flash · Germany → UN · through an 'international organization'
   germany —[host country for]→ un campus bonn —[site of]→ un volunteers —[program of]→ un
   → 'UN Volunteers' is a UN programme; rejecting it is defensible but not obvious.
```

**Exemplar noise.** Constraints show four data-derived exemplars, and some do not fit their cluster's
name: the *international-relations* class was presented with `"country"` as an exemplar,
*location-or-origin* with `"established"`, *membership* with `"signed"`. This is grading-consistent
(the judge sees the same class name and exemplars the model saw), but it means a cell's difficulty
partly reflects how coherent its cluster happened to be.

**Ambiguous endpoints weaken the pairing for ~3 of 30 bundles.** Because we strip disambiguating
parentheticals, different *senses* of a label can both count as the endpoint. In bundle A11 models
resolved "Brazil" as both the country and the 1985 Gilliam film across cells:
```
Sonnet baseline:    brazil (film) —[directed by]→ terry gilliam …
Sonnet categorical: brazil —[diplomatic relations with]→ france …   (the country)
```
For those bundles "same endpoints" does not strictly hold. It adds noise, not directional bias.

**A scorer bug the examples caught.** An earlier `_entity_matches` used bidirectional substring
matching, so a path ending at `australia group export controls` counted as reaching `Australia
Group`. 6.4 % of well-formed paths had inexact endpoints, ~81 % genuinely the wrong entity. Fixed to
require equality up to a trailing parenthetical; re-deriving offline moved 313 paths (4.4 %) and
changed no conclusion (ranking identical, categorical still 2/8, deltas ≤ 0.009). All numbers here
are the strict re-derivation.

---

## Summary

- **Creativity falls under every constraint**, from −16 % (categorical, n.s.) to −86 % (ordering,
  `***`), consistently across 8 models.
- **The cost is compliance, not knowledge.** Factuality is a flat ~32–37 % background tax; the added
  cost of a constraint is entirely in the constraint channel.
- **Models fail constraints two ways**: minimal surface edits that don't move the underlying relation,
  and clean rebuilds that never encode the demand — sometimes abandoning a baseline path that already
  satisfied it.
- **Ordering is a disguised conjunction**: 88 % of its failures never get both classes into the path;
  only 11.5 % are true order inversions.
- **Categorical can raise creativity** (2/8 models) because it constrains the waypoint, not the
  relation vocabulary — the Sylvia-Plath route is more novel *and* valid than the model's default.

Cost of the run: ~$6.6. Owed: human judge-reliability pass (load-bearing for the categorical cell),
and running the reframed single-stimulus blending task at scale.
