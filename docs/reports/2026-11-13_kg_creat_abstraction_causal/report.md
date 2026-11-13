# Abstraction bottleneck: a causal test (rebuttal S1; reviewers eN17, cGKy)

2026-11-13. The paper's claim is correlational: models find a valid generic space g only about 43% of the
time, but given one they write a coherent blend about 97% of the time. Here we intervene on g directly.

## Design

- **Models:** six, spanning frontier to small: claude-opus-5, gpt-5.2, deepseek-chat-v3-0324,
  llama-3.3-70b, qwen-2.5-72b, gpt-4o-mini. Their paper-run generic-space rates are 17–53%.
- **Items:** all 30 blending items.
- **Settings:** T = 0.9, one draw, the same settings as the paper.
- **Conditions:**
  - **control:** the original benchmark prompt, freshly re-sampled. Any gain must beat this baseline,
    since a model that failed once can succeed by chance on a second draw.
  - **hint:** the original prompt plus a *content-free* procedural hint: list each input's functional
    roles, find the ones that correspond, and abstract them.
  - **oracle:** step 1 is replaced by a supplied, panel-validated g, which the model is told to copy and
    build on. Each item's g is the same for every model: it comes from a model outside the test set and
    was unanimously accepted by the panel in the paper run (2/3 for one item, F25). Models copied it
    verbatim in 88% of answers.
- **Scoring:** the paper's three-judge panel (Haiku 4.5, GPT-5.4, o3), recomputed from per-judge votes.
  G = majority generic_ok, V = majority coherent, Q = median scope. "Success" = G ∧ V, the benchmark's
  definition of a valid blend.
- **Exclusions:** one cell (opus-5 on F24) had an API failure in some condition and is excluded from every
  comparison. Unparsed answers from successful calls count as failures.

## The three prompts, verbatim

All three conditions use the paper's full blending prompt: the definition of a genuine blend, the
"liquid franchise" and "cyborg" examples, the tagging rules and the output format. They differ only in
step 1 of "Build the blend in two moves", and the oracle also rewords the UTILITY reward line. Shown for
*Networks + Salt*.

**control** (the paper's prompt, unchanged):
> 1. GENERIC SPACE: name the shared schema both 'Networks' and 'Salt' instantiate -- the abstract structure that lets them fuse (for organism + machine: "a functional system whose parts can fail and be replaced"). Be specific; "both exist"/"both involve change" is vacuous, and a one-from-each conjunction does not count.

**hint**: the control text, plus:
> HINT for finding it: first list, for EACH input separately, its 3-4 core functional roles (what acts, on what, by what means, toward what end). Then look for roles that CORRESPOND across 'Networks' and 'Salt' (they play the same part in each). The generic space is the abstract structure those corresponding roles share -- state it so that it is true of BOTH inputs.

The hint is content-free: it gives a procedure for finding an abstraction, never the abstraction or any
fact about either input.

**oracle**: step 1 is replaced by:
> 1. GENERIC SPACE (GIVEN): this shared schema has already been identified, and both 'Networks' and 'Salt' instantiate it: "connected sites whose local occupancy controls long range transport". Use it as your generic space -- copy it VERBATIM into "generic_space" -- and build the blend on it.

and the UTILITY / ORIGINALITY reward lines become:
> - UTILITY: the blend is built on the given generic space, which BOTH 'Networks' and 'Salt' instantiate.

All 30 supplied generic spaces are listed in `data/kg_creat/abstraction_causal/oracle_g.json`.

## Results

**Primary contrast: items each model failed in the paper run** (n = 116 model × item cells; exact McNemar,
paired).

| Condition | Full blend success (G ∧ V) | vs control |
|---|---|---|
| control (re-sample) | 17.2% | — |
| hint (content-free) | 22.4% | +15 / −9, p = 0.31 |
| **oracle (g supplied)** | **65.5%** | **+59 / −3, p = 1.7e-14** |

On all items (n = 179), the figures are 30.2% / 33.0% / 70.9%, with oracle vs control +80 / −7,
p = 8.3e-17. **All six models improve under the oracle**:

| Model | Paper G | Control | Hint | Oracle |
|---|---|---|---|---|
| claude-opus-5 | 50.0 | 48.3 | 55.2 | 86.2 |
| gpt-5.2 | 53.3 | 36.7 | 43.3 | 80.0 |
| deepseek-chat-v3-0324 | 26.7 | 23.3 | 30.0 | 83.3 |
| llama-3.3-70b | 33.3 | 30.0 | 23.3 | 63.3 |
| qwen-2.5-72b | 33.3 | 26.7 | 30.0 | 60.0 |
| gpt-4o-mini | 16.7 | 16.7 | 16.7 | 53.3 |

**Elaboration is intact once an abstraction is in hand.** Previously failed items, n = 116:

| Condition | Coherent V | Double-scope Q ≥ 2 | Q = 3 | V given accepted g | Q ≥ 2 given accepted g |
|---|---|---|---|---|---|
| control | 80.2 | 37.1 | 22.4 | 90.9 (n = 22) | 72.7 |
| hint | 78.4 | 32.8 | 16.4 | 96.3 (n = 27) | 70.4 |
| oracle | **94.0** | **59.5** | **31.0** | 98.7 (n = 77) | 72.7 |

Given an accepted g, coherence (91–99%) and double-scope integration (70–73%) are the same whether the
model found g itself or was given it, and they match the paper run (V | self-found g = 96.9% for these
models). Supplying g does not make inventions less original: emergent originality among coherent blends is
0.525 (oracle) vs 0.526 (control), on a shared pool.

## Reading

- **The failure is in finding the abstraction.** On blends a model previously failed, supplying the shared
  schema nearly quadruples full success (17% → 66%), while re-sampling alone does not. Once a valid
  abstraction is in hand, models elaborate it as well as they do their own. This is the abstraction
  bottleneck, now shown by intervention rather than inferred from conditional rates.
- **A content-free procedural hint does not help** (+5 pp, n.s.). The step that fails is not a matter of
  following a procedure for abstraction. This is consistent with, but does not establish, a
  knowledge-based account such as the paper's experiential-knowledge hypothesis, which should stay
  labelled as a hypothesis.
- **The oracle effect is, if anything, understated.** Success under the oracle also requires the panel to
  re-accept the supplied g. Even for verbatim copies, it does so only 72% of the time: g strings the panel
  accepted in the paper run are judged afresh alongside a new blend. Four items (F7, F10, F20, F27) are
  mostly rejected on re-judging.

## Qualitative examples

The three examples are the same model on the same item under all three conditions, chosen from the 59
cells that failed under both control and hint but succeeded under the oracle. Tags are the model's own.
G / V are panel majorities; "scope" lists each judge's Q. Judge quotes are GPT-5.4's explanation,
truncated.

The pattern is the same throughout. Without the oracle, the model writes a **coherent blend** (V = yes in
every case below) **on an abstraction that fits only one input**, and is failed on the generic space. Given
a valid abstraction, the same model elaborates it into a double-scope blend.

### 1. claude-opus-5 — *Photosynthesis + Bread* (the item the paper uses to illustrate the bottleneck)

| Condition | Generic space | Concept | G | V | scope |
|---|---|---|---|---|---|
| control | "a gas-trading process that turns air into a solid starch structure" | photosynthetic loaf | ✗ | ✓ | 1, 1, 1 |
| hint | "a domesticated microbe turning cheap carbon into stored edible sugar while venting a gas that has to go somewhere" | solar sourdough | ✗ | ✓ | 1, 3, 3 |
| oracle | *(given)* "a process that transforms simple inputs into a stable, storable energy source using catalysis" | sunlight loaf | ✓ | ✓ | 2, 1, 3 |

- **Control judge:** "bread does not instantiate a process of 'turning air into a solid starch structure';
  breadmaking mainly transforms existing flour dough, with gas used for leavening rather than as the
  material source of the loaf."
- **Hint judge:** "bread/yeast fermentation does not fix cheap carbon into sugar, and bread is not itself a
  domesticated microbe … partly imported from photosynthesis and partly from bread."

The hint run follows the procedure (its abstraction is built from functional roles), but the roles it
abstracts come from photosynthesis. Oracle blend:

- (sunlight loaf | stores energy as | starch) [uv]
- (sunlight loaf | recycles | carbon dioxide) [uv]
- (sunlight loaf | catalyzed by | RuBisCO) [u]
- (sunlight loaf | proofed in | sourdough starter) [v]
- (sunlight loaf | re-rises in | sunlight) [emergent]
- (sunlight loaf | stales only in | darkness) [emergent]

(GPT-5.4 itself still calls this generic space "only weakly shared", but the panel majority accepts it.)

### 2. gpt-4o-mini — *Democracy + Banking* (a weak model)

| Condition | Generic space | Concept | G | V | scope |
|---|---|---|---|---|---|
| control | "a financial system that allocates resources based on collective input" | democratic banking | ✗ | ✓ | 1, 1, 1 |
| hint | "a participatory system that allocates resources based on collective input" | Democratic Banking | ✗ | ✓ | 2, 1, 1 |
| oracle | *(given)* "a system converting personal endowments into collective claims through trusted agents" | democratic banking | ✓ | ✓ | 2, 3, 2 |

- **Control judge:** "Democracy is not itself 'a financial system,' and Banking is not inherently based on
  collective input, so this is largely a one-from-each conjunction."
- **Effect of the hint:** it changes one word ("financial" → "participatory"), and the abstraction still
  describes only democracy.
- **The oracle blend** has a fused slot the judge credits: (democratic banking | allocates | voting shares)
  [uv]. "Democracy contributes the voting/participation frame and banking contributes the
  shares/financial-allocation frame, yielding a single allocative property rather than two separate ones."

### 3. gpt-5.2 — *Constellations + Cricket* (a frontier model)

| Condition | Generic space | Concept | G | V | scope |
|---|---|---|---|---|---|
| control | "a rule system that turns point positions into named patterns with value" | Star scorecard | ✗ | ✓ | 1, 1, 1 |
| hint | "a named formation of discrete points used to coordinate action and timing" | celestial eleven | ✗ | ✓ | 1, 3, 3 |
| oracle | *(given)* "discrete points arrayed across a bounded field forming a named pattern that guides action" | astral cricket | ✓ | ✓ | 3, 3, 3 |

- **Control judge:** "constellations … do not inherently form a 'rule system' with scored value; cricket
  has scoring rules, but it does not itself turn point positions into named patterns."
- **The hint brings the model close.** Its abstraction is near the oracle's, but is still judged "a
  constructed one-from-each abstraction".
- **The oracle blend is unanimously double-scope emergent:**
  - (astral cricket | arrays | fielder stars) [uv]
  - (astral cricket | names patterns as | Orion setting) [uv]
  - (astral cricket | projects | score constellations) [emergent]
  - (astral cricket | creates | dark lanes) [emergent]

### When the panel rejects a supplied generic space

For *Christianity + Beauty* (F27), the supplied generic space was unanimously accepted in the paper run:
"a system in which inexhaustible excess restructures the perceiver's desire into sustained pursuit beyond
any given form". Re-judged under the oracle, it passes the majority for only 1 of 6 models, for example:
- GPT-5.4: "too theory-laden and narrow to count as a clearly shared schema of Christianity and Beauty in general";
- Haiku 4.5: "reads as an imposed overlay rather than a real structural homology".

This is judge variance on hard, abstract schemas, not model failure. It is why oracle success (65.5%)
understates how much supplying the abstraction helps. Coherence under the oracle, which does not depend on
re-accepting the given schema, is 94.0%.

## Cost and reproduction

- **Cost:** about $27 of OpenRouter key budget for elicitation and judging (key remaining: $41.41). The
  judge ledger records $56.2 (o3 $30.45, GPT-5.4 $20.86, Haiku $4.92).
- **Model substitutions:** qwen3-max was deprecated on OpenRouter (Oct 9); phi-4 parsed only 5–10 of 30
  answers; gemini-2.5-flash was refused by the provider (403). They were replaced by deepseek-chat-v3-0324
  and qwen-2.5-72b.

```
.venv/bin/python -m src.kg_creat.scripts.build_abstraction_causal --seed 0
.venv/bin/python src/kg_creat/scripts/run_elicit.py configs/kg_creat/abstraction_causal/elicit_{control,hint,oracle}.yaml
# score_input/ = symlinks <condition>__<model> -> each condition's responses/<model>
.venv_mlx/bin/python src/kg_creat/scripts/score.py configs/kg_creat/abstraction_causal/score.yaml
.venv_mlx/bin/python -m src.kg_creat.scripts.rescore_split_originality data/kg_creat/abstraction_causal/scores data/kg_creat/abstraction_causal/score_input
.venv/bin/python -m src.kg_creat.scripts.analyze_abstraction_causal
```

Prompt change: `src/kg_creat/prompts.py` takes optional `blend_condition` / `given_generic_space`. Default
prompts are byte-identical to before for all 90 benchmark specs (checked).
