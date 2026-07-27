# 2026-07-27 · CDAT gate: fixed BH NaN-fragility bug (gated CDAT was T=1.5-only)

**Bug.** `benjamini_hochberg()` in `src/dat_eval/scripts/cdat_gate.py` was NaN-fragile: a single
NaN p-value (degenerate Welch t-test on some model at a temperature) poisoned every adjusted
p-value to NaN (np.argsort sorts NaN last; the monotonicity `np.minimum.accumulate` propagates it
backward). At T=1.0 and T=2.0 the FDR gate returned all-NaN, so 0/55 models passed there (despite
raw p ~1e-26 and appropriateness above the random baseline); only T=1.5 was clean (46/55). Since
gated CDAT = mean novelty over passing temperatures, the paper's CDAT was effectively computed
from T=1.5 alone.

**Fix.** BH now corrects over non-NaN p-values only, returns NaN for NaN inputs. Re-ran the gate.

**After fix (sbert):** T=1.0 65 pass, T=1.5 59, T=2.0 49; passed_temps {all-three:49, none:11,
T1.0-only:6, T1.0+1.5:10} — the gate now discriminates (T=2.0 fails more).

**Impact (gated CDAT, Overall composite), buggy -> fixed:** Arena +0.07/+0.21; EQ +0.17/+0.09;
Mazur +0.26/+0.40; Hivemind +0.23/+0.10; NoveltyBench +0.34/+0.20 -> +0.14/+0.03;
LiveIdeaBench +0.02/+0.34 -> -0.12/-0.08. Only gated CDAT changes. NoveltyBench specificity
trajectory: +0.60 (n=8) -> +0.20 (expanded, buggy) -> +0.03 (fixed). Weakens "CDAT best predicts
divergent thinking"; Table 1 CDAT row + both rebuttals updated.

Regenerate gitignored data via `PYTHONPATH=. python src/dat_eval/scripts/cdat_gate.py`.
