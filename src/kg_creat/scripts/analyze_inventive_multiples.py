"""Inventive multiples: how often do two independent models invent the same entity for the same anchors
(u,v), and what predicts it?

A multiple is two models RE-USING THE SAME PROPERTIES. The coined name never enters the calculation:
every triple is reduced to "relation object" (the invention's name is the subject of all of them and is
dropped), so what is compared is the properties asserted of the invention, not what it was called.

For a pair of inventions (same task, same anchor pair):

  SHARED PROPERTIES -- greedily match each property of one against an unused property of the other,
                       one-to-one; a pair of properties counts as the same when their "relation
                       object" texts are within COS_SLOT (theta).
  TAU-INVENTIVE     -- at least TAU shared properties. That is the whole criterion: property overlap
  MULTIPLE             and nothing else, matching the paper's Definition (tau-Inventive Multiples).

Two exclusions apply BEFORE matching, both because the excluded text is supplied by the item rather
than invented:
  * the invention's own name (the subject of every triple);
  * ANCHOR ECHO -- any property whose object IS one of the two anchors. `(X, resembles, Pi)` on the
    item (Don Quixote, Pi) is not something two models converged on; 8.7% of analogy properties are
    of this kind against 0.3% of blend properties, so leaving them in inflates analogy convergence
    with item-supplied matches. The test is exact match on the normalised object, not word overlap:
    `emits light` on (Mount Everest, The light bulb) is a real property and must survive. The
    unfiltered rates are still computed and reported (`anchor_echo`) as a diagnostic.

There is deliberately NO abstraction clause. An earlier version also required the underlying concept
(the projected source for analogy, the generic space for blending) to align at cosine >= COS_CON, but
conditioning on a shared generic space is a second, different claim. COS_CON is still computed and
reported as a descriptive diagnostic, never as an input -- exactly like the coined name.

TAU IS THE REPORTED AXIS, not a hidden constant. `tau_curve` gives the multiple rate for every tau the
data supports; the module-level TAU only says which point the prose quotes.

THETA (COS_SLOT) IS CALIBRATED against a cross-item null (see calibrate_theta.py): inventions on
different anchor pairs cannot share an item-specific property, so their greedy matches are a null,
and theta is the cosine that alpha = 0.25% of those null matches clear. `sensitivity` still sweeps it.

A FIXED TAU IS NOT A FAIR OPERATOR COMPARISON. Blends assert ~5 properties and analogy inventions
~2.6 (15% of analogy inventions carry only one and can never be a 2-multiple), and the chance of
sharing >= 2 grows superlinearly with how many there are to share. So the blend/analogy ratio at a
fixed tau mixes convergence with property count. The analysis therefore reports the task effect by
several routes that do not depend on property count, and quotes those as the operator comparison:
  * ELIGIBLE      -- the tau-multiple rate among pairs where both inventions carry >= tau properties;
  * PER PROPERTY  -- shared / min(k, k'): of the properties that could have been re-used, the share
                     that were (mean per task, paired per item);
  * NULL-CORRECTED-- the same, minus what unrelated inventions of the same task produce by chance;
  * EXACT MATCH   -- encoder-free: the share of the smaller invention's objects that appear verbatim
                     among the other's, after the anchor filter, minus its own null;
  * SIZE-MATCHED  -- the tau-multiple rate within strata of min(k, k').

NOMINAL -- the coined names matching -- is computed but is NEVER an input. It is the independent check:
same-name pairs re-use far more properties than the rest, yet only a minority of them qualify, so
name convergence and property convergence are separate phenomena rather than one measurement.

Judge-free; local MLX embeddings.

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_inventive_multiples
    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_inventive_multiples --prepost
"""
import argparse
import glob
import itertools
import json
import re
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon

from src.kg_creat.embed import get_embedder

NPZ = "data/kg_creat/kombine_test30/analysis/invention_vectors.npz"
RESP = "data/kg_creat/kombine_test30/responses"
OUT = "data/kg_creat/kombine_test30/analysis/inventive_multiples.json"
COS_SLOT = 0.545  # "relation object" cosine at which two models count as asserting the same
                  # property. CALIBRATED, not chosen: two inventions answering DIFFERENT anchor
                  # pairs cannot share an item-specific property, so their greedy matches are a
                  # null. 0.545 is the null's 99.75th percentile -- alpha = 0.25%, one property
                  # pair in 400 from unrelated inventions clears it. See calibrate_theta.py.
                  # (The previous 0.58 had no derivation; it implied alpha = 0.15%.)
TAU = 2     # the headline tau: a tau-inventive multiple re-uses >= TAU of the other invention's
            # properties. The rate is reported as a FUNCTION of tau (see tau_curve); TAU only picks
            # which point on that curve the prose quotes.
COS_CON = 0.50   # NOT part of the criterion. Retained only to report how often multiples also
                 # happen to share an abstraction -- a descriptive diagnostic, like the coined name.
N_NULL = 60000   # cross-item pairs drawn for the null (same task, different anchors)
NULL_SEED = 0
_PROV = ["openai", "anthropic", "google", "x-ai", "deepseek", "qwen", "z-ai", "meta-llama"]


def _provider(m):
    """Provider of a model key, falling back to the key's own prefix. Keying this on _PROV alone
    returned None for any provider missing from that list, which then crashed `sorted()` on the
    cluster's provider set -- so a new provider silently broke the whole analysis."""
    return next((p for p in _PROV if str(m).startswith(p)), str(m).split("_", 1)[0] or "unknown")


def _nn(s):
    s = str(s).lower().strip()
    s = re.sub(r"^(the|a|an|our)\s+", "", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    s = re.sub(r"s\b", "", s)
    return " ".join(sorted(s.split()))


def _find(p, x):
    while p[x] != x:
        p[x] = p[p[x]]
        x = p[x]
    return x


def _pct(x, q):
    return float(np.percentile(np.asarray(x, float), q)) if len(x) else float("nan")


def load_records():
    """Per (task, u, v, model): the underlying abstraction (display form and the text that is embedded),
    the invented structure, relation labels.

    The abstraction is what the STRUCTURAL criterion compares, so it must be semantic content, not a
    label. For blending that is automatic -- the generic space `g` is a sentence-long schema. For
    analogy the field `projected` is only the source concept's NAME (e.g. "adjuvant"), so we embed it
    together with the source triples the model asserted for it: the structure it projected, not the
    word it used for it. Two models that pick the same-sounding source but map different structure
    then do not count as having reached the invention the same way.
    """
    con, con_emb, struct, rels = {}, {}, {}, {}
    for f in glob.glob(f"{RESP}/*/responses.json"):
        m = f.split("/")[-2]
        for r in json.load(open(f)):
            mode = r.get("mode")
            if mode not in ("analogy", "blending") or not r.get("items"):
                continue
            it = r["items"][0]
            k = (mode, r.get("u_label"), r.get("v_label"), m)
            con[k] = ((it.get("projected") if mode == "analogy" else it.get("generic_space")) or "").strip()
            if mode == "analogy":
                src = " ; ".join(" ".join(str(x) for x in p["source"])
                                 for p in (it.get("projection") or []) if p.get("source"))
                con_emb[k] = (con[k] + " . " + src).strip(" .")
            else:
                con_emb[k] = con[k]
            if mode == "blending":
                tags = it.get("tags") or []
                tri = (it.get("paths") or [[]])[0]
                struct[k] = [list(t) + [tags[i] if i < len(tags) else ""] for i, t in enumerate(tri)]
                rels[k] = {str(t[1]).lower() for t in tri if len(t) > 1}
            else:
                pj = it.get("projection") or []
                struct[k] = [{"source": p.get("source"), "image": p.get("image")} for p in pj]
                rels[k] = {str(p["image"][1]).lower() for p in pj if p.get("image") and len(p["image"]) > 1}
    return con, con_emb, struct, rels


def calibrate(pairs_all, tasks_of):
    """How many properties pairs re-use, split by whether they also happen to share a name. The name
    is not part of the criterion; this is the check on it."""
    out = {}
    for scope in ("pooled", "analogy", "blending"):
        sel = [p for p in pairs_all if scope == "pooled" or p["task"] == scope]
        lex = [p["shared"] for p in sel if p["nominal"]]
        non = [p["shared"] for p in sel if not p["nominal"]]
        lexc = [p["cos_con"] for p in sel if p["nominal"] and np.isfinite(p["cos_con"])]
        nonc = [p["cos_con"] for p in sel if not p["nominal"] and np.isfinite(p["cos_con"])]
        out[scope] = {
            "n_pairs": len(sel), "n_lexical": len(lex),
            "shared_lexical_mean": float(np.mean(lex)) if lex else float("nan"),
            "shared_nonmatch_mean": float(np.mean(non)) if non else float("nan"),
            "shared_lexical_pct_ge_tau": float(np.mean(np.asarray(lex) >= TAU)) if lex else float("nan"),
            "shared_nonmatch_pct_ge_tau": float(np.mean(np.asarray(non) >= TAU)) if non else float("nan"),
            "con_lexical_mean": float(np.mean(lexc)) if lexc else float("nan"),
            "con_nonmatch_mean": float(np.mean(nonc)) if nonc else float("nan"),
            "con_frac_ge_cos_lexical": float(np.mean(np.asarray(lexc) >= COS_CON)) if lexc else float("nan"),
            "con_frac_ge_cos_nonmatch": float(np.mean(np.asarray(nonc) >= COS_CON)) if nonc else float("nan"),
        }
    print("\nCALIBRATION (properties re-used; the name is reported against the criterion, never in it)")
    for s, c in out.items():
        print(f"  {s:9s} same-name n={c['n_lexical']:4d} mean shared={c['shared_lexical_mean']:.2f} "
              f"({100*c['shared_lexical_pct_ge_tau']:.0f}% reach {TAU}) | other pairs "
              f"mean={c['shared_nonmatch_mean']:.2f} ({100*c['shared_nonmatch_pct_ge_tau']:.0f}%)")
    print(f"  abstraction >= {COS_CON}: same-name {100*out['pooled']['con_frac_ge_cos_lexical']:.0f}%"
          f" vs other pairs {100*out['pooled']['con_frac_ge_cos_nonmatch']:.0f}%")
    return out


def tau_curve(pairs_all, null, task_of_inv):
    """The headline as a function of tau: what fraction of co-response pairs are tau-inventive
    multiples, for every tau the data can support.

    A tau-inventive multiple is a pair re-using >= tau of each other's properties. tau = 1 is "one
    property in common"; raising tau makes the criterion strictly stricter, so the curve is monotone
    non-increasing by construction. Reporting the curve rather than a single tau is the point: the
    blending-vs-analogy gap should be visible at every tau if it is real, rather than being an
    artifact of where the bar was placed.

    Each row also carries, per task: the rate among ELIGIBLE pairs (both inventions have >= tau
    properties, so the pair can qualify at all), the share of INVENTIONS in at least one multiple,
    and the cross-item NULL rate with the excess over it -- so a reader can see how much of the
    blend/analogy gap at a fixed tau is property count rather than convergence.
    """
    bl = [p for p in pairs_all if p["task"] == "blending"]
    an = [p for p in pairs_all if p["task"] == "analogy"]
    same = [p for p in pairs_all if p["same_provider"]]
    cross = [p for p in pairs_all if not p["same_provider"]]
    n_inv = Counter(task_of_inv.values())
    tau_max = max((p["shared"] for p in pairs_all), default=0)
    rows = []
    for tau in range(1, int(tau_max) + 1):
        ok = lambda ps: [p for p in ps if p["shared"] >= tau]
        hit = ok(pairs_all)
        if not hit:
            break
        pct = lambda sub, tot: (100.0 * len(sub) / len(tot)) if tot else float("nan")
        touched = {i for p in hit for i in (p["a"], p["b"])}
        row = {"tau": tau, "n": len(hit), "overall_pct": pct(hit, pairs_all),
               "blending_pct": pct(ok(bl), bl), "analogy_pct": pct(ok(an), an),
               "same_provider_pct": pct(ok(same), same), "cross_provider_pct": pct(ok(cross), cross),
               "inventions_pct": 100.0 * len(touched) / len(task_of_inv)}
        for task, sub in (("blending", bl), ("analogy", an)):
            elig = [p for p in sub if p["kmin"] >= tau]
            row[f"{task}_eligible_pct"] = pct(ok(elig), elig)
            row[f"{task}_eligible_n"] = len(elig)
            row[f"inventions_{task}_pct"] = 100.0 * sum(1 for i in touched if task_of_inv[i] == task) / n_inv[task]
            nl = null[task]["rate_pct_by_tau"].get(str(tau), 0.0)
            row[f"null_{task}_pct"] = nl
            row[f"excess_{task}_pct"] = row[f"{task}_pct"] - nl
        rows.append(row)
    print("\nTAU-INVENTIVE MULTIPLES AS A FUNCTION OF TAU"
          f"  (properties matched one-to-one at cosine >= {COS_SLOT}; anchor echo removed)")
    print(f"  {'tau':>4}{'n pairs':>9}{'overall':>9}{'blend':>8}{'analogy':>9}{'bl/an':>7}"
          f"{'| elig bl':>10}{'elig an':>9}{'| null bl':>10}{'null an':>9}"
          f"{'| same':>8}{'cross':>8}{'| inv%':>7}{'inv bl':>8}{'inv an':>8}")
    print("  " + "-" * 128)
    for r in rows:
        ratio = (r["blending_pct"] / r["analogy_pct"]) if r["analogy_pct"] > 0 else float("inf")
        rs = f"{ratio:.1f}x" if np.isfinite(ratio) else "--"
        print(f"  {r['tau']:>4}{r['n']:>9}{r['overall_pct']:>8.2f}%{r['blending_pct']:>7.2f}%"
              f"{r['analogy_pct']:>8.2f}%{rs:>7}{r['blending_eligible_pct']:>9.2f}%"
              f"{r['analogy_eligible_pct']:>8.2f}%{r['null_blending_pct']:>9.2f}%{r['null_analogy_pct']:>8.2f}%"
              f"{r['same_provider_pct']:>7.2f}%{r['cross_provider_pct']:>7.2f}%{r['inventions_pct']:>6.1f}%"
              f"{r['inventions_blending_pct']:>7.1f}%{r['inventions_analogy_pct']:>7.1f}%")
    print(f"  tau = {TAU} is the value quoted in the prose.")
    return rows


def cross_item_null(SMAT, OBJS, tk, item_of, tau_max, rng):
    """Same task, DIFFERENT anchor pair: what two inventions share by chance. Two inventions answering
    different items cannot share an item-specific property, so their matches are what generic phrasing
    and property count alone produce. Returns, per task, the tau-multiple rate at every tau, the mean
    per-property re-use and the exact-match per-property re-use -- each the quantity the observed
    co-response pairs are compared against. Blends carry more properties, so their null is higher."""
    by_task = defaultdict(list)
    for i in range(len(tk)):
        by_task[str(tk[i])].append(i)
    out = {}
    for task, pool in sorted(by_task.items()):
        sh, fr, exf, draws = [], [], [], 0
        while draws < N_NULL // 2:
            i, j = pool[rng.integers(len(pool))], pool[rng.integers(len(pool))]
            if i == j or item_of[i] == item_of[j]:
                continue
            draws += 1
            kmin = min(len(SMAT[i]), len(SMAT[j]))
            s_ = shared_properties(SMAT[i], SMAT[j])
            sh.append(s_)
            if kmin:
                fr.append(s_ / kmin)
                exf.append(exact_shared(OBJS[i], OBJS[j]) / kmin)
        sh = np.asarray(sh)
        out[task] = {"n": draws,
                     "rate_pct_by_tau": {str(t): 100 * float(np.mean(sh >= t)) for t in range(1, tau_max + 1)},
                     "per_property_mean": float(np.mean(fr)), "exact_per_property_mean": float(np.mean(exf))}
    print(f"\nCROSS-ITEM NULL ({N_NULL // 2:,} pairs per task, same task, different anchors)")
    for task, o in out.items():
        print(f"  {task:9s} tau>=1 {o['rate_pct_by_tau']['1']:.2f}%  tau>=2 {o['rate_pct_by_tau'].get('2', 0):.2f}%"
              f"  tau>=3 {o['rate_pct_by_tau'].get('3', 0):.2f}%  per-property {o['per_property_mean']:.3f}"
              f"  exact per-property {o['exact_per_property_mean']:.4f}")
    return out


def task_routes(pairs_all, null, curve):
    """The blend-vs-analogy comparison by every route that does not depend on how many properties
    an invention carries. The fixed-tau ratio is reported first so the gap between it and the rest
    is visible: that gap is the property-count effect."""
    bl = [p for p in pairs_all if p["task"] == "blending"]
    an = [p for p in pairs_all if p["task"] == "analogy"]
    head = next(r for r in curve if r["tau"] == TAU)
    routes = {}

    def ratio(a, b):
        return (a / b) if b and np.isfinite(b) and b > 0 else float("nan")

    routes["fixed_tau"] = {"blending": head["blending_pct"], "analogy": head["analogy_pct"],
                           "ratio": ratio(head["blending_pct"], head["analogy_pct"]), "unit": "% of pairs"}
    routes["eligible"] = {"blending": head["blending_eligible_pct"], "analogy": head["analogy_eligible_pct"],
                          "ratio": ratio(head["blending_eligible_pct"], head["analogy_eligible_pct"]),
                          "n_blending": head["blending_eligible_n"], "n_analogy": head["analogy_eligible_n"],
                          "unit": f"% of pairs with min(k,k') >= {TAU}"}
    routes["null_corrected_fixed_tau"] = {
        "blending": head["excess_blending_pct"], "analogy": head["excess_analogy_pct"],
        "ratio": ratio(head["excess_blending_pct"], head["excess_analogy_pct"]), "unit": "% of pairs, minus null"}

    # per-property re-use, paired per item so the test respects the shared item
    def item_means(ps, key):
        acc = defaultdict(list)
        for p in ps:
            if np.isfinite(p[key]):
                acc[p["item"]].append(p[key])
        return {k: float(np.mean(v)) for k, v in acc.items()}
    for key, label, nkey in (("frac", "per_property", "per_property_mean"),
                             ("exact_frac", "exact_per_property", "exact_per_property_mean")):
        mb, ma = item_means(bl, key), item_means(an, key)
        items = sorted(set(mb) & set(ma))
        xb, xa = [mb[k] for k in items], [ma[k] for k in items]
        ob, oa = float(np.mean([p[key] for p in bl if np.isfinite(p[key])])), \
                 float(np.mean([p[key] for p in an if np.isfinite(p[key])]))
        nb, na = null["blending"][nkey], null["analogy"][nkey]
        routes[label] = {"blending": ob, "analogy": oa, "ratio": ratio(ob, oa),
                         "null_blending": nb, "null_analogy": na,
                         "excess_blending": ob - nb, "excess_analogy": oa - na,
                         "excess_ratio": ratio(ob - nb, oa - na),
                         "wilcoxon_p": float(wilcoxon(xb, xa).pvalue), "n_items": len(items),
                         "unit": "mean share of the smaller invention's properties re-used"}

    # size-matched strata: the tau-multiple rate at a fixed min(k, k')
    strata = []
    for kmin in sorted({p["kmin"] for p in pairs_all}):
        row = {"kmin": int(kmin)}
        for task, sub in (("blending", bl), ("analogy", an)):
            sel = [p for p in sub if p["kmin"] == kmin]
            row[f"n_{task}"] = len(sel)
            row[f"{task}_pct"] = 100 * float(np.mean([p["structural"] for p in sel])) if sel else float("nan")
        row["ratio"] = ratio(row["blending_pct"], row["analogy_pct"])
        strata.append(row)
    routes["size_matched"] = strata

    print(f"\nBLEND vs ANALOGY BY ROUTE (tau = {TAU})")
    for k in ("fixed_tau", "eligible", "null_corrected_fixed_tau"):
        r = routes[k]
        print(f"  {k:26s} blend {r['blending']:7.3f}  analogy {r['analogy']:7.3f}  ratio {r['ratio']:5.1f}x   [{r['unit']}]")
    for k in ("per_property", "exact_per_property"):
        r = routes[k]
        print(f"  {k:26s} blend {r['blending']:7.3f}  analogy {r['analogy']:7.3f}  ratio {r['ratio']:5.1f}x"
              f"   null {r['null_blending']:.3f}/{r['null_analogy']:.3f}  excess ratio {r['excess_ratio']:.1f}x"
              f"  paired Wilcoxon p={r['wilcoxon_p']:.1e} (n={r['n_items']} items)")
    print("  size-matched (rate at tau within min(k,k') strata):")
    for row in strata:
        rs = f"{row['ratio']:.1f}x" if np.isfinite(row["ratio"]) else "--"
        print(f"    min(k,k')={row['kmin']}: blend {row['blending_pct']:6.2f}% (n={row['n_blending']:5d})"
              f"  analogy {row['analogy_pct']:6.2f}% (n={row['n_analogy']:5d})  {rs}")
    return routes


def sensitivity(pairs_all, smat, groups, names):
    """Vary the ONE free bar the criterion still has: COS_SLOT, the cosine at which two
    "relation object" texts count as the same property. tau is reported as a curve, not swept here."""
    print("\nSENSITIVITY to the property-match cosine (multiple rate % overall / blending / analogy)")
    grid = []
    n = len(pairs_all)
    bl = [p for p in pairs_all if p["task"] == "blending"]
    an = [p for p in pairs_all if p["task"] == "analogy"]
    for cs in (0.53, 0.58, 0.63):
        sh_at = {}
        for (task, u, v), idx in groups.items():
            for a, b in itertools.combinations(idx, 2):
                sh_at[(a, b)] = shared_properties(smat[a], smat[b], tau=cs)
        for k in (1, 2, 3):
            hit = [p for p in pairs_all if sh_at[(p["a"], p["b"])] >= k]
            hb = sum(1 for p in hit if p["task"] == "blending")
            ha = sum(1 for p in hit if p["task"] == "analogy")
            grid.append({"tau": k, "cos_slot": cs, "overall_pct": 100*len(hit)/n,
                         "blending_pct": 100*hb/len(bl), "analogy_pct": 100*ha/len(an)})
        row = [g for g in grid if g["cos_slot"] == cs]
        cells = "  ".join(f"tau>={g['tau']}: {g['overall_pct']:.1f}/{g['blending_pct']:.1f}/"
                          f"{g['analogy_pct']:.1f}" for g in row)
        print(f"  cos>={cs:.2f}  {cells}")
    return grid


def prepost():
    """Blending-only, identical pipeline on the pre- and post-`uv` blends: what the re-elicitation
    changed. Requires the pre-v3 backups (responses.json.bak_pre_blendv3 + the backup .npz); fails
    loudly if they are gone, since a silent fall-through to the current data would compare a thing
    with itself."""
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    # Only models that were re-elicited have a pre-v3 backup. The current pool is larger, so comparing
    # every current model against that subset would confound the format change with a pool change --
    # restrict BOTH sides to the models present on both.
    keep = {f.split("/")[-2] for f in glob.glob(f"{RESP}/*/responses.json.bak_pre_blendv3")}
    if not keep:
        raise FileNotFoundError(f"FATAL: no pre-v3 backups under {RESP}")
    print(f"pre/post restricted to the {len(keep)} models with a pre-v3 backup")
    out = {}
    for label, npz, suf in (("pre-uv", NPZ + ".bak_pre_blendv3", ".bak_pre_blendv3"), ("post-uv", NPZ, "")):
        files = [f for f in sorted(glob.glob(f"{RESP}/*/responses.json{suf}"))
                 if f.split("/")[-2] in keep]
        if not files or not glob.glob(npz):
            raise FileNotFoundError(f"FATAL: missing {label} inputs ({npz}, {RESP}/*/responses.json{suf})")
        d = np.load(npz, allow_pickle=True)
        names, tk, us, vs, mo = d["names"], d["tasks"], d["u"], d["v"], d["models"]
        CL, ST = {}, {}
        for f in files:
            m = f.split("/")[-2]
            for r in json.load(open(f)):
                if r.get("mode") == "blending" and r.get("items"):
                    it = r["items"][0]
                    key = (r.get("u_label"), r.get("v_label"), m)
                    CL[key] = (it.get("generic_space") or "").strip()
                    tags = it.get("tags") or []
                    ST[key] = [list(t) + [tags[i] if i < len(tags) else ""]
                               for i, t in enumerate((it.get("paths") or [[]])[0])]
        idx_all = [i for i in range(len(names))
                   if str(tk[i]) == "blending" and str(mo[i]) in keep]
        con = {i: CL.get((str(us[i]), str(vs[i]), str(mo[i])), "") for i in idx_all}
        CE = {c: un(np.asarray(embed(c), float)) for c in sorted({c for c in con.values() if c})}
        dim = len(next(iter(CE.values())))
        CV = {i: (CE[con[i]] if con[i] else np.zeros(dim)) for i in idx_all}
        sv = {}                                            # the criterion is name-free here too
        SM = {}
        for i in idx_all:
            txts = [t for t, _ in slot_texts("blending", ST.get((str(us[i]), str(vs[i]), str(mo[i])), []),
                                             str(us[i]), str(vs[i]))]
            for t in txts:
                if t not in sv:
                    sv[t] = un(np.asarray(embed(t), float))
            SM[i] = np.vstack([sv[t] for t in txts]) if txts else np.zeros((0, dim))
        groups = defaultdict(list)
        for i in idx_all:
            groups[(str(us[i]), str(vs[i]))].append(i)
        pairs = []
        for k, idx in groups.items():
            for a, b in itertools.combinations(idx, 2):
                sh = shared_properties(SM[a], SM[b]); cc = float(CV[a] @ CV[b])
                pairs.append((k, a, b, bool(_nn(names[a]) == _nn(names[b]) and _nn(names[a])),
                              sh >= 1, sh >= TAU))
        tot = len(pairs)
        nclust, best = 0, 0
        for k, idx in groups.items():
            par = {i: i for i in idx}
            for p in pairs:
                if p[5] and p[0] == k:
                    par[_find(par, p[1])] = _find(par, p[2])
            comp = defaultdict(list)
            for i in idx:
                comp[_find(par, i)].append(i)
            sizes = [len(c) for c in comp.values() if len(c) >= 2]
            nclust += len(sizes); best = max([best] + sizes)
        sp, cp = [0, 0], [0, 0]
        for p in pairs:
            s = _provider(mo[p[1]]) == _provider(mo[p[2]])
            (sp if s else cp)[0] += p[5]; (sp if s else cp)[1] += 1
        AE = {s: un(np.asarray(embed(s), float)) for s in sorted({str(x) for i in idx_all for x in (us[i], vs[i])})}
        rate = defaultdict(lambda: [0, 0])
        for p in pairs:
            rate[p[0]][0] += p[5]; rate[p[0]][1] += 1
        ks = sorted(rate)
        xs = np.array([1 - float(AE[u] @ AE[v]) for u, v in ks]); ys = np.array([rate[k][0]/rate[k][1] for k in ks])
        rho, pp = spearmanr(xs, ys)
        out[label] = {"pairs": tot, "nominal_pct": 100*sum(p[3] for p in pairs)/tot,
                      "one_property_pct": 100*sum(p[4] for p in pairs)/tot,
                      "structural_pct": 100*sum(p[5] for p in pairs)/tot,
                      "n_clusters": nclust, "max_cluster": best,
                      "provider_same_pct": 100*sp[0]/sp[1], "provider_cross_pct": 100*cp[0]/cp[1],
                      "distance_rho": float(rho), "distance_p": float(pp)}
        c = out[label]
        print(f"{label:8s} pairs={c['pairs']:5d}  nominal={c['nominal_pct']:.1f}%  1prop={c['one_property_pct']:.1f}%  "
              f"structural={c['structural_pct']:.1f}%  clusters={c['n_clusters']} (max {c['max_cluster']})  "
              f"provider {c['provider_same_pct']:.1f}%/{c['provider_cross_pct']:.1f}%  "
              f"distance rho={c['distance_rho']:+.2f} (p={c['distance_p']:.3f})")
    return out


def _props(task, st):
    """Every property of an invention as (relation, object, tag), the invention's own name dropped."""
    if task == "blending":
        return [(str(p[1]), str(p[2]), (p[3] if len(p) > 3 else "")) for p in st if len(p) >= 3]
    return [(str(q["image"][1]), str(q["image"][2]), "projected")
            for q in st if q.get("image") and len(q["image"]) > 2]


def slot_texts(task, st, u, v, echo_filter=True):
    """Every property of an invention as ("relation object", tag).

    Two exclusions: the invention's own name (the subject of every triple, dropped by `_props`) and,
    with `echo_filter`, any property whose OBJECT is one of the anchors -- the anchor-echo filter.
    Such a property is supplied by the item, not invented, so two models asserting it have converged
    on nothing. Exact match on the normalised object, never word overlap.
    """
    anchors = {_nn(u), _nn(v)} if echo_filter else set()
    return [(f"{r} {o}", tag) for r, o, tag in _props(task, st) if _nn(o) not in anchors]


def slot_objects(task, st, u, v):
    """The normalised objects of an invention's properties, anchors excluded -- the encoder-free
    route matches these verbatim."""
    anchors = {_nn(u), _nn(v)}
    return [_nn(o) for _, o, _ in _props(task, st) if _nn(o) not in anchors]


def exact_shared(objs_a, objs_b):
    """Objects two inventions assert verbatim (normalised), as a multiset intersection: one-to-one
    by construction."""
    return sum((Counter(objs_a) & Counter(objs_b)).values())


def shared_properties(A, B, tau=COS_SLOT):
    """How many properties two inventions re-use, as a greedy one-to-one matching of their triples.
    One-to-one matters: without it a single generic property of A could match three of B's."""
    if not len(A) or not len(B):
        return 0
    M = A @ B.T
    used, n = set(), 0
    for ai in np.argsort(-M.max(axis=1)):
        cand = [(M[ai, j], j) for j in range(M.shape[1]) if j not in used]
        if not cand:
            break
        s, bj = max(cand)
        if s >= tau:
            n += 1
            used.add(bj)
    return n


def main():
    d = np.load(NPZ, allow_pickle=True)
    names, tk, us, vs, mo, orig = d["names"], d["tasks"], d["u"], d["v"], d["models"], d["orig"]
    embed = get_embedder("mlx-community/all-MiniLM-L6-v2-4bit")

    con_txt, con_emb, struct, rels = load_records()
    # the criterion runs on name-free properties, NOT on the saved invention vectors (those embed the
    # coined name alongside the structure, which would let a shared label carry a pair over the bar)
    un = lambda x: x / (np.linalg.norm(x) + 1e-9)
    slot_vec = {}

    def slots_of(i, echo_filter=True):
        st = slot_texts(str(tk[i]), struct.get((str(tk[i]), str(us[i]), str(vs[i]), str(mo[i])), []),
                        str(us[i]), str(vs[i]), echo_filter=echo_filter)
        for txt, _ in st:
            if txt not in slot_vec:
                slot_vec[txt] = un(np.asarray(embed(txt), float))
        return st

    def mat(slots):
        return {i: (np.vstack([slot_vec[t] for t, _ in slots[i]]) if slots[i] else np.zeros((0, 384)))
                for i in range(len(names))}

    SLOTS = {i: slots_of(i) for i in range(len(names))}
    SMAT = mat(SLOTS)
    SLOTS_RAW = {i: slots_of(i, echo_filter=False) for i in range(len(names))}   # diagnostic only
    SMAT_RAW = mat(SLOTS_RAW)
    OBJS = {i: slot_objects(str(tk[i]), struct.get((str(tk[i]), str(us[i]), str(vs[i]), str(mo[i])), []),
                            str(us[i]), str(vs[i])) for i in range(len(names))}
    concept = [con_emb.get((str(tk[i]), str(us[i]), str(vs[i]), str(mo[i])), "") for i in range(len(names))]
    uc = sorted({c for c in concept if c})
    CE = {c: np.asarray(embed(c), float) for c in uc}
    dim = len(next(iter(CE.values())))
    CV = np.array([un(CE[c]) if c else np.zeros(dim) for c in concept])

    groups = defaultdict(list)
    for i in range(len(names)):
        groups[(str(tk[i]), str(us[i]), str(vs[i]))].append(i)

    # A multiple = at least TAU re-used properties, nothing else. The name is not an input: `nominal`
    # is recorded only to be reported against the result. `kmin` is the smaller property count, the
    # number of properties the pair could at most have re-used; `frac` and `exact_frac` are the
    # per-property routes; `shared_raw` is the count WITHOUT the anchor-echo filter (diagnostic).
    pairs = []
    for (task, u, v), idx in groups.items():
        for a, b in itertools.combinations(idx, 2):
            sh = shared_properties(SMAT[a], SMAT[b])
            cc = float(CV[a] @ CV[b]) if (concept[a] and concept[b]) else float("nan")
            kmin = min(len(SLOTS[a]), len(SLOTS[b]))
            ex = exact_shared(OBJS[a], OBJS[b])
            pairs.append({"task": task, "item": (u, v), "a": a, "b": b, "shared": sh, "cos_con": cc,
                          "nominal": bool(_nn(names[a]) == _nn(names[b]) and _nn(names[a])),
                          "one_property": sh >= 1, "structural": sh >= TAU,
                          "same_provider": _provider(mo[a]) == _provider(mo[b]),
                          "kmin": kmin, "frac": (sh / kmin) if kmin else float("nan"),
                          "exact": ex, "exact_frac": (ex / kmin) if kmin else float("nan"),
                          "shared_raw": shared_properties(SMAT_RAW[a], SMAT_RAW[b])})

    tot = len(pairs)
    print(f"inventions: {len(names)}  |  co-response model-pairs (same task + anchors): {tot}")
    levels = {}
    # `one_property` and `structural` are the tau = 1 and tau = TAU points of the same curve; they are
    # kept as named keys because downstream consumers read them, but the label says which tau it is.
    for lvl, label in (("nominal", "nominal (name)"), ("one_property", "tau = 1"),
                       ("structural", f"tau = {TAU}")):
        c = sum(p[lvl] for p in pairs)
        levels[lvl] = {"count": c, "pct": 100*c/tot, "tau": (1 if lvl == "one_property"
                                                             else TAU if lvl == "structural" else None)}
        print(f"  {label:15s}: {c:5d}  ({100*c/tot:.1f}%)")
    named_hit = [p for p in pairs if p["nominal"]]
    print(f"  (of the {len(named_hit)} pairs that coined the SAME NAME, "
          f"{100*np.mean([p['structural'] for p in named_hit]):.0f}% are multiples -- the name is not "
          f"an input, and it does not stand in for one)")
    print(f"\nA tau-inventive multiple re-uses >= tau (relation, object) properties, matched one-to-one "
          f"at cosine >= {COS_SLOT}.\nHeadline tau = {TAU}; the full curve over tau is below. "
          f"Names are excluded throughout.")

    calib = calibrate(pairs, tk)

    # ANCHOR ECHO: what the filter removed, and what the headline would have been without it
    echo = {}
    print("\nANCHOR ECHO (properties whose object is an anchor; removed before matching)")
    for task in ("analogy", "blending"):
        idx = [i for i in range(len(names)) if str(tk[i]) == task]
        n_raw = sum(len(SLOTS_RAW[i]) for i in idx); n_kept = sum(len(SLOTS[i]) for i in idx)
        sub = [p for p in pairs if p["task"] == task]
        echo[task] = {"properties_raw": n_raw, "properties_kept": n_kept,
                      "pct_removed": 100 * (n_raw - n_kept) / n_raw,
                      "mean_properties_raw": n_raw / len(idx), "mean_properties": n_kept / len(idx),
                      "inventions_below_tau": sum(1 for i in idx if len(SLOTS[i]) < TAU),
                      "rate_pct_unfiltered": 100 * float(np.mean([p["shared_raw"] >= TAU for p in sub])),
                      "rate_pct": 100 * float(np.mean([p["structural"] for p in sub]))}
        e = echo[task]
        print(f"  {task:9s} removed {e['pct_removed']:.1f}% of properties ({n_raw - n_kept}/{n_raw}); "
              f"mean {e['mean_properties_raw']:.2f} -> {e['mean_properties']:.2f} per invention; "
              f"{e['inventions_below_tau']} inventions now carry < {TAU}; "
              f"tau={TAU} rate {e['rate_pct_unfiltered']:.2f}% -> {e['rate_pct']:.2f}%")

    item_of = {i: (str(us[i]), str(vs[i])) for i in range(len(names))}
    tau_max = int(max(p["shared"] for p in pairs))
    null = cross_item_null(SMAT, OBJS, tk, item_of, tau_max, np.random.default_rng(NULL_SEED))

    # clusters (connected components of structural pairs, per task+item)
    clusters = []
    for (task, u, v), idx in groups.items():
        par = {i: i for i in idx}
        for p in pairs:
            if p["structural"] and p["item"] == (u, v) and p["task"] == task:
                par[_find(par, p["a"])] = _find(par, p["b"])
        comp = defaultdict(list)
        for i in idx:
            comp[_find(par, i)].append(i)
        clusters += [(task, (u, v), c) for c in comp.values() if len(c) >= 2]
    # ---- consensus structure: the (relation, object) slots the invented concepts SHARE ----------
    # Names differ ("imperial lattice" / "Lattice Imperium" / "The Roman Lattice") while the properties
    # asserted of the invention often do not, so the compressed view of a cluster is its recurring
    # slots, not its names. The invention's own name is dropped from each triple (it is the subject of
    # all of them) and the remaining "relation object" text is embedded and grouped by an EXEMPLAR:
    # repeatedly take the slot with the most distinct models within COS_SLOT and remove that group.
    # Single-link would chain "builds ethical immunity" to "adjusts consent norms" through neighbours.
    # 0.58, not 0.62: at 0.62 a paraphrase like "splits politically along perfect cleavage planes"
    # (0.58 to "fractures along cleavage planes") fell just outside its own slot, so a model that had
    # said the same thing in other words read as sharing nothing. Exemplar grouping (below) is what
    # makes the looser bar safe -- single-link at this threshold chains unrelated properties together.
    def consensus(task, u, v, models, in_cluster):
        rows = []
        for m in models:
            for txt, tag in slot_texts(task, struct.get((task, u, v, m), []), u, v):
                if txt not in slot_vec:
                    slot_vec[txt] = un(np.asarray(embed(txt), float))
                rows.append((m, txt, tag))
        if not rows:
            return []
        X = np.vstack([slot_vec[r[1]] for r in rows])
        S = X @ X.T
        alive = set(range(len(rows)))
        out = []
        while alive:
            best, grp = None, None
            for i in alive:
                g = {j for j in alive if S[i, j] >= COS_SLOT}
                n = len({rows[j][0] for j in g})
                if best is None or n > best:
                    best, grp = n, g
            members = {rows[j][0] for j in grp}
            tags = Counter(rows[j][2] for j in grp if rows[j][2])
            texts = sorted({rows[j][1] for j in grp})
            if len(members) >= 3:
                # per-model assertion (with the tag that model gave it), so a figure can draw the
                # model x property matrix without regrouping the slots itself
                by_model = {}
                for j in grp:
                    by_model.setdefault(rows[j][0], rows[j][2])
                out.append({"gloss": rows[max(grp, key=lambda j: sum(S[j, k] for k in grp))][1],
                            "examples": texts[:4], "n_variants": len(texts),
                            "models": len(members), "models_in_cluster": len(members & in_cluster),
                            "tag": tags.most_common(1)[0][0] if tags else "",
                            "assertions": by_model})
            alive -= grp
        return sorted(out, key=lambda r: (-r["models"], -r["models_in_cluster"]))

    # For every model that answered an item but did NOT join a given cluster, record WHICH clause kept
    # it out: its best invention cosine and best abstraction cosine against that cluster's members. On
    # some items the abstraction is near-forced by the anchors -- outsiders share it and diverge only in
    # what they build on it -- and saying "a different abstraction" there would be false.
    def outsider_stats(comp, i):
        sh = max(shared_properties(SMAT[i], SMAT[j]) for j in comp)
        abs_ = max(float(CV[i] @ CV[j]) for j in comp) if concept[i] else float("nan")
        # The criterion is property overlap alone, so "blocked" can only ever mean too few shared
        # properties. The abstraction cosine is still recorded, purely as a descriptive diagnostic.
        return {"shared": sh, "abs_cos": None if not np.isfinite(abs_) else round(abs_, 3),
                "blocked_by": "properties" if sh < TAU else "none"}

    cluster_of = {}                                    # invention -> its cluster's shared name
    for task, (u, v), comp in clusters:
        lab = Counter(str(names[i]).lower() for i in comp).most_common(1)[0][0]
        for i in comp:
            cluster_of[i] = lab
    hit = len({(p["task"], p["item"]) for p in pairs if p["structural"]})
    print(f"(task,anchor) settings with >=1: {hit}/{len(groups)}; distinct rediscovered inventions: "
          f"{len(clusters)}; max multiplicity {max(len(c) for _, _, c in clusters)}")

    print("\nPREDICTORS (structural multiple):")
    bi = defaultdict(lambda: {"analogy": [0, 0], "blending": [0, 0]})
    for p in pairs:
        bi[p["item"]][p["task"]][0] += p["structural"]; bi[p["item"]][p["task"]][1] += 1
    keys = [k for k in bi if bi[k]["analogy"][1] and bi[k]["blending"][1]]
    an = [bi[k]["analogy"][0]/bi[k]["analogy"][1] for k in keys]
    bl = [bi[k]["blending"][0]/bi[k]["blending"][1] for k in keys]
    w_task = wilcoxon(bl, an).pvalue
    print(f"  task: blend {100*np.mean(bl):.1f}% vs analogy {100*np.mean(an):.1f}%  "
          f"(n={len(bl)} anchor pairs, paired Wilcoxon p={w_task:.1e})")
    # encoder-free cross-check on the same comparison
    nb = defaultdict(lambda: {"analogy": [0, 0], "blending": [0, 0]})
    for p in pairs:
        nb[p["item"]][p["task"]][0] += p["nominal"]; nb[p["item"]][p["task"]][1] += 1
    nan_ = [nb[k]["analogy"][0]/nb[k]["analogy"][1] for k in keys]
    nbl = [nb[k]["blending"][0]/nb[k]["blending"][1] for k in keys]
    print(f"  task (NOMINAL, encoder-free): blend {100*np.mean(nbl):.1f}% vs analogy {100*np.mean(nan_):.1f}%  "
          f"(paired Wilcoxon p={wilcoxon(nbl, nan_).pvalue:.1e})")

    models = sorted(set(mo.tolist())); pr0 = {m: _provider(m) for m in models}
    def diff(pm, key="structural"):
        sp, cp = [0.0, 0], [0.0, 0]
        for p in pairs:
            x = p[key]
            if not np.isfinite(x):
                continue
            s = pm[str(mo[p["a"]])] == pm[str(mo[p["b"]])]
            (sp if s else cp)[0] += x; (sp if s else cp)[1] += 1
        return sp[0]/sp[1] - cp[0]/cp[1], sp[0]/sp[1], cp[0]/cp[1]
    obs, rsp, rcp = diff(pr0)
    obs_f, fsp, fcp = diff(pr0, "frac")
    rng = np.random.default_rng(0); lab = [pr0[m] for m in models]; perm_null, perm_null_f = [], []
    for _ in range(2000):
        perm = list(lab); rng.shuffle(perm)
        pm = {m: perm[i] for i, m in enumerate(models)}
        perm_null.append(diff(pm)[0]); perm_null_f.append(diff(pm, "frac")[0])
    pv = (np.sum(np.abs(perm_null) >= abs(obs)) + 1) / 2001
    pv_f = (np.sum(np.abs(perm_null_f) >= abs(obs_f)) + 1) / 2001
    print(f"  provider: same {100*rsp:.1f}% vs cross {100*rcp:.1f}%  (RR {rsp/rcp:.1f}x, permutation p={pv:.4f})")
    print(f"  provider, per-property re-use: same {fsp:.3f} vs cross {fcp:.3f}  "
          f"(ratio {fsp/fcp:.2f}x, permutation p={pv_f:.4f})")
    prov_task = {}
    for task in ("blending", "analogy"):
        sp_ = [p["structural"] for p in pairs if p["task"] == task and p["same_provider"]]
        cp_ = [p["structural"] for p in pairs if p["task"] == task and not p["same_provider"]]
        prov_task[task] = {"same_pct": 100 * float(np.mean(sp_)), "cross_pct": 100 * float(np.mean(cp_))}
        print(f"    within {task}: same {prov_task[task]['same_pct']:.2f}% vs cross {prov_task[task]['cross_pct']:.2f}%")

    inm = {i for p in pairs if p["structural"] for i in (p["a"], p["b"])}
    om = [orig[i] for i in range(len(names)) if i in inm and np.isfinite(orig[i])]
    os_ = [orig[i] for i in range(len(names)) if i not in inm and np.isfinite(orig[i])]
    print(f"  originality: multiples {np.mean(om):.2f} vs singletons {np.mean(os_):.2f}")

    # invention-level view (the pair rate understates how much of the corpus is touched): how many
    # inventions are reinvented by at least one other model?
    inv_rate = {"all": 100*len(inm)/len(names)}
    for task in ("analogy", "blending"):
        idx = [i for i in range(len(names)) if str(tk[i]) == task]
        inv_rate[task] = 100*len({i for i in inm if i in set(idx)})/len(idx)
    print(f"  inventions in >=1 multiple: {len(inm)}/{len(names)} ({inv_rate['all']:.0f}%)  "
          f"[blending {inv_rate['blending']:.0f}%, analogy {inv_rate['analogy']:.0f}%]")

    # anchor-pair distance, PER TASK (the operator asymmetry)
    anchors = sorted(set(list(us) + list(vs)))
    AE = {s: un(np.asarray(embed(s), float)) for s in anchors}
    adist = {(u, v): 1.0 - float(AE[u] @ AE[v]) for (_, u, v) in groups}
    dist_out = {}
    print("  anchor-pair distance vs per-item structural rate:")
    for task in ("blending", "analogy"):
        rate = defaultdict(lambda: [0, 0])
        for p in pairs:
            if p["task"] == task:
                rate[p["item"]][0] += p["structural"]; rate[p["item"]][1] += 1
        ks = sorted(rate)
        xs = np.array([adist[k] for k in ks]); ys = np.array([rate[k][0]/rate[k][1] for k in ks])
        rho, pr = spearmanr(xs, ys); r, pp = pearsonr(xs, ys)
        loo = [spearmanr(np.delete(xs, i), np.delete(ys, i)).pvalue for i in range(len(xs))]
        o = np.argsort(xs); terc = [100*float(np.mean(ys[o[a:b]])) for a, b in
                                    ((0, len(xs)//3), (len(xs)//3, 2*len(xs)//3), (2*len(xs)//3, len(xs)))]
        dist_out[task] = {"n_items": len(xs), "spearman_rho": float(rho), "spearman_p": float(pr),
                          "pearson_r": float(r), "pearson_p": float(pp),
                          "loo_n_sig": int(sum(1 for q in loo if q < 0.05)), "loo_max_p": float(max(loo)),
                          "tercile_pct": terc}
        print(f"    {task:9s} rho={rho:+.2f} (p={pr:.3f})  r={r:+.2f} (p={pp:.3f})  "
              f"LOO {sum(1 for q in loo if q < 0.05)}/{len(loo)} keep p<0.05 (max p={max(loo):.3f})  "
              f"terciles {terc[0]:.1f}% -> {terc[1]:.1f}% -> {terc[2]:.1f}%")

    # relation-label Jaccard: is the agreement visible in the predicates themselves?
    jac_lex, jac_non = [], []
    for p in pairs:
        ka = (p["task"], p["item"][0], p["item"][1], str(mo[p["a"]]))
        kb = (p["task"], p["item"][0], p["item"][1], str(mo[p["b"]]))
        ra, rb = rels.get(ka, set()), rels.get(kb, set())
        if not ra or not rb:
            continue
        j = len(ra & rb) / len(ra | rb)
        (jac_lex if p["nominal"] else jac_non).append(j)
    print(f"  relation-label Jaccard: lexically-identical median {np.median(jac_lex):.2f} "
          f"(mean {np.mean(jac_lex):.2f}, n={len(jac_lex)}) vs non-matching median {np.median(jac_non):.2f}")

    task_of_inv = {i: str(tk[i]) for i in range(len(names))}
    curve = tau_curve(pairs, null, task_of_inv)
    routes = task_routes(pairs, null, curve)
    grid = sensitivity(pairs, SMAT, groups, names)

    # A cluster is a connected component of multiple-pairs, not a clique: report how far from one.
    dens = []
    for task, (u, v), c in clusters:
        cs = set(c)
        e = sum(1 for p in pairs if p["structural"] and p["task"] == task and p["item"] == (u, v)
                and p["a"] in cs and p["b"] in cs)
        dens.append(e / (len(c) * (len(c) - 1) / 2))
    print(f"\nMost-rediscovered inventions (largest structural clusters); mean within-cluster pair "
          f"density {np.mean(dens):.2f}:")
    for (task, (u, v), c), dn in sorted(zip(clusters, dens), key=lambda x: -len(x[0][2]))[:8]:
        nm = sorted({str(names[i]) for i in c})
        print(f"  [{task[:4]}] ({u}, {v}) x{len(c)} (density {dn:.2f}): {', '.join(nm[:6])}")

    dump = {
        "n_inventions": int(len(names)), "n_pairs": tot, "tau": TAU,
        "cos_slot": COS_SLOT, "cos_con": COS_CON,
        "levels": levels,
        "same_name_pairs": {"n": len(named_hit),
                            "pct_that_are_multiples": 100*float(np.mean([p["structural"] for p in named_hit]))},
        "calibration": calib, "tau_curve": curve, "sensitivity": grid,
        "anchor_echo": echo, "null": {"n_per_task": N_NULL // 2, "seed": NULL_SEED, **null},
        "task_routes": routes,
        "settings_with_multiple": hit, "n_settings": len(groups), "n_clusters": len(clusters),
        "task": {"blending_pct": 100*float(np.mean(bl)), "analogy_pct": 100*float(np.mean(an)),
                 "wilcoxon_p": float(w_task), "n_items": len(bl),
                 "nominal_blending_pct": 100*float(np.mean(nbl)), "nominal_analogy_pct": 100*float(np.mean(nan_))},
        "provider": {"same_pct": 100*rsp, "cross_pct": 100*rcp, "rr": rsp/rcp, "perm_p": float(pv),
                     "per_property_same": fsp, "per_property_cross": fcp, "per_property_ratio": fsp/fcp,
                     "per_property_perm_p": float(pv_f), "by_task": prov_task},
        "originality": {"multiples": float(np.mean(om)), "singletons": float(np.mean(os_))},
        "inventions_in_a_multiple": {"count": len(inm), "pct": inv_rate["all"],
                                     "blending_pct": inv_rate["blending"], "analogy_pct": inv_rate["analogy"]},
        "anchor_distance": dist_out,
        "relation_jaccard": {"lexical_median": float(np.median(jac_lex)), "lexical_mean": float(np.mean(jac_lex)),
                             "nonmatch_median": float(np.median(jac_non)), "n_lexical": len(jac_lex)},
        # Every cluster's OUTSIDERS: the models that answered the same item and did not join it. No
        # cluster ever holds the whole pool, so these are what show that a rediscovery is a property of
        # the models rather than of the anchor pair -- same inputs, a different abstraction.
        "n_models": len(set(map(str, mo))),   # the pool these rates are relative to; used by the showcase
        # `n_pairs` above counts CO-RESPONSE MODEL PAIRS, not anchor pairs -- keep both, named apart.
        "n_anchor_pairs": len({(str(a), str(b)) for a, b in zip(us, vs)}),
        # Computed, not asserted: at tau = 2 some blending items pull the WHOLE pool into one component.
        "clusters_have_outsiders": all(len(groups[(task, u, v)]) > len(c) for task, (u, v), c in clusters),
        "cluster_density_mean": float(np.mean(dens)),
        # `edges` are the actual structural pairs among a cluster's members (as member-list indices):
        # a cluster is the connected component of those pairs, not necessarily a clique, so plots that
        # draw the component must draw the edges rather than assume a blob. `density` is the share of
        # member pairs that are themselves multiples -- 1.0 would be a clique.
        "clusters": [
            {"task": task, "u": u, "v": v, "size": len(c),
             "providers": sorted({_provider(mo[i]) for i in c}),
             "density": len(edges) / (len(c) * (len(c) - 1) / 2),
             "members": [{"model": str(mo[i]), "name": str(names[i]),
                          "concept": con_txt.get((task, u, v, str(mo[i])), ""),
                          "structure": struct.get((task, u, v, str(mo[i])), [])} for i in sorted(c, key=lambda j: str(mo[j]))],
             "edges": edges,
             "consensus": consensus(task, u, v, [str(mo[i]) for i in groups[(task, u, v)]],
                                    {str(mo[i]) for i in c}),
             "outsiders": [{"model": str(mo[i]), "name": str(names[i]),
                            "concept": con_txt.get((task, u, v, str(mo[i])), ""),
                            "other_cluster": cluster_of.get(i), **outsider_stats(c, i)}
                           for i in sorted(set(groups[(task, u, v)]) - set(c), key=lambda j: str(mo[j]))]}
            for task, (u, v), c in sorted(clusters, key=lambda x: -len(x[2]))
            for order in [sorted(c, key=lambda j: str(mo[j]))]
            for edges in [[[order.index(p["a"]), order.index(p["b"])] for p in pairs
                           if p["structural"] and p["task"] == task and p["item"] == (u, v)
                           and p["a"] in set(c) and p["b"] in set(c)]]],
    }
    with open(OUT, "w") as f:
        json.dump(dump, f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepost", action="store_true",
                    help="blending-only comparison of the pre- vs post-`uv` re-elicitation")
    a = ap.parse_args()
    prepost() if a.prepost else main()
