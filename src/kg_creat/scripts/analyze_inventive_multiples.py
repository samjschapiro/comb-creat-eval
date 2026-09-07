"""Inventive multiples: how often do two independent models invent the same entity for the same anchors
(u,v), and what predicts it?

A multiple is two models RE-USING THE SAME STRUCTURE. The coined name never enters the calculation:
every triple is reduced to "relation object" (the invention's name is the subject of all of them and is
dropped), so what is compared is the properties asserted of the invention, not what it was called.

For a pair of inventions (same task, same anchor pair):

  SHARED PROPERTIES -- greedily match each triple of one against an unused triple of the other; a pair
                       counts as shared when their "relation object" texts are within TAU_SLOT.
  MULTIPLE          -- at least K_SHARED shared properties AND the underlying abstraction also aligns
                       (the projected source concept for analogy, the generic space for blending, at
                       cosine >= TAU_CON). Both clauses are structure; neither is a label.

NOMINAL -- the coined names matching -- is computed but is NEVER an input. It is the independent check:
under this definition only ~7% of same-name pairs qualify, so name convergence and structural
convergence are separate phenomena rather than one measurement.

Judge-free; local MLX embeddings. SENSITIVITY re-runs the headline over K and the abstraction bar.

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
TAU_SLOT = 0.58  # "relation object" cosine at which two models count as asserting the same property
K_SHARED = 2     # properties two inventions must share to be a multiple ("the same properties", plural)
TAU_CON = 0.50   # concept (phi / generic-space) cosine the abstraction clause requires
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
            "shared_lexical_pct_ge_k": float(np.mean(np.asarray(lex) >= K_SHARED)) if lex else float("nan"),
            "shared_nonmatch_pct_ge_k": float(np.mean(np.asarray(non) >= K_SHARED)) if non else float("nan"),
            "con_lexical_mean": float(np.mean(lexc)) if lexc else float("nan"),
            "con_nonmatch_mean": float(np.mean(nonc)) if nonc else float("nan"),
            "con_frac_ge_tau_lexical": float(np.mean(np.asarray(lexc) >= TAU_CON)) if lexc else float("nan"),
            "con_frac_ge_tau_nonmatch": float(np.mean(np.asarray(nonc) >= TAU_CON)) if nonc else float("nan"),
        }
    print("\nCALIBRATION (properties re-used; the name is reported against the criterion, never in it)")
    for s, c in out.items():
        print(f"  {s:9s} same-name n={c['n_lexical']:4d} mean shared={c['shared_lexical_mean']:.2f} "
              f"({100*c['shared_lexical_pct_ge_k']:.0f}% reach {K_SHARED}) | other pairs "
              f"mean={c['shared_nonmatch_mean']:.2f} ({100*c['shared_nonmatch_pct_ge_k']:.0f}%)")
    print(f"  abstraction >= {TAU_CON}: same-name {100*out['pooled']['con_frac_ge_tau_lexical']:.0f}%"
          f" vs other pairs {100*out['pooled']['con_frac_ge_tau_nonmatch']:.0f}%")
    return out


def sensitivity(pairs_all):
    print("\nSENSITIVITY (multiple rate % overall / blending / analogy)")
    grid = []
    n = len(pairs_all)
    bl = [p for p in pairs_all if p["task"] == "blending"]
    an = [p for p in pairs_all if p["task"] == "analogy"]
    for k in (1, 2, 3):
        row = []
        for tc in (0.45, 0.50, 0.55):
            hit = [p for p in pairs_all if p["shared"] >= k and np.isfinite(p["cos_con"]) and p["cos_con"] >= tc]
            hb = sum(1 for p in hit if p["task"] == "blending")
            ha = sum(1 for p in hit if p["task"] == "analogy")
            row.append((tc, 100*len(hit)/n, 100*hb/len(bl), 100*ha/len(an)))
            grid.append({"k_shared": k, "tau_con": tc, "overall_pct": 100*len(hit)/n,
                         "blending_pct": 100*hb/len(bl), "analogy_pct": 100*ha/len(an)})
        cells = "  ".join(f"tc={tc:.2f}: {o:.1f}/{b:.1f}/{a:.1f}" for tc, o, b, a in row)
        print(f"  k>={k}  {cells}")
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
            txts = [t for t, _ in slot_texts("blending", ST.get((str(us[i]), str(vs[i]), str(mo[i])), []))]
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
                              sh >= 1 and cc >= TAU_CON, sh >= K_SHARED and cc >= TAU_CON))
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


def slot_texts(task, st):
    """Every property of an invention as ("relation object", tag), the invention's own name dropped."""
    if task == "blending":
        return [(f"{s[1]} {s[2]}", (s[3] if len(s) > 3 else "")) for s in st if len(s) >= 3]
    return [(f"{q['image'][1]} {q['image'][2]}", "projected")
            for q in st if q.get("image") and len(q["image"]) > 2]


def shared_properties(A, B, tau=TAU_SLOT):
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

    def slots_of(i):
        st = slot_texts(str(tk[i]), struct.get((str(tk[i]), str(us[i]), str(vs[i]), str(mo[i])), []))
        for txt, _ in st:
            if txt not in slot_vec:
                slot_vec[txt] = un(np.asarray(embed(txt), float))
        return st

    SLOTS = {i: slots_of(i) for i in range(len(names))}
    SMAT = {i: (np.vstack([slot_vec[t] for t, _ in SLOTS[i]]) if SLOTS[i] else np.zeros((0, 384)))
            for i in range(len(names))}
    concept = [con_emb.get((str(tk[i]), str(us[i]), str(vs[i]), str(mo[i])), "") for i in range(len(names))]
    uc = sorted({c for c in concept if c})
    CE = {c: np.asarray(embed(c), float) for c in uc}
    dim = len(next(iter(CE.values())))
    CV = np.array([un(CE[c]) if c else np.zeros(dim) for c in concept])

    groups = defaultdict(list)
    for i in range(len(names)):
        groups[(str(tk[i]), str(us[i]), str(vs[i]))].append(i)

    # A multiple = at least K_SHARED re-used properties AND an aligned abstraction. An AND, not an OR,
    # and the name is in neither clause: `nominal` is recorded only to be reported against the result.
    pairs = []
    for (task, u, v), idx in groups.items():
        for a, b in itertools.combinations(idx, 2):
            sh = shared_properties(SMAT[a], SMAT[b])
            cc = float(CV[a] @ CV[b]) if (concept[a] and concept[b]) else float("nan")
            con = bool(np.isfinite(cc) and cc >= TAU_CON)
            pairs.append({"task": task, "item": (u, v), "a": a, "b": b, "shared": sh, "cos_con": cc,
                          "nominal": bool(_nn(names[a]) == _nn(names[b]) and _nn(names[a])),
                          "one_property": sh >= 1 and con, "structural": sh >= K_SHARED and con,
                          "same_provider": _provider(mo[a]) == _provider(mo[b])})

    tot = len(pairs)
    print(f"inventions: {len(names)}  |  co-response model-pairs (same task + anchors): {tot}")
    levels = {}
    for lvl in ("nominal", "one_property", "structural"):
        c = sum(p[lvl] for p in pairs)
        levels[lvl] = {"count": c, "pct": 100*c/tot}
        print(f"  {lvl:13s}: {c:5d}  ({100*c/tot:.1f}%)")
    named_hit = [p for p in pairs if p["nominal"]]
    print(f"  (of the {len(named_hit)} pairs that coined the SAME NAME, "
          f"{100*np.mean([p['structural'] for p in named_hit]):.0f}% are multiples -- the name is not "
          f"an input, and it does not stand in for one)")
    print(f"\nMultiple = >={K_SHARED} shared properties (matched at {TAU_SLOT}) AND abstraction "
          f">= {TAU_CON}. Names excluded throughout.")

    calib = calibrate(pairs, tk)

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
    # repeatedly take the slot with the most distinct models within TAU_SLOT and remove that group.
    # Single-link would chain "builds ethical immunity" to "adjusts consent norms" through neighbours.
    # 0.58, not 0.62: at 0.62 a paraphrase like "splits politically along perfect cleavage planes"
    # (0.58 to "fractures along cleavage planes") fell just outside its own slot, so a model that had
    # said the same thing in other words read as sharing nothing. Exemplar grouping (below) is what
    # makes the looser bar safe -- single-link at this threshold chains unrelated properties together.
    def consensus(task, u, v, models, in_cluster):
        rows = []
        for m in models:
            for txt, tag in slot_texts(task, struct.get((task, u, v, m), [])):
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
                g = {j for j in alive if S[i, j] >= TAU_SLOT}
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
        ok_s, ok_a = sh >= K_SHARED, np.isfinite(abs_) and abs_ >= TAU_CON
        return {"shared": sh, "abs_cos": None if not np.isfinite(abs_) else round(abs_, 3),
                "blocked_by": "abstraction" if ok_s and not ok_a else "properties" if ok_a and not ok_s
                else "both"}

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
    def diff(pm):
        sp, cp = [0, 0], [0, 0]
        for p in pairs:
            s = pm[str(mo[p["a"]])] == pm[str(mo[p["b"]])]
            (sp if s else cp)[0] += p["structural"]; (sp if s else cp)[1] += 1
        return sp[0]/sp[1] - cp[0]/cp[1], sp[0]/sp[1], cp[0]/cp[1]
    obs, rsp, rcp = diff(pr0)
    rng = np.random.default_rng(0); lab = [pr0[m] for m in models]; null = []
    for _ in range(2000):
        perm = list(lab); rng.shuffle(perm)
        null.append(diff({m: perm[i] for i, m in enumerate(models)})[0])
    pv = (np.sum(np.abs(null) >= abs(obs)) + 1) / 2001
    print(f"  provider: same {100*rsp:.1f}% vs cross {100*rcp:.1f}%  (RR {rsp/rcp:.1f}x, permutation p={pv:.4f})")

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

    grid = sensitivity(pairs)

    print("\nMost-rediscovered inventions (largest structural clusters):")
    for task, (u, v), c in sorted(clusters, key=lambda x: -len(x[2]))[:8]:
        nm = sorted({str(names[i]) for i in c})
        print(f"  [{task[:4]}] ({u}, {v}) x{len(c)}: {', '.join(nm[:6])}")

    dump = {
        "n_inventions": int(len(names)), "n_pairs": tot, "k_shared": K_SHARED,
        "tau_slot": TAU_SLOT, "tau_con": TAU_CON,
        "levels": levels,
        "same_name_pairs": {"n": len(named_hit),
                            "pct_that_are_multiples": 100*float(np.mean([p["structural"] for p in named_hit]))},
        "calibration": calib, "sensitivity": grid,
        "settings_with_multiple": hit, "n_settings": len(groups), "n_clusters": len(clusters),
        "task": {"blending_pct": 100*float(np.mean(bl)), "analogy_pct": 100*float(np.mean(an)),
                 "wilcoxon_p": float(w_task), "n_items": len(bl),
                 "nominal_blending_pct": 100*float(np.mean(nbl)), "nominal_analogy_pct": 100*float(np.mean(nan_))},
        "provider": {"same_pct": 100*rsp, "cross_pct": 100*rcp, "rr": rsp/rcp, "perm_p": float(pv)},
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
        "clusters_have_outsiders": True,
        # `edges` are the actual structural pairs among a cluster's members (as member-list indices):
        # a cluster is the connected component of those pairs, not necessarily a clique, so plots that
        # draw the component must draw the edges rather than assume a blob.
        "clusters": [
            {"task": task, "u": u, "v": v, "size": len(c),
             "providers": sorted({_provider(mo[i]) for i in c}),
             "members": [{"model": str(mo[i]), "name": str(names[i]),
                          "concept": con_txt.get((task, u, v, str(mo[i])), ""),
                          "structure": struct.get((task, u, v, str(mo[i])), [])} for i in sorted(c, key=lambda j: str(mo[j]))],
             "edges": [[order.index(p["a"]), order.index(p["b"])] for p in pairs
                       if p["structural"] and p["task"] == task and p["item"] == (u, v)
                       and p["a"] in set(c) and p["b"] in set(c)],
             "consensus": consensus(task, u, v, [str(mo[i]) for i in groups[(task, u, v)]],
                                    {str(mo[i]) for i in c}),
             "outsiders": [{"model": str(mo[i]), "name": str(names[i]),
                            "concept": con_txt.get((task, u, v, str(mo[i])), ""),
                            "other_cluster": cluster_of.get(i), **outsider_stats(c, i)}
                           for i in sorted(set(groups[(task, u, v)]) - set(c), key=lambda j: str(mo[j]))]}
            for task, (u, v), c in sorted(clusters, key=lambda x: -len(x[2]))
            for order in [sorted(c, key=lambda j: str(mo[j]))]],
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
