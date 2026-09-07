"""Analysis #2: cross-model artifact homogeneity ("artificial hivemind") + RSA.

Two views, because association and analogy/blending are structurally different:
  BASE      (all 3 tasks)  the core combinatorial artifact each task emits, apples-to-apples:
                           association = bridge path (anchors removed); analogy = the mapping
                           path_a u path_b; blending = the projected blend structure (u/v triples).
  EMERGENT  (analogy+blend) the invented concept only: analogy h (name + projected image); blending
                           the elaborated emergent structure Delta (concept + emergent triples).
                           Association has no emergent artifact, so it is absent from this view.

Per view/task we report: convergence (1 - mean pairwise cosine distance of the artifacts) vs a
cross-item shuffled null; anchor->artifact RSA (Mantel test); anchor separation vs convergence; and the
full model x model similarity matrix. No API: local MLX embeddings only.

    .venv_mlx/bin/python -m src.kg_creat.scripts.analyze_invention_homogeneity
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.kg_creat.embed import get_embedder  # noqa: E402

RESP = Path("data/kg_creat/kombine_test30/responses")
OUT = Path("data/kg_creat/kombine_test30/analysis")
OUT.mkdir(parents=True, exist_ok=True)

LAB = lambda m: m.split("/")[0]  # noqa: E731


def tri_txt(t):
    return " ".join(str(x) for x in t) if isinstance(t, list) else str(t)


def _norm(s):
    return str(s).strip().lower()


def base_text(mode, item, u=None, v=None):
    """BASE artifact -- the thing utility/surprise/originality score for each task: association = the
    bridge path; analogy = the mapping (both domain paths); blending = the GENERIC SPACE g."""
    if mode == "analogy":                                   # the mapping: both domain paths
        trs = [tri_txt(t) for p in (item.get("paths") or [])[:2] for t in p]
        return " ; ".join(trs)
    if mode == "blending":                                  # the shared schema g (a textual description)
        return (item.get("generic_space") or "").strip()
    # association: bridge = intermediate concepts + relations, anchors removed
    anchors = {_norm(u), _norm(v)}
    toks = []
    for tr in (item.get("paths") or [[]])[0]:
        if not isinstance(tr, list) or len(tr) < 3:
            continue
        h, r, t = tr[0], tr[1], tr[2]
        toks.append(str(r))
        if _norm(h) not in anchors:
            toks.append(str(h))
        if _norm(t) not in anchors:
            toks.append(str(t))
    return " ".join(toks).strip()


def emergent_text(mode, item, u=None, v=None):
    """EMERGENT invention: the invented concept. Analogy = h (name + projected image); blending = the
    blended concept c' (name + full blended-space structure + emergent). Association has none -> ''."""
    if mode == "analogy":
        name = item.get("invention") or ""
        imgs = [tri_txt(p.get("image")) for p in (item.get("projection") or []) if isinstance(p, dict)]
        return (name + " . " + " ; ".join(imgs)).strip(" .")
    if mode == "blending":                                  # c' = concept + blended structure + Delta
        name = item.get("concept") or ""
        struct = " ; ".join(tri_txt(t) for t in (item.get("paths") or [[]])[0])
        emg = " ; ".join(item.get("inferences") or [])
        return (name + " . " + struct + " . " + emg).strip(" .")
    return ""


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def cos_dist(a, b):
    return 1.0 - cos(a, b)


def upper(M):
    n = M.shape[0]
    return np.array([M[i, j] for i, j in combinations(range(n), 2)])


def spearman(x, y):
    xr = np.argsort(np.argsort(x)); yr = np.argsort(np.argsort(y))
    xr = xr - xr.mean(); yr = yr - yr.mean()
    return float((xr @ yr) / (np.sqrt(xr @ xr) * np.sqrt(yr @ yr) + 1e-9))


def mantel_p(rdm_a, rdm_b, obs, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = rdm_a.shape[0]
    ua = upper(rdm_a)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        ub = upper(rdm_b[np.ix_(perm, perm)])
        if abs(spearman(ua, ub)) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def analyze_view(inv, modes, all_mids, E, anchors):
    """Compute the homogeneity metrics for one view (dict mode -> item -> model -> vec)."""
    results = {}
    for mode in modes:
        conv, npm, anchor_sep = {}, {}, {}
        for iid in sorted(inv[mode]):
            vs = list(inv[mode][iid].values())
            if len(vs) < 3:
                continue
            conv[iid] = 1.0 - float(np.mean([cos_dist(a, b) for a, b in combinations(vs, 2)]))
            npm[iid] = len(vs)
            u, v, _, _ = anchors[iid]
            anchor_sep[iid] = cos_dist(E(u), E(v))
        ci = np.array([conv[i] for i in conv])
        allvecs = [x for iid in conv for x in inv[mode][iid].values()]
        rng = np.random.default_rng(0)
        null = []
        for iid in conv:
            idx = rng.choice(len(allvecs), size=npm[iid], replace=False)
            g = [allvecs[j] for j in idx]
            null.append(1.0 - float(np.mean([cos_dist(a, b) for a, b in combinations(g, 2)])))
        null = np.array(null)

        cent = {iid: np.mean(list(inv[mode][iid].values()), axis=0) for iid in conv}
        ids = list(conv); n = len(ids)
        RDM_inv = np.zeros((n, n)); RDM_anchor = np.zeros((n, n)); RDM_domain = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                ia, ib = ids[a], ids[b]
                RDM_inv[a, b] = RDM_inv[b, a] = cos_dist(cent[ia], cent[ib])
                ua, va, dua, dva = anchors[ia]; ub, vb, dub, dvb = anchors[ib]
                RDM_anchor[a, b] = RDM_anchor[b, a] = cos_dist((E(ua) + E(va)) / 2, (E(ub) + E(vb)) / 2)
                RDM_domain[a, b] = RDM_domain[b, a] = cos_dist((E(str(dua)) + E(str(dva))) / 2,
                                                               (E(str(dub)) + E(str(dvb))) / 2)
        r_anchor = spearman(upper(RDM_anchor), upper(RDM_inv))
        r_domain = spearman(upper(RDM_domain), upper(RDM_inv))
        p_anchor = mantel_p(RDM_anchor, RDM_inv, r_anchor)
        r_sep_conv = spearman(np.array([anchor_sep[i] for i in ids]), np.array([conv[i] for i in ids]))

        allmodels = sorted({m for iid in conv for m in inv[mode][iid]})
        msim = {}
        for m1, m2 in combinations(allmodels, 2):
            sims = [cos(inv[mode][iid][m1], inv[mode][iid][m2])
                    for iid in conv if m1 in inv[mode][iid] and m2 in inv[mode][iid]]
            if sims:
                msim[(m1, m2)] = float(np.mean(sims))
        same = [s for (a, b), s in msim.items() if LAB(a) == LAB(b)]
        cross = [s for (a, b), s in msim.items() if LAB(a) != LAB(b)]
        sim_mat = [[(1.0 if i == j else msim.get(tuple(sorted([all_mids[i], all_mids[j]])), float("nan")))
                    for j in range(len(all_mids))] for i in range(len(all_mids))]

        conv_sorted = sorted(conv.items(), key=lambda kv: kv[1])
        results[mode] = {
            "n_items": n, "n_models": len(allmodels),
            "hivemind_index": float(ci.mean()), "null_mean": float(null.mean()),
            "conv_min": float(ci.min()), "conv_max": float(ci.max()),
            "rsa_anchor_r": r_anchor, "rsa_anchor_p": p_anchor, "rsa_domain_r": r_domain,
            "anchor_sep_vs_convergence_r": r_sep_conv,
            "same_lab_sim": float(np.mean(same)) if same else None,
            "cross_lab_sim": float(np.mean(cross)) if cross else None,
            "most_convergent": [(anchors[i][0] + " | " + anchors[i][1], round(c, 3)) for i, c in conv_sorted[-5:][::-1]],
            "most_divergent": [(anchors[i][0] + " | " + anchors[i][1], round(c, 3)) for i, c in conv_sorted[:5]],
            "plot": {
                "conv_items": [float(conv[i]) for i in ids],
                "anchor_sep_items": [float(anchor_sep[i]) for i in ids],
                "rdm_anchor_upper": [float(x) for x in upper(RDM_anchor)],
                "rdm_inv_upper": [float(x) for x in upper(RDM_inv)],
                "model_order": [m.split("/")[-1] for m in all_mids],
                "sim_matrix": sim_mat,
            },
        }
    return results


def main():
    embed = get_embedder()
    models = sorted(d.name for d in RESP.iterdir() if (d / "responses.json").exists())
    model_ids, anchors, cache = {}, {}, {}

    def E(s):
        if s not in cache:
            cache[s] = np.asarray(embed(s), dtype=float)
        return cache[s]

    base_inv = {"baseline": {}, "analogy": {}, "blending": {}}
    emg_inv = {"analogy": {}, "blending": {}}
    for mdir in models:
        recs = json.loads((RESP / mdir / "responses.json").read_text())
        mid = json.loads((RESP / mdir / "summary.json").read_text())["model_id"]
        model_ids[mdir] = mid
        for r in recs:
            if not r.get("items"):
                continue
            mode, it = r["mode"], r["items"][0]
            iid, u, v = r["prompt_id"], r.get("u_label"), r.get("v_label")
            anchors[iid] = (u, v, r.get("domain_u"), r.get("domain_v"))
            if mode in base_inv:
                bt = base_text(mode, it, u, v)
                if len(bt) >= 3:
                    base_inv[mode].setdefault(iid, {})[mid] = E(bt)
            if mode in emg_inv:
                et = emergent_text(mode, it, u, v)
                if len(et) >= 3:
                    emg_inv[mode].setdefault(iid, {})[mid] = E(et)

    LAB_ORDER = ["openai", "anthropic", "google", "x-ai", "deepseek", "qwen", "z-ai", "meta-llama"]
    all_mids = sorted(set(model_ids.values()),
                      key=lambda m: (LAB_ORDER.index(LAB(m)) if LAB(m) in LAB_ORDER else 99, m))

    out = {
        "base": analyze_view(base_inv, ["baseline", "analogy", "blending"], all_mids, E, anchors),
        "emergent": analyze_view(emg_inv, ["analogy", "blending"], all_mids, E, anchors),
    }

    for view in ("base", "emergent"):
        print(f"\n############  VIEW: {view.upper()}  ############")
        for mode, R in out[view].items():
            excess = R["hivemind_index"] - R["null_mean"]
            sl, cl = R["same_lab_sim"], R["cross_lab_sim"]
            print(f"  {mode:9s}  conv={R['hivemind_index']:.3f} (null {R['null_mean']:.3f}, "
                  f"excess {excess:+.3f})  RSA anchor r={R['rsa_anchor_r']:+.2f} p={R['rsa_anchor_p']:.4f}  "
                  f"sep->conv r={R['anchor_sep_vs_convergence_r']:+.2f}  "
                  f"lab {sl:.3f}/{cl:.3f}" if sl and cl else f"  {mode}: conv={R['hivemind_index']:.3f}")

    (OUT / "invention_homogeneity.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nsaved -> {OUT/'invention_homogeneity.json'}")


if __name__ == "__main__":
    main()
