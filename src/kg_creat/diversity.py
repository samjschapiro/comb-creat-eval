"""Set-level diversity D over the M resamples of a prompt (design.md's deferred D term).

Instance novelty R asks "is one artifact remote?"; diversity D asks "across the model's M
independent attempts at the same prompt, how varied are they?" -- the creativity notion that
needs a *set*, which is exactly what M>1 resampling produces. D is a pure embedding measure
(local MLX), so it is free and judge-independent: it can be computed straight off the elicitation
responses, before any utility judging.

Unit of the set:
  Regime A  -- item = one emitted path; pool all paths across the M samples of (prompt, temp).
  Regime B  -- item = one analogy/blend STRUCTURE (its whole 2-path output), one per sample.
D = mean pairwise cosine distance over the item embeddings. Reported two ways (per the design
decision): over ALL emitted items, and over only the structurally-VALID ones.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from src.kg_creat import scoring
from src.kg_creat.scoring import EmittedPath
from src.kg_creat import regime_b as RB


def _emb(triples, embed):
    """Unit-norm centroid of a path's/structure's triple-sentence embeddings."""
    p = EmittedPath([tuple(t) for t in triples])
    sents = p.triple_sentences()
    if not sents:
        return None
    v = np.mean([np.asarray(embed(s), dtype=float) for s in sents], axis=0)
    n = np.linalg.norm(v)
    return v / n if n else None


def _mean_pairwise_distance(embs):
    embs = [e for e in embs if e is not None]
    if len(embs) < 2:
        return float("nan"), len(embs)
    d = [1.0 - float(a @ b) for a, b in itertools.combinations(embs, 2)]
    return float(np.mean(d)), len(embs)


def _regime_a_items(samples):
    """(all_items, valid_items) as lists of triple-lists: one item per emitted path."""
    all_items, valid_items = [], []
    for r in samples:
        for triples in r["paths"]:
            if not triples:
                continue
            all_items.append(triples)
            wf, _ = scoring.well_formed(EmittedPath([tuple(t) for t in triples]),
                                        r["u_label"], r["v_label"], h=None)
            if wf:
                valid_items.append(triples)
    return all_items, valid_items


def _regime_b_items(samples, mode):
    """(all_items, valid_items) as flattened triple-lists: one item per 2-path STRUCTURE.

    The structure's embedding is the centroid of BOTH its paths' triples, so a whole analogy/blend
    is one point; validity is the structural predicate (same-relation + disjointness / antanaclasis
    shape), judge-independent.
    """
    all_items, valid_items = [], []
    for r in samples:
        ps = r["paths"]
        if len(ps) < 2 or not ps[0] or not ps[1]:
            continue
        p1, p2 = ps[0], ps[1]
        combined = list(p1) + list(p2)
        all_items.append(combined)
        ok = (RB.analogy_structural_ok(p1, p2)[0] if mode == "analogy"
              else RB.blend_structural_ok(p1, p2, r["u_label"])[0])
        if ok:
            valid_items.append(combined)
    return all_items, valid_items


def per_prompt_diversity(responses, embed):
    """responses = one model's response records (with temperature/sample_idx). Returns a list of
    per-(prompt_id, temperature) rows with D_all / D_valid and the set sizes."""
    groups = defaultdict(list)
    for r in responses:
        groups[(r["prompt_id"], r.get("temperature"))].append(r)

    rows = []
    for (pid, temp), samples in groups.items():
        mode = samples[0]["mode"]
        regime = samples[0]["regime"]
        if regime == "A":
            all_items, valid_items = _regime_a_items(samples)
        else:
            all_items, valid_items = _regime_b_items(samples, mode)
        d_all, n_all = _mean_pairwise_distance([_emb(t, embed) for t in all_items])
        d_val, n_val = _mean_pairwise_distance([_emb(t, embed) for t in valid_items])
        rows.append({"prompt_id": pid, "mode": mode, "regime": regime, "temperature": temp,
                     "n_samples": len(samples), "D_all": d_all, "n_items_all": n_all,
                     "D_valid": d_val, "n_items_valid": n_val})
    return rows


def aggregate_diversity(rows):
    """Pool per-prompt D up to (mode, temperature): mean D_all / D_valid over prompts."""
    by = defaultdict(lambda: {"D_all": [], "D_valid": []})
    for r in rows:
        for k in ("D_all", "D_valid"):
            v = r[k]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                by[(r["mode"], r["temperature"])][k].append(v)
    out = {}
    for (mode, temp), d in by.items():
        out[f"{mode}@T{temp}"] = {
            "mode": mode, "temperature": temp,
            "mean_D_all": round(float(np.mean(d["D_all"])), 4) if d["D_all"] else None,
            "mean_D_valid": round(float(np.mean(d["D_valid"])), 4) if d["D_valid"] else None,
            "n_prompts_all": len(d["D_all"]), "n_prompts_valid": len(d["D_valid"])}
    return out
