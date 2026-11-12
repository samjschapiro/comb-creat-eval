"""Diversity of SCTT responses, for humans (DRAT study package) and LLMs (human-battery run).

Every SCTT scenario asks for three different responses. That gives three diversity measures:

- within-set: how different the three responses one individual gives to one scenario are
  (humans: their one set; LLMs: each call's set, averaged over calls);
- across-call (LLMs only): how different responses from separate calls on the same scenario are,
  i.e. the paper's distributional creativity, restricted to the SCTT items;
- population: how different one individual's responses are from everyone else's in the same
  population on the same scenario.

Two pairwise scorers: SBERT cosine distance (1 - cosine) and the NoveltyBench DeBERTa
equivalence classifier (share of pairs judged distinct). Raw scores only: no adjustment for
response length, by design.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.new_tests.human_battery import parse_list


def load_human_sets(responses_csv, keep_pids, items):
    """One row per (pid, item_id, response_index) for the given items; empty responses dropped."""
    r = pd.read_csv(responses_csv, low_memory=False)
    r = r[(r["test"] == "sctt") & r["pid"].isin(keep_pids) & r["item_id"].isin(items)]
    r = r.assign(text=r["response"].fillna("").astype(str).str.strip())
    n_empty = int((r["text"] == "").sum())
    r = r[r["text"] != ""]
    out = r[["pid", "item_id", "response_index", "text"]].rename(columns={"pid": "unit"})
    return out.assign(call=1).reset_index(drop=True), n_empty


def load_llm_sets(responses_dir, items, models, n_responses):
    """One row per (model, item_id, call, response_index), for the given models only.

    As in prep_sctt_judging_llm.py (which defines the responses the SCTT judges scored), a call
    contributes the first `n_responses` items of its parsed list. Calls with an API error, no
    parsable list, or fewer than `n_responses` items are dropped; every dropped or truncated call
    is returned in a log table with the reason.
    """
    rows, log = [], []
    for f in sorted(Path(responses_dir).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            rec = json.loads(line)
            if rec["test"] != "sctt" or rec["item_id"] not in items or rec["model"] not in models:
                continue
            key = {"unit": rec["model"], "item_id": rec["item_id"], "call": int(rec["sample"])}
            parsed = parse_list(rec["content"]) if rec["content"] else None
            if not rec["content"]:
                log.append(key | {"n_parsed": None, "action": "dropped", "reason": str(rec.get("error"))[:120]})
                continue
            if parsed is None or len(parsed) < n_responses:
                log.append(key | {"n_parsed": None if parsed is None else len(parsed), "action": "dropped",
                                  "reason": "unparsed" if parsed is None else f"fewer_than_{n_responses}"})
                continue
            if len(parsed) > n_responses:
                log.append(key | {"n_parsed": len(parsed), "action": f"truncated_to_{n_responses}", "reason": ""})
            for i, text in enumerate(parsed[:n_responses], 1):
                rows.append(key | {"response_index": i, "text": " ".join(text.split())})
    return pd.DataFrame(rows), pd.DataFrame(log, columns=["unit", "item_id", "call", "n_parsed", "action", "reason"])


def embed(texts, model_name, device):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name, device=device)
    v = st.encode(list(texts), batch_size=64, show_progress_bar=False, convert_to_numpy=True).astype(np.float64)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def within_and_across_pairs(df):
    """Pairs of row indices of the same unit and item: kind 'within' (same call) or 'across'."""
    pairs = []
    for (unit, item), g in df.groupby(["unit", "item_id"]):
        idx, calls = g.index.to_numpy(), g["call"].to_numpy()
        for a, b in itertools.combinations(range(len(idx)), 2):
            pairs.append((unit, item, "within" if calls[a] == calls[b] else "across", idx[a], idx[b]))
    return pd.DataFrame(pairs, columns=["unit", "item_id", "kind", "i", "j"])


def score_pairs(pairs, df, vecs, classifier_device, classifier_threshold):
    from src.new_tests.distinctness import batch_distinct
    i, j = pairs["i"].to_numpy(), pairs["j"].to_numpy()
    cos_dist = 1 - np.einsum("ij,ij->i", vecs[i], vecs[j])
    texts = df["text"].to_numpy()
    distinct = batch_distinct(list(zip(texts[i], texts[j])), threshold=classifier_threshold,
                              device=classifier_device, batch_size=32)
    return pairs.assign(sbert_dist=cos_dist, distinct=np.array(distinct, dtype=float))


def population_distance(df, vecs):
    """Per (unit, item): mean SBERT distance from the unit's responses to all other units'
    responses on that item, in the same population; plus each item's overall mean pairwise
    distance between responses of different units."""
    per_unit, per_item = [], []
    for item, g in df.groupby("item_id"):
        v = vecs[g.index.to_numpy()]
        units = g["unit"].to_numpy()
        d = 1 - v @ v.T
        other = units[:, None] != units[None, :]
        per_item.append({"item_id": item, "mean_between_unit_dist": float(d[other].mean())})
        for u in np.unique(units):
            rows = units == u
            per_unit.append({"unit": u, "item_id": item, "population_dist": float(d[rows][:, ~rows].mean())})
    return pd.DataFrame(per_unit), pd.DataFrame(per_item)


def bootstrap_between_unit(df, vecs, n_boot, seed):
    """Bootstrap (over units) the item-averaged mean between-unit SBERT distance."""
    rng = np.random.default_rng(seed)
    units = df["unit"].unique()
    by_item = {item: (vecs[g.index.to_numpy()], g["unit"].to_numpy()) for item, g in df.groupby("item_id")}
    stats_ = []
    for _ in range(n_boot):
        draw = rng.choice(units, size=len(units), replace=True)
        weight = pd.Series(draw).value_counts()
        vals = []
        for v, u in by_item.values():
            w = weight.reindex(u).fillna(0).to_numpy()
            keep = w > 0
            vv, uu, ww = v[keep], u[keep], w[keep]
            d = 1 - vv @ vv.T
            other = uu[:, None] != uu[None, :]
            ww2 = np.outer(ww, ww) * other
            vals.append(float((d * ww2).sum() / ww2.sum()))
        stats_.append(np.mean(vals))
    return np.percentile(stats_, [2.5, 97.5]).tolist()


def item_alpha_and_split_half(wide):
    """Cronbach's alpha over items and odd/even split-half (Spearman-Brown) for a unit x item table.
    Units missing any item are dropped and counted."""
    from src.new_tests.drat_human import cronbach_alpha, spearman_brown
    full = wide.dropna()
    cols = sorted(full.columns)
    odd, even = full[cols[0::2]].mean(axis=1), full[cols[1::2]].mean(axis=1)
    r = float(np.corrcoef(odd, even)[0, 1])
    return {"n": int(len(full)), "n_dropped_missing_item": int(len(wide) - len(full)),
            "alpha": round(float(cronbach_alpha(full)), 3), "split_half_r": round(r, 3),
            "spearman_brown": round(float(spearman_brown(r)), 3)}
