"""Corpus contamination check for Kombine (rebuttal, reviewer En17 Q1: memorisation).

Asks whether Kombine's combinations are *documented* in open pretraining text, using the public
infini-gram API (https://api.infini-gram.io; free, no key). We cannot search closed models' training data,
so two open corpora stand in for it:
  BIG    v4_olmo-mix-1124_llama      4.6T tokens (OLMo 2 pretraining mix; includes DCLM web text)
  SMALL  v4_dolma-v1_6-sample_llama  9.2B tokens (a Dolma sample, small enough that counts are EXACT)

Counting rules (from probing the API, 2026-11-14):
  * single-phrase counts are exact on both indexes;
  * co-occurrence ("a AND b" within W tokens) on BIG is *sampled* when a term has > 500k matches. A sampled
    nonzero is a real match, but the magnitude is unreliable and a sampled zero is inconclusive.
  => "documented" = nonzero on BIG; a zero is accepted only if BIG says it is exact; otherwise we fall back to
     the exact SMALL count, and if that is zero too the link is recorded as 'inconclusive'.
  => magnitudes (anchor co-occurrence strength) use SMALL, where every count is exact and comparable.
Strings are matched exactly (Llama-2 tokens, case-sensitive), as an OR over case variants and with a leading
"The " stripped. Exact matching UNDERcounts documentation (paraphrases are missed); every statement below is
read with that in mind.

Checks
  1. anchors     for each item's two input entities: exact co-occurrence within 100 / 1000 tokens (SMALL),
                 lift over independence, and documented-or-not on BIG.
  2. hops        "each step known, the combination not" (Swanson's undiscovered public knowledge): for a
                 stratified sample of utility-passing association chains, is every hop documented while the
                 chain's two endpoints are not?
     names       does the invented concept's name (analogy invention h, blend c') occur verbatim (BIG)?
  3. predictors  per item: does anchor co-occurrence predict pass rate, originality, or inventive multiples
                 (Spearman, and partial on entity prominence = mean log sitelinks)?

    .venv/bin/python -m src.kg_creat.scripts.corpus_novelty            # queries are cached; reruns are free
"""
import json
import random
import re
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

API = "https://api.infini-gram.io/"
BIG, SMALL = "v4_olmo-mix-1124_llama", "v4_dolma-v1_6-sample_llama"
PAPER = Path("data/kg_creat/human_study/paper_scores")
RESP = Path("data/kg_creat/kombine_test30/responses")
POOL = Path("data/kg_creat/pool_wikidata.json")
MULT = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT = Path("data/kg_creat/corpus_novelty")
CACHE = OUT / "cache.jsonl"
N_PATHS = 1000
WORKERS = 6
_lock = threading.Lock()


# ---------------------------------------------------------------- API + cache
def _load_cache():
    c = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                c[r["key"]] = r["res"]
    return c


CACHE_D = {}


def query(index, q, window=100):
    cnf = " AND " in q
    key = f"{index}|{window if cnf else ''}|{q}"
    if key in CACHE_D:
        return CACHE_D[key]
    body = {"index": index, "query_type": "count", "query": q}
    if cnf:
        body |= {"max_clause_freq": 500000, "max_diff_tokens": window}
    res = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            d = json.loads(urllib.request.urlopen(req, timeout=120).read())
            if d.get("error"):
                raise RuntimeError(d["error"])
            res = {"count": d["count"], "approx": bool(d.get("approx"))}
            break
        except Exception as e:  # noqa: BLE001 -- transient server / network errors: back off and retry
            err = str(e)
            time.sleep(2 * 2 ** attempt)
    if res is None:
        return {"count": None, "approx": None, "error": err}
    with _lock:
        CACHE_D[key] = res
        with CACHE.open("a") as f:
            f.write(json.dumps({"key": key, "res": res}) + "\n")
    return res


def run_all(jobs):
    """jobs: list of (index, q, window); fills the cache in parallel."""
    todo = sorted({j for j in jobs if f"{j[0]}|{j[2] if ' AND ' in j[1] else ''}|{j[1]}" not in CACHE_D})
    print(f"  {len(jobs)} queries, {len(todo)} uncached")
    done = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for _ in ex.map(lambda j: query(*j), todo):
            done += 1
            if done % 500 == 0:
                print(f"    {done}/{len(todo)}", flush=True)


# ---------------------------------------------------------------- strings
_BAD = re.compile(r"\b(AND|OR)\b")


def clause(s):
    """OR-clause over case variants of one entity / phrase."""
    s = re.sub(r"\s+", " ", str(s)).strip().strip(".,;:'\"")
    s = _BAD.sub(lambda m: m.group(0).lower(), s)
    vs = {s, s.lower(), s[:1].upper() + s[1:]}
    if s.lower().startswith("the ") and len(s) > 4:
        t = s[4:]
        vs |= {t, t.lower(), t[:1].upper() + t[1:]}
    return " OR ".join(sorted(v for v in vs if v))


def cooc(a, b):
    return f"{clause(a)} AND {clause(b)}"


def documented(a, b, window=100):
    """True / False / None (inconclusive), per the counting rules in the module docstring."""
    big = query(BIG, cooc(a, b), window)
    if big.get("count") is None:
        return None
    if big["count"] > 0:
        return True
    if not big["approx"]:
        return False
    small = query(SMALL, cooc(a, b), window)
    if small.get("count"):
        return True
    return None


# ---------------------------------------------------------------- data
def load_items():
    """(task, prompt_id) -> (u, v), and per-item per-model artifact records from the paper run."""
    recs = {m.name: json.loads((m / "path_scores.json").read_text())
            for m in sorted(PAPER.iterdir()) if (m / "path_scores.json").exists() and not m.name.startswith("human")}
    items = {}
    for p in sorted(RESP.glob("*/responses.json")):
        for r in json.loads(p.read_text()):
            items[r["prompt_id"]] = (r["mode"], r["u_label"], r["v_label"])
    return items, recs


def artifacts(recs):
    """per (prompt_id): list of (model, passed, originality, em_originality, triples) at the artifact level."""
    out = defaultdict(list)
    for m, rs in recs.items():
        for r in rs:
            if r["mode"] == "analogy":
                if "pair_sat" not in r:
                    continue
                ok = r.get("pair_sat") is True
            elif r.get("triples"):
                ok = r.get("sat") is True
            else:
                continue
            out[r["prompt_id"]].append((m, ok, r.get("originality"), r.get("em_originality"), r.get("triples")))
    return out


def invention_names():
    names = []
    for p in sorted(RESP.glob("*/responses.json")):
        for r in json.loads(p.read_text()):
            it = (r.get("items") or [None])[0]
            if not it:
                continue
            n = it.get("invention") if r["mode"] == "analogy" else it.get("concept") if r["mode"] == "blending" else None
            if n and str(n).strip():
                names.append((p.parent.name, r["mode"], r["prompt_id"], str(n).strip()))
    return names


def partial_spearman(x, y, z):
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def perm_p(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(spearmanr(x, y).statistic)
    y = np.asarray(y)
    return float((1 + sum(abs(spearmanr(x, rng.permutation(y)).statistic) >= obs for _ in range(n))) / (n + 1))


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE_D.update(_load_cache())
    items, recs = load_items()
    arts = artifacts(recs)
    pool = {r["label"]: r for r in json.loads(POOL.read_text())["rows"]}
    res = {"indexes": {"big": BIG, "small": SMALL}}

    # ---- 1. anchors
    print("1. anchor co-occurrence")
    pairs = sorted({(u, v) for (_m, u, v) in items.values()})
    jobs = [(SMALL, clause(a), 100) for p in pairs for a in p] + [(SMALL, "", 100)]
    jobs += [(SMALL, cooc(u, v), w) for u, v in pairs for w in (100, 1000)] + [(BIG, cooc(u, v), 100) for u, v in pairs]
    run_all(jobs)
    n_tok = query(SMALL, "")["count"]
    anchor = {}
    for u, v in pairs:
        cu, cv = query(SMALL, clause(u))["count"], query(SMALL, clause(v))["count"]
        c100, c1000 = query(SMALL, cooc(u, v), 100)["count"], query(SMALL, cooc(u, v), 1000)["count"]
        exp100 = cu * cv * 200 / n_tok if n_tok else None      # expected within +-100 tokens under independence
        anchor[f"{u} || {v}"] = {"u": u, "v": v, "count_u": cu, "count_v": cv, "cooc100": c100, "cooc1000": c1000,
                                 "lift100": (c100 / exp100) if exp100 else None, "documented_big": documented(u, v)}
    res["anchors"] = anchor

    # reference: pairs that ARE known to be linked -- the factual hops of passing association chains
    # ---- 2a. hops (stratified sample of passing association chains)
    print("2a. association hops")
    rng = random.Random(0)
    passing = defaultdict(list)
    for pid, a in arts.items():
        if items[pid][0] == "baseline":
            passing[pid] += [(m, t) for (m, ok, _o, _e, t) in a if ok and t]
    per_item = max(1, N_PATHS // len(passing))
    sample = []
    for pid in sorted(passing):
        rng.shuffle(passing[pid])
        sample += [(pid, m, t) for m, t in passing[pid][:per_item]]
    hop_pairs = sorted({(h[0], h[2]) for _p, _m, t in sample for h in t if h[0] != h[2]})
    run_all([(BIG, cooc(a, b), 100) for a, b in hop_pairs])
    run_all([(SMALL, cooc(a, b), 100) for a, b in hop_pairs if query(BIG, cooc(a, b)).get("count") == 0
             and query(BIG, cooc(a, b)).get("approx")])
    hop_doc = {(a, b): documented(a, b) for a, b in hop_pairs}
    chains = []
    for pid, m, t in sample:
        _mode, u, v = items[pid]
        hd = [hop_doc.get((h[0], h[2]), True) for h in t if h[0] != h[2]]
        chains.append({"prompt_id": pid, "model": m, "n_hops": len(t), "hops_documented": hd,
                       "all_hops_documented": all(x is True for x in hd),
                       "endpoints_documented": anchor[f"{u} || {v}"]["documented_big"],
                       "endpoints_cooc100_small": anchor[f"{u} || {v}"]["cooc100"]})
    res["hops"] = {"n_chains": len(chains), "n_unique_hops": len(hop_pairs),
                   "hop_documented": sum(x is True for x in hop_doc.values()) / len(hop_doc),
                   "hop_undocumented": sum(x is False for x in hop_doc.values()) / len(hop_doc),
                   "hop_inconclusive": sum(x is None for x in hop_doc.values()) / len(hop_doc),
                   "chains": chains}

    # ---- 2b. invention names
    print("2b. invention names")
    names = invention_names()
    run_all([(BIG, clause(n), 100) for *_x, n in names])
    nm = []
    for model, mode, pid, n in names:
        c = query(BIG, clause(n))["count"]
        nm.append({"model": model, "mode": mode, "prompt_id": pid, "name": n, "count_big": c})
    res["names"] = nm

    # ---- 3. predictors, per item
    print("3. does anchor co-occurrence predict performance?")
    mult = {(r["task"], r["u"], r["v"]): r for r in json.loads(MULT.read_text())["per_item"]}
    rows = []
    for pid, (mode, u, v) in sorted(items.items()):
        a = arts.get(pid, [])
        if not a:
            continue
        by_model = defaultdict(list)
        for m, ok, o, e, _t in a:
            by_model[m].append((ok, o, e))
        pass_rate = float(np.mean([np.mean([x[0] for x in xs]) for xs in by_model.values()]))
        orig = [o for xs in by_model.values() for ok, o, _e in xs if ok and o is not None]
        emo = [e for xs in by_model.values() for ok, _o, e in xs if ok and e is not None]
        task = {"baseline": "association"}.get(mode, mode)
        mu = mult.get((task, u, v), {})
        an = anchor[f"{u} || {v}"]
        rows.append({"prompt_id": pid, "task": task, "u": u, "v": v,
                     "log_cooc100": float(np.log1p(an["cooc100"])), "log_cooc1000": float(np.log1p(an["cooc1000"])),
                     "log_lift100": float(np.log1p(an["lift100"] or 0)),
                     "prominence": float(np.mean([np.log1p((pool.get(x) or {}).get("sitelinks") or 0) for x in (u, v)])),
                     "pass_rate": pass_rate, "originality": float(np.mean(orig)) if orig else None,
                     "em_originality": float(np.mean(emo)) if emo else None,
                     "multiples_tau1": mu.get("rate_tau1_pct"), "multiples_per_property": mu.get("per_property_mean")})
    res["per_item"] = rows
    pred = {}
    for task in ("association", "analogy", "blending"):
        rs = [r for r in rows if r["task"] == task]
        for x in ("log_cooc100", "log_lift100"):
            for y in ("pass_rate", "originality", "em_originality", "multiples_tau1", "multiples_per_property"):
                ok = [r for r in rs if r[y] is not None]
                if len(ok) < 8:
                    continue
                xs, ys, zs = [r[x] for r in ok], [r[y] for r in ok], [r["prominence"] for r in ok]
                pred[f"{task}|{x}|{y}"] = {"n": len(ok), "spearman": float(spearmanr(xs, ys).statistic),
                                           "p_perm": perm_p(xs, ys, 5000), "partial_on_prominence": partial_spearman(xs, ys, zs)}
    res["predictors"] = pred

    (OUT / "corpus_novelty.json").write_text(json.dumps(res, indent=1, default=str))
    summarize(res)


def summarize(res):
    A = list(res["anchors"].values())
    c100 = np.array([a["cooc100"] for a in A])
    print(f"\n== 1. anchors ({len(A)} pairs; exact counts in the {SMALL} 9.2B-token sample)")
    print(f"  co-occur within 100 tokens: 0 times {np.mean(c100 == 0):.0%}, <10 {np.mean(c100 < 10):.0%}, "
          f"median {np.median(c100):.0f}, max {c100.max()}")
    print(f"  documented anywhere in {BIG} (4.6T, within 100 tokens): "
          f"yes {sum(a['documented_big'] is True for a in A)}, no {sum(a['documented_big'] is False for a in A)}, "
          f"inconclusive {sum(a['documented_big'] is None for a in A)}")
    lifts = [a["lift100"] for a in A if a["lift100"] is not None]
    print(f"  lift over independence (100 tok): median {np.median(lifts):.2f}")
    H = res["hops"]
    ch = H["chains"]
    print(f"\n== 2a. association chains ({H['n_chains']} sampled passing chains, {H['n_unique_hops']} unique hops)")
    print(f"  hops documented {H['hop_documented']:.1%}, undocumented {H['hop_undocumented']:.1%}, inconclusive {H['hop_inconclusive']:.1%}")
    allh = [c for c in ch if c["all_hops_documented"]]
    print(f"  chains with EVERY hop documented: {len(allh) / len(ch):.1%}")
    for lab, f in (("endpoints co-occur 0x in 9.2B sample", lambda c: c["endpoints_cooc100_small"] == 0),
                   ("endpoints undocumented in 4.6T", lambda c: c["endpoints_documented"] is False)):
        k = [c for c in allh if f(c)]
        print(f"  ... of which {lab}: {len(k) / max(1, len(allh)):.1%}")
    N = res["names"]
    for mode in ("analogy", "blending"):
        ns = [n for n in N if n["mode"] == mode and n["count_big"] is not None]
        cs = np.array([n["count_big"] for n in ns])
        print(f"\n== 2b. {mode} invention names (n={len(ns)}): verbatim in 4.6T tokens 0x {np.mean(cs == 0):.1%}, "
              f"1-9x {np.mean((cs > 0) & (cs < 10)):.1%}, >=10x {np.mean(cs >= 10):.1%}, >=1000x {np.mean(cs >= 1000):.1%}")
    print("\n== 3. per-item Spearman of anchor co-occurrence with outcomes (perm p; partial on prominence)")
    for k, v in res["predictors"].items():
        print(f"  {k:52s} n={v['n']:2d} rho={v['spearman']:+.2f} p={v['p_perm']:.3f} partial={v['partial_on_prominence']:+.2f}")


if __name__ == "__main__":
    main()
