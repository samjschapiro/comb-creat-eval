"""Score the LLMs' human-battery answers exactly as the human study scores its participants.

DRAT and DAT go through the data package's own scorer (scoring/drat_scoring.py: noun-pool threshold,
>= 3 survivors, GloVe 840B / fastText crawl-2M / SBERT all-mpnet-base-v2) with its word cleaning
copied from scoring/score_dataset.py (clean / resolve / duplicate and multi-word removal). The same
code re-scores the human DRAT sets first and must reproduce the package's human scores; the run
stops if it does not. RAT: first word of the answer against the human answer key, plural-folded,
as in the human scoring. The DAT words are the dat_eval pool's existing runs (temperature 1.0).
"""
import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.human_battery import parse_list
from src.utils import init_directory, load_config, save_config

MODELS = ("glove", "fasttext", "sbert")


def load_package_scorer(scoring_dir):
    spec = importlib.util.spec_from_file_location("drat_scoring", Path(scoring_dir) / "drat_scoring.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DictVectors:
    """Unit vectors for a word subset, with the interface drat_scoring expects."""

    def __init__(self, vecs, name):
        self.v, self.name = vecs, name

    def __contains__(self, w):
        return w in self.v

    def __getitem__(self, w):
        return self.v[w]


def load_text_vectors(path, words):
    words, out = set(words), {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w, rest = line.split(" ", 1)
            if w in words and w not in out:
                v = np.asarray(rest.split(), dtype=np.float32)
                if v.size == 300:
                    out[w] = v / (np.linalg.norm(v) + 1e-9)
    return out


# --- copied from the package's scoring/score_dataset.py so cleaning is identical ---
def clean(raw):
    w = (raw or "").strip().lower()
    w = re.sub(r"[^a-z\- ]", "", w).strip()
    return w, {"multiword": " " in w, "empty": w == ""}


def resolve(w, model, wn, sbert_cls):
    if w in model and not isinstance(model, sbert_cls):
        return w
    if isinstance(model, sbert_cls):
        return w if w else None
    m = wn.morphy(w, wn.NOUN)
    if m and m in model:
        return m
    w2 = w.replace("-", "")
    if w2 in model:
        return w2
    return None
# ------------------------------------------------------------------------------------


def candidate_forms(words, wn):
    out = set()
    for w in words:
        out |= {w, w.replace("-", "")}
        m = wn.morphy(w, wn.NOUN)
        if m:
            out.add(m)
    return out


def drat_sets_table(human_sets, llm_rows):
    """One row per (respondent, set): respondent = pid or model, words = up to 10 typed/parsed words."""
    rows = []
    for _, s in human_sets.iterrows():
        words = [s[f"w{i:02d}"] for i in range(1, 11) if isinstance(s[f"w{i:02d}"], str) and s[f"w{i:02d}"].strip()]
        rows.append({"population": "human", "respondent": s["pid"], "set_id": s["set_id"], "anchors": s["anchors"],
                     "relation": s["relation"], "k": int(s["k"]), "words": words, "parsed": True})
    for rec in llm_rows:
        words = parse_list(rec["content"]) if rec["content"] else None
        rows.append({"population": "llm", "respondent": rec["model"], "set_id": rec["item_id"],
                     "anchors": rec["item_id"], "relation": rec["relation"], "k": rec["k"],
                     "words": (words or [])[:10], "parsed": words is not None})
    return pd.DataFrame(rows)


def score_drat_sets(t, scorer, vec, pool, wn, q, n_min):
    tau = {}
    out = []
    for _, s in t.iterrows():
        anchors = s["anchors"].split("|")
        seen, clean_words = set(), []
        for raw in s["words"]:
            w, fl = clean(raw)
            if not w or fl["multiword"] or w in seen:
                seen.add(w)
                continue
            seen.add(w)
            clean_words.append(w)
        rec = {k: s[k] for k in ("population", "respondent", "set_id", "anchors", "relation", "k", "parsed")}
        rec["n_typed"] = len(s["words"])
        scores = []
        for name in MODELS:
            m = vec[name]
            key = (name, s["anchors"])
            if key not in tau:
                tau[key] = scorer.pool_threshold(anchors, pool, m, q, "max")[0]
            forms = [f for f in (resolve(w, m, wn, scorer.SbertVectors) for w in clean_words) if f]
            res = scorer.score_drat(forms, anchors, m, tau[key], n_min)
            rec[f"{name}_n_surv"] = res["n_survivors"]
            rec[f"{name}_score"] = res["score"]
            rec[f"{name}_dist_all"] = scorer.pairwise_distance(forms, m) if len(forms) >= 2 else np.nan
            if not np.isnan(res["score"]):
                scores.append(res["score"])
        rec["composite_score"] = float(np.mean(scores)) if len(scores) == len(MODELS) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


def check_human_reproduction(scored, package_sets):
    h = scored[scored["population"] == "human"].set_index(["respondent", "set_id"])
    p = package_sets.set_index(["pid", "set_id"]).loc[h.index]
    res = {}
    for name in MODELS:
        same = (h[f"{name}_n_surv"].values == p[f"{name}_n_surv"].values).mean()
        diff = np.nanmax(np.abs(h[f"{name}_score"].values - p[f"{name}_score"].values))
        nan_mismatch = int((h[f"{name}_score"].isna().values != p[f"{name}_score"].isna().values).sum())
        res[name] = {"share_same_n_surv": round(float(same), 4), "max_abs_score_diff": round(float(diff), 4),
                     "scorable_mismatches": nan_mismatch}
    comp = np.nanmax(np.abs(h["composite_score"].values - p["composite_score"].values))
    res["composite_max_abs_diff"] = round(float(comp), 4)
    bad = [n for n in MODELS if res[n]["share_same_n_surv"] < 0.995 or res[n]["scorable_mismatches"] > 0.005 * len(h)]
    if bad or comp > 0.05:
        raise ValueError(f"FATAL: re-scoring does not reproduce the package's human DRAT scores: {res}")
    return res


def rat_correct(content, solution):
    m = re.search(r"[A-Za-z]+", content or "")
    if not m:
        return False
    w = m.group(0).lower()
    fold = lambda x: x[:-2] if x.endswith("es") else x[:-1] if x.endswith("s") else x
    return w == solution or fold(w) == fold(solution)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "llm_run_dir", "dat_dir", "dat_file", "output_dir", "glove_path", "fasttext_path",
                "nltk_data", "q", "n_min"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    import nltk
    nltk.data.path.insert(0, config["nltk_data"])
    from nltk.corpus import wordnet as wn
    wn.ensure_loaded()

    up = Path(config["upstream_dir"])
    scorer = load_package_scorer(up / "scoring")
    pool = [w for w in (up / "stimuli" / "drat_noun_pool_v1.txt").read_text().split() if w]
    package_sets = pd.read_csv(up / "drat_sets.csv")
    human_sets = package_sets[(package_sets["sample"] == "main") & (~package_sets["exclude"].astype(bool))]
    llm_rows = [json.loads(l) for f in sorted((Path(config["llm_run_dir"]) / "responses").glob("*.jsonl"))
                for l in f.read_text().splitlines()]
    llm_models = sorted({r["model"] for r in llm_rows if r["content"]})
    t = drat_sets_table(human_sets, [r for r in llm_rows if r["test"] == "drat" and r["model"] in llm_models])

    dat = {}
    for m in llm_models:
        f = Path(config["dat_dir"]) / m.replace("/", "_").replace(".", "-") / config["dat_file"]
        if not f.exists():
            raise FileNotFoundError(f"FATAL: no DAT run for {m} at {f}")
        dat[m] = [[w for w in trial["words"] if isinstance(w, str)] for trial in json.loads(f.read_text())]

    all_clean = {clean(w)[0] for ws in t["words"] for w in ws} | {clean(w)[0] for trials in dat.values() for ws in trials for w in ws}
    all_clean.discard("")
    anchors = {a for x in t["anchors"] for a in x.split("|")}
    needed = candidate_forms(all_clean, wn) | set(pool) | anchors
    vec = {"glove": DictVectors(load_text_vectors(config["glove_path"], needed), "glove"),
           "fasttext": DictVectors(load_text_vectors(config["fasttext_path"], needed), "fasttext"),
           "sbert": scorer.SbertVectors(cache=out / "sbert_cache.npz")}
    vec["sbert"].preload(set(pool) | anchors | all_clean)

    scored = score_drat_sets(t, scorer, vec, pool, wn, config["q"], config["n_min"])
    res = {"human_reproduction": check_human_reproduction(scored, package_sets)}
    scored.to_csv(out / "drat_sets_scored.csv", index=False)

    rat = pd.DataFrame([r for r in llm_rows if r["test"] == "rat" and r["model"] in llm_models])
    rat["correct"] = [rat_correct(c, s) for c, s in zip(rat["content"], rat["solution"])]
    rat.drop(columns=["content"]).to_csv(out / "rat_scored.csv", index=False)

    dat_rows = []
    for m, trials in dat.items():
        for i, ws in enumerate(trials):
            cw = [clean(w)[0] for w in ws if clean(w)[0] and " " not in clean(w)[0]]
            rec = {"model": m, "trial": i, "n_words": len(cw)}
            sc = []
            for name in MODELS:
                forms = [f for f in (resolve(w, vec[name], wn, scorer.SbertVectors) for w in cw) if f]
                sc.append(scorer.score_dat(forms, vec[name])["score"])
                rec[f"{name}_dat"] = sc[-1]
            rec["dat_composite"] = float(np.mean(sc))
            dat_rows.append(rec)
    dat_scored = pd.DataFrame(dat_rows)
    dat_scored.to_csv(out / "dat_scored.csv", index=False)

    llm = scored[scored["population"] == "llm"]
    g = llm.groupby("respondent")
    models = pd.DataFrame({
        "drat_score": g["composite_score"].mean(), "drat_sets_scorable": g["composite_score"].count(),
        "drat_relevance": g["glove_n_surv"].mean(), "drat_ungated": g["glove_dist_all"].mean(),
        "drat_words": g["n_typed"].mean(), "drat_parse_failures": g["parsed"].apply(lambda x: int((~x).sum())),
        "rat_correct": rat.groupby("model")["correct"].sum(),
        "dat_composite": dat_scored.groupby("model")["dat_composite"].mean(),
    })
    for k in (2, 3):
        models[f"drat_score_k{k}"] = llm[llm["k"] == k].groupby("respondent")["composite_score"].mean()
    models.index.name = "model"
    models.to_csv(out / "model_scores.csv")
    res["n_models"] = int(len(models))
    res["drat_pct_scorable_llm"] = round(100 * llm["composite_score"].notna().mean(), 1)
    res["drat_pct_scorable_human"] = round(100 * scored.loc[scored["population"] == "human", "composite_score"].notna().mean(), 1)
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(models.round(2).sort_values("drat_score").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
