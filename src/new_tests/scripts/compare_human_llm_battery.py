"""Same battery, two populations: how the creativity tests relate to each other in humans vs LLMs.

Humans: person-level scores from analyze_drat_human_psychometrics.py (240 analysed participants).
LLMs: model-level scores from score_human_battery_llm.py and score_sctt_judging_llm.py (same items,
same wording, same scorer, same SCTT judges). General ability is Raven's 2 for humans and MMLU-Pro
for LLMs (as in the NeurIPS paper). Reports each population's Spearman matrix, per-pair differences
(Fisher z for independent samples), and the incremental-validity regression for SCTT creativity.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.drat_human import heatmap, save, set_camera_ready_style
from src.utils import init_directory, load_config, save_config

VARS = {"drat_score": "DRAT score", "drat_relevance": "DRAT-A", "drat_ungated": "DRAT-N",
        "dat": "DAT", "rat": "RAT", "sctt_creativity": "SCTT creativity", "ability": "General ability"}


def rat_lenient(content, solution):
    """Bolded or quoted answer word if the model wrapped its answer in a sentence, else the first word."""
    c = content or ""
    m = re.search(r"\*\*([A-Za-z]+)\*\*", c) or re.search(r"[\"“]([A-Za-z]+)[\"”]", c) or re.search(r"[A-Za-z]+", c)
    if not m:
        return False
    w = (m.group(1) if m.groups() else m.group(0)).lower()
    fold = lambda x: x[:-2] if x.endswith("es") else x[:-1] if x.endswith("s") else x
    return w == solution or fold(w) == fold(solution)


def load_humans(path):
    t = pd.read_csv(path, index_col="pid")
    return t.rename(columns={"raven_standard_score": "ability"})[list(VARS) + ["sctt_length"]]


def load_llms(scores_dir, sctt_dir, run_dir, benchmarks):
    m = pd.read_csv(Path(scores_dir) / "model_scores.csv", index_col="model")
    s = pd.read_csv(Path(sctt_dir) / "model_scores.csv", index_col="model")
    m = m.join(s[["sctt_creativity", "sctt_length"]])
    m = m.rename(columns={"dat_composite": "dat", "rat_correct": "rat"})
    bench = json.load(open(benchmarks))
    m["ability"] = [bench.get(i.replace("/", "_").replace(".", "-"), {}).get("mmlu_pro", np.nan) for i in m.index]
    raw = [json.loads(l) for f in sorted((Path(run_dir) / "responses").glob("*.jsonl")) for l in f.read_text().splitlines()]
    rat = pd.DataFrame([r for r in raw if r["test"] == "rat" and r["model"] in m.index])
    m["rat_lenient"] = rat.assign(c=[rat_lenient(c, s) for c, s in zip(rat["content"], rat["solution"])]).groupby("model")["c"].sum()
    return m


def spearman_table(df, label):
    rows = []
    keys = list(VARS)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            d = df[[a, b]].dropna()
            rho, p = stats.spearmanr(d[a], d[b])
            rows.append({"population": label, "a": a, "b": b, "n": len(d), "rho": rho, "p": p})
    return pd.DataFrame(rows)


def compare(h, l):
    t = h.merge(l, on=["a", "b"], suffixes=("_human", "_llm"))
    z = lambda r: np.arctanh(np.clip(r, -0.999, 0.999))
    se = np.sqrt(1 / (t["n_human"] - 3) + 1 / (t["n_llm"] - 3))
    t["z_diff"] = (z(t["rho_human"]) - z(t["rho_llm"])) / se
    t["p_diff"] = 2 * stats.norm.sf(np.abs(t["z_diff"]))
    return t.round(4)


def incremental(df, label):
    d = df[["sctt_creativity", "dat", "rat", "drat_score", "ability", "sctt_length"]].dropna()
    z = (d - d.mean()) / d.std()
    out = {"population": label, "n": int(len(z))}
    for name, base, add in (("drat_over_dat_rat", "dat + rat", "drat_score"),
                            ("drat_over_dat_rat_ability", "dat + rat + ability", "drat_score"),
                            ("dat_rat_over_drat", "drat_score", "dat + rat")):
        m0 = smf.ols(f"sctt_creativity ~ {base}", z).fit()
        m1 = smf.ols(f"sctt_creativity ~ {base} + {add}", z).fit()
        f, p, _ = m1.compare_f_test(m0)
        out[name] = {"r2_base": round(m0.rsquared, 3), "r2_full": round(m1.rsquared, 3),
                     "delta_r2": round(m1.rsquared - m0.rsquared, 3), "p": float(p)}
    full = smf.ols("sctt_creativity ~ dat + rat + drat_score", z).fit()
    out["betas"] = {k: round(v, 3) for k, v in full.params.items() if k != "Intercept"}
    out["beta_p"] = {k: float(v) for k, v in full.pvalues.items() if k != "Intercept"}
    return out


def plot(h, l, path):
    keys = list(VARS)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.2), gridspec_kw={"wspace": 0.42})
    for ax, (t, title) in zip(axes, ((h, "(a) Humans"), (l, "(b) LLMs"))):
        mat = np.full((len(keys), len(keys)), np.nan)
        stars = [[""] * len(keys) for _ in keys]
        for _, r in t.iterrows():
            i, j = keys.index(r["a"]), keys.index(r["b"])
            mat[j, i] = r["rho"]
            stars[j][i] = "*" if r["p"] < 0.05 else ""
        # lower triangle only: drop the empty first row and last column
        im = heatmap(ax, mat[1:, :-1], [row[:-1] for row in stars[1:]], vmax=0.8)
        ax.set_xticks(range(len(keys) - 1), [VARS[k] for k in keys[:-1]], rotation=35, ha="right")
        ax.set_yticks(range(len(keys) - 1), [VARS[k] for k in keys[1:]])
        n = int(t["n"].max())
        ax.set_title(f"{title} (n = {n})")
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label(r"Spearman $\rho$")
    cbar.outline.set_visible(False)
    save(fig, path)


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("human_measures", "llm_scores_dir", "llm_sctt_dir", "llm_run_dir", "benchmarks", "output_dir"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()

    hum = load_humans(config["human_measures"])
    llm = load_llms(config["llm_scores_dir"], config["llm_sctt_dir"], config["llm_run_dir"], config["benchmarks"])
    llm.to_csv(out / "llm_model_table.csv")
    th, tl = spearman_table(hum, "human"), spearman_table(llm, "llm")
    comp = compare(th, tl)
    comp.to_csv(out / "correlations_compared.csv", index=False)
    tl_len = spearman_table(llm.assign(rat=llm["rat_lenient"]), "llm_rat_lenient")
    tl_len.to_csv(out / "llm_correlations_rat_lenient.csv", index=False)
    res = {"n_human": int(len(hum)), "n_llm": int(len(llm)), "n_llm_with_mmlu_pro": int(llm["ability"].notna().sum()),
           "incremental": [incremental(hum, "human"), incremental(llm, "llm")],
           "sctt_vs_length": {"human": round(hum[["sctt_creativity", "sctt_length"]].corr("spearman").iloc[0, 1], 3),
                              "llm": round(llm[["sctt_creativity", "sctt_length"]].corr("spearman").iloc[0, 1], 3)}}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    set_camera_ready_style()
    plot(th, tl, out / "figures" / "fig_human_vs_llm")
    print(json.dumps(res, indent=2))
    print(comp[["a", "b", "n_human", "rho_human", "p_human", "n_llm", "rho_llm", "p_llm", "z_diff", "p_diff"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
