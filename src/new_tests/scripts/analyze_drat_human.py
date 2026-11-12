"""Analyse the human DRAT study export (main run): cell effects, reliability, validity.

Reads the three master tables of a data package (participants / drat_sets / responses) and writes
JSON + CSV tables and figures to the config's output_dir. Nothing is rescored here; all DRAT
scores are the package's own (q = .90 gate, n_min = 3, unscorable = missing).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.new_tests.drat_human import RELATIONS, crossed_mixed, load_tables, save, set_camera_ready_style
from src.utils import init_directory, load_config, save_config

CRITERIA = {
    "dat_composite": "DAT",
    "rat_correct": "RAT",
    "raven_standard_score": "Raven's 2",
    "sctt9_words_mean": "SCTT words/resp.",
    "bfi_openness": "BFI openness",
}


def cell_table(s):
    """Per relation x k: fluency, survivors, scorable rate, composite among scorable, difficulty."""
    g = s.groupby(["relation", "k"])
    t = pd.DataFrame({
        "n_sets": g.size(),
        "anchor_cosine": g["anchor_cosine"].mean(),
        "words_typed": g["n_words"].mean(),
        "survivors_glove": g["glove_n_surv"].mean(),
        "survive_rate_glove": g["glove_n_surv"].sum() / g["glove_n_valid"].sum(),
        "pct_scorable": 100 * g["scorable"].mean(),
        "composite_if_scorable": g["composite_score"].mean(),
        "dist_all_glove": g["glove_dist_all"].mean(),
        "difficulty": g["difficulty"].mean(),
    })
    return t.reindex(RELATIONS, level=0).round(3)


def person_facets(s, p, min_sets):
    """Person-level DRAT facets recomputed from the set file (so every facet uses the same sets)."""
    g = s.groupby("pid")
    f = pd.DataFrame({
        "drat_words": g["n_words"].mean(),
        "drat_survivors": g["glove_n_surv"].mean(),
        "drat_scorable": g["scorable"].sum(),
        "drat_composite": g["composite_score"].mean(),
        "drat_dist_all": g["glove_dist_all"].mean(),
        "drat_difficulty": g["difficulty"].mean(),
    })
    f.loc[f["drat_scorable"] < min_sets, "drat_composite"] = np.nan
    for k in (2, 3):
        sk = s[s["k"] == k].groupby("pid")
        f[f"drat_composite_k{k}"] = sk["composite_score"].mean()
        f[f"drat_survivors_k{k}"] = sk["glove_n_surv"].mean()
    return f.join(p.set_index("pid")[list(CRITERIA) + ["block_order", "sctt9_minutes"]])


def split_half(s, col):
    """Odd vs even presentation position, person means, Spearman-Brown corrected Pearson r."""
    s = s.assign(half=s["presentation_index"] % 2)
    h = s.groupby(["pid", "half"])[col].mean().unstack().dropna()
    r = h[0].corr(h[1])
    return {"n": int(len(h)), "r_half": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3)}


def k_consistency(f, base):
    d = f[[f"{base}_k2", f"{base}_k3"]].dropna()
    rho, pval = stats.spearmanr(d.iloc[:, 0], d.iloc[:, 1])
    return {"n": int(len(d)), "rho": round(rho, 3), "p": float(pval)}


def validity_table(f, facets):
    rows = []
    for fac in facets:
        for crit, label in CRITERIA.items():
            d = f[[fac, crit]].dropna()
            rho, pval = stats.spearmanr(d[fac], d[crit])
            rows.append({"facet": fac, "criterion": label, "n": len(d), "rho": rho, "p": pval})
    t = pd.DataFrame(rows)
    t["p_bh"] = multipletests(t["p"], method="fdr_bh")[1]
    return t.round(4)


def partial_spearman(f, x, y, z):
    """Spearman partial r(x, y | z) via rank residuals."""
    d = f[[x, y, z]].dropna().rank()
    rx = d[x] - np.polyval(np.polyfit(d[z], d[x], 1), d[z])
    ry = d[y] - np.polyval(np.polyfit(d[z], d[y], 1), d[z])
    rho, pval = stats.pearsonr(rx, ry)
    return {"n": int(len(d)), "rho": round(rho, 3), "p": float(pval)}


def word_level(r):
    """Survival of nouns vs non-nouns and of anchor repeats (GloVe)."""
    w = r[(r["test"] == "drat_word") & r["glove_survives"].notna()].copy()
    w["glove_survives"] = w["glove_survives"].astype(str) == "True"
    w["is_noun"] = w["is_noun"].astype(str) == "True"
    out = {"n_words_scored_glove": int(len(w)), "pct_non_noun": round(100 * (~w["is_noun"]).mean(), 1)}
    for name, sub in (("noun", w[w["is_noun"]]), ("non_noun", w[~w["is_noun"]])):
        out[f"survive_rate_{name}"] = round(sub["glove_survives"].mean(), 3)
    # Pilot report's all-anchor check: of the max-gate survivors, how many also clear tau on
    # their MIN similarity (related to every anchor, as the instruction asks).
    surv = w[w["glove_survives"]].assign(min_pass=lambda d: d["glove_u_min"] > d["glove_tau"])
    out["n_survivors_glove"] = int(len(surv))
    out["survivors_passing_min_by_k"] = surv.groupby("k")["min_pass"].mean().round(3).to_dict()
    out["survivors_passing_min_by_cell"] = {
        f"{rel}_k{int(k)}": round(v, 3) for (rel, k), v in surv.groupby(["relation", "k"])["min_pass"].mean().items()}
    return out


K_COLORS = {2: "#0072B2", 3: "#D55E00"}  # Okabe-Ito blue / vermillion


def plot_cells(s, path):
    """Relation x k cell means (± 1 SE over participants) for the two DRAT facets and the score."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    panels = [("glove_n_surv", "(a) Relevance", "Survivors per set"),
              ("scorable", "(b) Scorable sets", "% of sets"),
              ("composite_score", "(c) Diversity", "DRAT score")]
    x = np.arange(len(RELATIONS))
    for ax, (col, title, ylabel) in zip(axes, panels):
        for i, k in enumerate((2, 3)):
            g = s[s["k"] == k].groupby("relation")[col]
            m, se = g.mean().reindex(RELATIONS), g.sem().reindex(RELATIONS)
            if col == "scorable":
                m, se = 100 * m, 100 * se
            ax.errorbar(x + (i - 0.5) * 0.18, m, yerr=se, marker="o", ms=7, lw=1.8, capsize=3,
                        color=K_COLORS[k], label=f"$k = {k}$")
        ax.set_xticks(x, [r[:5] + "." if len(r) > 8 else r for r in RELATIONS])
        ax.set_xlim(-0.4, len(RELATIONS) - 0.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout(w_pad=2.0)
    save(fig, path)


def plot_validity(t, facets, labels, path):
    """Spearman rho of each DRAT facet with each criterion; * = BH-corrected p < .05."""
    cols = list(CRITERIA.values())
    mat = t.pivot(index="facet", columns="criterion", values="rho").loc[facets, cols]
    sig = t.pivot(index="facet", columns="criterion", values="p_bh").loc[facets, cols]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            star = "*" if sig.values[i, j] < 0.05 else ""
            ax.text(j, i, f"{v:+.2f}{star}".replace("-", "−"), ha="center", va="center",
                    fontsize=14, color="white" if abs(v) > 0.3 else "black")
    ax.set_xticks(range(len(cols)), [CRITERIA_SHORT[c] for c in cols])
    ax.set_yticks(range(len(facets)), [labels[f] for f in facets])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"Spearman $\rho$")
    cbar.outline.set_visible(False)
    save(fig, path)


CRITERIA_SHORT = {"DAT": "DAT", "RAT": "RAT", "Raven's 2": "Raven's", "SCTT words/resp.": "SCTT",
                  "BFI openness": "Openness"}


def main(config_path, overwrite=False, debug=False):
    config = load_config(config_path)
    for key in ("upstream_dir", "output_dir", "sample", "apply_exclude", "sensitivity_exclude", "min_sets_scored"):
        if key not in config:
            raise ValueError(f"FATAL: '{key}' is required in config")
    upstream = Path(config["upstream_dir"])
    if not upstream.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream}")
    out = init_directory(config["output_dir"], overwrite=overwrite)
    save_config(config, out)
    (out / "figures").mkdir()

    p, s, r = load_tables(upstream, config["sample"], config["apply_exclude"])
    res = {"n_participants": int(len(p)), "n_sets": int(len(s)),
           "n_with_block_a": int(p["raven_standard_score"].notna().sum()),
           "block_order": p["block_order"].value_counts().to_dict(),
           "pct_sets_scorable": round(100 * s["scorable"].mean(), 1)}

    cells = cell_table(s)
    cells.to_csv(out / "cell_table.csv")

    # Set-level models: relation x k, then with graded anchor similarity, then order/position.
    res["models"] = {}
    for outcome in ("n_words", "glove_n_surv", "composite_score", "difficulty", "glove_dist_all"):
        res["models"][outcome] = {
            "relation_x_k": crossed_mixed(s, outcome, "C(relation, Treatment('categorical')) * C(k)")[1],
            "anchor_cosine_x_k": crossed_mixed(s, outcome, "anchor_cosine * C(k)")[1],
            "position_and_order": crossed_mixed(s, outcome, "presentation_index + drat_first")[1],
        }
    s_items = s.groupby("anchors")[["anchor_cosine", "glove_n_surv", "difficulty", "scorable"]].mean()
    res["item_level_spearman"] = {
        f"{a}~{b}": round(stats.spearmanr(s_items[a], s_items[b])[0], 3)
        for a, b in (("anchor_cosine", "glove_n_surv"), ("anchor_cosine", "difficulty"),
                     ("glove_n_surv", "difficulty"), ("anchor_cosine", "scorable"))}
    res["item_level_spearman"]["n_anchor_sets"] = int(len(s_items))

    # Reliability.
    res["split_half"] = {c: split_half(s, c) for c in
                         ("n_words", "glove_n_surv", "composite_score", "glove_dist_all", "difficulty")}
    f = person_facets(s, p, config["min_sets_scored"])
    f.to_csv(out / "person_facets.csv")
    res["k2_vs_k3"] = {b: k_consistency(f, b) for b in ("drat_composite", "drat_survivors")}

    # Validity: each facet against the Block A measures and the SCTT.
    facets = ["drat_composite", "drat_composite_k2", "drat_composite_k3", "drat_survivors",
              "drat_words", "drat_dist_all", "drat_difficulty"]
    labels = {"drat_composite": "DRAT score", "drat_composite_k2": "DRAT score, $k = 2$",
              "drat_composite_k3": "DRAT score, $k = 3$", "drat_survivors": "Survivors per set",
              "drat_words": "Words typed per set", "drat_dist_all": "Ungated distance",
              "drat_difficulty": "Rated difficulty"}
    val = validity_table(f, facets)
    val.to_csv(out / "validity.csv", index=False)
    res["composite_vs_fluency"] = partial_spearman(f, "drat_composite", "drat_words", "drat_survivors")
    res["partial_given_words"] = {
        lab: partial_spearman(f, "drat_composite", crit, "drat_words") for crit, lab in CRITERIA.items()}
    res["composite_vs_n_scorable"] = round(f[["drat_composite", "drat_scorable"]].corr("spearman").iloc[0, 1], 3)

    keep = ~f.index.isin(config["sensitivity_exclude"])
    val_sens = validity_table(f[keep], facets)
    val_sens.to_csv(out / "validity_sensitivity.csv", index=False)

    res["words"] = word_level(r)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))

    set_camera_ready_style()
    plot_cells(s, out / "figures" / "fig_cells")
    plot_validity(val, facets, labels, out / "figures" / "fig_validity")
    print(cells.to_string())
    print(json.dumps(res, indent=2, default=float))
    print(val.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
