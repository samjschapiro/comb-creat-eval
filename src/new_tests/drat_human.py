"""Shared helpers for analysing the human DRAT study data package (Stony Brook, IRB2026-00448)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

RELATIONS = ["categorical", "constitutive", "causal", "unrelated"]


def load_tables(upstream, sample, apply_exclude):
    """The package's three master tables restricted to one sample; adds set-level flags.

    apply_exclude drops the study team's `exclude == True` participants (packages from
    10 Nov 2026 on carry the column; earlier packages do not, so it must be False for them).
    """
    upstream = Path(upstream)
    p = pd.read_csv(upstream / "participants.csv")
    s = pd.read_csv(upstream / "drat_sets.csv")
    r = pd.read_csv(upstream / "responses.csv", low_memory=False)
    p, s, r = p[p["sample"] == sample], s[s["sample"] == sample], r[r["sample"] == sample]
    if apply_exclude:
        if "exclude" not in p.columns:
            raise ValueError(f"FATAL: apply_exclude is set but {upstream} has no `exclude` column")
        keep = set(p.loc[~p["exclude"].astype(bool), "pid"])
        p, s, r = p[p["pid"].isin(keep)], s[s["pid"].isin(keep)], r[r["pid"].isin(keep)]
    s = s.assign(scorable=s["composite_score"].notna(), drat_first=s["session"] == 1)
    return p.reset_index(drop=True), s.reset_index(drop=True), r.reset_index(drop=True)


def crossed_mixed(s, outcome, fixed, reml=True):
    """Outcome ~ fixed + crossed random intercepts for participant and anchor set.

    Returns the fitted result alongside a summary dict (fixed effects, variance shares).
    """
    d = s.dropna(subset=[outcome]).assign(one=1)
    model = smf.mixedlm(f"{outcome} ~ {fixed}", d, groups="one", re_formula="0",
                        vc_formula={"pid": "0 + C(pid)", "item": "0 + C(anchors)"})
    fit = model.fit(reml=reml)
    vc = dict(zip(model.exog_vc.names, fit.vcomp))
    var = {"participant": vc["pid"], "anchor_set": vc["item"], "residual": fit.scale}
    total = sum(var.values())
    summary = {
        "n_obs": int(len(d)),
        "n_participants": int(d["pid"].nunique()),
        "n_anchor_sets": int(d["anchors"].nunique()),
        "fixed": {k: {"b": round(fit.params[k], 3), "se": round(fit.bse[k], 3), "p": float(fit.pvalues[k])}
                  for k in fit.params.index if k not in ("pid Var", "item Var")},
        "variance_share": {k: round(v / total, 3) for k, v in var.items()},
    }
    return fit, summary


def cronbach_alpha(items):
    """Alpha over the columns of a complete person x item matrix."""
    items = items.dropna()
    k = items.shape[1]
    return k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))


def spearman_brown(r, factor=2):
    return factor * r / (1 + (factor - 1) * r)


def load_glove_subset(glove_path, words):
    """Scan GloVe 840B for just the needed (lower-case) words; returns {word: unit vector}."""
    words, vecs = set(words), {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            w, rest = line.split(" ", 1)
            if w in words and w not in vecs:
                v = np.asarray(rest.split(), dtype=np.float32)
                vecs[w] = v / np.linalg.norm(v)
    return vecs


def mean_pairwise_distance(vectors):
    """100 x mean pairwise cosine distance (the DAT/DRAT score) over unit vectors; NaN if < 2."""
    if len(vectors) < 2:
        return np.nan
    m = np.vstack(vectors)
    sims = m @ m.T
    iu = np.triu_indices(len(m), 1)
    return 100 * float(np.mean(1 - sims[iu]))


def set_camera_ready_style():
    """Paper style shared with dat_eval/make_figures.py: Times, TrueType, no top/right spines."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def heatmap(ax, mat, stars, vmax=0.5):
    """Annotated RdBu_r correlation heatmap; `stars` is a same-shape array of suffix strings."""
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.2f}{stars[i][j]}".replace("-", "−"), ha="center", va="center",
                    fontsize=14, color="white" if abs(v) > 0.6 * vmax else "black")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im
