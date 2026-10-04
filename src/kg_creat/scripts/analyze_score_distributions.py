"""How performance is DISTRIBUTED across humans vs. models, per task: composite and every dimension.

Units. Humans: each kept participant (their own ~14 test answers). Models: (a) each model (all ~90 answers) and
(b) model PSEUDO-PARTICIPANTS -- a model answering a random set of items the same size as a real person's set
(per task, n items drawn from the human per-task counts, one answer per item), 30 per model. (b) is the
like-for-like comparison: a person's score rests on ~14 answers, so person-level scores are noisy and spread out
for that reason alone; pseudo-participants carry the same noise. Skipped items count as failed (0).

Fits (scores on [0, 1]; the composite / 100):
  zero-inflated beta  P(x = 0) = pi, otherwise Beta(a, b)       -> pi, mean of the positive part, precision a+b
  normal              mean and SD only (its likelihood is not comparable with the zero-inflated fit's point mass)
  beta-binomial       per-unit pass COUNT k of n (utility)      -> mean pass rate and rho, the share of variance due to
                                                                   real differences between units (0 = all alike)
Group differences, humans vs. pseudo-participants: two-sample KS and Anderson-Darling, and bootstrap 95% CIs
(over units) for the differences in pi, positive-part mean, and SD.

Writes data/kg_creat/human_study/analysis/score_distributions.json and figures/score_distributions.png.

    .venv/bin/python -m src.kg_creat.scripts.analyze_score_distributions
"""
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import optimize, stats

from src.kg_creat.scripts.compute_composite import TASK_DIMS, _mean, _ok, artifact_dims

SCORES = Path("data/kg_creat/human_study/combined_scores_skip0")
OUT = Path("data/kg_creat/human_study/analysis/score_distributions.json")
FIG = Path("data/kg_creat/human_study/figures/score_distributions.png")
MOCK = Path("scratch/human_vs_model_space/score_distributions.png")
MODE = {"association": "baseline", "analogy": "analogy", "blending": "blending"}
PSEUDO_PER_MODEL, BOOT, SEED = 30, 2000, 0
HUMAN, MODEL = "#d95f02", "#1b6ca8"


def heads(recs, mode):
    """The artifacts compute_composite scores for one task (path / pair head / blend head)."""
    rs = [r for r in recs if r.get("mode") == mode]
    return [r for r in rs if "pair_sat" in r] if mode == "analogy" else [r for r in rs if r.get("triples")]


def unit_scores(recs):
    """Per-task composite (0-1), every gated dimension, and the pass count k / n, plus Overall."""
    raw = artifact_dims(recs)
    out = {}
    for task, dims in TASK_DIMS.items():
        arts = heads(recs, MODE[task])
        if not arts:
            continue
        vals = {k: raw[task][k] for k in dims if _ok(raw[task].get(k))}
        ok = [(r.get("pair_sat") if task == "analogy" else r.get("sat")) is True for r in arts]
        out[task] = {"composite": _mean(list(vals.values())), **vals, "_k": sum(ok), "_n": len(ok)}
    if out:
        out["overall"] = {"composite": _mean([v["composite"] for v in out.values()])}
    return out


def humans():
    by = defaultdict(list)
    for r in json.loads((SCORES / "human_kept" / "path_scores.json").read_text()):
        by[r.get("sample_idx")].append(r)
    return [unit_scores(v) for v in by.values()], {
        t: [len({r["prompt_id"] for r in heads(v, MODE[t])}) for v in by.values()] for t in MODE}


def models(n_items, rng):
    whole, pseudo = [], []
    for f in sorted(SCORES.glob("*/path_scores.json")):
        if f.parent.name.startswith("human"):
            continue
        recs = json.loads(f.read_text())
        whole.append(unit_scores(recs))
        by_item = {t: defaultdict(list) for t in MODE}
        for t, m in MODE.items():
            for r in heads(recs, m):
                by_item[t][r["prompt_id"]].append(r)
        for _ in range(PSEUDO_PER_MODEL):
            pick = []
            for t in MODE:
                n = int(rng.choice(n_items[t]))
                items = rng.choice(sorted(by_item[t]), size=min(n, len(by_item[t])), replace=False)
                for it in items:
                    cands = by_item[t][it]
                    r = dict(cands[rng.integers(len(cands))])
                    if t == "analogy":   # a pair head stands for the whole pair: keep it alone
                        r["path_idx"] = 0
                    pick.append(r)
            pseudo.append(unit_scores(pick))
    return whole, pseudo


def fit_zib(x):
    x = np.asarray(x, float)
    pi = float(np.mean(x <= 0))
    pos = np.clip(x[x > 0], 1e-4, 1 - 1e-4)
    if len(pos) < 3 or np.ptp(pos) == 0:
        return {"pi": pi, "a": None, "b": None, "pos_mean": float(pos.mean()) if len(pos) else None, "aic": None}
    a, b, _, _ = stats.beta.fit(pos, floc=0, fscale=1)
    ll = (np.sum(x <= 0) * np.log(pi) if pi > 0 else 0) + len(pos) * np.log(1 - pi) + stats.beta.logpdf(pos, a, b).sum()
    return {"pi": pi, "a": float(a), "b": float(b), "pos_mean": float(a / (a + b)), "precision": float(a + b),
            "aic": float(2 * 3 - 2 * ll)}


def fit_normal(x):
    x = np.asarray(x, float)
    mu, sd = x.mean(), x.std() or 1e-6
    return {"mean": float(mu), "sd": float(sd), "aic": float(2 * 2 - 2 * stats.norm.logpdf(x, mu, sd).sum())}


def fit_betabinom(k, n):
    """MLE beta-binomial over units with different n: mean pass rate mu and rho = 1 / (a + b + 1)."""
    k, n = np.asarray(k), np.asarray(n)

    def nll(th):
        a, b = np.exp(th)
        return -stats.betabinom.logpmf(k, n, a, b).sum()
    res = optimize.minimize(nll, x0=[0.0, 0.0], method="Nelder-Mead")
    a, b = np.exp(res.x)
    return {"mu": float(a / (a + b)), "rho": float(1 / (a + b + 1)), "aic": float(2 * 2 + 2 * res.fun)}


def compare(h, m, rng):
    h, m = np.asarray(h, float), np.asarray(m, float)
    feats = {"pi": lambda x: np.mean(x <= 0), "pos_mean": lambda x: x[x > 0].mean() if np.any(x > 0) else np.nan,
             "sd": np.std, "mean": np.mean}
    out = {"ks_p": float(stats.ks_2samp(h, m).pvalue),
           "ad_p": float(stats.anderson_ksamp([h, m], variant="midrank").pvalue) if np.ptp(np.r_[h, m]) > 0 else None}
    for name, f in feats.items():
        d = [f(h[rng.integers(0, len(h), len(h))]) - f(m[rng.integers(0, len(m), len(m))]) for _ in range(BOOT)]
        out[f"diff_{name}"] = {"est": float(f(h) - f(m)), "ci95": [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]}
    return out


RAWKEY = {"surprise": "R", "originality": "originality", "em_originality": "em_originality"}


def passed_values(recs, task, dim):
    """Raw (un-gated) values of one continuous dimension over the answers that PASSED utility."""
    arts = heads(recs, MODE[task])
    ok = [(r.get("pair_sat") if task == "analogy" else r.get("sat")) is True for r in arts]
    return [float(np.clip(r[RAWKEY[dim]], 0, 1)) for r, p in zip(arts, ok)
            if p and not r.get("_missing") and _ok(r.get(RAWKEY[dim]))]


def artifact_level(rng):
    """Beta fits to each continuous dimension over passed answers: humans (kept) vs. all models' answers."""
    hum = json.loads((SCORES / "human_kept" / "path_scores.json").read_text())
    mod = [r for f in sorted(SCORES.glob("*/path_scores.json")) if not f.parent.name.startswith("human")
           for r in json.loads(f.read_text())]
    out = {}
    for task, dims in TASK_DIMS.items():
        for dim in dims:
            if dim not in RAWKEY:
                continue
            h, m = passed_values(hum, task, dim), passed_values(mod, task, dim)
            if len(h) < 5:
                continue
            fit = {}
            for g, x in (("human", h), ("model", m)):
                a, b, _, _ = stats.beta.fit(np.clip(x, 1e-4, 1 - 1e-4), floc=0, fscale=1)
                fit[g] = {"n": len(x), "a": float(a), "b": float(b), "mean": float(np.mean(x)), "sd": float(np.std(x)),
                          "median": float(np.median(x))}
            boot = [np.mean(rng.choice(h, len(h))) - np.mean(rng.choice(m, len(m))) for _ in range(BOOT)]
            fit["diff_mean"] = {"est": float(np.mean(h) - np.mean(m)), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
            fit["ks_p"] = float(stats.ks_2samp(h, m).pvalue)
            out[f"{task}.{dim}"] = fit
    return out


def main():
    rng = np.random.default_rng(SEED)
    H, n_items = humans()
    Mw, Mp = models(n_items, rng)
    views = [("overall", "composite")] + [(t, d) for t, ds in TASK_DIMS.items() for d in ["composite", *ds]]
    res = {"n": {"humans": len(H), "models": len(Mw), "pseudo": len(Mp)}, "metrics": {}}
    for task, dim in views:
        get = lambda units: [u[task][dim] for u in units if task in u and _ok(u[task].get(dim))]
        h, mw, mp = get(H), get(Mw), get(Mp)
        r = {g: {"n": len(x), "zib": fit_zib(x), "normal": fit_normal(x)} for g, x in (("human", h), ("model", mw), ("pseudo", mp))}
        r["human_vs_pseudo"] = compare(h, mp, rng)
        if dim == "utility":
            for g, units in (("human", H), ("model", Mw), ("pseudo", Mp)):
                kk = [u[task]["_k"] for u in units if task in u]
                nn = [u[task]["_n"] for u in units if task in u]
                r[g]["betabinom"] = fit_betabinom(kk, nn)
        res["metrics"][f"{task}.{dim}"] = r
    res["passed_answers"] = artifact_level(rng)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=1)

    print(f"units: {res['n']}")
    print(f"{'metric':30s} {'P(0) H / pseudo':>16s} {'pos-mean H / pseudo':>20s} {'SD H / pseudo':>14s}  KS p")
    for k, r in res["metrics"].items():
        zh, zp = r["human"]["zib"], r["pseudo"]["zib"]
        fmt = lambda v: f"{v:.2f}" if v is not None else "  - "
        print(f"{k:30s} {zh['pi']:.2f} / {zp['pi']:.2f}       {fmt(zh['pos_mean'])} / {fmt(zp['pos_mean'])}        "
              f"{r['human']['normal']['sd']:.2f} / {r['pseudo']['normal']['sd']:.2f}    {r['human_vs_pseudo']['ks_p']:.1e}")
    for t in TASK_DIMS:
        r = res["metrics"][f"{t}.utility"]
        print(f"beta-binomial {t:11s} human mu {r['human']['betabinom']['mu']:.2f} rho {r['human']['betabinom']['rho']:.2f} | "
              f"pseudo mu {r['pseudo']['betabinom']['mu']:.2f} rho {r['pseudo']['betabinom']['rho']:.2f} | "
              f"models mu {r['model']['betabinom']['mu']:.2f} rho {r['model']['betabinom']['rho']:.2f}")

    print("\npassed answers only (raw values, beta fits): humans vs. models")
    for k, r in res["passed_answers"].items():
        print(f"  {k:28s} n {r['human']['n']:4d}/{r['model']['n']:5d}  mean {r['human']['mean']:.3f} / {r['model']['mean']:.3f}  "
              f"diff {r['diff_mean']['est']:+.3f} [{r['diff_mean']['ci95'][0]:+.3f},{r['diff_mean']['ci95'][1]:+.3f}]  "
              f"sd {r['human']['sd']:.3f}/{r['model']['sd']:.3f}  KS p {r['ks_p']:.1e}")
    plot(res, H, Mp)
    plot_dims(res, H, Mp)


def plot(res, H, Mp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if "Nimbus Roman" not in {f.name for f in font_manager.fontManager.ttflist}:
        raise SystemExit("FATAL: Nimbus Roman is not installed")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman"], "mathtext.fontset": "custom",
                         "mathtext.rm": "Nimbus Roman", "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
                         "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12})
    views = [("Overall", "overall"), ("Association", "association"), ("Analogy", "analogy"), ("Blending", "blending")]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.4))
    xs = np.linspace(0.005, 0.995, 300)
    bins = np.linspace(0, 1, 26)
    for ax, letter, (title, t) in zip(axes, "abcd", views):
        r = res["metrics"][f"{t}.composite"]
        for g, units, c in (("pseudo", Mp, MODEL), ("human", H, HUMAN)):
            x = np.array([u[t]["composite"] for u in units if t in u])
            z = r[g]["zib"]
            w = np.ones_like(x) / len(x) / (bins[1] - bins[0])
            ax.hist(x[x > 0], bins=bins, weights=w[x > 0], color=c, alpha=0.35)
            if z["a"]:
                ax.plot(xs, (1 - z["pi"]) * stats.beta.pdf(xs, z["a"], z["b"]), color=c, lw=2.5,
                        label=f"{'humans' if g == 'human' else 'model pseudo-participants'}: P(0) = {z['pi']:.2f}")
            # the point mass at 0 on the same density scale as the bars: P(0) spread over one bin width
            ax.bar(0 if g == "pseudo" else (bins[1] - bins[0]) / 2, z["pi"] / (bins[1] - bins[0]),
                   width=(bins[1] - bins[0]) / 2, align="edge", color=c, alpha=0.9)
        ax.set_title(f"({letter}) {title}")
        ax.set_xlabel("composite (fraction of max)")
        if letter == "a":
            ax.set_ylabel("density")
        ax.set_xlim(0, 1)
        ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=180)
    shutil.copy(FIG, MOCK)
    print(f"wrote {OUT}, {FIG} (copy {MOCK})")


def plot_dims(res, H, Mp):
    """Per-unit, utility-gated: every dimension of every task (rows = tasks), humans vs. pseudo-participants."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"utility": "utility", "surprise": "surprise", "originality": "originality",
             "em_originality": "invention orig." , "em_utility": "emergent utility", "em_integration": "integration"}
    tasks = list(TASK_DIMS)
    ncol = max(len(d) for d in TASK_DIMS.values())
    fig, axes = plt.subplots(3, ncol, figsize=(4.2 * ncol, 12))
    bins = np.linspace(0, 1, 21)
    w0 = bins[1] - bins[0]
    xs = np.linspace(0.005, 0.995, 300)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for i, t in enumerate(tasks):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(TASK_DIMS[t]):
                ax.axis("off")
                continue
            d = TASK_DIMS[t][j]
            r = res["metrics"][f"{t}.{d}"]
            for g, units, c in (("pseudo", Mp, MODEL), ("human", H, HUMAN)):
                x = np.array([u[t][d] for u in units if t in u and _ok(u[t].get(d))])
                z = r[g]["zib"]
                ax.hist(x[x > 0], bins=bins, weights=np.ones((x > 0).sum()) / len(x) / w0, color=c, alpha=0.35)
                if z["a"]:
                    ax.plot(xs, (1 - z["pi"]) * stats.beta.pdf(xs, z["a"], z["b"]), color=c, lw=2)
                ax.bar(0 if g == "pseudo" else w0 / 2, z["pi"] / w0, width=w0 / 2, align="edge", color=c, alpha=0.9)
            label = "emergent orig." if (t == "blending" and d == "em_originality") else names[d]
            ax.set_title(f"({next(letters)}) {t}: {label}")
            ax.set_xlim(0, 1)
            if j == 0:
                ax.set_ylabel("density")
            ax.text(0.97, 0.95, f"P(0) {r['human']['zib']['pi']:.2f} / {r['pseudo']['zib']['pi']:.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=12)
    from matplotlib.lines import Line2D
    axes[0, ncol - 1].legend(handles=[Line2D([], [], color=HUMAN, lw=3, label="humans"),
                                      Line2D([], [], color=MODEL, lw=3, label="model pseudo-participants")],
                             loc="center", frameon=False, fontsize=15)
    fig.tight_layout()
    f = FIG.with_name("score_distributions_per_dim.png")
    fig.savefig(f, dpi=160)
    shutil.copy(f, MOCK.with_name(f.name))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
