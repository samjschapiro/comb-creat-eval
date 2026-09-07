"""Composite creativity score as a function of thinking effort.

Two subjects (gpt-5.6-sol, gpt-6-astra-flex) at low/medium/high reasoning effort, scored in one
pooled pass. Writes three camera-ready figures into --out-dir:

  fig_effort_composite    overall + per-task composite vs effort, bootstrap 95% CIs, shared y-scale
  fig_effort_dimensions   all six constituent dimensions vs effort (dimension x task grid)
  fig_effort_delta        paired effect size, high effort minus low, per task and dimension

BOOTSTRAP. The anchor pair is the sampling unit -- paths within a pair are correlated -- and it is
resampled WITHIN each task, because association is posed over a different set of 30 pairs than
analogy and blending (union 60), so resampling the union jointly would let each task's item count
drift binomially instead of holding at its true 30. For the delta figure the SAME resampled items are
applied to every effort level, making it a paired bootstrap: effort levels share their anchor pairs,
and pairing removes the item-difficulty variance that dominates the marginal CIs.

CAVEAT, stated on each figure: originality is pool-relative and this pool is the six effort configs,
not the 35-model leaderboard pool. Effort levels are comparable to each other, not to the leaderboard.

    .venv/bin/python -m src.kg_creat.scripts.plot_effort_composite
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.kg_creat.scripts.compute_composite import TASK_DIMS, artifact_dims

EFFORTS = ["low", "medium", "high"]
TASK_ORDER = ["association", "analogy", "blending"]
MODE_OF_TASK = {"association": "baseline", "analogy": "analogy", "blending": "blending"}
DIM_ORDER = ["utility", "surprise", "originality", "em_originality", "em_utility", "em_integration"]
DIM_LABEL = {"utility": "utility", "surprise": "surprise", "originality": "originality",
             "em_originality": "emergent\noriginality", "em_utility": "emergent\nutility",
             "em_integration": "emergent\nintegration"}
# Okabe-Ito: colourblind-safe, and separable in greyscale print.
COLOR = {"gpt-5-6-sol": "#0072B2", "gpt-6-astra-flex": "#D55E00"}
MARKER = {"gpt-5-6-sol": "o", "gpt-6-astra-flex": "s"}
NICE = {"gpt-5-6-sol": "gpt-5.6-sol", "gpt-6-astra-flex": "gpt-6-astra-flex"}
GRID = "#E4E4E4"
MUTED = "#555555"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#333333",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.labelcolor": "#111111", "text.color": "#111111",
    "legend.frameon": False,
    "figure.dpi": 200,
})

CAVEAT = ("Originality is pool-relative to these six configurations, so effort levels are comparable "
          "to each other but not to the 35-model leaderboard.")


def measure(recs):
    """(per-task composite %, overall %, per-task dimension dict) -- the compute_composite formula."""
    raw = artifact_dims(recs)
    per_task, dims = {}, {}
    for task in TASK_ORDER:
        keys = TASK_DIMS[task]
        vals = [raw[task][k] for k in keys
                if raw[task].get(k) is not None and not np.isnan(raw[task][k])]
        per_task[task] = 100.0 * float(np.mean(vals)) if vals else np.nan
        dims[task] = {k: (None if raw[task].get(k) is None else 100.0 * raw[task][k]) for k in keys}
    overall = float(np.nanmean([per_task[t] for t in TASK_ORDER]))
    return per_task, overall, dims


def index_items(recs):
    """{mode: {(u,v): [records]}} -- the anchor pair is the resampling unit."""
    idx = defaultdict(lambda: defaultdict(list))
    for r in recs:
        idx[r.get("mode")][(r.get("u_label"), r.get("v_label"))].append(r)
    return idx


def analyse(scores_dir, n_boot, seed):
    by_model = defaultdict(dict)
    for d in sorted(Path(scores_dir).iterdir()):
        f = d / "path_scores.json"
        if not f.exists():
            continue
        model, effort = d.name.rsplit("__", 1)
        by_model[model][effort] = json.loads(f.read_text())

    out = {}
    for model, per_effort in by_model.items():
        idx = {e: index_items(rs) for e, rs in per_effort.items()}
        point = {e: measure(rs) for e, rs in per_effort.items()}
        # union of items per mode across effort levels, so every replicate indexes the same pairs
        modes = sorted({m for e in idx for m in idx[e]})
        items = {m: sorted({it for e in idx for it in idx[e].get(m, {})}) for m in modes}

        rng = np.random.default_rng(seed)
        boot = {e: {"overall": [], "task": defaultdict(list), "dim": defaultdict(list)}
                for e in per_effort}
        for _ in range(n_boot):
            picks = {m: rng.choice(len(items[m]), size=len(items[m]), replace=True) for m in modes}
            for e in per_effort:
                sample = [r for m in modes for i in picks[m]
                          for r in idx[e].get(m, {}).get(items[m][i], [])]
                pt, ov, dm = measure(sample)
                boot[e]["overall"].append(ov)
                for t in TASK_ORDER:
                    boot[e]["task"][t].append(pt[t])
                    for k, v in dm[t].items():
                        boot[e]["dim"][(t, k)].append(np.nan if v is None else v)

        ci = lambda xs: (float(np.nanpercentile(xs, 2.5)), float(np.nanpercentile(xs, 97.5)))
        res = {"point": point, "n_items": {m: len(v) for m, v in items.items()},
               "ci": {e: {"overall": ci(boot[e]["overall"]),
                          "task": {t: ci(boot[e]["task"][t]) for t in TASK_ORDER},
                          "dim": {k: ci(v) for k, v in boot[e]["dim"].items()}} for e in per_effort}}
        # paired deltas: same resampled items on both effort levels, so item difficulty cancels
        if "low" in boot and "high" in boot:
            d_ov = np.array(boot["high"]["overall"]) - np.array(boot["low"]["overall"])
            res["delta"] = {
                "overall": (point["high"][1] - point["low"][1], *ci(d_ov)),
                "task": {t: (point["high"][0][t] - point["low"][0][t],
                             *ci(np.array(boot["high"]["task"][t]) - np.array(boot["low"]["task"][t])))
                         for t in TASK_ORDER},
                "dim": {k: ((point["high"][2][k[0]].get(k[1]) or np.nan)
                            - (point["low"][2][k[0]].get(k[1]) or np.nan),
                            *ci(np.array(boot["high"]["dim"][k]) - np.array(boot["low"]["dim"][k])))
                        for k in boot["high"]["dim"]}}
        out[model] = res
    return out


def _axes_style(ax):
    ax.set_xticks(range(len(EFFORTS)))
    ax.set_xticklabels(EFFORTS)
    ax.set_xlim(-0.28, len(EFFORTS) - 0.72)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _line(ax, model, ys, lo=None, hi=None, ms=6.0):
    c = COLOR[model]
    xs = np.arange(len(EFFORTS))
    if lo is not None:
        ax.fill_between(xs, lo, hi, color=c, alpha=0.13, lw=0)
    ax.plot(xs, ys, color=c, lw=1.8, marker=MARKER[model], ms=ms, mfc="white", mec=c, mew=1.6,
            label=NICE[model], clip_on=False, solid_capstyle="round", zorder=3)


def fig_composite(res, out):
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    panels = [("overall", "Overall composite")] + [(t, t.capitalize()) for t in TASK_ORDER]
    for ax, (key, title) in zip(axes, panels):
        for model, r in res.items():
            if key == "overall":
                ys = [r["point"][e][1] for e in EFFORTS]
                b = [r["ci"][e]["overall"] for e in EFFORTS]
            else:
                ys = [r["point"][e][0][key] for e in EFFORTS]
                b = [r["ci"][e]["task"][key] for e in EFFORTS]
            _line(ax, model, ys, [x[0] for x in b], [x[1] for x in b])
        _axes_style(ax)
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_xlabel("thinking effort")
    lo = min(min(l.get_ydata().min() for l in ax.lines) for ax in axes)
    hi = max(max(l.get_ydata().max() for l in ax.lines) for ax in axes)
    pad = 0.10 * (hi - lo)
    for ax in axes:
        ax.set_ylim(lo - pad, hi + pad)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel("composite (% of max)")
    h, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.8, columnspacing=1.8)
    fig.suptitle("Composite creativity score vs. thinking effort", fontsize=11.5, y=1.13)
    n = next(iter(res.values()))["n_items"]
    n_txt = ", ".join(f"{t}: {n.get(MODE_OF_TASK[t], 0)}" for t in TASK_ORDER)
    fig.text(0.5, -0.15,
             f"Bands are bootstrap 95% CIs over anchor pairs, resampled within task ({n_txt}); "
             "association is posed over a different pair set than analogy and blending.\n"
             f"All four panels share one y-scale. {CAVEAT}",
             ha="center", fontsize=7.8, color=MUTED, linespacing=1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_dimensions(res, out):
    nrow, ncol = len(DIM_ORDER), len(TASK_ORDER)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.8, 9.6))
    for i, dim in enumerate(DIM_ORDER):
        row_axes = []
        for j, task in enumerate(TASK_ORDER):
            ax = axes[i][j]
            if dim not in TASK_DIMS[task]:
                ax.axis("off")
                continue
            row_axes.append(ax)
            for model, r in res.items():
                ys = [r["point"][e][2][task].get(dim) for e in EFFORTS]
                ys = [np.nan if y is None else y for y in ys]
                b = [r["ci"][e]["dim"][(task, dim)] for e in EFFORTS]
                _line(ax, model, ys, [x[0] for x in b], [x[1] for x in b], ms=5.0)
            _axes_style(ax)
            ax.tick_params(labelsize=8)
            if i == 0:
                ax.set_title(task.capitalize(), fontsize=10, pad=8)
        # one y-scale per DIMENSION row, so the three tasks are directly comparable
        if row_axes:
            lo = min(min(l.get_ydata().min() for l in a.lines) for a in row_axes)
            hi = max(max(l.get_ydata().max() for l in a.lines) for a in row_axes)
            pad = 0.12 * (hi - lo) or 1.0
            for a in row_axes:
                a.set_ylim(lo - pad, hi + pad)
            for a in row_axes[1:]:
                a.tick_params(labelleft=False)
            row_axes[0].set_ylabel(DIM_LABEL[dim], fontsize=9)
    # x tick labels on the LAST VISIBLE axis of each column (association stops three rows early)
    for j, task in enumerate(TASK_ORDER):
        rows = [i for i, d in enumerate(DIM_ORDER) if d in TASK_DIMS[task]]
        for i in rows[:-1]:
            axes[i][j].tick_params(labelbottom=False)
        axes[rows[-1]][j].set_xlabel("thinking effort")
    h, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.963),
               handlelength=1.8, columnspacing=1.8)
    fig.suptitle("Composite dimensions vs. thinking effort", fontsize=11.5, y=0.995)
    fig.text(0.5, 0.028,
             "Every dimension is a percentage of its maximum, utility-gated; bands are bootstrap 95% "
             "CIs over anchor pairs.\nEach row shares one y-scale across tasks. Emergent dimensions "
             f"are undefined for association.\n{CAVEAT}",
             ha="center", fontsize=7.8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=[0, 0.10, 1, 0.945])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_delta(res, out):
    """High minus low, paired over anchor pairs. The question the study exists to answer."""
    rows = [("Overall composite", ("overall",)), ("", None)]
    for t in TASK_ORDER:
        rows.append((f"{t.capitalize()} composite", ("task", t)))
    rows.append(("", None))
    for t in TASK_ORDER:
        for d in TASK_DIMS[t]:
            rows.append((f"{t.capitalize()}: {DIM_LABEL[d].replace(chr(10), ' ')}", ("dim", (t, d))))
    labels = [r[0] for r in rows]
    ys = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.30 * len(rows) + 1.9))
    ax.axvline(0, color="#888888", lw=0.9, zorder=1)
    off = {m: (0.19 if i == 0 else -0.19) for i, m in enumerate(res)}
    for model, r in res.items():
        if "delta" not in r:
            continue
        for y, (_, key) in zip(ys, rows):
            if key is None:
                continue
            v = r["delta"][key[0]] if len(key) == 1 else r["delta"][key[0]][key[1]]
            est, lo, hi = v
            if not np.isfinite(est):
                continue
            yy = y + off[model]
            sig = lo > 0 or hi < 0
            c = COLOR[model]
            ax.plot([lo, hi], [yy, yy], color=c, lw=1.5, alpha=0.85, solid_capstyle="round", zorder=2)
            ax.plot([est], [yy], marker=MARKER[model], ms=5.5, color=c,
                    mfc=c if sig else "white", mec=c, mew=1.5, zorder=3,
                    label=NICE[model] if y == ys[0] else None)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("high effort $-$ low effort (percentage points)")
    ax.set_title("Effect of thinking effort, paired over anchor pairs", fontsize=11, pad=18)
    lo_x = min(min(l.get_xdata()) for l in ax.lines if len(l.get_xdata()) == 2)
    hi_x = max(max(l.get_xdata()) for l in ax.lines if len(l.get_xdata()) == 2)
    pad = 0.06 * (hi_x - lo_x)
    ax.set_xlim(lo_x - pad, hi_x + pad)
    h, l_ = ax.get_legend_handles_labels()
    ax.legend(h, l_, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=8.5,
              handlelength=1.4, columnspacing=1.6, borderaxespad=0.0)
    fig.text(0.5, -0.055 - 0.004 * len(rows),
             "Bars are paired bootstrap 95% CIs: the same resampled anchor pairs are scored at both "
             "effort levels.\nFilled markers mark intervals excluding zero. " + CAVEAT,
             ha="center", fontsize=7.8, color=MUTED, linespacing=1.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def report(res):
    print(f"\n{'config':<26}{'assoc':>9}{'analogy':>9}{'blend':>9}{'OVERALL':>10}{'95% CI':>16}")
    print("-" * 79)
    for model, r in res.items():
        for e in EFFORTS:
            pt, ov, _ = r["point"][e]
            lo, hi = r["ci"][e]["overall"]
            print(f"{model + '__' + e:<26}{pt['association']:>9.2f}{pt['analogy']:>9.2f}"
                  f"{pt['blending']:>9.2f}{ov:>10.2f}{f'[{lo:.1f}, {hi:.1f}]':>16}")
        d = r["delta"]
        sig = lambda v: "  *" if (v[1] > 0 or v[2] < 0) else ""
        ov = d["overall"]
        ov_ci = f"[{ov[1]:.1f}, {ov[2]:.1f}]"
        print(f"{'  high - low (paired)':<26}"
              f"{d['task']['association'][0]:>+9.2f}{d['task']['analogy'][0]:>+9.2f}"
              f"{d['task']['blending'][0]:>+9.2f}{ov[0]:>+10.2f}{ov_ci:>16}{sig(ov)}")
        print(f"{'    task CIs':<26}"
              + "  ".join(f"{t[:5]} [{d['task'][t][1]:+.1f},{d['task'][t][2]:+.1f}]{sig(d['task'][t])}"
                          for t in TASK_ORDER))
        print()
    print("* = paired 95% CI excludes zero")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", default="data/kg_creat/effort_study/scores")
    ap.add_argument("--out-dir", default="data/kg_creat/effort_study/figures")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    res = analyse(a.scores_dir, a.n_boot, a.seed)
    report(res)
    fig_composite(res, str(out / "fig_effort_composite"))
    fig_dimensions(res, str(out / "fig_effort_dimensions"))
    fig_delta(res, str(out / "fig_effort_delta"))
    (out / "effort_composite.json").write_text(json.dumps(
        {m: {"point": {e: {"per_task": r["point"][e][0], "overall": r["point"][e][1],
                           "dims": r["point"][e][2]} for e in EFFORTS},
             "ci": {e: {"overall": r["ci"][e]["overall"],
                        "task": r["ci"][e]["task"]} for e in EFFORTS},
             "delta_high_minus_low": {"overall": r["delta"]["overall"], "task": r["delta"]["task"]}}
         for m, r in res.items()}, indent=2))
    for f in ("fig_effort_composite", "fig_effort_dimensions", "fig_effort_delta"):
        print(f"wrote {out}/{f}.{{pdf,png}}")
    print(f"wrote {out}/effort_composite.json")


if __name__ == "__main__":
    main()
