"""Regime-A figures: the ideation-execution 2x2 and the failure-channel decomposition.

Downstream of score.py. Endpoints are held FIXED across cells, so within a bundle the only thing
that changes is the constraint -- which makes the baseline->constrained displacement causal in
constraint type rather than confounded by which entity pair was drawn.

Two figures:
  fig_regimeA_2x2.png       per-constraint mean within-bundle displacement (dR_emit, dsat). The
                            quadrant a constraint lands in IS the claim: down-right = the constraint
                            bought novelty at the cost of compliance (the ideation-execution gap).
  fig_regimeA_channels.png  where satisfaction was lost -- structural / factual / constraint -- since
                            a low success rate means something different in each channel.

    .venv_mlx/bin/python src/kg_creat/scripts/plot_regime_a.py data/kg_creat/scores_regimeA_all
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

MODES = ["exclusion", "inclusion", "inclusion_rare", "categorical"]
MODE_LABEL = {"exclusion": "Exclusion", "inclusion": "Inclusion (common class)",
              "inclusion_rare": "Inclusion (rare class)", "ordering": "Ordering",
              "categorical": "Categorical"}
CHANNELS = ["structural", "factual", "constraint"]
CHANNEL_COLOR = {"structural": "#94A3B8", "factual": "#CE5E5E", "constraint": "#E0A458",
                 "ok": "#388C3C", "unjudged": "#D8DEE6"}
SEGMENTS = ["ok", "structural", "factual", "constraint", "unjudged"]
INK, MUTED, GRID = "#1f2933", "#66727f", "#e3e8ee"

# Colour by model family, shade by capability within family: family structure is the comparison
# a reader actually makes, and eight arbitrary hues would not be separable.
FAMILY = [("anthropic", "#7C3AED"), ("openai", "#059669"), ("google", "#2563EB"), ("meta-llama", "#EA580C")]
PRETTY = {"anthropic_claude-sonnet-4-6": "Claude Sonnet 4.6", "anthropic_claude-haiku-4-5": "Claude Haiku 4.5",
          "openai_gpt-4-1-mini": "GPT-4.1-mini", "openai_gpt-4o-mini": "GPT-4o-mini",
          "google_gemini-2-5-flash": "Gemini 2.5 Flash", "google_gemini-2-5-flash-lite": "Gemini 2.5 Flash-Lite",
          "meta-llama_llama-3-3-70b-instruct": "Llama 3.3 70B", "meta-llama_llama-3-1-8b-instruct": "Llama 3.1 8B"}
# Strong -> weak within each family, so the shade gradient tracks capability.
ORDER = ["anthropic_claude-sonnet-4-6", "anthropic_claude-haiku-4-5", "openai_gpt-4-1-mini",
         "openai_gpt-4o-mini", "google_gemini-2-5-flash", "google_gemini-2-5-flash-lite",
         "meta-llama_llama-3-3-70b-instruct", "meta-llama_llama-3-1-8b-instruct"]


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def _colors(models):
    out, seen = {}, {}
    for m in models:
        fam = next((f for f, _ in FAMILY if m.startswith(f)), None)
        base = dict(FAMILY).get(fam, "#666666")
        i = seen.get(fam, 0)
        seen[fam] = i + 1
        out[m] = (base, 1.0 if i == 0 else 0.45)  # first (stronger) model solid, second faded
    return out


def fig_2x2(summ, models, colors, out):
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    _style(ax)
    ax.axhline(0, color=MUTED, lw=1.0, zorder=1)
    ax.axvline(0, color=MUTED, lw=1.0, zorder=1)
    markers = dict(zip(MODES, ["o", "s", "D", "^", "P"]))
    for m in models:
        c, a = colors[m]
        for mode in MODES:
            cell = summ[m].get("two_by_two", {}).get(mode)
            if not cell or cell["mean_dR_emit"] is None or cell["mean_dsat"] is None:
                continue
            ax.scatter([cell["mean_dR_emit"]], [cell["mean_dsat"]], s=88, marker=markers[mode],
                       color=c, alpha=a, edgecolors="white", linewidths=0.8, zorder=5)
    xs = [abs(x) for m in models for mode in MODES
          if (x := (summ[m].get("two_by_two", {}).get(mode) or {}).get("mean_dR_emit")) is not None]
    pad = max(xs) * 1.35 if xs else 0.05
    ax.set_xlim(-pad, pad)
    ax.set_xlabel("Δ novelty vs. baseline  (same endpoints)", fontsize=10.5, color=MUTED)
    ax.set_ylabel("Δ success rate vs. baseline", fontsize=10.5, color=MUTED)
    ax.set_title("What each constraint costs and buys", fontsize=13, color=INK, pad=10)
    # Label the quadrants the data actually occupies. Every constraint costs compliance, so the
    # interpretive question is which ones at least buy novelty for it.
    lo, hi = ax.get_ylim()
    for x, va, y, txt in ((pad * 0.97, "bottom", lo, "traded compliance\nFOR novelty"),
                          (-pad * 0.97, "bottom", lo, "lost compliance,\ngained nothing"),
                          (pad * 0.97, "top", hi, "more novel\nmore compliant"),
                          (-pad * 0.97, "top", hi, "less novel\nmore compliant")):
        ax.annotate(txt, (x, y), ha="right" if x > 0 else "left", va=va,
                    fontsize=8.5, color=MUTED, style="italic")
    mode_h = [Line2D([0], [0], marker=markers[md], color="w", markerfacecolor=MUTED,
                     markersize=8, label=MODE_LABEL[md]) for md in MODES]
    model_h = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m][0],
                      alpha=colors[m][1], markersize=8, label=PRETTY.get(m, m)) for m in models]
    ax.legend(handles=mode_h + model_h, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


def fig_creativity(summ, models, colors, out):
    """Creativity (the dependent variable) against constraint type (the independent variable).

    Left: E[R·U] per cell, with each model's own unconstrained baseline as its reference line —
    the quantity creativity research actually cares about, novelty and utility jointly.
    Right: the same data decomposed into its two factors, with iso-creativity hyperbolas. A point's
    creativity is its position relative to those curves, which makes visible whether a constraint
    sits on the novelty-utility frontier or genuinely beats it.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 5.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    _style(axL)
    _style(axR)

    # Order the levels by the dependent variable, so the panel reads as the ranking of constraint
    # types it is rather than as a shape imposed by the order the taxonomy happens to list them in.
    def _pooled(cell):
        v = [summ[m]["per_mode"].get(cell, {}).get("creativity") for m in models]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else -1
    cells = ["baseline"] + sorted(MODES, key=_pooled, reverse=True)
    xs = range(len(cells))
    for m in models:
        c, a = colors[m]
        ys = [summ[m]["per_mode"].get(k, {}).get("creativity") for k in cells]
        axL.plot(list(xs), ys, marker="o", ms=6, lw=1.6, color=c, alpha=a,
                 label=PRETTY.get(m, m), zorder=4)
        axL.axhline(ys[0], color=c, alpha=a * 0.28, lw=0.9, ls=":", zorder=2)
    axL.set_xticks(list(xs))
    axL.set_xticklabels(["Baseline\n(no constraint)"] + [MODE_LABEL[m].replace(" (", "\n(") for m in cells[1:]],
                        fontsize=8.5, color=INK)
    axL.set_ylabel("Creativity   E[R · U]", fontsize=10.5, color=MUTED)
    axL.set_title("Creativity by constraint type\n(dotted = that model's own baseline)",
                  fontsize=12, color=INK, pad=10)
    axL.legend(frameon=False, fontsize=8, ncol=2)

    # Iso-creativity contours: creativity = sat × R_valid, so equal-creativity sets are hyperbolas.
    import numpy as np
    sat = np.linspace(0.02, 0.85, 200)
    for level in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        r = level / sat
        keep = (r >= 0.30) & (r <= 0.60)
        if keep.any():
            axR.plot(sat[keep], r[keep], color=GRID, lw=1.0, zorder=1)
            i = int(np.argmax(keep))
            axR.annotate(f"{level:.2f}", (sat[keep][0], r[keep][0]), fontsize=7.5,
                         color=MUTED, va="bottom", ha="left", zorder=1)
    markers = {"baseline": "*", "exclusion": "o", "inclusion": "s",
               "inclusion_rare": "D", "ordering": "^", "categorical": "P"}
    for m in models:
        c, a = colors[m]
        for cell in cells:
            pm = summ[m]["per_mode"].get(cell, {})
            x, y = pm.get("sat_rate"), pm.get("R_valid")
            if x is None or y is None:
                continue
            axR.scatter([x], [y], s=130 if cell == "baseline" else 78, marker=markers[cell],
                        color=c, alpha=a, edgecolors="white", linewidths=0.8, zorder=5)
    axR.set_xlabel("Utility   (success rate)", fontsize=10.5, color=MUTED)
    axR.set_ylabel("Novelty of the useful artifacts   (R_valid)", fontsize=10.5, color=MUTED)
    axR.set_title("Creativity decomposed\n(grey curves = equal creativity)", fontsize=12, color=INK, pad=10)
    axR.legend(handles=[Line2D([0], [0], marker=markers[c], color="w", markerfacecolor=MUTED,
                               markersize=9, label=("Baseline" if c == "baseline" else MODE_LABEL[c]))
                        for c in cells],
               loc="upper right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


def fig_channels(summ, models, out):
    fig, axes = plt.subplots(1, len(MODES) + 1, figsize=(15, 4.2), sharey=True)
    for ax, mode in zip(axes, ["baseline"] + MODES):
        _style(ax)
        ax.set_title(MODE_LABEL.get(mode, "Baseline"), fontsize=10.5, color=INK, pad=8)
        names, bottoms = [], []
        for m in models:
            ch = summ[m]["per_mode"].get(mode, {}).get("channels", {})
            tot = sum(ch.values()) or 1
            names.append(PRETTY.get(m, m))
            bottoms.append({k: 100 * ch.get(k, 0) / tot for k in SEGMENTS})
        y = list(range(len(names)))
        left = [0.0] * len(names)
        for key in SEGMENTS:
            vals = [b[key] for b in bottoms]
            ax.barh(y, vals, left=left, color=CHANNEL_COLOR[key], height=0.72,
                    hatch="//" if key == "unjudged" else None,  # a judge hole is not a result
                    label=key if ax is axes[0] else None, zorder=3)
            left = [a + b for a, b in zip(left, vals)]
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of paths", fontsize=9, color=MUTED)
    axes[0].set_yticks(list(range(len(models))))
    axes[0].set_yticklabels([PRETTY.get(m, m) for m in models], fontsize=8.5, color=INK)
    axes[0].invert_yaxis()  # once only: sharey propagates, and an even number of flips is a no-op
    fig.legend(loc="upper center", ncol=5, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Where satisfaction is lost, by constraint type", fontsize=13, color=INK, y=1.14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


def main(scores_dir):
    scores_dir = Path(scores_dir)
    summ = json.loads((scores_dir / "scores_summary.json").read_text())
    models = [m for m in ORDER if m in summ] + [m for m in summ if m not in ORDER]
    colors = _colors(models)
    fig_creativity(summ, models, colors, scores_dir / "fig_regimeA_creativity.png")
    fig_2x2(summ, models, colors, scores_dir / "fig_regimeA_2x2.png")
    fig_channels(summ, models, scores_dir / "fig_regimeA_channels.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_regimeA_all")
