"""NeurIPS 2026 camera-ready: Table 1 body, headline figure, inter-benchmark/inter-test figure.

Changes from the submitted version (all in one place so the table and figures cannot drift apart):
  * specificity controls for MMLU-Pro ONLY (k = 1). Arena Overall is dropped from the control set
    (common-method bias with Arena CW; rebuttal to F63x). The frontier's R for a benchmark is
    therefore the signed correlation of the benchmark with MMLU-Pro.
  * a fourth construct, Combinatorial Creativity, measured by the Kombine benchmark overall score
    (data/kg_creat/kombine_test30/scores/composite.json).
  * the RAT is a row everywhere (zero-shot strict accuracy; no embedding, so absent from the
    per-embedding blocks).
  * every number is computed live from the current data, which includes NoveltyBench n = 23 and the
    2026-07-27 CDAT gate fix. The submitted Table 1 predates both.
  * Figure 3 colours every cell but prints the value only for correlations significant at p < .05.
  * Table 1 (moved to the appendix) has no best-test shading; bold marks p < .05.

MMLU-Pro comes from configs/comb_eval/benchmarks.json plus exact-name TIGER-Lab matches
(data/dat_eval/mmlu_pro_raw.json) for models missing there.

Outputs (new files; the shared figures used by the ICML/preprint versions are not touched):
  papers/iccc-2026/tables_neurips/main_result_body.tex     tabular only, no caption
  papers/iccc-2026/figures/neurips/fig_headline.pdf
  papers/iccc-2026/figures/neurips/fig_benchmark_correlations.pdf
  data/dat_eval/neurips_camera_ready/cells.json            every cell with r, p, n

    .venv/bin/python -m src.dat_eval.scripts.neurips_camera_ready
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from src.dat_eval.scripts.make_figures import (
    C_CAPP, C_CDAT, C_CNOV, C_DAT, C_GREY, C_PACE, _draw_corr_triangle, _panel_max_ceiling, plt)

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "data/dat_eval/run_v1/downstream/scores_v1/results"
BENCH = json.load(open(ROOT / "configs/comb_eval/benchmarks.json"))
PAPER = ROOT / "papers/iccc-2026"
OUT_FIG = PAPER / "figures/neurips"
OUT_TAB = PAPER / "tables_neurips/main_result_body.tex"
OUT_JSON = ROOT / "data/dat_eval/neurips_camera_ready/cells.json"

MMLU_RAW_MATCH = {
    "moonshotai_kimi-k2": "Kimi-K2-Instruct",
    "z-ai_glm-4-5-air": "GLM-4.5-Air",
    "google_gemini-3-1-pro-preview": "Gemini-3.1-Pro",
    "google_gemini-3-flash-preview": "Gemini-3-Flash(12/25)",
    "openai_gpt-5-2": "GPT-5.2",
}

TESTS = ["DAT", "CDAT", "CDAT-N", "CDAT-A", "PACE", "RAT"]
TEST_KEY = {"DAT": "dat", "CDAT": "cdat", "CDAT-N": "cdat_novelty",
            "CDAT-A": "cdat_appropriateness", "PACE": "pace"}
TEST_COLORS = {"DAT": C_DAT, "CDAT": C_CDAT, "CDAT-N": C_CNOV, "CDAT-A": C_CAPP,
               "PACE": C_PACE, "RAT": "#8C5A3D"}
CONSTRUCTS = {  # construct -> [(label, key)]
    "Creative Writing": [("Arena CW", "arena_cw"), ("EQ-Bench CW", "eq_bench_cw"),
                         ("Mazur CW", "mazur_cw_v2")],
    "Output Diversity": [("Hivemind Div.", "hivemind_diversity"),
                         ("NovBench Util.", "noveltybench_utility")],
    "Combinatorial Creativity": [("Kombine", "kombine")],
    "Scientific Ideation": [("LiveIdeaBench", "liveideabench")],
}
BENCHMARKS = [b for bs in CONSTRUCTS.values() for b in bs]
EMB_LABEL = {"glove": "GloVe 840B", "fasttext": "FastText crawl-300d-2M",
             "sbert": "Sentence-BERT all-mpnet-base-v2"}


def norm(m: str) -> str:
    return str(m).replace("/", "_").replace(".", "-").replace(":", "_").lower()


def ok(v):
    return v is not None and v not in ("", "---") and not (isinstance(v, float) and math.isnan(v))


# ---------------------------------------------------------------- data

def load_tests():
    """{block: {test: {model: score}}}; block 'overall' = mean z over the three embeddings."""
    me = json.load(open(RESULTS / "multi_embed_scores.json"))
    embs = sorted(me)
    models = sorted({m for e in embs for m in me[e]})
    valid = lambda v: ok(v) and v != 0
    blocks = {e: {t: {norm(m): float(me[e][m][k]) for m in me[e] if valid(me[e][m].get(k))}
                  for t, k in TEST_KEY.items()} for e in embs}
    overall = {}
    for t, k in TEST_KEY.items():
        st = {}
        for e in embs:
            vals = [me[e].get(m, {}).get(k) for m in models]
            vals = [v for v in vals if valid(v)]
            if vals:
                st[e] = (float(np.mean(vals)), float(np.std(vals)) or 1.0)
        col = {}
        for m in models:
            zs = [(me[e][m][k] - st[e][0]) / st[e][1] for e in embs
                  if e in st and valid(me[e].get(m, {}).get(k))]
            if zs:
                col[norm(m)] = float(np.mean(zs))
        overall[t] = col
    rat = {}
    for p in ("expansion_v1", "pilot_v1"):  # pilot wins on overlap, as in make_figures
        f = ROOT / f"data/new_tests/rat/{p}/summary.json"
        for m, s in json.load(open(f)).items():
            if s.get("n_total", 0) > 0 and s.get("n_errors", 0) < s["n_total"]:
                rat[norm(m)] = float(s["zs_accuracy_strict"])
    overall["RAT"] = rat
    return {"overall": overall, **blocks}


def load_benchmarks():
    out = {k: {norm(m): float(b[k]) for m, b in BENCH.items() if ok(b.get(k))}
           for _, k in BENCHMARKS if k != "kombine"}
    comp = json.load(open(ROOT / "data/kg_creat/kombine_test30/scores/composite.json"))
    out["kombine"] = {norm(m): float(v["overall"]) for m, v in comp["per_model"].items()
                      if ok(v.get("overall"))}
    for k in ("mmlu_pro", "arena_overall"):
        out[k] = {norm(m): float(b[k]) for m, b in BENCH.items() if ok(b.get(k))}
    raw = json.load(open(ROOT / "data/dat_eval/mmlu_pro_raw.json"))
    for m, row in MMLU_RAW_MATCH.items():
        out["mmlu_pro"].setdefault(m, float(raw[row]))
    return out


# ---------------------------------------------------------------- statistics

def p_of(r, n, k):
    df = n - 2 - k
    if df <= 0 or not -1 < r < 1:
        return float("nan")
    return float(2 * stats.t.sf(abs(r * math.sqrt(df / (1 - r * r))), df))


def cell(test: dict, bench: dict, mmlu: dict) -> dict:
    out = {}
    v_ms = sorted(set(test) & set(bench))
    if len(v_ms) >= 5:
        r = float(stats.pearsonr([test[m] for m in v_ms], [bench[m] for m in v_ms]).statistic)
        out["v"] = {"r": r, "p": p_of(r, len(v_ms), 0), "n": len(v_ms)}
    s_ms = sorted(set(v_ms) & set(mmlu))
    if len(s_ms) >= 5:
        y = np.array([bench[m] for m in s_ms]); g = np.array([mmlu[m] for m in s_ms])
        X = np.column_stack([np.ones(len(g)), g])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        r = float(stats.pearsonr([test[m] for m in s_ms], resid).statistic)
        out["s"] = {"r": r, "p": p_of(r, len(s_ms), 1), "n": len(s_ms)}
    return out


def signed_R(bench: dict, mmlu: dict):
    """Frontier R for k = 1: the benchmark's signed correlation with MMLU-Pro."""
    ms = sorted(set(bench) & set(mmlu))
    if len(ms) < 5:
        return None, len(ms)
    return float(stats.pearsonr([bench[m] for m in ms], [mmlu[m] for m in ms]).statistic), len(ms)


def compute():
    tests, bench = load_tests(), load_benchmarks()
    mmlu = bench["mmlu_pro"]
    cells = {blk: {t: {lab: cell(tests[blk][t], bench[key], mmlu) for lab, key in BENCHMARKS}
                   for t in tests[blk]} for blk in tests}
    R = {lab: dict(zip(("R", "n"), signed_R(bench[key], mmlu))) for lab, key in BENCHMARKS}
    # header n: models with the benchmark and ANY test (validity) / and MMLU-Pro (specificity)
    any_test = set().union(*tests["overall"].values())
    header_n = {lab: (len(set(bench[key]) & any_test), len(set(bench[key]) & any_test & set(mmlu)))
                for lab, key in BENCHMARKS}
    return cells, R, header_n, bench, tests


# ---------------------------------------------------------------- table

def fmt_tex(c, kind, green):
    if kind not in c:
        return "--"
    r, p = c[kind]["r"], c[kind]["p"]
    s = f"{r:+.2f}".replace("0.", ".")
    st = "^{***}" if p < .001 else "^{**}" if p < .01 else "^{*}" if p < .05 else ""
    body = f"${s}{st}$"
    if p < .05:
        body = (r"\good{" if r > 0 else r"\bad{") + body + "}"
    return body


def write_table(cells, header_n):
    nb = len(BENCHMARKS)
    cols = "l" + " c >{\\columncolor{gray!10}}c" * nb
    lines = [r"\begin{tabular}{@{}" + cols + "@{}}", r"\toprule"]
    grp, cm, c0 = [], [], 2
    for cname, bs in CONSTRUCTS.items():
        w = 2 * len(bs)
        # a header wider than its two columns would stretch them; stack it on two lines
        head = (rf"\begin{{tabular}}[b]{{@{{}}c@{{}}}}{cname.replace(' ', r' \\[-1pt] ', 1)}\end{{tabular}}"
                if len(bs) == 1 and len(cname) > 20 else cname)
        grp.append(rf"\multicolumn{{{w}}}{{c}}{{\textit{{{head}}}}}")
        cm.append(rf"\cmidrule(lr){{{c0}-{c0 + w - 1}}}"); c0 += w
    lines += [" & " + "\n & ".join(grp) + r" \\", " " + " ".join(cm)]
    lines.append(" & " + "\n & ".join(rf"\multicolumn{{2}}{{c}}{{\textbf{{{lab} $\uparrow$}}}}"
                                      for lab, _ in BENCHMARKS) + r" \\")
    lines.append(" & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{\footnotesize $n{{=}}{header_n[lab][0]}/{header_n[lab][1]}$}}"
        for lab, _ in BENCHMARKS) + r" \\")
    lines.append(" " + " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(nb)))
    lines.append(" & " + " & ".join(["Valid. & Spec."] * nb) + r" \\")
    lines.append(r"\midrule")
    # best test per benchmark in the Overall block: highest v + s
    best = {}
    for lab, _ in BENCHMARKS:
        cand = [(c[lab]["v"]["r"] + c[lab]["s"]["r"], t) for t, c in cells["overall"].items()
                if "v" in c[lab] and "s" in c[lab]]
        best[lab] = max(cand)[1] if cand else None
    blocks = [("overall", r"\textbf{Overall (mean $z$-score across 3 embeddings)}")] + \
             [(e, rf"\textit{{{EMB_LABEL[e]}}}") for e in ("glove", "fasttext", "sbert")]
    for bi, (blk, title) in enumerate(blocks):
        if bi:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{1 + 2 * nb}}}{{@{{}}l}}{{{title}}} \\")
        for t in TESTS:
            if t not in cells[blk]:
                continue
            name = rf"\quad {t}" if t in ("CDAT-N", "CDAT-A") else t
            row = [name]
            for lab, _ in BENCHMARKS:
                c = cells[blk][t][lab]
                row += [fmt_tex(c, "v", False), fmt_tex(c, "s", False)]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    OUT_TAB.write_text("% generated by src/dat_eval/scripts/neurips_camera_ready.py -- do not edit\n"
                       + "\n".join(lines) + "\n")
    return best



# ---------------------------------------------------------------- headline figure

def _place_labels(ax, pts, obstacles, xr, yr, fs):
    """Greedy label placement next to each construct-mean point: try 8 directions at a fixed
    offset in points, keep the one whose label box is farthest from every dot (means and cells)
    and from labels already placed. Distances are in axis-fraction units."""
    cands = [(1, 0, "left", "center"), (-1, 0, "right", "center"), (0, 1, "center", "bottom"),
             (0, -1, "center", "top"), (1, 1, "left", "bottom"), (-1, 1, "right", "bottom"),
             (1, -1, "left", "top"), (-1, -1, "right", "top")]
    fx = lambda x: (x - xr[0]) / (xr[1] - xr[0])
    fy = lambda y: (y - yr[0]) / (yr[1] - yr[0])
    off = 13  # points
    w_char, h = 0.0115 * fs / 15, 0.05 * fs / 15  # rough label size in axis fraction
    dots = [(fx(x), fy(y)) for x, y in obstacles]
    placed = []
    for name, (x, y) in pts.items():
        px, py = fx(x), fy(y)
        w = w_char * len(name)
        best, best_score = None, -1e9
        for dx, dy, ha, va in cands:
            ox, oy = px + dx * 0.022, py + dy * 0.035
            cx = ox + (w / 2 if ha == "left" else -w / 2 if ha == "right" else 0)
            cy = oy + (h / 2 if va == "bottom" else -h / 2 if va == "top" else 0)
            def gap(qx, qy):  # distance from a point to the label box
                return math.hypot(max(abs(qx - cx) - w / 2, 0), max(abs(qy - cy) - h / 2, 0))
            d_dots = min([gap(qx, qy) for qx, qy in dots if (qx, qy) != (px, py)] or [1])
            d_lab = min([math.hypot(max(abs(cx - lx) - (w + lw) / 2, 0), max(abs(cy - ly) - h, 0))
                         for lx, ly, lw in placed] or [1])
            inside = 0.01 < cx - w / 2 and cx + w / 2 < 0.99 and 0.01 < cy - h / 2 and cy + h / 2 < 0.99
            score = min(d_dots, d_lab) - (0 if inside else 1) - 0.002 * (abs(dx) + abs(dy))
            if score > best_score:
                best, best_score = (dx, dy, ha, va, cx, cy), score
        dx, dy, ha, va, cx, cy = best
        placed.append((cx, cy, w))
        n = math.hypot(dx, dy)
        ax.annotate(name, xy=(x, y), xytext=(dx / n * off, dy / n * off), textcoords="offset points",
                    ha=ha, va=va, fontsize=fs, fontweight="bold", color="black", zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.0, alpha=0.85))


def fig_headline(cells, R):
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    import matplotlib.transforms as mtransforms

    ov = cells["overall"]
    v_grid = np.linspace(-1, 1, 400)

    def pts_of(labels, t):
        out = []
        for lab in labels:
            c = ov[t][lab]
            if "v" in c and "s" in c:
                out.append((c["v"]["r"], c["s"]["r"], c["v"]["p"] < .05, c["s"]["p"] < .05))
        return out

    def ceiling(Rs):
        return _panel_max_ceiling([r for r in Rs if r is not None], v_grid)

    fig = plt.figure(figsize=(22.0, 12.4))
    nc = len(CONSTRUCTS)
    gs_top = GridSpec(1, nc, figure=fig, left=0.045, right=0.99, top=0.935, bottom=0.575, wspace=0.12)
    gs_bot = GridSpec(1, len(BENCHMARKS), figure=fig, left=0.045, right=0.99, top=0.405,
                      bottom=0.165, wspace=0.16)
    ax_top = [fig.add_subplot(gs_top[0, 0])]
    ax_top += [fig.add_subplot(gs_top[0, i], sharex=ax_top[0], sharey=ax_top[0]) for i in range(1, nc)]
    axes_bot = [fig.add_subplot(gs_bot[0, 0])]
    axes_bot += [fig.add_subplot(gs_bot[0, i], sharex=axes_bot[0], sharey=axes_bot[0])
                 for i in range(1, len(BENCHMARKS))]
    fig.text(0.5, 0.975, "(a) Prediction by construct", ha="center", va="bottom",
             fontsize=28.0, fontweight="bold")
    fig.text(0.5, 0.455, "(b) Prediction by benchmark", ha="center", va="bottom",
             fontsize=28.0, fontweight="bold")

    # shared limits for the top row
    allx, ally = [], []
    for cname, bs in CONSTRUCTS.items():
        labs = [b for b, _ in bs]
        for t in TESTS:
            p = pts_of(labs, t)
            allx += [q[0] for q in p] + ([np.mean([q[0] for q in p])] if p else [])
            ally += [q[1] for q in p] + ([np.mean([q[1] for q in p])] if p else [])
        cy = ceiling([R[b]["R"] for b in labs]); j = int(np.argmax(v_grid + cy))
        allx.append(v_grid[j]); ally.append(cy[j] + 0.06)
    xpad, ypad = 0.12 * (max(allx) - min(allx)), 0.10 * (max(ally) - min(ally))
    xr, yr = (min(allx) - xpad, max(allx) + xpad), (min(ally) - ypad, max(ally) + ypad)

    for ax, (cname, bs) in zip(ax_top, CONSTRUCTS.items()):
        labs = [b for b, _ in bs]
        ax.add_patch(Rectangle((0, 0), 10, 10, facecolor="#7ec587", alpha=0.10,
                               edgecolor="none", zorder=-1))
        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        cy = ceiling([R[b]["R"] for b in labs])
        ax.plot(v_grid, cy, color="black", linewidth=1.6, alpha=0.9, zorder=1)
        j = int(np.argmax(v_grid + cy))
        ax.scatter(v_grid[j], cy[j], marker="D", s=70, facecolor="black", edgecolor="black",
                   linewidth=0.8, zorder=7)
        ax.annotate(f"$(v^*, s^*)=({v_grid[j]:+.2f}, {cy[j]:+.2f})$", xy=(v_grid[j], cy[j]),
                    xytext=(-12, -8), textcoords="offset points", fontsize=17.0, ha="right",
                    va="top", zorder=8, bbox=dict(facecolor="white", edgecolor="none", pad=2.0, alpha=0.85))
        star_trans = mtransforms.offset_copy(ax.transData, fig=fig, x=8, y=8, units="points")
        means, obstacles = {}, []
        for t in TESTS:
            p = pts_of(labs, t)
            if not p:
                continue
            for v, s_, vs, ss in p:
                obstacles.append((v, s_))
                anys = vs or ss
                ax.scatter(v, s_, marker="o", s=60, c=[TEST_COLORS[t]],
                           edgecolor=("black" if anys else "white"),
                           linewidth=(1.0 if anys else 0.4), zorder=3, alpha=0.55)
                if vs and ss and v > 0 and s_ > 0:
                    ax.scatter(v, s_, marker="*", s=65, facecolor="#ffcc00", edgecolor="black",
                               linewidth=0.4, zorder=5, transform=star_trans)
            mx, my = float(np.mean([q[0] for q in p])), float(np.mean([q[1] for q in p]))
            ax.scatter(mx, my, marker="o", s=290, c=[TEST_COLORS[t]], edgecolor="black",
                       linewidth=1.4, zorder=4)
            means[t] = (mx, my); obstacles.append((mx, my))
        ax.set_xlim(*xr); ax.set_ylim(*yr)
        _place_labels(ax, means, obstacles, xr, yr, 15.0)
        ax.set_title(cname, fontsize=23.0)
        ax.tick_params(axis="both", labelsize=17.0)
        ax.set_xlabel(r"Validity  ($r$)", fontsize=21.0)
    ax_top[0].set_ylabel(r"Specificity  ($r \mid g$)", fontsize=21.0)
    for ax in ax_top[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    for ax, (lab, _) in zip(axes_bot, BENCHMARKS):
        c = R[lab]["R"]
        r1c = math.sqrt(max(0.0, 1 - c ** 2)); r1v = np.sqrt(np.clip(1 - v_grid ** 2, 0, None))
        upper, lower = v_grid * r1c + abs(c) * r1v, v_grid * r1c - abs(c) * r1v
        ax.fill_between(v_grid, lower, upper, color="#d0d0d0", alpha=0.35, zorder=0)
        ax.plot(v_grid, upper, color="black", linewidth=1.6, zorder=1)
        ax.plot(v_grid, lower, color="black", linewidth=0.7, linestyle=":", alpha=0.5, zorder=1)
        ax.axhline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        ax.axvline(0, color=C_GREY, linewidth=0.5, linestyle=":", alpha=0.7, zorder=0)
        j = int(np.argmax(v_grid + upper))
        ax.scatter(v_grid[j], upper[j], marker="D", s=55, facecolor="black", edgecolor="black",
                   linewidth=0.8, zorder=7)
        for t in TESTS:
            cc = ov[t][lab]
            if "v" in cc and "s" in cc:
                ax.scatter(cc["v"]["r"], cc["s"]["r"], marker="o", s=85, c=[TEST_COLORS[t]],
                           edgecolor="black", linewidth=0.9, zorder=3, alpha=0.95)
        ax.set_title(lab, fontsize=20.0)
        ax.tick_params(axis="both", labelsize=16.0)
        ax.set_xlabel(r"Validity  ($r$)", fontsize=20.0)
    axes_bot[0].set_xlim(-1.02, 1.02); axes_bot[0].set_ylim(-1.05, 1.05)
    axes_bot[0].set_ylabel(r"Specificity  ($r \mid g$)", fontsize=21.0)
    for ax in axes_bot[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    handles = [Line2D([], [], marker="o", linestyle="none", markerfacecolor=TEST_COLORS[t],
                      markeredgecolor="black", markeredgewidth=1.0, markersize=15, label=t)
               for t in TESTS]
    handles += [Line2D([], [], color="black", linewidth=1.4, label="theoretical ceiling"),
                Line2D([], [], marker="D", linestyle="none", markerfacecolor="black",
                       markeredgecolor="black", markersize=10,
                       label=r"$(v^*, s^*) = \arg\max\,(v + s)$")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=len(handles),
               frameon=False, fontsize=22, handletextpad=0.4, columnspacing=1.4)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    out = OUT_FIG / "fig_headline.pdf"
    plt.savefig(out); plt.savefig(out.with_suffix(".png")); plt.close()
    return out


# ---------------------------------------------------------------- Figure 3: significant cells only

def _sig_triangle(ax, mat, pmat, labels, fs=14):
    """Lower triangle coloured by r for every cell; the value is printed only where p < .05."""
    from matplotlib.patches import Rectangle
    n = len(labels)
    disp = np.ma.masked_array(np.nan_to_num(mat), mask=~np.tri(n, dtype=bool) | np.isnan(mat))
    im = ax.imshow(disp, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    for i in range(n):
        for j in range(i + 1):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black",
                                   linewidth=0.2, zorder=4))
            v = mat[i, j]
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=fs, color="white", zorder=3)
            elif not np.isnan(v) and pmat[i, j] < .05:  # value printed only when significant
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=fs,
                        color=("white" if abs(v) > 0.6 else "black"), zorder=3)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=fs)
    ax.set_yticklabels(labels, fontsize=fs)
    ax.tick_params(axis="both", which="major", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def _corr_mats(series: list[dict]):
    n = len(series)
    mat, pm, nm = np.full((n, n), np.nan), np.full((n, n), np.nan), np.zeros((n, n), int)
    for i in range(n):
        for j in range(n):
            ms = sorted(set(series[i]) & set(series[j]))
            nm[i, j] = len(ms)
            if len(ms) >= 5:
                r, p = stats.pearsonr([series[i][m] for m in ms], [series[j][m] for m in ms])
                mat[i, j], pm[i, j] = r, p
    return mat, pm, nm


def fig_benchmark_correlations(bench, tests):
    from matplotlib.gridspec import GridSpec
    keys_a = ["mmlu_pro", "arena_cw", "eq_bench_cw", "mazur_cw_v2",
              "hivemind_diversity", "noveltybench_utility", "kombine", "liveideabench"]
    labels_a = ["MMLU-Pro", "Arena CW", "EqBench CW", "Mazur V2", "HiveMind",
                "NoveltyBench", "Kombine", "LiveIdea"]
    mat_a, p_a, n_a = _corr_mats([bench[k] for k in keys_a])
    mat_b, p_b, n_b = _corr_mats([tests["overall"][t] for t in TESTS])
    na, nb = len(keys_a), len(TESTS)
    fig = plt.figure(figsize=(16.0, 7.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[na, nb], wspace=0.20)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    im = _sig_triangle(ax1, mat_a, p_a, labels_a)
    _sig_triangle(ax2, mat_b, p_b, TESTS)
    fig.canvas.draw()
    pos1 = ax1.get_position(); cs = pos1.height / na; pos2 = ax2.get_position()
    ax2.set_position([pos2.x0, pos1.y0, pos2.width, cs * nb])
    ax1.set_title("(a) Inter-benchmark correlations", fontsize=19, loc="left", pad=6)
    ax2.set_title("(b) Inter-test correlations", fontsize=19, loc="left", pad=6)
    cb = fig.colorbar(im, ax=[ax1, ax2], shrink=0.7, pad=0.03)
    cb.set_label("Pearson $r$", fontsize=14); cb.ax.tick_params(labelsize=12)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    out = OUT_FIG / "fig_benchmark_correlations.pdf"
    plt.savefig(out, bbox_inches="tight"); plt.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close()
    return out, {"benchmarks": {"labels": labels_a, "r": mat_a.tolist(), "p": p_a.tolist(), "n": n_a.tolist()},
                 "tests": {"labels": TESTS, "r": mat_b.tolist(), "p": p_b.tolist(), "n": n_b.tolist()}}


# ---------------------------------------------------------------- main

def main():
    cells, R, header_n, bench, tests = compute()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    best = write_table(cells, header_n)
    f1 = fig_headline(cells, R)
    f3, fig3 = fig_benchmark_correlations(bench, tests)
    OUT_JSON.write_text(json.dumps({"cells": cells, "R_mmlu": R, "header_n": header_n,
                                    "best_overall": best, "fig3": fig3}, indent=1))
    print(f"wrote {f1}, {f3}")
    print("frontier R (signed corr with MMLU-Pro):",
          {k: (round(v["R"], 2) if v["R"] is not None else None, v["n"]) for k, v in R.items()})
    print("header n:", header_n)
    print("best:", best)
    for t in TESTS:
        row = f"{t:7s}"
        for lab, _ in BENCHMARKS:
            c = cells["overall"][t][lab]
            f = lambda k: (f"{c[k]['r']:+.2f}{'*' if c[k]['p'] < .05 else ' '}" if k in c else "  -- ")
            row += f" | {f('v')} {f('s')}"
        print(row)
    print(f"wrote {OUT_TAB}, {OUT_JSON}")


if __name__ == "__main__":
    main()
