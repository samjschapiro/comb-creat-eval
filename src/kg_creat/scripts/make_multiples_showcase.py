"""Render the inventive-multiple clusters found by analyze_inventive_multiples.py.

Everything is generated from `data/kg_creat/kombine_test30/analysis/inventive_multiples.json`, so the
prose in the report can never drift from the data. Three artifacts:

  <report>/multiples_showcase.html   -- standalone catalogue of every cluster (open locally)
  <scratch>/multiples_artifact.html  -- the same catalogue as an Artifact page (no document wrapper;
                                        the Artifact host supplies <!doctype>, charset and viewport)
  <report>/examples_section.md       -- the worked-examples section of report.md, largest cross-family
                                        clusters only

Design: a specimen catalogue. Cool-biased paper, Spectral for the coined inventions, IBM Plex Sans for
running text, IBM Plex Mono for the triples, and the benchmark's own tag vocabulary (u / v / uv /
emergent) as the colour system -- `uv`, the slot both inputs organize, is the one that carries the
accent, because it is what separates a fusion from a concatenation.

    .venv/bin/python -m src.kg_creat.scripts.make_multiples_showcase
"""
import html
import json
from collections import Counter
from pathlib import Path

from src.kg_creat.scripts.plot_radar import BRAND, DISPLAY

SRC = Path("data/kg_creat/kombine_test30/analysis/inventive_multiples.json")
OUT = Path("docs/reports/2026-09-01_kg_creat_inventive_multiples")
ARTIFACT = Path("/private/tmp/claude-501/-Users-schapiro-Desktop-Experiments-comb-creat-eval/"
                "3bb7078c-f53d-4e87-a8ec-3fa06af28eb9/scratchpad/multiples_artifact.html")
N_MD = 5          # cross-family clusters written into the markdown section
TAGS = ["u", "v", "uv", "emergent"]
PROV_LABEL = {"openai": "OpenAI", "anthropic": "Anthropic", "google": "Google", "x-ai": "xAI",
              "deepseek": "DeepSeek", "qwen": "Qwen", "z-ai": "Z-AI", "meta-llama": "Meta"}


def _prov(model_key):
    return next((p for p in PROV_LABEL if str(model_key).startswith(p)), "other")


def _brand(prov):
    return BRAND.get({"meta-llama": "meta"}.get(prov, prov), "#8A94A3")


def _disp(model_key):
    return DISPLAY.get(model_key, model_key.split("_", 1)[-1])


def cluster_label(members):
    """The cluster's shared invention: the most common coined name, lower-cased."""
    return Counter(str(m["name"]).lower() for m in members).most_common(1)[0][0]


# ---------------------------------------------------------------- markdown (report section)
def _triples_md(task, structure):
    if task == "blending":
        return [f"- ({t[0]}, {t[1]}, {t[2]}) `[{t[3] or '?'}]`" for t in structure if len(t) >= 3]
    out = []
    for p in structure:
        s, i = p.get("source"), p.get("image")
        if s and i:
            out.append(f"- ({s[0]}, {s[1]}, {s[2]}) ⇒ ({i[0]}, {i[1]}, {i[2]})")
    return out


def markdown(clusters):
    xf = [c for c in clusters if len(c["providers"]) >= 2][:N_MD]
    L = ["## Cross-family worked examples", ""]
    L.append(f"Of the {sum(1 for c in clusters if len(c['providers']) >= 2)} structural clusters that span two or "
             f"more provider families, the {len(xf)} largest are shown in full: every member's invention, with its "
             f"generic space (blend) or projected source (analogy) and its structure. The agreement is not only the "
             f"coined name but the shared abstraction, and repeatedly the same emergent mapping.")
    L.append("")
    for c in xf:
        fams = ", ".join(PROV_LABEL[p] for p in c["providers"] if p in PROV_LABEL)
        L.append(f"### {c['u']} + {c['v']} — {c['task']}")
        L.append("")
        L.append(f"*{c['size']} models across {len(c['providers'])} families: {fams}.*")
        L.append("")
        for m in c["members"]:
            key = "g" if c["task"] == "blending" else "φ"
            L.append(f"**{_disp(m['model'])}** ({PROV_LABEL.get(_prov(m['model']), '—')})  ")
            L.append(f"{key}: *{m['concept']}* · invention: **{m['name']}**  ")
            L += _triples_md(c["task"], m["structure"])
            L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- HTML catalogue
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root{
  --paper:#F4F6F9; --panel:#FFFFFF; --ink:#14161B; --muted:#5C6472; --faint:#8A94A3;
  --rule:#DCE1E9; --rule-soft:#E9EDF3; --accent:#2E5FA3;
  --u:#A8476A; --v:#2F7D6E; --uv:#2E5FA3; --emergent:#6A5FC0;
  --chip:#EDF1F7; --shadow:0 1px 2px rgba(20,30,50,.06);
  --sans:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
  --serif:'Spectral',Georgia,'Times New Roman',serif;
  --mono:'IBM Plex Mono',ui-monospace,'SF Mono',Menlo,monospace;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --paper:#0F1116; --panel:#171A21; --ink:#E7EAF1; --muted:#9AA4B4; --faint:#767F8E;
    --rule:#262B35; --rule-soft:#1E222B; --accent:#7FA9E8;
    --u:#D98099; --v:#63B6A2; --uv:#7FA9E8; --emergent:#A79BEE;
    --chip:#1E222B; --shadow:0 1px 2px rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"]{
  --paper:#0F1116; --panel:#171A21; --ink:#E7EAF1; --muted:#9AA4B4; --faint:#767F8E;
  --rule:#262B35; --rule-soft:#1E222B; --accent:#7FA9E8;
  --u:#D98099; --v:#63B6A2; --uv:#7FA9E8; --emergent:#A79BEE;
  --chip:#1E222B; --shadow:0 1px 2px rgba(0,0,0,.4);
}

body{background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.55;
     -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:44px 24px 96px}
a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}

/* ---- masthead ---- */
.eyebrow{font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--faint);
         font-weight:600;margin:0 0 10px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(30px,4.4vw,46px);line-height:1.08;
   margin:0 0 14px;text-wrap:balance;letter-spacing:-.012em}
.stand{max-width:66ch;color:var(--muted);font-size:16.5px;margin:0 0 6px}
.stand em{color:var(--ink);font-style:italic}
.crit{max-width:66ch;color:var(--faint);font-size:14px;margin:14px 0 0}
.crit code{font-family:var(--mono);font-size:12.5px;color:var(--muted)}

.stats{display:flex;flex-wrap:wrap;gap:34px;margin:30px 0 6px;padding:20px 0;
       border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.stat{display:flex;flex-direction:column;gap:2px}
.stat b{font-family:var(--serif);font-size:27px;font-weight:600;font-variant-numeric:tabular-nums;
        line-height:1}
.stat span{font-size:12px;color:var(--faint);letter-spacing:.04em}

/* ---- controls ---- */
.controls{position:sticky;top:0;z-index:8;display:flex;flex-wrap:wrap;gap:10px;align-items:center;
          padding:14px 0;margin:0 0 26px;background:var(--paper);border-bottom:1px solid var(--rule-soft)}
.controls input{flex:1 1 260px;min-width:200px;font:inherit;font-size:14px;color:var(--ink);
   background:var(--panel);border:1px solid var(--rule);border-radius:7px;padding:8px 11px}
.controls input::placeholder{color:var(--faint)}
.seg{display:flex;border:1px solid var(--rule);border-radius:7px;overflow:hidden;background:var(--panel)}
.seg button{font:inherit;font-size:13px;color:var(--muted);background:none;border:0;cursor:pointer;
            padding:8px 14px;border-right:1px solid var(--rule)}
.seg button:last-child{border-right:0}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}
.count{font-size:12.5px;color:var(--faint);font-variant-numeric:tabular-nums;margin-left:auto}

/* ---- cluster ---- */
.cluster{margin:0 0 40px;scroll-margin-top:72px}
.chead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;padding-bottom:10px;
       border-bottom:1px solid var(--rule)}
.task{font-size:10.5px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
      padding:3px 8px;border-radius:4px;color:#fff;flex:none;position:relative;top:-2px}
.task.blending{background:var(--uv)}
.task.analogy{background:var(--emergent)}
.anchors{font-family:var(--serif);font-size:22px;font-weight:600;letter-spacing:-.005em}
.mult{font-family:var(--mono);font-size:15px;color:var(--accent);font-weight:500}
.fams{font-size:12.5px;color:var(--faint);margin-left:auto}

.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px;margin-top:16px}
.card{background:var(--panel);border:1px solid var(--rule-soft);border-radius:9px;padding:15px 16px 14px;
      box-shadow:var(--shadow);display:flex;flex-direction:column;gap:9px}
.mhead{display:flex;align-items:center;gap:8px}
.dot{width:9px;height:9px;border-radius:50%;flex:none}
.model{font-size:13px;font-weight:600}
.lab{font-size:11.5px;color:var(--faint);margin-left:auto}
.gs{font-family:var(--serif);font-style:italic;font-size:14px;color:var(--muted);
    border-left:2px solid var(--rule);padding-left:10px}
.inv{font-family:var(--serif);font-size:19px;font-weight:600;line-height:1.2}
.trips{display:flex;flex-direction:column;gap:5px;font-family:var(--mono);font-size:12px;
       line-height:1.45;overflow-x:auto}
.trip{display:flex;gap:7px;align-items:baseline}
.tag{flex:none;width:54px;font-size:9.5px;font-weight:500;letter-spacing:.05em;text-transform:uppercase;
     padding-top:2px}
.tag.u{color:var(--u)} .tag.v{color:var(--v)} .tag.uv{color:var(--uv);font-weight:600}
.tag.emergent{color:var(--emergent)}
.proj{white-space:nowrap}
.arrow{color:var(--faint);padding:0 4px}
.names{font-size:13px;color:var(--muted);margin:12px 0 2px}
.names b{font-family:var(--serif);font-size:15px;color:var(--ink);font-weight:600}
.names i{font-style:normal;color:var(--faint);font-family:var(--mono);font-size:11.5px}

.cons{margin:16px 0 0;display:grid;gap:0}
.cons-h{font-size:11px;letter-spacing:.11em;text-transform:uppercase;color:var(--faint);font-weight:600;
        margin:0 0 8px}
.slot{display:grid;grid-template-columns:1fr auto;gap:6px 16px;align-items:baseline;padding:9px 0;
      border-bottom:1px solid var(--rule-soft)}
.slot:last-child{border-bottom:0}
.slot-g{font-family:var(--serif);font-size:16px;font-weight:600;line-height:1.25}
.slot-g .tag{display:inline-block;width:auto;margin-left:8px;vertical-align:1px}
.slot-alt{grid-column:1;font-size:12.5px;color:var(--faint);font-family:var(--mono);line-height:1.5}
.cov{display:flex;align-items:center;gap:8px;grid-row:1;grid-column:2;white-space:nowrap}
.cov-n{font-family:var(--mono);font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
.pips{display:flex;gap:2px}
.pip{width:6px;height:12px;border-radius:1.5px;background:var(--rule)}
.pip.inc{background:var(--uv)} .pip.out{background:var(--faint);opacity:.55}

details{margin-top:14px}
summary{cursor:pointer;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
        font-weight:600;padding:6px 0;list-style:none;display:flex;align-items:center;gap:7px}
summary::-webkit-details-marker{display:none}
summary::before{content:"+";font-family:var(--mono);font-size:14px;color:var(--accent)}
details[open] summary::before{content:"\2212"}
summary:hover{color:var(--ink)}

.outs{margin-top:18px;padding-top:14px;border-top:1px dashed var(--rule)}
.outs-h{display:flex;align-items:baseline;gap:10px;font-size:11px;letter-spacing:.11em;
        text-transform:uppercase;color:var(--faint);font-weight:600;margin:0 0 10px}
.outs-h b{font-family:var(--mono);font-size:11.5px;letter-spacing:0;text-transform:none;color:var(--muted);
          font-weight:500}
.outs-note{font-size:12.5px;color:var(--faint);margin:-4px 0 10px;max-width:78ch}
.why{font-family:var(--sans);font-style:normal;font-size:10.5px;letter-spacing:.05em;
     text-transform:uppercase;white-space:nowrap;padding:1px 6px;border-radius:4px;
     background:var(--chip);color:var(--muted);margin-left:8px}
.why.inv{color:var(--uv)} .why.abs{color:var(--u)} .why.both{color:var(--faint)}
.why span{font-family:var(--mono);text-transform:none;letter-spacing:0;opacity:.8}
.outs ul{list-style:none;margin:0;padding:0;display:grid;gap:0}
.outs li{display:grid;grid-template-columns:11px minmax(96px,auto) minmax(120px,max-content) 1fr;
         gap:10px;align-items:baseline;padding:7px 0;border-bottom:1px solid var(--rule-soft);font-size:13px}
.outs li:last-child{border-bottom:0}
.outs .dot{position:relative;top:1px}
.outs .om{font-size:12px;color:var(--muted)}
.outs .oname{font-family:var(--serif);font-size:15px;font-weight:600}
.outs .og{color:var(--muted);font-family:var(--serif);font-style:italic;font-size:13.5px}
.outs .og b{font-style:normal;font-family:var(--sans);font-size:10.5px;letter-spacing:.05em;
            text-transform:uppercase;color:var(--faint);white-space:nowrap}
.empty{color:var(--faint);font-size:14px;padding:40px 0}
.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:12.5px;color:var(--muted);margin:0 0 30px}
.legend span{display:flex;gap:6px;align-items:baseline}
.legend b{font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase}
.legend .lu{color:var(--u)} .legend .lv{color:var(--v)} .legend .luv{color:var(--uv)}
.legend .lem{color:var(--emergent)}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);font-size:13px;color:var(--faint);
       max-width:70ch}
@media (max-width:640px){.fams{margin-left:0;width:100%}.stats{gap:22px}}
</style>
"""


def _triple_html(t):
    tag = (t[3] if len(t) > 3 and t[3] else "?").lower()
    cls = tag if tag in TAGS else "u"
    body = html.escape(f"({t[0]}, {t[1]}, {t[2]})")
    return f'<div class="trip"><span class="tag {cls}">{html.escape(tag)}</span><span>{body}</span></div>'


def _proj_html(p):
    s, i = p.get("source"), p.get("image")
    if not (s and i):
        return ""
    return (f'<div class="trip"><span>{html.escape(f"({s[0]}, {s[1]}, {s[2]})")}'
            f'<span class="arrow">&rarr;</span>{html.escape(f"({i[0]}, {i[1]}, {i[2]})")}</span></div>')


def catalogue(d):
    lv, tk, pr, im = d["levels"], d["task"], d["provider"], d["inventions_in_a_multiple"]
    P = ['<title>Inventive Multiples</title>', STYLE, '<div class="wrap">',
         f'<p class="eyebrow">Kombine &middot; {d["n_models"]} models &middot; '
         f'{d["n_anchor_pairs"]} anchor pairs</p>',
         '<h1>When two models invent the same thing</h1>',
         f'<p class="stand">Give {d["n_inventions"]:,} inventions the same starting pair of concepts and models '
         f'rediscover each other\'s work. An <em>inventive multiple</em> is two models that invented the same '
         f'entity <em>by re-using the same properties</em> &mdash; the model analogue of Newton and '
         f'Leibniz. Every one found in the benchmark is catalogued below, and under each one, the models '
         f'that saw the same two concepts and invented something else &mdash; no cluster ever takes the '
         f'whole pool. Often those models share the cluster\'s abstraction and still build something '
         f'different on it: the anchors can force the schema, the invention stays the model\'s.</p>',
         f'<p class="crit">Each cluster leads with what its models actually <em>say</em> about the '
         f'invention &mdash; the (relation, object) slots that recur across them, counted over all '
         f'{max(c["size"] + len(c.get("outsiders", [])) for c in d["clusters"])} models that answered '
         f'the item, however each one worded it. The coined names differ far more than the properties '
         f'do. Individual inventions, in full, are one click away.</p>',
         f'<p class="crit">A pair qualifies when the two inventions <strong>re-use at least '
         f'{d["k_shared"]} of the same properties</strong> &mdash; every triple reduced to its '
         f'&ldquo;relation object&rdquo;, the coined name dropped, matched one-to-one at cosine '
         f'<code>&ge; {d["tau_slot"]}</code> &mdash; <em>and</em> their underlying abstractions align '
         f'(<code>&ge; {d["tau_con"]}</code>). The name is never an input: of the '
         f'{d["same_name_pairs"]["n"]} pairs that coined the identical name, only '
         f'{d["same_name_pairs"]["pct_that_are_multiples"]:.0f}% qualify.</p>',
         '<div class="stats">',
         f'<div class="stat"><b>{d["n_clusters"]}</b><span>clusters of rediscovery</span></div>',
         f'<div class="stat"><b>{im["pct"]:.0f}%</b><span>of inventions are in one</span></div>',
         f'<div class="stat"><b>{max(c["size"] for c in d["clusters"])}</b><span>models on one invention</span></div>',
         f'<div class="stat"><b>{tk["blending_pct"]/tk["analogy_pct"]:.1f}&times;</b>'
         f'<span>blending over analogy</span></div>',
         f'<div class="stat"><b>{pr["rr"]:.1f}&times;</b><span>same provider over cross</span></div>',
         '</div>',
         '<div class="controls">',
         '<input id="q" type="search" placeholder="Search an anchor, an invention, a model…" '
         'aria-label="Search clusters">',
         '<div class="seg" role="group" aria-label="Filter by task">',
         '<button data-task="all" aria-pressed="true">All</button>',
         '<button data-task="blending" aria-pressed="false">Blends</button>',
         '<button data-task="analogy" aria-pressed="false">Analogies</button>',
         '</div><span class="count" id="count"></span></div>',
         '<p class="legend"><span><b class="lu">u</b> from the first input</span>'
         '<span><b class="lv">v</b> from the second</span>'
         '<span><b class="luv">uv</b> one slot both inputs organize</span>'
         '<span><b class="lem">emergent</b> true of the blend, of neither input</span>'
         '<span>Analogy rows read source &rarr; image.</span></p>']

    for c in d["clusters"]:
        fams = ", ".join(PROV_LABEL.get(p, p) for p in c["providers"])
        sep = "+" if c["task"] == "blending" else "&#8759;"
        search = " ".join([c["u"], c["v"], c["task"], fams] +
                          [m["name"] for m in c["members"]] + [_disp(m["model"]) for m in c["members"]] +
                          [m["concept"] for m in c["members"]] +
                          [o["name"] for o in c.get("outsiders", [])]).lower()
        P.append(f'<section class="cluster" data-task="{c["task"]}" data-search="{html.escape(search, quote=True)}">')
        label = "blend" if c["task"] == "blending" else "analogy"
        P.append(f'<div class="chead"><span class="task {c["task"]}">{label}</span>'
                 f'<span class="anchors">{html.escape(c["u"])} {sep} {html.escape(c["v"])}</span>'
                 f'<span class="mult">&times;{c["size"]}</span>'
                 f'<span class="fams">{html.escape(fams)}</span></div>')
        names = Counter(str(m["name"]) for m in c["members"]).most_common()
        strip = " &middot; ".join(f'<b>{html.escape(n)}</b>' + (f' <i>&times;{k}</i>' if k > 1 else "")
                                 for n, k in names)
        P.append(f'<p class="names">{strip}</p>')

        cons = c.get("consensus", [])[:6]
        if cons:
            total = c["size"] + len(c.get("outsiders", []))
            P.append('<div class="cons"><p class="cons-h">What they all say about it &mdash; '
                     'shared (relation, object) slots, however they are worded</p>')
            for r in cons:
                pips = ("".join('<span class="pip inc"></span>' for _ in range(r["models_in_cluster"])) +
                        "".join('<span class="pip out"></span>'
                                for _ in range(r["models"] - r["models_in_cluster"])) +
                        "".join('<span class="pip"></span>' for _ in range(total - r["models"])))
                alt = [e for e in r["examples"] if e != r["gloss"]][:3]
                tag = (f'<span class="tag {r["tag"]}">{r["tag"]}</span>'
                       if r["tag"] in TAGS else "")
                P.append('<div class="slot">'
                         f'<div class="slot-g">{html.escape(r["gloss"])}{tag}</div>'
                         f'<div class="cov"><span class="pips">{pips}</span>'
                         f'<span class="cov-n">{r["models"]}/{total}</span></div>'
                         + (f'<div class="slot-alt">also: {html.escape(", ".join(alt))}'
                            + (f' &hellip; {r["n_variants"]} phrasings' if r["n_variants"] > 4 else "")
                            + '</div>' if alt else "")
                         + '</div>')
            P.append('</div>')

        P.append(f'<details><summary>Each model&rsquo;s invention in full &middot; {c["size"]} models'
                 f'</summary><div class="grid">')
        for m in c["members"]:
            pv = _prov(m["model"])
            P.append('<article class="card"><div class="mhead">'
                     f'<span class="dot" style="background:{_brand(pv)}"></span>'
                     f'<span class="model">{html.escape(_disp(m["model"]))}</span>'
                     f'<span class="lab">{PROV_LABEL.get(pv, "")}</span></div>')
            if m["concept"]:
                P.append(f'<p class="gs">{html.escape(m["concept"])}</p>')
            P.append(f'<p class="inv">{html.escape(m["name"] or "&mdash;")}</p><div class="trips">')
            if c["task"] == "blending":
                P += [_triple_html(t) for t in m["structure"] if len(t) >= 3]
            else:
                P += [_proj_html(p) for p in m["structure"]]
            P.append('</div></article>')
        P.append('</div></details>')

        outs = sorted(c.get("outsiders", []), key=lambda o: (o["other_cluster"] is not None, o["model"]))
        if outs:
            total = len(outs) + c["size"]
            shares = sum(1 for o in outs if o.get("blocked_by") == "properties")
            if shares == len(outs):
                note = ("Every one of them writes an abstraction close to the cluster's and still "
                        "re-uses too few of its properties: on this pair the schema is close to forced "
                        "by the anchors, and the models part company over what they build on it.")
            elif shares:
                note = (f"{shares} of {len(outs)} share the cluster's abstraction and diverge in the "
                        f"invention; the rest reached for a different schema altogether.")
            else:
                note = "None of them organizes the two inputs the way the cluster does."
            P.append(f'<details class="outs"><summary>Same anchors, another invention &middot; '
                     f'{len(outs)} models</summary><p class="outs-h">'
                     f'<b>{len(outs)} of {total} models</b></p>'
                     f'<p class="outs-note">{note} Each row is tagged with the clause that keeps it '
                     f'out, and its best cosine to a cluster member on the other one.</p><ul>')
            for o in outs:
                pv = _prov(o["model"])
                other = (f'<b>&nbsp;&middot; joined &ldquo;{html.escape(o["other_cluster"])}&rdquo;</b>'
                         if o["other_cluster"] else "")
                blocked, sh, ac = o.get("blocked_by"), o.get("shared", 0), o.get("abs_cos")
                if blocked == "properties":
                    why = (f'<span class="why inv">{sh} shared propert{"y" if sh == 1 else "ies"}'
                           f'<span> schema {ac:.2f}</span></span>')
                elif blocked == "abstraction":
                    why = f'<span class="why abs">different schema <span>{sh} shared</span></span>'
                else:
                    why = f'<span class="why both">{sh} shared, different schema</span>'
                other += why
                P.append('<li>'
                         f'<span class="dot" style="background:{_brand(pv)}"></span>'
                         f'<span class="om">{html.escape(_disp(o["model"]))}</span>'
                         f'<span class="oname">{html.escape(o["name"] or "—")}</span>'
                         f'<span class="og">{html.escape(o["concept"])}{other}</span></li>')
            P.append('</ul></details>')
        P.append('</section>')

    P.append('<p class="empty" id="empty" hidden>No cluster matches that search.</p>')
    P.append(f'<footer>Kombine, a three-task benchmark of combinatorial creativity. '
             f'{d["levels"]["structural"]["count"]} of {d["n_pairs"]:,} co-response model pairs are structural '
             f'multiples ({lv["structural"]["pct"]:.1f}%); {d["settings_with_multiple"]} of {d["n_settings"]} '
             f'(task, anchor-pair) settings produced at least one. Blends and analogies were elicited once per '
             f'model at temperature 0.9; every invention shown is verbatim model output.</footer>')
    P.append('</div>')
    P.append("""
<script>
const clusters = Array.from(document.querySelectorAll('.cluster'));
const q = document.getElementById('q'), count = document.getElementById('count'),
      empty = document.getElementById('empty');
let task = 'all';
function apply(){
  const term = q.value.trim().toLowerCase();
  let shown = 0;
  for (const el of clusters){
    const ok = (task === 'all' || el.dataset.task === task) &&
               (!term || el.dataset.search.includes(term));
    el.hidden = !ok;
    if (ok) shown++;
  }
  count.textContent = shown + (shown === 1 ? ' cluster' : ' clusters');
  empty.hidden = shown > 0;
}
q.addEventListener('input', apply);
for (const b of document.querySelectorAll('.seg button')){
  b.addEventListener('click', () => {
    task = b.dataset.task;
    document.querySelectorAll('.seg button').forEach(x =>
      x.setAttribute('aria-pressed', String(x === b)));
    apply();
  });
}
apply();
</script>""")
    return "\n".join(P)


def main():
    d = json.loads(SRC.read_text())
    body = catalogue(d)
    (OUT / "multiples_showcase.html").write_text(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>"
        f"{body}</head></html>")
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(body)                      # Artifact host supplies the document wrapper
    (OUT / "examples_section.md").write_text(markdown(d["clusters"]))
    print(f"wrote {OUT/'multiples_showcase.html'}, {ARTIFACT} ({d['n_clusters']} clusters), "
          f"{OUT/'examples_section.md'}")


if __name__ == "__main__":
    main()
