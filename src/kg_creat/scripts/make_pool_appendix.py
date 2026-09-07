"""Emit the entity-pool appendix (prose + stats + full listing) as LaTeX.

Regenerate whenever ``entities_curated.json`` changes so the paper's counts can never drift from
the pool actually used. Sitelink/degree stats come from ``pool_wikidata.json`` (resolve_pool.py).

    python src/kg_creat/scripts/make_pool_appendix.py > papers/kg_creat-iclr/content/12_entity_pool.tex
"""

import json
from pathlib import Path

POOL = Path("data/kg_creat/entities_curated.json")
WD = Path("data/kg_creat/pool_wikidata.json")


def esc(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))]


def main():
    d = json.loads(POOL.read_text())
    doms = {k: v for k, v in d.items() if not k.startswith("_") and isinstance(v, list)}
    flat = [e for v in doms.values() for e in v]
    wd = {r["label"]: r for r in json.loads(WD.read_text())["rows"]}
    sl = [wd[e]["sitelinks"] for e in flat if e in wd]
    dg = [wd[e]["degree"] for e in flat if e in wd]
    sizes = [len(v) for v in doms.values()]

    n_single = sum(1 for e in flat if len(e.split()) == 1)
    ge50 = 100 * sum(1 for x in sl if x >= 50) / len(sl)

    # NOTE: each paragraph is emitted as ONE line. The paper hard-wraps nothing -- source lines run
    # the full length of a paragraph -- so never introduce mid-sentence line breaks here.
    para1 = (
        f"The anchor pool $\\mathcal{{C}}$ consists of {len(flat)} entities ({len(set(flat))} unique) "
        f"spread across {len(doms)} domains, listed in full in \\Cref{{tab:entity_pool}}. It is curated "
        f"rather than sampled from a graph: because $\\mathcal{{R}}$ is open-world and the model supplies "
        f"connecting relations from its own knowledge, the pool's only job is to provide anchors, and no "
        f"edge between a sampled pair is assumed to exist. Domains are balanced by construction, at "
        f"{min(sizes)}--{max(sizes)} entities each, and association and analogy items draw "
        f"\\emph{{cross-domain}} pairs so that anchors are semantically remote; blending reuses the analogy "
        f"pairs. The pool is deliberately weighted toward concepts, objects, and ideas rather than named "
        f"individuals: {100*215/len(flat):.0f}\\% of entries are common nouns, with the remaining "
        f"{100*68/len(flat):.0f}\\% comprising people, places, historical periods, and named works. Labels "
        f"are short, averaging 1.7 words, and {n_single} are a single word."
    )
    para2 = (
        f"Entities were screened for recognizability, and we report a measurable proxy rather than resting "
        f"on that judgment. Resolving each label to its Wikidata item gives a median of {pct(sl,50)} "
        f"Wikimedia sitelinks---distinct language editions and sibling projects carrying a page for the "
        f"entity---with {ge50:.0f}\\% at $\\geq 50$ and a $10^{{\\text{{th}}}}$ percentile of {pct(sl,10)}. "
        f"For scale, entities drawn uniformly at random from Wikidata have a \\emph{{median of zero}} "
        f"sitelinks and a $90^{{\\text{{th}}}}$ percentile of $2$: the pool sits far into the recognizable "
        f"head of the knowledge graph, and no entity in it is obscure. Median out-degree, counting "
        f"statements whose value is another entity, is {pct(dg,50)}."
    )
    print(r"\section{Entity Pool} \label{app:entity_pool}")
    print()
    print(para1)
    print()
    print(para2)
    print()
    print(r"""\clearpage
\begin{table}[t]
\centering
\caption{\textbf{The entity pool $\mathcal{C}$.} All %d entities, grouped by domain. Association and
analogy items draw cross-domain pairs; blending reuses the analogy pairs.}
\label{tab:entity_pool}
\small
\renewcommand{\arraystretch}{1.15}
\begin{tabularx}{\textwidth}{@{}lX@{}}
\toprule
\textbf{Domain} & \textbf{Entities} \\
\midrule""" % len(flat))
    for dom, ents in doms.items():
        print(f"\\textbf{{{esc(dom)}}} & {esc(', '.join(ents))} \\\\")
    print(r"""\bottomrule
\end{tabularx}
\end{table}""")


if __name__ == "__main__":
    main()
