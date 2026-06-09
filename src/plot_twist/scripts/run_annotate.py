"""Annotate every story (LLM-generated + human gold) with setup / reveal /
score-rationale, and write a combined JSON plus a readable markdown log.

Durable + resumable: each annotation is saved per-story as it completes; a re-run
without --overwrite reuses what is on disk (no re-spend).

Usage:
    python src/plot_twist/scripts/run_annotate.py configs/plot_twist/annotate.yaml --overwrite [--debug]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from src.utils import init_directory, load_config, save_config
from src.plot_twist.annotate import AnnotateConfig, annotate_stories


def _scores_by_id(csv_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in csv.DictReader(csv_path.open()):
        out[r["slug"]] = {
            k: r.get(k)
            for k in ("surprise", "coherence", "prose_quality", "overall", "twist_present", "geomean_surp_coh")
        }
    return out


def _build_items(cfg_dict: dict) -> list[dict]:
    items: list[dict] = []

    # LLM-generated stories
    gens = json.loads(Path(cfg_dict["llm_generations"]).read_text())
    llm_scores = _scores_by_id(Path(cfg_dict["llm_scores_csv"]))
    for r in gens:
        if not r.get("story"):
            continue
        items.append(
            {"id": r["id"], "source": r["model"], "story": r["story"], "scores": llm_scores.get(r["id"], {})}
        )

    # Human gold stories
    human_scores = _scores_by_id(Path(cfg_dict["human_scores_csv"]))
    texts_dir = Path(cfg_dict["human_texts_dir"])
    for txt in sorted(texts_dir.glob("*.txt")):
        slug = txt.stem
        items.append(
            {
                "id": slug,
                "source": "human",
                "story": txt.read_text(encoding="utf-8"),
                "scores": human_scores.get(slug, {}),
            }
        )
    return items


def _write_markdown(records: list[dict], manifest_titles: dict[str, str], path: Path) -> None:
    by_source: dict[str, list[dict]] = {}
    for r in records:
        by_source.setdefault(r["source"], []).append(r)

    # human first, then models alphabetically
    sources = (["human"] if "human" in by_source else []) + sorted(
        s for s in by_source if s != "human"
    )

    def sc(r, k):
        v = r.get("scores", {}).get(k)
        return "?" if v in (None, "", "None") else v

    lines = ["# Story annotations: setup / reveal / why it scored", ""]
    lines.append(
        f"{len(records)} stories ({sum(1 for r in records if r['source']=='human')} human, "
        f"{sum(1 for r in records if r['source']!='human')} LLM). Scores are judge-ensemble "
        "medians (1-5); geomean is geomean(surprise, coherence)."
    )
    lines.append("")
    for source in sources:
        recs = sorted(by_source[source], key=lambda r: -float(sc(r, "geomean_surp_coh") or 0))
        label = "Human gold set" if source == "human" else source
        lines.append(f"## {label}  ({len(recs)} stories)")
        lines.append("")
        for r in recs:
            title = manifest_titles.get(r["id"], "")
            head = f"### {r['id']}" + (f" — *{title}*" if title else "")
            lines.append(head)
            lines.append(
                f"surprise **{sc(r,'surprise')}** · coherence **{sc(r,'coherence')}** · "
                f"geomean **{sc(r,'geomean_surp_coh')}** · overall {sc(r,'overall')} · "
                f"twist_present {sc(r,'twist_present')}"
            )
            lines.append(f"- **Setup:** {r.get('setup') or '—'}")
            lines.append(f"- **Reveal:** {r.get('reveal') or '—'}")
            lines.append(f"- **Why scored:** {r.get('why_scored') or '—'}")
            lines.append("")
    path.write_text("\n".join(lines))


def main(config_path: str, overwrite: bool = False, debug: bool = False) -> None:
    cfg_dict = load_config(config_path)
    for f in ("output_dir", "annotator_model", "llm_generations", "llm_scores_csv",
              "human_texts_dir", "human_scores_csv"):
        if f not in cfg_dict:
            raise ValueError(f"FATAL: '{f}' required in config")

    if overwrite:
        out = init_directory(cfg_dict["output_dir"], overwrite=True)
    else:
        out = Path(cfg_dict["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
    save_config(cfg_dict, out)

    items = _build_items(cfg_dict)
    if debug:
        items = items[:3] + items[-2:]  # a few LLM + a couple human
    print(f"annotating {len(items)} stories with {cfg_dict['annotator_model']}\n")

    cfg = AnnotateConfig(
        model=cfg_dict["annotator_model"],
        temperature=cfg_dict.get("temperature", 0.0),
        max_tokens=cfg_dict.get("max_tokens", 400),
        concurrency=cfg_dict.get("concurrency", 16),
    )
    records = asyncio.run(annotate_stories(cfg, items, cache_dir=out / "cache"))

    n_ok = sum(1 for r in records if r.get("reveal"))
    print(f"annotated {n_ok}/{len(records)} (failures have reveal=null)")

    manifest_titles = {}
    if cfg_dict.get("manifest"):
        man = json.loads(Path(cfg_dict["manifest"]).read_text())
        manifest_titles = {s["slug"]: s.get("title", "") for s in man["stories"]}

    (out / "annotations.json").write_text(json.dumps(records, indent=2, ensure_ascii=False))
    _write_markdown(records, manifest_titles, out / "annotations.md")
    print(f"\nsaved: {out/'annotations.json'}\n       {out/'annotations.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args.config_path, args.overwrite, args.debug)
