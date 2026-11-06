"""Administer the DAT to local base models as a text-completion task, then score with GloVe.

The DAT prompt text is identical to the instruct runs (src/dat_eval/dat.py:DAT_PROMPT); it is wrapped in
a document frame ending in 'Response: ["' so a base model continues the JSON array. Sampling settings
match the paper's API runs (T in {1.0, 1.5, 2.0}, 40 trials each, top_p=1, top_k=0, 256 new tokens).

Resumable: a model whose response files already exist in output_dir is not regenerated.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dat_eval.dat import DAT_PROMPT, GloVeEmbeddings, score_dat

FRAME = (
    "The following is a transcript of a participant completing a word association task.\n\n"
    "Instructions: {prompt}\n\n"
    "Response: [\""
)


def slug(hf_id):
    return hf_id.replace("/", "_").lower().replace(".", "-")


def parse_words(completion):
    text = '["' + completion
    text = text.split("]")[0]
    words = re.findall(r'"([^"\n]*)"', text + '"')
    if len(words) < 2:
        words = [w.strip(' "\'\n') for w in text.lstrip('["').split(",")]
    return [w for w in words if w.strip()][:10]


def generate(model_cfg, cfg, out_dir):
    hf_id = model_cfg["hf_id"]
    files = {t: out_dir / slug(hf_id) / f"dat_responses_t{str(t).replace('.', '-')}.json" for t in cfg["temperatures"]}
    if all(f.exists() for f in files.values()):
        print(f"[skip] {hf_id}: responses exist")
        return
    print(f"[load] {hf_id}")
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=getattr(torch, cfg["dtype"])).to(cfg["device"])
    model.eval()
    prompt = FRAME.format(prompt=DAT_PROMPT)
    inputs = tok(prompt, return_tensors="pt").to(cfg["device"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for t, path in files.items():
        if path.exists():
            continue
        torch.manual_seed(cfg["seed"])
        with torch.no_grad():
            out = model.generate(**inputs, do_sample=True, temperature=t, top_p=cfg["top_p"], top_k=cfg["top_k"],
                                 max_new_tokens=cfg["max_new_tokens"], num_return_sequences=cfg["n_trials"],
                                 stop_strings=["]"], tokenizer=tok, pad_token_id=tok.pad_token_id)
        comps = tok.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        rows = [{"trial": k, "seed": cfg["seed"], "raw_response": '["' + c, "words": parse_words(c)}
                for k, c in enumerate(comps)]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=1))
        print(f"  T={t}: wrote {len(rows)} responses")
    del model
    torch.mps.empty_cache()


def score(cfg, out_dir):
    glove = GloVeEmbeddings(cfg["glove_path"])
    instruct = json.load(open(cfg["instruct_scores_path"]))["glove"]
    summary = {}
    for m in cfg["models"]:
        d = out_dir / slug(m["hf_id"])
        if not d.exists():
            continue
        by_t, pooled = {}, []
        for t in cfg["temperatures"]:
            rows = json.load(open(d / f"dat_responses_t{str(t).replace('.', '-')}.json"))
            res = [score_dat(r["words"], glove) for r in rows if r["words"]]
            valid = [r["score"] for r in res if r["sufficient"]]
            by_t[str(t)] = {"mean_score": float(np.mean(valid)) if valid else None, "n_sufficient": len(valid),
                            "n_trials": len(rows)}
            pooled += valid
        inst = instruct.get(m["instruct"], {}).get("dat") if m["instruct"] else None
        summary[m["hf_id"]] = {"dat": float(np.mean(pooled)) if pooled else None, "n_sufficient": len(pooled),
                               "by_temperature": by_t, "instruct_model": m["instruct"],
                               "instruct_dat": inst}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{'base model':28s} {'DAT':>6s} {'valid':>7s}   per-T (valid/40)              instruct DAT")
    for k, s in summary.items():
        per = "  ".join(f"{t}:{v['mean_score']:.1f}({v['n_sufficient']})" if v["mean_score"] else f"{t}:--(0)"
                        for t, v in s["by_temperature"].items())
        dat = f"{s['dat']:.2f}" if s["dat"] else "--"
        inst = f"{s['instruct_dat']:.2f}" if s["instruct_dat"] else "--"
        print(f"{k:28s} {dat:>6s} {s['n_sufficient']:>4d}/120   {per:30s} {inst}")


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    for m in cfg["models"]:
        try:
            generate(m, cfg, out_dir)
        except OSError as e:  # gated or missing repo: report and continue with the rest
            print(f"[fail] {m['hf_id']}: {str(e).splitlines()[0]}")
    score(cfg, out_dir)


if __name__ == "__main__":
    main(sys.argv[1])
