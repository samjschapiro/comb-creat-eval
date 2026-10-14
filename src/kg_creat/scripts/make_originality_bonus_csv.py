"""Turn the originality-bonus draft into Prolific's bulk-bonus upload (participant_id,amount).

Takes the top N eligible participants of originality_bonus_draft.csv (draft_originality_bonus.py), skipping the
sessions excluded as likely AI, maps each session_id to its Prolific ID through the identity files downloaded
from the Drive folder "Kombine study data (private)/identity" (one <session_id>.json each, with prolific_pid),
and checks each participant's submission status on Prolific (read only): a bonus can only be paid on an approved
submission, so any that are not approved are listed for approval first.

    .venv/bin/python -m src.kg_creat.scripts.make_originality_bonus_csv [--top 15] [--amount 2.00]
"""
import argparse
import csv
import json
import urllib.request
from pathlib import Path

COMP = Path("data/kg_creat/human_study/compensation")
IDENT = Path("data/kg_creat/human_study/identity")
STUDY = "6abd23503d9299654c1d658e"
EXCLUDE = {"5fcbab1199998a4b37d8a431", "eac00f68a7135a2daa8b88f8"}   # likely AI (README, 2026-10-14)


def prolific(path, tok):
    req = urllib.request.Request(path if path.startswith("http") else "https://api.prolific.com" + path,
                                 headers={"Authorization": "Token " + tok})
    return json.load(urllib.request.urlopen(req, timeout=60))


def main(top, amount):
    files = list(IDENT.glob("*.json"))
    if not files:
        raise SystemExit(f"FATAL: no identity files in {IDENT}; download the Drive identity folder there first")
    pid = {}
    for f in files:
        d = json.loads(f.read_text())
        sid = d.get("session_id") or f.stem.replace("identity_", "")
        p = d.get("prolific_pid") or (d.get("prolific") or {}).get("prolific_pid")
        if p:
            pid[sid] = p
    draft = [r for r in csv.DictReader(open(COMP / "originality_bonus_draft.csv"))
             if r["session_id"] not in EXCLUDE and float(r["score"]) > 0][:top]
    missing = [r["session_id"] for r in draft if r["session_id"] not in pid]
    if missing:
        raise SystemExit(f"FATAL: no Prolific ID for {missing}")
    env = dict(l.split("=", 1) for l in Path(".env").read_text().splitlines() if "=" in l and not l.startswith("#"))
    tok = env["PROLIFIC_API_TOKEN"].strip().strip('"')
    subs, r = [], prolific(f"/api/v1/studies/{STUDY}/submissions/?limit=200", tok)
    subs += r["results"]
    while r.get("_links", {}).get("next", {}).get("href"):
        r = prolific(r["_links"]["next"]["href"], tok)
        subs += r["results"]
    status = {}
    for s in subs:
        status.setdefault(s["participant_id"], []).append((s["id"], s["status"]))
    rows, to_approve = [], []
    for r in draft:
        p = pid[r["session_id"]]
        st = status.get(p, [])
        if not any(x == "APPROVED" for _i, x in st):
            to_approve.append((p, st))
        rows.append((p, amount, r["rank"], r["score"]))
    out = COMP / "originality_bonus_upload.csv"
    out.write_text("".join(f"{p},{a:.2f}\n" for p, a, _r, _s in rows))
    print(f"wrote {out}: {len(rows)} bonuses x ${amount:.2f} = ${len(rows) * amount:.2f} "
          f"(${len(rows) * amount * 4 / 3:.2f} with Prolific's 33.3% fee)")
    for p, a, rk, sc in rows:
        print(f"  {p},{a:.2f}   (draft rank {rk}, score {sc}, status {status.get(p)})")
    if to_approve:
        print("NOT YET APPROVED (approve before paying their bonus):")
        for p, st in to_approve:
            print(f"  {p} {st}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--amount", type=float, default=2.00)
    a = ap.parse_args()
    main(a.top, a.amount)
