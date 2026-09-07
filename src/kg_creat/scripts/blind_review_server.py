"""Local BLIND-review web app: rate analogy/blend subjective dimensions yourself, without seeing the
model or the LLM panel's verdicts. Mirrors the judge rubric (blend: generic_ok/coherent/scope;
analogy: valid/coherent). Ratings append to human_review/ratings.jsonl (resumable). The /results page
compares your ratings to the hidden panel majority once you have rated some items.

    .venv/bin/python -m src.kg_creat.scripts.blind_review_server        # then open http://127.0.0.1:8000
"""
import argparse
import html
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DIR = Path("data/kg_creat/kombine_test30/human_review")
ITEMS = json.loads((DIR / "items.json").read_text())
KEY = json.loads((DIR / "key.json").read_text())
RATINGS = DIR / "ratings.jsonl"
BY_ID = {it["id"]: it for it in ITEMS}

CSS = """
:root{--bg:#faf9f7;--fg:#222;--mut:#777;--card:#fff;--line:#e5e2dc;--acc:#3f6f8f;--u:#A8476A;--v:#2F7D6E;--em:#7B76C4;--inv:#9a7d2e}
*{box-sizing:border-box}body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);color:var(--fg);line-height:1.5}
.wrap{max-width:760px;margin:0 auto;padding:26px 20px 80px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;margin:8px 0 22px}.bar>i{display:block;height:100%;background:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:22px 24px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
h1{font-size:20px;margin:0 0 4px}.sub{color:var(--mut);font-size:13px;margin:0 0 18px}
.tasktag{display:inline-block;font-size:12px;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--acc);margin-bottom:10px}
.anchors{font-size:19px;font-weight:600;margin:2px 0 16px}.u{color:var(--u)}.v{color:var(--v)}.inv{color:var(--inv);font-weight:700}.em{color:var(--em)}
.lbl{font-size:12px;font-weight:700;letter-spacing:.03em;color:var(--mut);text-transform:uppercase;margin:16px 0 6px}
.trip{font-family:'SF Mono',ui-monospace,Menlo,monospace;font-size:13.5px;margin:3px 0;padding:4px 8px;background:#f4f2ee;border-radius:6px;display:inline-block}
.map{font-family:'SF Mono',ui-monospace,Menlo,monospace;font-size:13px;margin:4px 0}
.gs{font-style:italic;font-size:15px;background:#f4f2ee;padding:8px 12px;border-radius:8px}
.q{margin:20px 0 6px;font-weight:600}.help{color:var(--mut);font-size:13px;margin:0 0 8px}
.opts label{display:inline-block;margin-right:14px;padding:6px 14px;border:1px solid var(--line);border-radius:20px;cursor:pointer;font-size:14px;background:#fff}
.opts input{margin-right:6px}
textarea{width:100%;border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit;margin-top:6px}
.btn{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:11px 22px;font-size:15px;font-weight:600;cursor:pointer;margin-top:22px}
.btn:hover{opacity:.92}a{color:var(--acc)}
table{border-collapse:collapse;width:100%;margin-top:10px;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left}
.ok{color:#2e7d32;font-weight:600}.no{color:#c0392b;font-weight:600}
"""


def esc(s):
    return html.escape(str(s))


def trip(t):
    return f"({esc(t[0])}, <b>{esc(t[1])}</b>, {esc(t[2])})" if len(t) == 3 else esc(t)


def rated_ids():
    if not RATINGS.exists():
        return set()
    return {json.loads(l)["id"] for l in RATINGS.read_text().splitlines() if l.strip()}


def render_item(it):
    u, v = f'<span class="u">{esc(it["u"])}</span>', f'<span class="v">{esc(it["v"])}</span>'
    inv = f'<span class="inv">{esc(it["invention"])}</span>'
    if it["task"] == "analogy":
        pa = "".join(f'<div class="trip">{trip(t)}</div> ' for t in it["path_a"])
        pb = "".join(f'<div class="trip">{trip(t)}</div> ' for t in it["path_b"])
        mp = "".join(f'<div class="map">{trip(p["source"])} &rarr; {trip(p["image"])}</div>'
                     for p in it["projection"])
        body = f"""<div class="tasktag">Analogy</div>
        <div class="anchors">{u} &nbsp;::&nbsp; {v}</div>
        <div class="lbl">Structure of {u}</div>{pa}
        <div class="lbl">Structure of {v}</div>{pb}
        <div class="lbl">Projected source concept</div><div class="trip">{esc(it["projected"])}</div>
        <div class="lbl">Invention</div><div class="anchors">{inv}</div>
        <div class="lbl">Projection (source &rarr; image)</div>{mp}"""
        form = """
        <div class="q">1. Integration quality &mdash; was the mapping actually used to invent it?</div>
        <div class="help">Each source triple is factually true of the projected concept, AND each image is that source carried across the alignment (same relation, every entity replaced by its counterpart). Judge only whether the projection was genuinely applied &mdash; not how novel it is.</div>
        <div class="opts"><label><input type="radio" name="valid" value="1" required>Yes</label><label><input type="radio" name="valid" value="0">No</label></div>
        <div class="q">2. Utility &mdash; is the invention feasible and coherent?</div>
        <div class="help">The image triples are each individually plausible and together describe a single usable concept (not a disjointed list). It need not be a real/existing thing.</div>
        <div class="opts"><label><input type="radio" name="coherent" value="1" required>Yes</label><label><input type="radio" name="coherent" value="0">No</label></div>"""
    else:
        rows = ""
        tagcol = {"u": ("u", it["u"]), "v": ("v", it["v"]), "emergent": ("em", "emergent")}
        for s in it["structure"]:
            cls, _ = tagcol.get(s["tag"], ("", s["tag"]))
            rows += f'<div class="trip">{trip(s["triple"])} <span class="{cls}">[{esc(s["tag"])}]</span></div> '
        body = f"""<div class="tasktag">Blend</div>
        <div class="anchors">{u} &nbsp;+&nbsp; {v}</div>
        <div class="lbl">Generic space (claimed shared schema)</div><div class="gs">{esc(it["generic_space"])}</div>
        <div class="lbl">Invention</div><div class="anchors">{inv}</div>
        <div class="lbl">Blended space (each triple tagged u / v / emergent)</div>{rows}"""
        form = """
        <div class="q">1. Generic space &mdash; is it a real schema BOTH inputs instantiate?</div>
        <div class="help">A specific schema that both inputs genuinely instantiate &mdash; not vacuous ("both exist", "both involve change") and not a one-from-each conjunction.</div>
        <div class="opts"><label><input type="radio" name="generic_ok" value="1" required>Yes</label><label><input type="radio" name="generic_ok" value="0">No</label></div>
        <div class="q">2. Utility &mdash; are the blended triples coherent as one concept?</div>
        <div class="help">Each triple individually plausible/sensible, and together a single coherent new concept rather than two lists of properties side by side. Need not be real.</div>
        <div class="opts"><label><input type="radio" name="coherent" value="1" required>Yes</label><label><input type="radio" name="coherent" value="0">No</label></div>
        <div class="q">3. Integration scope</div>
        <div class="help">1 = single-scope (only ONE input contributes organizing structure); 2 = double-scope (BOTH do); 3 = double-scope emergent (both, AND a genuine emergent triple true of neither input alone).</div>
        <div class="opts"><label><input type="radio" name="scope" value="1" required>1</label><label><input type="radio" name="scope" value="2">2</label><label><input type="radio" name="scope" value="3">3</label></div>"""
    return body, form


def page(inner):
    return f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><style>{CSS}</style></head><body><div class=wrap>{inner}</div></body></html>"


def index_page():
    done = len(rated_ids())
    inner = f"""<div class="card"><h1>Blind review &mdash; analogy &amp; blend</h1>
    <p class="sub">Rate each artifact on the same dimensions the LLM judges scored. You will not see the model or the judges' verdicts.</p>
    <div class="bar"><i style="width:{100*done//len(ITEMS)}%"></i></div>
    <p>{done} of {len(ITEMS)} rated.</p>
    <a class="btn" href="/next" style="text-decoration:none">{'Continue rating' if done else 'Start rating'}</a>
    &nbsp;&nbsp;<a href="/results">See agreement with the panel &rarr;</a></div>"""
    return page(inner)


def rate_page(idx):
    it = ITEMS[idx]
    body, form = render_item(it)
    done = len(rated_ids())
    inner = f"""<div class="bar"><i style="width:{100*done//len(ITEMS)}%"></i></div>
    <p class="sub">{done+1} of {len(ITEMS)} &nbsp;&middot;&nbsp; <a href="/">home</a></p>
    <div class="card">{body}
    <form method="POST" action="/submit">
    <input type="hidden" name="id" value="{esc(it['id'])}">
    <input type="hidden" name="idx" value="{idx}">
    {form}
    <div class="q" style="font-weight:500">Notes (optional)</div>
    <textarea name="notes" rows="2"></textarea>
    <br><button class="btn" type="submit">Submit &amp; next</button></form></div>"""
    return page(inner)


def results_page():
    rated = {}
    if RATINGS.exists():
        for l in RATINGS.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                rated[r["id"]] = r["ratings"]
    dims = {"analogy": ["valid", "coherent"], "blending": ["generic_ok", "coherent", "scope"]}
    agree = {}
    for rid, rr in rated.items():
        panel = KEY[rid]["panel"]
        task = BY_ID[rid]["task"]
        for d in dims[task]:
            hv, pv = rr.get(d), panel.get(d)
            if hv is None or pv is None:
                continue
            k = f"{task}.{d}"
            a = agree.setdefault(k, [0, 0])
            a[1] += 1
            a[0] += int(int(hv) == int(pv))
    rows = "".join(f"<tr><td>{esc(k)}</td><td>{a[0]}/{a[1]}</td><td>{100*a[0]/a[1]:.0f}%</td></tr>"
                   for k, a in sorted(agree.items()))
    tot = [sum(a[0] for a in agree.values()), sum(a[1] for a in agree.values())]
    pct = f"{100*tot[0]/tot[1]:.0f}%" if tot[1] else "n/a"
    inner = f"""<div class="card"><h1>Your agreement with the LLM panel</h1>
    <p class="sub">Exact-match rate between your rating and the 3-judge panel majority, on the {len(rated)} items you have rated so far.</p>
    <table><tr><th>Dimension</th><th>Match</th><th>Agreement</th></tr>{rows}
    <tr><td><b>Overall</b></td><td><b>{tot[0]}/{tot[1]}</b></td><td><b>{pct}</b></td></tr></table>
    <p style="margin-top:18px"><a href="/">&larr; back</a> &nbsp; Raw ratings: <code>{esc(str(RATINGS))}</code></p></div>"""
    return page(inner)


class H(BaseHTTPRequestHandler):
    def _send(self, body, code=200, ctype="text/html"):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/":
            self._send(index_page())
        elif p.path == "/next":
            done = rated_ids()
            nxt = next((i for i, it in enumerate(ITEMS) if it["id"] not in done), None)
            if nxt is None:
                self._send(page('<div class="card"><h1>All done &mdash; thank you!</h1><p><a href="/results">See your agreement with the panel &rarr;</a></p></div>'))
            else:
                self._send(rate_page(nxt))
        elif p.path == "/rate":
            i = int(parse_qs(p.query).get("i", ["0"])[0])
            self._send(rate_page(max(0, min(i, len(ITEMS) - 1))))
        elif p.path == "/results":
            self._send(results_page())
        else:
            self._send(page('<div class="card"><a href="/">home</a></div>'), 404)

    def do_POST(self):
        if urlparse(self.path).path != "/submit":
            self._send("no", 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(n).decode())
        g = lambda k: form.get(k, [None])[0]
        rid, task = g("id"), BY_ID.get(g("id"), {}).get("task")
        ratings = {}
        for d in (["valid", "coherent"] if task == "analogy" else ["generic_ok", "coherent", "scope"]):
            val = g(d)
            ratings[d] = int(val) if val is not None else None
        rec = {"id": rid, "task": task, "ratings": ratings, "notes": g("notes") or "", "ts": time.time()}
        with RATINGS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        self.send_response(303)
        self.send_header("Location", "/next")
        self.end_headers()


def main(port):
    print(f"Blind review: {len(ITEMS)} items. Open  http://127.0.0.1:{port}")
    print(f"Ratings saved to {RATINGS}  (resumable; delete to restart)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    main(ap.parse_args().port)
