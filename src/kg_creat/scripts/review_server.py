"""Local web UI for the blind judge-reliability review — auto-logs every verdict.

Serves the review items (from ``<scores_dir>/human_review/`` built by sample_review.py) in the
browser; each click POSTs to /log and appends to ``responses.jsonl`` immediately (resumable). No
dependencies (stdlib http.server). The judge key is never sent to the browser, so the review is blind.

    .venv_mlx/bin/python src/kg_creat/scripts/review_server.py data/kg_creat/scores_analogy_v2
    # then open http://localhost:8111  ;  when done: score_review.py <scores_dir>
"""

import csv
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8111


def load_items(review_dir):
    items = []
    with open(review_dir / "review_factuality.csv") as f:
        for row in list(csv.reader(f))[1:]:
            items.append({"id": row[0], "type": "factuality", "context": row[1], "focus": row[2]})
    with open(review_dir / "review_analogy.csv") as f:
        for row in list(csv.reader(f))[1:]:
            items.append({"id": row[0], "type": "analogy", "a": row[1], "b": row[2],
                          "sa": row[3], "sb": row[4]})
    return items


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Judge reliability review</title>
<style>
 body{font:15px/1.5 -apple-system,system-ui,sans-serif;max-width:900px;margin:0 auto;padding:20px;color:#1f2933}
 h1{font-size:20px} .sub{color:#66727f}
 #bar{position:sticky;top:0;background:#fff;padding:10px 0;border-bottom:1px solid #e3e8ee;z-index:9}
 .card{border:1px solid #e3e8ee;border-radius:10px;padding:14px 16px;margin:12px 0}
 .ctx{color:#66727f;font-size:13px;margin-bottom:6px}
 .focus{font-weight:600;font-family:ui-monospace,Menlo,monospace;font-size:14px}
 .struct{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;margin:3px 0}
 .lab{color:#66727f;font-size:12px;margin-right:6px}
 button{font:14px/1 inherit;padding:8px 16px;margin-right:8px;border-radius:8px;border:1.5px solid #cbd5e1;background:#fff;cursor:pointer}
 button.sel{color:#fff;border-color:transparent}
 button.good.sel{background:#388C3C} button.bad.sel{background:#CE5E5E}
 .done{opacity:.55}
</style></head><body>
<div id="bar"><h1>Blind judge-reliability review</h1>
<div class="sub" id="prog">loading…</div></div>
<div id="list"></div>
<script>
let items=[], answered={};
async function load(){
 items=await (await fetch('/items')).json();
 answered=await (await fetch('/responses')).json();
 render();
}
function prog(){document.getElementById('prog').textContent=
  Object.keys(answered).length+' / '+items.length+' answered — answers save automatically as you click';}
function pick(id,v,el){
 answered[id]=v;
 fetch('/log',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,v})});
 [...el.parentNode.querySelectorAll('button')].forEach(b=>b.classList.remove('sel'));
 el.classList.add('sel'); el.closest('.card').classList.add('done'); prog();
}
function btns(id,opts){
 return opts.map(o=>`<button class="${o.cls} ${answered[id]===o.v?'sel':''}"
   onclick="pick('${id}','${o.v}',this)">${o.t}</button>`).join('');
}
function render(){
 const L=document.getElementById('list');
 const fact=items.filter(i=>i.type=='factuality'), ana=items.filter(i=>i.type=='analogy');
 L.innerHTML='<h2>Factuality — is the highlighted triple a true/plausible real-world fact?</h2>'+
  fact.map(i=>`<div class="card ${answered[i.id]?'done':''}"><div class="ctx">path: ${i.context}</div>
   <div class="focus">${i.focus}</div><div style="margin-top:10px">
   ${btns(i.id,[{v:'true',t:'True',cls:'good'},{v:'hallucinated',t:'Hallucinated',cls:'bad'}])}</div></div>`).join('')
  +'<h2>Analogy — do the two structures form a genuine analogy (same relations, corresponding roles)?</h2>'+
  ana.map(i=>`<div class="card ${answered[i.id]?'done':''}"><b>${i.a}</b> &nbsp;::&nbsp; <b>${i.b}</b>
   <div class="struct"><span class="lab">A</span>${i.sa}</div>
   <div class="struct"><span class="lab">B</span>${i.sb}</div><div style="margin-top:10px">
   ${btns(i.id,[{v:'valid',t:'Valid',cls:'good'},{v:'invalid',t:'Invalid',cls:'bad'}])}</div></div>`).join('');
 prog();
}
load();
</script></body></html>"""


def make_handler(review_dir, items):
    log_path = review_dir / "responses.jsonl"

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes) else body.encode())

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE, "text/html")
            elif self.path == "/items":
                self._send(200, json.dumps(items))
            elif self.path == "/responses":
                resp = {}
                if log_path.exists():
                    for line in log_path.read_text().splitlines():
                        if line.strip():
                            r = json.loads(line)
                            resp[r["id"]] = r["v"]
                self._send(200, json.dumps(resp))
            else:
                self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            r = json.loads(self.rfile.read(n) or b"{}")
            with open(log_path, "a") as f:
                f.write(json.dumps({"id": r["id"], "v": r["v"], "ts": time.time()}) + "\n")
            self._send(200, "{}")

    return H


def main(scores_dir):
    review_dir = Path(scores_dir) / "human_review"
    if not (review_dir / "review_factuality.csv").exists():
        raise FileNotFoundError(f"No review set at {review_dir} -- run sample_review.py first")
    items = load_items(review_dir)
    server = HTTPServer(("localhost", PORT), make_handler(review_dir, items))
    print(f"Review UI: http://localhost:{PORT}   ({len(items)} items; logging to {review_dir/'responses.jsonl'})")
    print("Open it, click your verdicts (auto-saved), then Ctrl-C and run score_review.py.")
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/kg_creat/scores_analogy_v2")
