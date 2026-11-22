"""Progress page for every K.run_pool experiment (reads OUT/_progress/*.json). Stdlib only.

python3 progress_server.py [port]   -> http://localhost:8790
"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROG = Path(__file__).resolve().parents[2] / 'outputs' / 'pilots_v4_20261103' / '_progress'
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8790


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def runs():
    out = []
    for f in PROG.glob('*.json'):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if r['finished']:
            r['state'] = 'done' if r['done'] >= r['of'] else 'stopped'
        else:
            r['state'] = 'running' if alive(r['pid']) else 'stopped'
        if r['state'] != 'running' and time.time() - r['updated'] > 48 * 3600:
            continue
        out.append(r)
    out.sort(key=lambda r: (r['state'] != 'running', -r['updated']))
    return out[:40]


PAGE = r'''<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>实验进度</title><style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1f;--mut:#6e6e73;--line:#e5e5ea;--run:#2f6fde;--ok:#2e9b57;--bad:#c9473b;--track:#ececf0}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--card:#1f1f22;--fg:#f2f2f4;--mut:#9a9aa1;--line:#2c2c30;--run:#5b8cff;--ok:#4cc27a;--bad:#ef6a5e;--track:#2c2c31}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}
main{max-width:860px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--mut);font-size:12px;margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:10px}
.top{display:flex;justify-content:space-between;gap:12px;align-items:baseline}
.lab{font-family:ui-monospace,Menlo,monospace;font-size:13px;word-break:break-all}
.pct{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}
.bar{height:8px;background:var(--track);border-radius:4px;margin:8px 0 6px;overflow:hidden}
.bar>i{display:block;height:100%;border-radius:4px;background:var(--run);transition:width .4s}
.done .bar>i{background:var(--ok)}.stopped .bar>i{background:var(--bad)}
.meta{color:var(--mut);font-size:12px;display:flex;flex-wrap:wrap;gap:4px 14px;font-variant-numeric:tabular-nums}
.tag{font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid currentColor;margin-right:6px}
.running .tag{color:var(--run)}.done .tag{color:var(--ok)}.stopped .tag{color:var(--bad)}
.empty{color:var(--mut);text-align:center;padding:40px 0}
</style></head><body><main><h1>实验进度</h1><div class="sub" id="sub">加载中…</div><div id="list"></div></main>
<script>
const S={running:'运行中',done:'已完成',stopped:'已停止'};
const dur=s=>{s=Math.max(0,Math.round(s));const h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}时${m}分`:m?`${m}分${s%60}秒`:`${s}秒`};
const esc=t=>String(t).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function tick(){
  try{
    const rs=await (await fetch('api',{cache:'no-store'})).json(), now=Date.now()/1000;
    document.getElementById('sub').textContent=`${rs.filter(r=>r.state==='running').length} 个运行中 · 每 3 秒刷新 · ${new Date().toLocaleTimeString('zh-CN')}`;
    document.getElementById('list').innerHTML=rs.length?rs.map(r=>{
      const p=r.of?r.done/r.of:1, el=(r.finished||now)-r.started, rate=r.done/Math.max(el,1);
      const eta=r.state==='running'&&rate>0?dur((r.of-r.done)/rate):'—';
      const st=Object.entries(r.statuses).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`${esc(k)} ${v}`).join(' · ');
      return `<div class="card ${r.state}"><div class="top"><div class="lab"><span class="tag">${S[r.state]}</span>${esc(r.label)}</div>
      <div class="pct">${(p*100).toFixed(1)}%</div></div><div class="bar"><i style="width:${(p*100).toFixed(2)}%"></i></div>
      <div class="meta"><span>${r.done} / ${r.of}</span><span>并发 ${r.workers}</span><span>已用 ${dur(el)}</span><span>剩余约 ${eta}</span><span>${st}</span></div></div>`}).join('')
      :'<div class="empty">暂无实验</div>';
  }catch(e){document.getElementById('sub').textContent='无法连接进度服务'}
}
tick();setInterval(tick,3000);
</script></body></html>'''


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api'):
            body, ctype = json.dumps(runs()).encode(), 'application/json'
        else:
            body, ctype = PAGE.encode(), 'text/html; charset=utf-8'
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
