"""Local review server: static files plus an append-only annotation log.

  GET  /                  index.html
  GET  /data/...          payload built by build.py
  GET  /api/annotations   latest answer per (reviewer, task, item)
  POST /api/annotations   {reviewer, task, item, answer, comment} -> appended to annotations.jsonl

Run:  python serve.py [port]   (binds 127.0.0.1 only)
"""
from __future__ import annotations

import json
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG = HERE / 'annotations.jsonl'


def latest():
    out = {}
    if LOG.exists():
        for line in LOG.read_text().splitlines():
            rec = json.loads(line)
            out[f"{rec['reviewer']}|{rec['task']}|{rec['item']}"] = rec
    return list(out.values())


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/api/annotations'):
            return self._json(latest())
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith('/api/annotations'):
            return self._json({'error': 'not found'}, 404)
        try:
            rec = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            assert all(k in rec for k in ('reviewer', 'task', 'item', 'answer'))
        except Exception:
            return self._json({'error': 'bad request'}, 400)
        rec['saved_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        with LOG.open('a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        return self._json({'ok': True})

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    print(f'review server on http://127.0.0.1:{port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
