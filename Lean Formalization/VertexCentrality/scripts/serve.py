"""Read-only loopback dashboard. Only public dashboard and generated blueprint files are served."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from urllib.parse import unquote, urlsplit
from project import ROOT, snapshot


def asset_path(url):
    path = unquote(urlsplit(url).path)
    if path in ('/', '/index.html'):
        return ROOT/'dashboard/index.html'
    if path == '/blueprint.pdf':
        return ROOT/'blueprint/print/print.pdf'
    for prefix, base in (('/blueprint/', ROOT/'blueprint/web'),):
        if path.startswith(prefix):
            candidate=(base/path[len(prefix):]).resolve()
            if not candidate.is_relative_to(base.resolve()):
                return None
            return candidate/'index.html' if candidate.is_dir() else candidate
    return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if urlsplit(self.path).path == '/api/status':
                data=json.dumps(snapshot()).encode()
                mime='application/json'
            else:
                path=asset_path(self.path)
                if path is None or not path.is_file():
                    self.send_error(404)
                    return
                data=path.read_bytes()
                mime=mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
            self.send_response(200)
            self.send_header('Content-Type',mime+'; charset=utf-8' if mime.startswith('text/') else mime)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers()
            self.wfile.write(data)
        except (ValueError, OSError, RuntimeError) as exc:
            print(f'Dashboard error: {exc}', flush=True)
            self.send_error(500,'Dashboard data could not be loaded')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8766)
    args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Dashboard: http://127.0.0.1:{args.port}/',flush=True)
    server.serve_forever()
