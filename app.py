"""Serve the GENESIS AI static chat interface on Amvera without ML dependencies."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "80"))


class GenesisHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, fmt, *args):
        print("[GENESIS WEB]", fmt % args, flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), GenesisHandler)
    print(f"GENESIS chat interface listening on 0.0.0.0:{PORT}", flush=True)
    server.serve_forever()
