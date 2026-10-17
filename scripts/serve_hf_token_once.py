"""Hand the local HF token to the Playwright process once, over loopback, so it never enters a transcript.

Usage: python scripts/serve_hf_token_once.py
Prints the one-time URL (a random path, not the token), then serves ~/.cache/huggingface/token there
for one request and exits, or exits after 60 s. Other paths, other Host headers and any request after
the first successful one get 404.
"""

import http.server
import pathlib
import secrets
import threading
from collections.abc import Callable

HOST, PORT = "127.0.0.1", 8765
TIMEOUT_S = 60


def make_server(
    token: bytes,
    host: str = HOST,
    port: int = PORT,
    on_served: Callable[[http.server.HTTPServer], None] = lambda server: None,
) -> tuple[http.server.HTTPServer, str]:
    """Server that returns `token` once, at a random path; returns the server and its one-time URL."""
    expected_path = "/" + secrets.token_urlsafe(16)

    class OneShot(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            expected_host = f"{host}:{self.server.server_port}"
            if self.server.served or self.path != expected_path or self.headers.get("Host") != expected_host:
                self.send_response(404)
                self.end_headers()
                return
            self.server.served = True
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(token)
            on_served(self.server)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer((host, port), OneShot)
    server.served = False
    return server, f"http://{host}:{server.server_port}{expected_path}"


def main() -> None:
    token = (pathlib.Path.home() / ".cache/huggingface/token").read_bytes().strip()
    server, url = make_server(
        token, on_served=lambda s: threading.Thread(target=s.shutdown, daemon=True).start()
    )
    timer = threading.Timer(TIMEOUT_S, server.shutdown)
    timer.daemon = True
    timer.start()
    print(url, flush=True)
    try:
        server.serve_forever()
    finally:
        timer.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
