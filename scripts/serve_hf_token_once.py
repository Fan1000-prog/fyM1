"""Hand the local HF token to the Playwright process once, over loopback, so it never enters a transcript.

Usage: python scripts/serve_hf_token_once.py
Prints the one-time URL (a random path, not the token), then serves ~/.cache/huggingface/token there
for one request and exits, or exits after 60 s. Other paths or Host headers get 404.
"""

import http.server
import pathlib
import secrets
import threading

HOST, PORT = "127.0.0.1", 8765
TIMEOUT_S = 60


def main() -> None:
    token = (pathlib.Path.home() / ".cache/huggingface/token").read_bytes().strip()
    expected_path = "/" + secrets.token_urlsafe(16)

    class OneShot(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != expected_path or self.headers.get("Host") != f"{HOST}:{PORT}":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(token)
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer((HOST, PORT), OneShot)
    timer = threading.Timer(TIMEOUT_S, server.shutdown)
    timer.daemon = True
    timer.start()
    print(f"http://{HOST}:{PORT}{expected_path}", flush=True)
    try:
        server.serve_forever()
    finally:
        timer.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
