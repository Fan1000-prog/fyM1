"""Hand the local HF token to the Playwright process once, over loopback, so it never enters a transcript.

Usage: python scripts/serve_hf_token_once.py <random-path>
Serves ~/.cache/huggingface/token at http://127.0.0.1:8765/<random-path> for one request, then exits
(or after 60 s). Any other path gets 404.
"""

import http.server
import pathlib
import sys
import threading

PORT = 8765
TIMEOUT_S = 60


def main(secret_path: str) -> None:
    token = (pathlib.Path.home() / ".cache/huggingface/token").read_bytes().strip()
    expected = "/" + secret_path

    class OneShot(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != expected:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(token)
            threading.Thread(target=self.server.shutdown).start()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", PORT), OneShot)
    threading.Timer(TIMEOUT_S, server.shutdown).start()
    server.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1])
