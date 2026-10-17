"""One-shot token server: only the exact path and Host get the (fake) token, and only once."""

import http.client
import importlib.util
import threading
from pathlib import Path
from urllib.parse import urlsplit

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "serve_hf_token_once.py"
FAKE_TOKEN = b"not-a-real-token"


def _load_script():
    spec = importlib.util.spec_from_file_location("serve_hf_token_once", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def served():
    server, url = _load_script().make_server(FAKE_TOKEN, port=0)  # port 0: any free port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server, urlsplit(url)
    server.shutdown()
    server.server_close()


def _get(url, path=None, host=None):
    conn = http.client.HTTPConnection(url.hostname, url.port, timeout=5)
    conn.request("GET", path or url.path, headers={"Host": host or url.netloc})
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, body


def test_url_has_random_path_not_token(served):
    _, url = served
    assert len(url.path) > 10
    assert FAKE_TOKEN.decode() not in url.geturl()


def test_wrong_path_gets_404(served):
    _, url = served
    assert _get(url, path="/wrong") == (404, b"")


def test_wrong_host_gets_404(served):
    _, url = served
    assert _get(url, host=f"evil.example:{url.port}") == (404, b"")


def test_serves_token_only_once(served):
    _, url = served
    assert _get(url) == (200, FAKE_TOKEN)
    assert _get(url) == (404, b"")


def test_failed_attempts_do_not_use_up_the_token(served):
    _, url = served
    _get(url, path="/wrong")
    _get(url, host="evil.example")
    assert _get(url) == (200, FAKE_TOKEN)
