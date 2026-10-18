import importlib.util
from pathlib import Path

import pytest

BUILD_KNOWN_WORDS = Path(__file__).resolve().parents[2] / "scripts" / "build_known_words.py"


@pytest.fixture(scope="session")
def build_known_words():
    """scripts/build_known_words.py as a module; skips when the optional `data` group (wordfreq) is not installed."""
    pytest.importorskip("wordfreq")
    spec = importlib.util.spec_from_file_location("build_known_words", BUILD_KNOWN_WORDS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
