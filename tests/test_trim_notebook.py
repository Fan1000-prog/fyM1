"""notebooks/03_trim.ipynb builds the keep set from public text only (security finding #13)."""

import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "03_trim.ipynb"
CODE = "\n".join("".join(c["source"]) for c in json.loads(NOTEBOOK.read_text("utf-8"))["cells"] if c["cell_type"] == "code")


def test_keep_set_comes_from_the_public_builder():
    assert "public_keep_ids(tokenizer, DEFAULT_SOURCES" in CODE
    # The lower-level builders take any text; only fym1.shrink.public_keep may call them.
    assert not re.search(r"\b(tokens_to_keep|chat_texts)\b", CODE)


def test_chat_lines_are_only_held_out():
    calls = re.findall(r"public_keep_ids\(([^)]*)\)", CODE)
    assert calls and not any("chat" in args or "held_out" in args for args in calls)


def test_keep_ids_stay_private():
    uploads = re.findall(r"path_in_repo=['\"]([^'\"]*keep_ids[^'\"]*)['\"]", CODE)
    assert uploads == ["private/keep_ids.json"]
