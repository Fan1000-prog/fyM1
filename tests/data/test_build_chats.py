import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_chats.py"
spec = importlib.util.spec_from_file_location("build_chats", SCRIPT)
build_chats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_chats)


def test_private_path_refuses_any_git_work_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    with pytest.raises(SystemExit, match="git work tree"):
        build_chats.private_path(tmp_path / "not" / "yet" / "chats.jsonl")


def test_private_path_refuses_this_repo():
    with pytest.raises(SystemExit, match="git work tree"):
        build_chats.private_path(SCRIPT.parent / "chats.jsonl")


def test_private_path_allows_folders_outside_git(tmp_path):
    target = tmp_path / "private" / "chats.jsonl"

    assert build_chats.private_path(target) == target.resolve()
