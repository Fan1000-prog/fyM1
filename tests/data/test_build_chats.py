import importlib.util
import json
import subprocess
import sys
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


def _flat_export(path: Path) -> Path:
    def msg(i: int, sender: str, text: str, ts: str) -> dict:
        return {"id": i, "sender": sender, "text": text, "timestamp": ts}

    messages = [
        msg(1, "Alpha", "inona no vaovao", "2026-07-01T08:00:00"),
        msg(2, "Owner", "tsy misy", "2026-07-01T08:05:00"),
        msg(3, "Alpha", "aiza ianao", "2026-07-01T14:00:00"),
        msg(4, "Owner", "ao an-trano", "2026-07-01T14:10:00"),
    ]
    conversation = {"title": "Synthetic", "participants": ["Alpha", "Bravo", "Owner"], "timezone": "UTC"}
    path.write_text(json.dumps({"conversation": conversation, "messages": messages}), encoding="utf-8")
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_dry_run_prints_counts_and_writes_nothing(tmp_path):
    export = _flat_export(tmp_path / "group.json")
    stats = tmp_path / "stats.json"

    result = _run("--export", str(export), "--owner", "Owner", "--dry-run", "--stats", str(stats))

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"threads": 1, "messages": 4, "sessions": 2, "conversations": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["group.json"]


def test_dry_run_still_checks_the_owner(tmp_path):
    export = _flat_export(tmp_path / "group.json")

    result = _run("--export", str(export), "--owner", "Nobody", "--dry-run")

    assert result.returncode != 0
    assert "not a participant" in result.stderr


def test_full_run_still_requires_out_and_sheet(tmp_path):
    export = _flat_export(tmp_path / "group.json")

    result = _run("--export", str(export), "--owner", "Owner")

    assert result.returncode != 0
    assert "--out" in result.stderr and "--sheet" in result.stderr
