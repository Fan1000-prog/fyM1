import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_code_to_hub.py"
spec = importlib.util.spec_from_file_location("sync_code_to_hub", SCRIPT)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_clean_checkout_of_origin_main_is_allowed():
    assert sync.refusal_reason(head="abc", origin_main="abc", porcelain="") is None


def test_other_commit_is_refused():
    assert "origin/main" in sync.refusal_reason(head="abc", origin_main="def", porcelain="")


def test_local_changes_or_untracked_files_are_refused():
    assert "not clean" in sync.refusal_reason(head="abc", origin_main="abc", porcelain=" M src/x.py\n")
    assert "not clean" in sync.refusal_reason(head="abc", origin_main="abc", porcelain="?? notes.txt\n")


def test_commit_operations_include_code_and_sha_in_one_list(tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    (tmp_path / "src" / "pkg" / "words.txt").write_text("synced: package data\n")
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"cache")
    (tmp_path / ".env").write_text("HF_TOKEN=fake\n")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "prompts.jsonl").write_text("{}\n")
    ops = sync.commit_operations(tmp_path, "abc1234")
    assert [op.path_in_repo for op in ops] == ["code/eval/prompts.jsonl", "code/src/pkg/a.py", "code/src/pkg/words.txt",
                                                "code/SHA"]
    assert ops[-1].path_or_fileobj == b"abc1234"


def test_every_package_file_under_src_is_synced():
    """A new data file the package reads (word list, rubric) must reach the GPU, not fail with FileNotFoundError."""
    root = SCRIPT.parents[1]
    tracked = subprocess.run(["git", "ls-files", "src"], cwd=root, check=True, capture_output=True,
                             text=True).stdout.split()
    assert any(not f.endswith(".py") for f in tracked), "expected package data files under src/"
    synced = {op.path_in_repo.removeprefix("code/") for op in sync.commit_operations(root, "abc1234")}
    assert sorted(set(tracked) - synced) == []
    assert not any("__pycache__" in f or f.endswith(".pyc") for f in synced)
