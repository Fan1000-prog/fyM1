import importlib.util
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
    (tmp_path / "src" / "pkg" / "notes.txt").write_text("not synced\n")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "prompts.jsonl").write_text("{}\n")
    ops = sync.commit_operations(tmp_path, "abc1234")
    assert [op.path_in_repo for op in ops] == ["code/eval/prompts.jsonl", "code/src/pkg/a.py", "code/SHA"]
    assert ops[-1].path_or_fileobj == b"abc1234"
