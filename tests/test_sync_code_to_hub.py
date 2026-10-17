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
