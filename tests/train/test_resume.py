from fym1.train.resume import latest_checkpoint


def test_latest_checkpoint_picks_highest_step_numerically():
    files = ["checkpoint-50/a", "checkpoint-200/a", "checkpoint-100/a"]
    assert latest_checkpoint("x", files) == "checkpoint-200"


def test_latest_checkpoint_empty_is_none():
    assert latest_checkpoint("x", []) is None


def test_latest_checkpoint_ignores_non_checkpoint_paths():
    files = [
        "README.md",
        "final/adapter_model.safetensors",
        "checkpoint-abc/a",
        "results/checkpoint-900/a",
        "checkpoint-30/trainer_state.json",
    ]
    assert latest_checkpoint("x", files) == "checkpoint-30"


def test_latest_checkpoint_lists_repo_when_files_missing(monkeypatch):
    seen = {}

    def fake_list(repo_id):
        seen["repo_id"] = repo_id
        return ["checkpoint-7/a", "checkpoint-10/a"]

    monkeypatch.setattr("fym1.train.resume.list_repo_files", fake_list)
    assert latest_checkpoint("owner/ckpt") == "checkpoint-10"
    assert seen["repo_id"] == "owner/ckpt"
