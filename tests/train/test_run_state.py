from fym1.train.run_state import config_mismatch, session_progress


def test_config_mismatch_lists_changed_and_missing_keys():
    saved = {"lr": 2e-4, "model": "m", "epochs": 2}
    assert config_mismatch(saved, {"lr": 2e-4, "model": "m", "epochs": 2}) == []
    assert config_mismatch(saved, {"lr": 1e-4, "model": "m", "epochs": 2}) == ["lr: 0.0002 -> 0.0001"]
    assert config_mismatch(saved, {"lr": 2e-4, "model": "m"}) == ["epochs: 2 -> None"]


def test_session_progress_adds_this_session_once():
    assert session_progress(None, 100.4) == {"wall_time_s": 100, "sessions": 1}
    assert session_progress({"wall_time_s": 500, "sessions": 2}, 50) == {"wall_time_s": 550, "sessions": 3}


def test_resume_on_different_code_is_refused():
    saved = {"lr": 2e-4, "code_sha": "abc1234"}
    assert config_mismatch(saved, {"lr": 2e-4, "code_sha": "def5678"}) == ["code_sha: abc1234 -> def5678"]


def test_train_notebook_puts_code_sha_in_the_checked_run_config():
    import json
    from pathlib import Path

    nb = json.loads((Path(__file__).resolve().parents[2] / "notebooks" / "01_train.ipynb").read_text())
    source = "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    run_config = source[source.index("RUN_CONFIG = {"):source.index("}", source.index("RUN_CONFIG = {"))]
    assert '"code_sha": CODE_SHA' in run_config
    assert "config_mismatch(saved_config, RUN_CONFIG)" in source
