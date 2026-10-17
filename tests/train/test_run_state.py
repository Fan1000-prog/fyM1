from fym1.train.run_state import config_mismatch, session_progress


def test_config_mismatch_lists_changed_and_missing_keys():
    saved = {"lr": 2e-4, "model": "m", "epochs": 2}
    assert config_mismatch(saved, {"lr": 2e-4, "model": "m", "epochs": 2}) == []
    assert config_mismatch(saved, {"lr": 1e-4, "model": "m", "epochs": 2}) == ["lr: 0.0002 -> 0.0001"]
    assert config_mismatch(saved, {"lr": 2e-4, "model": "m"}) == ["epochs: 2 -> None"]


def test_session_progress_adds_this_session_once():
    assert session_progress(None, 100.4) == {"wall_time_s": 100, "sessions": 1}
    assert session_progress({"wall_time_s": 500, "sessions": 2}, 50) == {"wall_time_s": 550, "sessions": 3}
