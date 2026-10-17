import sys
import types

import pytest

from fym1 import runtime


@pytest.fixture
def no_runtimes(monkeypatch):
    for name in ("google", "google.colab", "kaggle_secrets"):
        monkeypatch.setitem(sys.modules, name, None)  # None makes `import name` raise ImportError
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return monkeypatch


def _fake_colab(monkeypatch, value):
    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=lambda key: {"HF_TOKEN": value}[key])
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def _fake_kaggle(monkeypatch, value):
    secrets = types.ModuleType("kaggle_secrets")
    secrets.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=lambda key: {"HF_TOKEN": value}[key])
    monkeypatch.setitem(sys.modules, "kaggle_secrets", secrets)


def test_colab_secret_wins(no_runtimes):
    _fake_colab(no_runtimes, "colab-fake")
    _fake_kaggle(no_runtimes, "kaggle-fake")
    assert runtime.hf_token() == "colab-fake"


def test_kaggle_secret_when_not_on_colab(no_runtimes):
    _fake_kaggle(no_runtimes, "kaggle-fake")
    assert runtime.hf_token() == "kaggle-fake"


def test_environment_variable_last(no_runtimes):
    no_runtimes.setenv("HF_TOKEN", "env-fake")
    assert runtime.hf_token() == "env-fake"


def test_no_token_anywhere_fails_without_echoing_anything(no_runtimes):
    with pytest.raises(RuntimeError, match="no HF_TOKEN"):
        runtime.hf_token()


def test_work_dir_prefers_kaggle_working(tmp_path, monkeypatch):
    kaggle_working = tmp_path / "kaggle" / "working"
    monkeypatch.setattr(runtime, "KAGGLE_WORKING", str(kaggle_working))
    assert runtime.work_dir() == "/content"
    kaggle_working.mkdir(parents=True)
    assert runtime.work_dir() == str(kaggle_working)
