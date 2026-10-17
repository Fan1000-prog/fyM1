import sys
import types

import pytest

from fym1 import runtime


@pytest.fixture
def no_runtimes(monkeypatch):
    for name in ("google", "google.colab", "kaggle_secrets"):
        monkeypatch.setitem(sys.modules, name, None)  # None makes `import name` raise ImportError
    for var in ("HF_TOKEN", "KAGGLE_KERNEL_RUN_TYPE"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _fake_colab(monkeypatch, value):
    def get(key):
        if value is None:
            raise LookupError("secret not found")
        return {"HF_TOKEN": value}[key]

    colab = types.ModuleType("google.colab")
    colab.userdata = types.SimpleNamespace(get=get)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def _fake_kaggle(monkeypatch, value):
    def get_secret(key):
        if value is None:
            raise RuntimeError("not attached")
        return {"HF_TOKEN": value}[key]

    secrets = types.ModuleType("kaggle_secrets")
    secrets.UserSecretsClient = lambda: types.SimpleNamespace(get_secret=get_secret)
    monkeypatch.setitem(sys.modules, "kaggle_secrets", secrets)


def test_kaggle_wins_when_colab_also_imports(no_runtimes):
    _fake_colab(no_runtimes, "colab-fake")
    _fake_kaggle(no_runtimes, "kaggle-fake")
    assert runtime.hf_token() == "kaggle-fake"


def test_kaggle_env_var_alone_marks_kaggle(no_runtimes):
    no_runtimes.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    _fake_kaggle(no_runtimes, "kaggle-fake")
    assert runtime.on_kaggle()
    assert runtime.hf_token() == "kaggle-fake"


def test_colab_secret_when_not_on_kaggle(no_runtimes):
    _fake_colab(no_runtimes, "colab-fake")
    assert runtime.hf_token() == "colab-fake"


def test_environment_variable_elsewhere(no_runtimes):
    no_runtimes.setenv("HF_TOKEN", "env-fake")
    assert runtime.hf_token() == "env-fake"


@pytest.mark.parametrize("runtime_setup", ["kaggle", "colab", "none"])
def test_missing_or_empty_token_raises_one_hint(no_runtimes, runtime_setup):
    if runtime_setup == "kaggle":
        _fake_kaggle(no_runtimes, None)
    elif runtime_setup == "colab":
        _fake_colab(no_runtimes, "")
    with pytest.raises(RuntimeError, match="Add-ons → Secrets") as err:
        runtime.hf_token()
    assert "not attached" not in str(err.value) and "secret not found" not in str(err.value)


def test_dirs_on_colab(no_runtimes):
    assert runtime.output_dir() == "/content" and runtime.scratch_dir() == "/content"


def test_dirs_on_kaggle_keep_big_files_out_of_saved_output(no_runtimes, tmp_path):
    no_runtimes.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    no_runtimes.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    assert runtime.output_dir() == "/kaggle/working"
    assert runtime.scratch_dir() == str(tmp_path / "fym1") and (tmp_path / "fym1").is_dir()
