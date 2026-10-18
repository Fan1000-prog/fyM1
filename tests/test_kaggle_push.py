import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_push.py"
spec = importlib.util.spec_from_file_location("kaggle_push", SCRIPT)
push = importlib.util.module_from_spec(spec)
spec.loader.exec_module(push)


def test_metadata_is_private_gpu_t4_with_internet():
    meta = push.kernel_metadata("someone", "00_baseline")
    assert meta["id"] == "someone/fym1-00-baseline"
    assert meta["code_file"] == "00_baseline.ipynb"
    assert meta["kernel_type"] == "notebook" and meta["language"] == "python"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"


def test_04_synth_is_a_private_gpu_t4_kernel_with_internet():
    meta = push.kernel_metadata("croixrousse69", "04_synth")
    assert meta["id"] == "croixrousse69/fym1-04-synth"
    assert meta["code_file"] == "04_synth.ipynb"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["machine_shape"] == "NvidiaTeslaT4"


def test_every_kaggle_notebook_exists():
    for notebook in push.NOTEBOOKS:
        assert (push.REPO_ROOT / "notebooks" / f"{notebook}.ipynb").is_file()


def test_unknown_notebook_is_refused():
    with pytest.raises(KeyError):
        push.kernel_metadata("someone", "99_nope")


def test_build_folder_holds_notebook_and_metadata(tmp_path):
    notebook = tmp_path / "00_baseline.ipynb"
    notebook.write_text("{}")
    folder = push.build_folder(tmp_path / "out", push.kernel_metadata("someone", "00_baseline"), notebook)
    assert sorted(p.name for p in folder.iterdir()) == ["00_baseline.ipynb", "kernel-metadata.json"]
    assert json.loads((folder / "kernel-metadata.json").read_text())["id"] == "someone/fym1-00-baseline"


def test_main_pushes_one_folder_with_notebook_and_metadata_only(tmp_path, monkeypatch):
    (tmp_path / "notebooks").mkdir()
    (tmp_path / "notebooks" / "01_train.ipynb").write_text('{"cells": []}')
    monkeypatch.setattr(push, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "fake-kaggle-key")  # must never end up in the pushed folder
    seen = {}

    def fake_run(cmd, check):
        folder = Path(cmd[-1])
        seen["cmd"] = cmd
        seen["files"] = {p.name: p.read_text() for p in folder.iterdir()}
        return None

    monkeypatch.setattr(push.subprocess, "run", fake_run)
    push.main(["01_train", "--user", "someone"])
    assert seen["cmd"][:4] == ["uvx", "--from", "kaggle==2.2.4", "kaggle"]
    assert seen["cmd"][4:7] == ["kernels", "push", "-p"]
    assert sorted(seen["files"]) == ["01_train.ipynb", "kernel-metadata.json"]
    assert json.loads(seen["files"]["kernel-metadata.json"])["id"] == "someone/fym1-01-train"
    assert not any("fake-kaggle-key" in text for text in seen["files"].values())


def test_main_needs_a_username(monkeypatch):
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(SystemExit):
        push.main(["00_baseline"])


def test_main_refuses_unknown_notebook():
    with pytest.raises(SystemExit):
        push.main(["99_nope", "--user", "someone"])
