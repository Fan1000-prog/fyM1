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


def test_unknown_notebook_is_refused():
    with pytest.raises(KeyError):
        push.kernel_metadata("someone", "99_nope")


def test_username_is_read_without_the_key(tmp_path):
    cfg = tmp_path / "kaggle.json"
    cfg.write_text(json.dumps({"username": "someone", "key": "fake-key"}))
    assert push.kaggle_username(cfg) == "someone"


def test_build_folder_holds_notebook_and_metadata(tmp_path):
    notebook = tmp_path / "00_baseline.ipynb"
    notebook.write_text("{}")
    folder = push.build_folder(tmp_path / "out", push.kernel_metadata("someone", "00_baseline"), notebook)
    assert sorted(p.name for p in folder.iterdir()) == ["00_baseline.ipynb", "kernel-metadata.json"]
    assert json.loads((folder / "kernel-metadata.json").read_text())["id"] == "someone/fym1-00-baseline"
