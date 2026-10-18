"""The notebooks run on Colab and Kaggle: their bootstrap cell must match fym1.runtime exactly."""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ["notebooks/00_baseline.ipynb", "notebooks/01_train.ipynb"]
COPIED = ("MISSING_TOKEN", "on_kaggle", "_secret", "hf_token")


def runtime_segments() -> dict[str, str]:
    source = (ROOT / "src" / "fym1" / "runtime.py").read_text()
    segments = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        if name in COPIED:
            segments[name] = ast.get_source_segment(source, node)
    return segments


def code_cells(notebook: str) -> list[str]:
    nb = json.loads((ROOT / notebook).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def bootstrap_cell(notebook: str) -> str:
    cells = [c for c in code_cells(notebook) if "def hf_token" in c]
    assert len(cells) == 1
    return cells[0]


def test_every_copied_name_still_exists_in_runtime():
    assert set(runtime_segments()) == set(COPIED)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_bootstrap_is_a_verbatim_copy_of_runtime(notebook):
    bootstrap = bootstrap_cell(notebook)
    for name, segment in runtime_segments().items():
        assert segment in bootstrap, f"{notebook}: {name} differs from src/fym1/runtime.py"


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_token_only_through_the_runtime_helper(notebook):
    source = "\n".join(code_cells(notebook))
    assert 'os.environ["HF_TOKEN"] = hf_token()' in source
    assert "userdata.get" not in source.replace("_secret(userdata.get)", "")


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_one_gpu_is_pinned_before_torch_and_no_colab_paths(notebook):
    cells = code_cells(notebook)
    pin = next(i for i, c in enumerate(cells) if 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"' in c)
    torch = next(i for i, c in enumerate(cells) if "import torch" in c)
    assert pin < torch
    assert "/content" not in "\n".join(cells)


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_bootstrap_runs_and_overrides_a_preset_gpu_list(notebook, monkeypatch):
    import sys

    for name in ("google", "google.colab", "kaggle_secrets"):
        monkeypatch.setitem(sys.modules, name, None)  # neither Colab nor Kaggle: falls back to the env var
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "env-fake")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")  # what Kaggle may preset
    namespace = {}
    exec(bootstrap_cell(notebook), namespace)
    assert namespace["hf_token"]() == "env-fake"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_code_dir_is_fresh_each_run_not_a_fixed_tmp_path(notebook):
    source = "\n".join(code_cells(notebook))
    assert 'CODE_DIR = tempfile.mkdtemp(prefix="fym1code-")' in bootstrap_cell(notebook)
    assert "gettempdir()" not in source and '"fym1code"' not in source
