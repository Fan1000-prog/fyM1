"""The notebooks run on Colab and Kaggle: their bootstrap cell must match fym1.runtime exactly."""

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ONE_GPU_NOTEBOOKS = ["notebooks/00_baseline.ipynb", "notebooks/01_train.ipynb"]
TWO_GPU_NOTEBOOKS = ["notebooks/04_synth.ipynb"]  # the synthetic-chat generator spreads a 27B model over 2 T4s
NOTEBOOKS = ONE_GPU_NOTEBOOKS + TWO_GPU_NOTEBOOKS
GPU_PINS = [(n, "0") for n in ONE_GPU_NOTEBOOKS] + [(n, "0,1") for n in TWO_GPU_NOTEBOOKS]
_spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "scripts" / "build_notebooks.py")
build_notebooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_notebooks)
COPIED = build_notebooks.COPIED


def runtime_segments() -> dict[str, str]:
    return dict(zip(COPIED, build_notebooks.runtime_segments()))


def code_cells(notebook: str) -> list[str]:
    nb = json.loads((ROOT / notebook).read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def bootstrap_cell(notebook: str) -> str:
    cells = [c for c in code_cells(notebook) if "def hf_token" in c]
    assert len(cells) == 1
    return cells[0]


def test_every_copied_name_still_exists_in_runtime():
    assert set(runtime_segments()) == set(COPIED)


def test_copied_code_needs_no_other_runtime_names():
    """Every module-level name of runtime.py that the copied code uses must itself be copied."""
    tree = ast.parse((ROOT / "src" / "fym1" / "runtime.py").read_text())
    module_names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)} | {
        t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    copied_nodes = [n for n in tree.body if getattr(n, "name", None) in COPIED]
    used = {node.id for fn in copied_nodes for node in ast.walk(fn) if isinstance(node, ast.Name)}
    assert used & module_names <= set(COPIED)


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


@pytest.mark.parametrize("notebook, gpus", GPU_PINS)
def test_gpus_are_pinned_before_torch_and_no_colab_paths(notebook, gpus):
    cells = code_cells(notebook)
    pin = next(i for i, c in enumerate(cells) if f'os.environ["CUDA_VISIBLE_DEVICES"] = "{gpus}"' in c)
    torch = next(i for i, c in enumerate(cells) if "import torch" in c)
    assert pin < torch
    assert "/content" not in "\n".join(cells)


@pytest.mark.parametrize("notebook, gpus", GPU_PINS)
def test_bootstrap_runs_and_overrides_a_preset_gpu_list(notebook, gpus, monkeypatch, tmp_path):
    import sys
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # the bootstrap's mkdtemp lands here, not in /tmp

    for name in ("google", "google.colab", "kaggle_secrets"):
        monkeypatch.setitem(sys.modules, name, None)  # neither Colab nor Kaggle: falls back to the env var
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    monkeypatch.setenv("HF_TOKEN", "env-fake")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1" if gpus == "0" else "1")  # Kaggle may preset "0,1"
    namespace = {}
    exec(bootstrap_cell(notebook), namespace)
    assert namespace["hf_token"]() == "env-fake"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == gpus


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_code_dir_is_fresh_each_run_not_a_fixed_tmp_path(notebook):
    source = "\n".join(code_cells(notebook))
    assert 'CODE_DIR = tempfile.mkdtemp(prefix="fym1code-")' in bootstrap_cell(notebook)
    assert "gettempdir()" not in source and '"fym1code"' not in source
