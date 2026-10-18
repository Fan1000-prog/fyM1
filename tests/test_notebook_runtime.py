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


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_bootstrap_is_a_verbatim_copy_of_runtime(notebook):
    bootstrap = [c for c in code_cells(notebook) if "def hf_token" in c]
    assert len(bootstrap) == 1
    for name, segment in runtime_segments().items():
        assert segment in bootstrap[0], f"{notebook}: {name} differs from src/fym1/runtime.py"


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_token_only_through_the_runtime_helper(notebook):
    source = "\n".join(code_cells(notebook))
    assert 'os.environ["HF_TOKEN"] = hf_token()' in source
    assert "userdata.get" not in source.replace("_secret(userdata.get)", "")


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_one_gpu_is_pinned_before_torch_and_no_colab_paths(notebook):
    cells = code_cells(notebook)
    pin = next(i for i, c in enumerate(cells) if 'os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")' in c)
    torch = next(i for i, c in enumerate(cells) if "import torch" in c)
    assert pin < torch
    assert "/content" not in "\n".join(cells)
