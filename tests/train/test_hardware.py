import json
from pathlib import Path

import pytest

from fym1.train.hardware import EFFECTIVE_BATCH, batch_shape, precision_flags

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("gpu", ["NVIDIA A100-SXM4-40GB", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"])
def test_a100_trains_8_rows_in_one_step(gpu):
    assert batch_shape(gpu) == (8, 1)


@pytest.mark.parametrize("gpu", ["Tesla T4", "NVIDIA L4", ""])
def test_other_gpus_keep_2_x_4(gpu):
    assert batch_shape(gpu) == (2, 4)


@pytest.mark.parametrize("gpu", ["NVIDIA A100-SXM4-40GB", "Tesla T4"])
def test_effective_batch_is_the_same_on_every_gpu(gpu):
    batch, grad_accum = batch_shape(gpu)
    assert batch * grad_accum == EFFECTIVE_BATCH


def test_precision_follows_the_compute_dtype():
    assert precision_flags("torch.bfloat16") == {"fp16": False, "bf16": True}
    assert precision_flags("torch.float16") == {"fp16": True, "bf16": False}
    assert precision_flags("torch.float32") == {"fp16": False, "bf16": False}


def code_source(notebook: str) -> str:
    nb = json.loads((ROOT / "notebooks" / notebook).read_text())
    return "".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


@pytest.mark.parametrize("notebook", ["00_baseline.ipynb", "01_train.ipynb"])
def test_notebooks_take_precision_from_the_dtype_not_a_hard_coded_flag(notebook):
    source = code_source(notebook)
    assert "**precision_flags(str(compute_dtype))" in source
    assert "bf16=False" not in source and "fp16=" not in source


def test_train_notebook_picks_and_records_the_batch_shape_by_gpu():
    source = code_source("01_train.ipynb")
    assert "BATCH, GRAD_ACCUM = batch_shape(GPU)" in source
    assert '"hardware": HARDWARE' in source
    assert 'HARDWARE["compute_dtype"] = str(compute_dtype)' in source
