import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lora_budget", REPO_ROOT / "scripts" / "lora_budget.py")
lora_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lora_budget)


def test_budget_counts_one_up_projection_and_its_adapter():
    out = lora_budget.budget({"text_config": {"hidden_size": 8, "intermediate_size": 24, "num_hidden_layers": 2}}, 4, 8)
    assert out["example"] == {"matrix": "mlp.up_proj", "d": 24, "k": 8, "full_params": 192, "lora_params": 128,
                              "lora_share": 128 / 192}
    assert out["scale"] == 2


def test_budget_reads_a_flat_config_too():
    out = lora_budget.budget({"hidden_size": 2, "intermediate_size": 3, "num_hidden_layers": 1}, 1, 1)
    assert out["example"]["lora_params"] == 5


def test_budget_refuses_a_zero_rank():
    with pytest.raises(ValueError):
        lora_budget.budget({"hidden_size": 2, "intermediate_size": 3, "num_hidden_layers": 1}, 0, 1)


def test_training_lora_needs_exactly_one_settings_line():
    assert lora_budget.training_lora("x\nLORA_R, LORA_ALPHA = 16, 32\n") == (16, 32)
    with pytest.raises(ValueError):
        lora_budget.training_lora("no settings here")


def test_recorded_budget_matches_the_training_settings():
    """results/lora_budget.json must use the rank and alpha 01_train trains with."""
    recorded = json.loads((REPO_ROOT / "results" / "lora_budget.json").read_text())
    r, alpha = lora_budget.training_lora(lora_budget.BUILDER.read_text(encoding="utf-8"))
    assert (recorded["lora_r"], recorded["lora_alpha"]) == (r, alpha)
    ex = recorded["example"]
    assert ex["full_params"] == ex["d"] * ex["k"] and ex["lora_params"] == r * (ex["d"] + ex["k"])
