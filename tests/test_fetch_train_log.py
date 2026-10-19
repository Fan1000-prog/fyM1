"""fetch_train_log copies counts and losses only, in the shape the paper scripts read. Fake hub, made-up values."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)  # dataclasses look the module up while building
    spec.loader.exec_module(module)
    return module


fetch_train_log = load("fetch_train_log", ROOT / "scripts" / "fetch_train_log.py")
make_figures = load("make_figures_for_fetch", ROOT / "paper" / "make_figures.py")
make_numbers = load("make_numbers_for_fetch", ROOT / "paper" / "make_numbers.py")

RUN = "run-x"
LOG = {
    "run_id": RUN, "model": "org/base-9b", "loaded_from": "org/base-9b", "code_sha": "abc1234", "gpu": "NVIDIA A100",
    "steps": 4, "epochs": 2, "wall_time_s": 1000, "sessions": 1, "final_train_loss": 1.1, "best_eval_loss": 1.3,
    "best_step": 2, "final_adapter": f"{RUN}/checkpoint-2", "stopped_early": False,
    "data": {"files": ["chats.jsonl"], "revision": "deadbeef", "examples": 40, "train": 39, "eval": 1,
             "tokens": 5000, "over_max_len": 0},
    "config": {"lora_r": 16, "lora_alpha": 16, "lr": 2e-4, "lr_scheduler": "cosine", "warmup_steps": 1,
               "max_seq_len": 2048, "batch": 8, "grad_accum": 1, "effective_batch": 8,
               "hardware": {"gpu": "NVIDIA A100", "compute_dtype": "torch.bfloat16"}, "data_revision": "deadbeef",
               "chat_template_kwargs": {"enable_thinking": False}},
    "loss_curve": [{"step": 1, "loss": 1.5}, {"step": 2, "loss": 1.2}],
    "eval_curve": [{"step": 2, "eval_loss": 1.3}],
}
STATE = {"global_step": 4, "max_steps": 10, "epoch": 0.8, "num_train_epochs": 2,
         "log_history": [{"step": 1, "loss": 1.5}]}


@pytest.fixture
def hub(monkeypatch):
    files = {f"{RUN}/train_log.json": LOG, f"{RUN}/checkpoint-2/trainer_state.json": {"global_step": 2},
             f"{RUN}/checkpoint-4/trainer_state.json": STATE, "checkpoint-9/trainer_state.json": {}}
    fetched = []

    def fake_fetch(path):
        fetched.append(path)
        return json.loads(json.dumps(files[path]))

    monkeypatch.setattr(fetch_train_log, "fetch_json", fake_fetch)
    monkeypatch.setattr(fetch_train_log, "list_files", lambda: list(files))
    return files, fetched


def run(tmp_path, *extra) -> tuple[int, Path]:
    out = tmp_path / "results" / "train_log.json"
    return fetch_train_log.main(["--run-id", RUN, "--out", str(out), *extra]), out


def test_writes_counts_losses_and_sec_per_step_from_the_newest_checkpoint(hub, tmp_path):
    _, fetched = hub
    code, out = run(tmp_path)
    assert code == 0
    assert fetched == [f"{RUN}/train_log.json", f"{RUN}/checkpoint-4/trainer_state.json"]
    result = json.loads(out.read_text())
    assert result["sec_per_step"] == 250.0
    assert result["best_step"] == 2 and result["best_eval_loss"] == 1.3
    assert result["loss_curve"] == LOG["loss_curve"] and result["eval_curve"] == LOG["eval_curve"]
    assert result["checkpoint"] == {"name": "checkpoint-4", "global_step": 4, "max_steps": 10, "epoch": 0.8,
                                    "num_train_epochs": 2}
    assert result["config"]["compute_dtype"] == "torch.bfloat16"
    assert result["data"] == {"examples": 40, "train": 39, "eval": 1, "tokens": 5000, "over_max_len": 0}


def test_leaves_out_file_names_revisions_and_paths(hub, tmp_path):
    _, out = run(tmp_path)
    text = out.read_text()
    for private in ("chats.jsonl", "deadbeef", "abc1234", "checkpoint-2", "log_history", "enable_thinking"):
        assert private not in text, private


def test_paper_scripts_read_every_train_log_number(hub, tmp_path):
    _, out = run(tmp_path)
    result = json.loads(out.read_text())
    for number in make_numbers.NUMBERS:
        if number.file == "train_log.json":
            assert isinstance(number.key, str), number.key
            value = result
            for part in number.key.split("."):
                value = value[part]
            assert value is not None, number.key
    figures = make_figures.render(out.parent)
    assert figures["loss.csv"] == "step,loss\n1,1.5\n2,1.2\n"
    assert "\\fyFigTrainReady}{1}" in figures["flags.tex"]


def test_refuses_a_log_without_steps_and_writes_nothing(hub, tmp_path, capsys):
    files, _ = hub
    files[f"{RUN}/train_log.json"] = {**LOG, "steps": 0}
    code, out = run(tmp_path)
    assert code == 1 and not out.exists()
    assert "refused" in capsys.readouterr().err


def test_fails_when_the_run_has_no_checkpoint(hub, tmp_path, capsys):
    code, out = fetch_train_log.main(["--run-id", "other", "--out", str(tmp_path / "t.json")]), tmp_path / "t.json"
    assert code == 1 and not out.exists()
    assert "no other/checkpoint-*" in capsys.readouterr().err
