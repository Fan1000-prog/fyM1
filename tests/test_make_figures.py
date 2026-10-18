"""paper/make_figures.py: figure data comes from results/ only, and the committed copies are up to date."""

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("make_figures", ROOT / "paper" / "make_figures.py")
figures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(figures)

TRAIN_LOG = {
    "epochs": 2,
    "data": {"train": 20},
    "config": {"lr": 2e-4, "lr_scheduler": "cosine", "warmup_steps": 2, "batch": 8, "grad_accum": 1},
    "loss_curve": [{"step": 1, "loss": 1.5}, {"step": 2, "loss": 1.25}],
    "eval_curve": [{"step": 2, "eval_loss": 1.4}],
}


def write(results: Path, name: str, data: dict) -> None:
    (results / name).write_text(json.dumps(data))


@pytest.fixture
def results(tmp_path):
    write(tmp_path, "data_gate.json", {"conversations_before_gate": 10, "kept": 7})
    write(tmp_path, "synth_stats.json", {"raw": 100})
    write(tmp_path, "synth_gate.json", {"checked": 40, "kept": 38})
    return tmp_path


def test_funnel_reads_each_count_from_its_file(results):
    assert [count for _, _, count in figures.funnel(results)] == [10, 7, 100, 40, 38]


def test_funnel_leaves_a_missing_count_out_and_is_not_ready(results):
    (results / "synth_stats.json").unlink()
    out = figures.render(results)
    assert "synth-raw" not in out["funnel.csv"]
    assert "\\fyFigFunnelReady}{0}" in out["flags.tex"]


def test_funnel_refuses_a_non_count(results):
    write(results, "synth_gate.json", {"checked": 4.5, "kept": 3})
    with pytest.raises(TypeError):
        figures.funnel(results)


def test_training_figure_is_pending_without_a_log(results):
    out = figures.render(results)
    assert out["loss.csv"] == "step,loss\n" and out["lr.csv"] == "step,lr\n"
    assert "\\fyFigTrainReady}{0}" in out["flags.tex"]


def test_training_figure_uses_the_logged_curves_and_settings(results):
    write(results, "train_log.json", TRAIN_LOG)
    out = figures.render(results)
    assert out["loss.csv"] == "step,loss\n1,1.5\n2,1.25\n"
    assert out["eval.csv"] == "step,eval_loss\n2,1.4\n"
    lr_rows = out["lr.csv"].splitlines()[1:]
    assert len(lr_rows) == figures.planned_steps(20, 8, 1, 2) + 1 == 7
    assert "\\fyFigTrainReady}{1}" in out["flags.tex"]


def test_lr_curve_needs_a_cosine_schedule():
    assert figures.lr_curve({**TRAIN_LOG, "config": {**TRAIN_LOG["config"], "lr_scheduler": "linear"}}) is None


def test_cosine_lr_warms_up_then_falls_to_zero():
    assert figures.cosine_lr(0, 1.0, 10, 110) == 0
    assert figures.cosine_lr(5, 1.0, 10, 110) == 0.5
    assert figures.cosine_lr(10, 1.0, 10, 110) == 1.0
    assert math.isclose(figures.cosine_lr(60, 1.0, 10, 110), 0.5)
    assert math.isclose(figures.cosine_lr(110, 1.0, 10, 110), 0.0, abs_tol=1e-12)


def test_planned_steps_rounds_up_partial_batches_and_accumulation():
    assert figures.planned_steps(17, 8, 1, 2) == 6
    assert figures.planned_steps(17, 2, 4, 1) == 3


def test_mermaid_funnel_says_pending_for_a_missing_count(results):
    (results / "synth_gate.json").unlink()
    block = figures.mermaid_funnel(results)
    assert block.startswith("```mermaid") and '"pending kept by the filter"' in block


def test_inject_funnel_needs_the_markers():
    with pytest.raises(ValueError):
        figures.inject_funnel("no markers", "x")
    out = figures.inject_funnel(f"a\n{figures.FUNNEL_START}\nold\n{figures.FUNNEL_END}\nb", "new")
    assert out == f"a\n{figures.FUNNEL_START}\nnew\n{figures.FUNNEL_END}\nb"


def test_committed_figure_data_matches_results():
    """paper/figdata and the README funnel are what make_figures.py writes from the committed results/."""
    assert figures.main(["--check", "--readme", str(ROOT / "README.md")]) == 0


def test_funnel_data_equals_the_result_files():
    rows = dict(line.split(",") for line in (ROOT / "paper" / "figdata" / "funnel.csv").read_text().splitlines()[1:])
    gate = json.loads((ROOT / "results" / "data_gate.json").read_text())
    synth_gate = json.loads((ROOT / "results" / "synth_gate.json").read_text())
    assert int(rows["real-cleaned"]) == gate["conversations_before_gate"]
    assert int(rows["real-kept"]) == gate["kept"]
    assert int(rows["synth-filtered"]) == synth_gate["checked"]
    # The sample gate checked exactly what the filter kept; a stale synth_stats.json would break this.
    assert json.loads((ROOT / "results" / "synth_stats.json").read_text())["kept"] == synth_gate["checked"]
    assert int(rows["synth-kept"]) == synth_gate["kept"] == synth_gate["checked"] - synth_gate["dropped"]
