import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_notebooks", ROOT / "scripts" / "build_notebooks.py")
build_notebooks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_notebooks)


@pytest.mark.parametrize("name", sorted(build_notebooks.NOTEBOOKS))
def test_committed_notebook_is_exactly_what_the_builder_makes(name):
    committed = (ROOT / "notebooks" / f"{name}.ipynb").read_text()
    assert committed == build_notebooks.build(name), f"run: python scripts/build_notebooks.py"


def test_builds_are_deterministic():
    assert build_notebooks.build("00_baseline") == build_notebooks.build("00_baseline")


def test_check_mode_reports_up_to_date(capsys):
    assert build_notebooks.main(["--check"]) == 0
    assert "up to date" in capsys.readouterr().out


def test_check_mode_fails_when_a_notebook_is_stale(tmp_path, monkeypatch, capsys):
    (tmp_path / "notebooks").mkdir()
    for name in build_notebooks.NOTEBOOKS:
        (tmp_path / "notebooks" / f"{name}.ipynb").write_text(build_notebooks.build(name))
    (tmp_path / "notebooks" / "01_train.ipynb").write_text("{}")
    monkeypatch.setattr(build_notebooks, "REPO_ROOT", tmp_path)
    assert build_notebooks.main(["--check"]) == 1
    assert "01_train.ipynb" in capsys.readouterr().out
    assert (tmp_path / "notebooks" / "01_train.ipynb").read_text() == "{}"  # --check never writes
