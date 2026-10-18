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


def synth_code_lines() -> list[str]:
    import json

    nb = json.loads((ROOT / "notebooks" / "04_synth.ipynb").read_text())
    return [line for c in nb["cells"] if c["cell_type"] == "code" for line in "".join(c["source"]).splitlines()]


def test_synth_notebook_writes_training_jsonl_only_when_promoted():
    """01_train trains on every *.jsonl in fyM1-chats: unchecked synthetic chats must not be one."""
    lines = [line.strip() for line in synth_code_lines() if ".jsonl" in line]
    assert lines[0].startswith('REAL_FILES = ["chats.jsonl"]')
    assert lines[1].startswith("PROMOTE = False")
    assert lines[2:] == ['uploads["synth/synth_chats.jsonl"] = uploads["synth/filtered.ndjson"]']
    promote = next(i for i, line in enumerate(synth_code_lines()) if "synth_chats.jsonl\"] =" in line)
    assert synth_code_lines()[promote - 1].strip() == "if PROMOTE:"


def test_synth_notebook_never_prints_chat_text():
    prints = [line for line in synth_code_lines() if "print(" in line]
    assert prints
    for needle in ('["output"]', '["messages"]', '["content"]', "real[", "kept[", "rows[", "smoke"):
        assert not any(needle in line for line in prints), needle


def test_synth_notebook_generators_are_permissive_and_ordered():
    """Owner decision 2026-10-18: synthetic chats only from permissive models; AfriqueQwen 9B first, Gemma 4 last."""
    import ast
    import re

    from fym1.data.synth import generator_licence

    code = "\n".join(synth_code_lines())
    listed = re.search(r"^GENERATORS = (\[.*?^\])", code, re.S | re.M)[1]
    generators = ast.literal_eval(re.sub(r"#.*", "", listed))
    for hub_id, dtype in generators:
        generator_licence(hub_id)
        assert dtype in ("float16", "float32")
    assert generators[0] == ("McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1", "float16")
    assert generators[-1] == ("google/gemma-4-12B-it", "float32")
    assert "gemma-3" not in code and "Gemma license" not in code
