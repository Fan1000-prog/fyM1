import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "paper" / "numbers.py"

# Loaded by path: paper/ is not a package, and "numbers" would shadow the stdlib module.
_spec = importlib.util.spec_from_file_location("paper_numbers", SCRIPT)
numbers = importlib.util.module_from_spec(_spec)
sys.modules["paper_numbers"] = numbers  # dataclasses look the module up while building Number
_spec.loader.exec_module(numbers)


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    (tmp_path / "eval").mkdir()
    (tmp_path / "data_stats.json").write_text(json.dumps({"threads": 12, "conversations": 3456}))
    (tmp_path / "eval" / "scores.json").write_text(
        json.dumps({"n_prompts": 150, "base": {"drift_rate": 0.4}, "fym1": {"drift_rate": 0.0533}})
    )
    (tmp_path / "model_selection.json").write_text(
        json.dumps(
            {
                "chosen": "Some_Model-8B",
                "candidates": [
                    {"model": "Some_Model-8B", "license": "Apache-2.0", "belebele_acc": 0.5, "tokens_per_word": 2.345},
                ],
            }
        )
    )
    return tmp_path


def macro(tex: str, name: str) -> str:
    match = re.search(r"\\newcommand\{\\" + numbers.latex_name(name) + r"\}\{(.*)\}$", tex, re.M)
    assert match, f"no macro for {name}"
    return match.group(1)


def test_present_values_are_formatted(results_dir):
    tex = numbers.render_tex(*_resolve(results_dir))
    assert macro(tex, "data-conversations") == "3,456"
    assert macro(tex, "eval-prompts") == "150"
    assert macro(tex, "drift-base") == r"40.0\%"
    assert macro(tex, "drift-fym1") == r"5.3\%"


def test_text_values_are_latex_escaped(results_dir):
    tex = numbers.render_tex(*_resolve(results_dir))
    assert macro(tex, "base-model") == r"Some\_Model-8B"


def test_missing_values_render_pending(results_dir):
    tex = numbers.render_tex(*_resolve(results_dir))
    assert macro(tex, "data-approved") == r"\textbf{pending}"  # key missing from a present file
    assert macro(tex, "arena-win-rate") == r"\textbf{pending}"  # whole file missing
    assert macro(tex, "ppl-fym1") == r"\textbf{pending}"  # nested key missing


def test_every_number_gets_a_macro_even_with_no_results(tmp_path):
    tex = numbers.render_tex(*_resolve(tmp_path))
    assert tex.count(r"\textbf{pending}") == len(numbers.NUMBERS) + 1  # +1: empty selection table
    assert r"\multicolumn" in macro(tex, "selection-rows")


def test_selection_rows_fill_missing_columns_with_pending(results_dir):
    rows = macro(numbers.render_tex(*_resolve(results_dir)), "selection-rows")
    assert rows.startswith(r"Some\_Model-8B & Apache-2.0 & 50.0\% & \textbf{pending}")
    assert "2.35" in rows


def test_macro_names_are_letters_only_and_unique():
    names = [numbers.latex_name(n.name) for n in numbers.NUMBERS] + [numbers.latex_name("selection-rows")]
    assert all(re.fullmatch(r"fy[A-Za-z]+", n) for n in names)
    assert len(names) == len(set(names))


def test_latex_name_spells_digits():
    assert numbers.latex_name("drift-fym1") == "fyDriftFymOne"
    assert numbers.latex_name("chrf-en-mg_base") == "fyChrfEnMgBase"


def test_wrong_type_fails_loudly(tmp_path):
    (tmp_path / "shrink.json").write_text(json.dumps({"gates_passed": "yes"}))
    with pytest.raises(TypeError):
        _resolve(tmp_path)


def test_readme_table_injected_between_markers(results_dir):
    values, _ = _resolve(results_dir)
    readme = f"# x\n{numbers.README_START}\nold\n{numbers.README_END}\nafter\n"
    out = numbers.inject_readme(readme, numbers.render_readme_table(values))
    assert "old" not in out and out.endswith(f"{numbers.README_END}\nafter\n")
    assert "| 40.0% | 5.3% |" in out
    assert "| - | pending |" in out  # arena row: no base column, not measured yet


def test_readme_without_markers_is_rejected():
    with pytest.raises(ValueError):
        numbers.inject_readme("# no markers\n", "table")


def test_cli_writes_tex(results_dir, tmp_path):
    out = tmp_path / "numbers.tex"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--results", str(results_dir), "--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert "pending" in run.stdout
    assert macro(out.read_text(), "data-threads") == "12"


def _resolve(results_dir: Path):
    results = numbers.load_results(results_dir)
    return numbers.resolve(results), numbers.selection_rows(results)
