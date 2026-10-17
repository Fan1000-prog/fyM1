import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "paper" / "make_numbers.py"

# paper/ is not a package, so load the script by path.
_spec = importlib.util.spec_from_file_location("make_numbers", SCRIPT)
numbers = importlib.util.module_from_spec(_spec)
sys.modules["make_numbers"] = numbers  # dataclasses look the module up while building Number
_spec.loader.exec_module(numbers)


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    (tmp_path / "eval").mkdir()
    (tmp_path / "data_stats.json").write_text(json.dumps({"threads": 12, "conversations": 3456}))
    # Same shape scripts/score.py writes (PR #6): metrics not measured yet are left out.
    (tmp_path / "eval" / "scores.json").write_text(
        json.dumps({"n_prompts": 150, "base": {"drift_rate": 0.4}, "fym1": {"drift_rate": 0.0533, "leak_rate": 0.0}})
    )
    (tmp_path / "eval" / "judge.json").write_text(
        json.dumps(
            {
                "judge": "Some judge, blind",
                "base": {"naturalness": 2.5, "correctness": 3.0, "n": 150},
                "fym1": {"naturalness": 4.125, "correctness": 3.5, "n": 150},
            }
        )
    )
    (tmp_path / "model_selection.json").write_text(
        json.dumps(
            {
                "chosen": "org/Some_Model-8B",
                "chosen_size": "8B",
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


def resolve(results_dir: Path):
    return numbers.resolve(numbers.load_results(results_dir))


def test_present_values_are_formatted(results_dir):
    tex = numbers.render_tex(resolve(results_dir))
    assert macro(tex, "data-conversations") == "3,456"
    assert macro(tex, "eval-prompts") == "150"
    assert macro(tex, "drift-base") == r"40.0\%"
    assert macro(tex, "drift-fym1") == r"5.3\%"
    assert macro(tex, "natural-fym1") == "4.12"
    assert macro(tex, "model-repo") == "Fan1000-Prog/fyM1-8B"


def test_text_values_are_latex_escaped(results_dir):
    tex = numbers.render_tex(resolve(results_dir))
    assert macro(tex, "base-model") == r"org/Some\_Model-8B"


def test_missing_values_render_pending(results_dir):
    tex = numbers.render_tex(resolve(results_dir))
    assert macro(tex, "data-approved") == r"\textbf{pending}"  # key missing from a present file
    assert macro(tex, "ppl-fym1") == r"\textbf{pending}"  # nested key missing
    assert macro(tex, "train-steps") == r"\textbf{pending}"  # whole file missing


def test_only_later_release_items_say_v02(results_dir):
    tex = numbers.render_tex(resolve(results_dir))
    assert macro(tex, "arena-win-rate") == r"\textbf{pending (v0.2)}"
    assert "v0.2" not in macro(tex, "train-steps")


def test_every_number_gets_a_macro_even_with_no_results(tmp_path):
    tex = numbers.render_tex(resolve(tmp_path))
    assert tex.count(r"\textbf{pending") == len(numbers.NUMBERS) + 1  # +1: empty selection table
    assert r"\multicolumn" in macro(tex, "selection-rows")


def test_selection_rows_fill_missing_columns_with_pending(results_dir):
    rows = macro(numbers.render_tex(resolve(results_dir)), "selection-rows")
    assert rows.startswith(r"Some\_Model-8B & Apache-2.0 & 50.0\% & \textbf{pending}")
    assert "2.35" in rows


def test_missing_keys_listed_only_for_files_that_exist(results_dir):
    resolved = resolve(results_dir)
    assert "eval/scores.json: fym1.perplexity" in resolved.missing_keys
    assert "model_selection.json: candidates[0].chrf_plt_eng" in resolved.missing_keys
    assert not any(w.startswith(("train_log.json", "eval/arena.json")) for w in resolved.missing_keys)


def test_strict_needs_every_v01_number_but_not_v02(results_dir):
    missing = resolve(results_dir).missing_now
    assert "train_log.json: steps" in missing
    assert not any(w.startswith("eval/arena.json") for w in missing)


def test_macro_names_are_letters_only_and_unique():
    names = [numbers.latex_name(n.name) for n in numbers.NUMBERS] + [numbers.latex_name("selection-rows")]
    assert all(re.fullmatch(r"fy[A-Za-z]+", n) for n in names)
    assert len(names) == len(set(names))


def test_latex_name_spells_digits():
    assert numbers.latex_name("drift-fym1") == "fyDriftFymOne"
    assert numbers.latex_name("chrf-en-mg_base") == "fyChrfEnMgBase"


@pytest.mark.parametrize(
    "fmt, value",
    [
        (numbers.passed, "yes"),
        (numbers.count, True),
        (numbers.count, 2.5),
        (numbers.percent, 1.5),
        (numbers.percent, -0.1),
        (numbers.percent, "0.4"),
        (numbers.text, 3),
    ],
)
def test_bad_values_fail_loudly(fmt, value):
    with pytest.raises((TypeError, ValueError)):
        fmt(value)


def test_markdown_table_inline_and_yaml_values_refreshed(results_dir):
    values = resolve(results_dir).values
    doc = (
        "---\nbase_model: pending # n:base-model\nlicense: pending # n:base-license-hf\n---\n"
        f"Winner: <!-- n:model-size -->old<!-- /n -->.\n{numbers.README_START}\nold\n{numbers.README_END}\nafter\n"
    )
    out = numbers.inject_readme(doc, values)
    assert "base_model: org/Some_Model-8B # n:base-model" in out
    assert "license: pending # n:base-license-hf" in out
    assert "Winner: <!-- n:model-size -->8B<!-- /n -->." in out
    assert "| 40.0% | 5.3% |" in out
    assert "| - | pending (v0.2) |" in out  # arena row: no base column, due in v0.2
    assert out.endswith(f"{numbers.README_END}\nafter\n")
    assert numbers.inject_readme(out, values) == out  # rerunning changes nothing


def test_markdown_rejects_missing_markers_and_unknown_names(results_dir):
    values = resolve(results_dir).values
    with pytest.raises(ValueError):
        numbers.inject_readme("# no markers\n", values)
    with pytest.raises(ValueError):
        numbers.inject_readme(f"<!-- n:nope -->x<!-- /n -->{numbers.README_START}{numbers.README_END}", values)


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_cli_writes_tex_and_warns_on_missing_keys(results_dir, tmp_path):
    out = tmp_path / "numbers.tex"
    run = run_cli("--results", str(results_dir), "--out", str(out))
    assert run.returncode == 0
    assert "warning: missing key eval/scores.json: fym1.perplexity" in run.stderr
    assert macro(out.read_text(), "data-threads") == "12"


def test_cli_strict_fails_and_writes_nothing(results_dir, tmp_path):
    out = tmp_path / "numbers.tex"
    run = run_cli("--results", str(results_dir), "--out", str(out), "--strict")
    assert run.returncode != 0
    assert "train_log.json: steps" in run.stderr
    assert not out.exists()
