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
    (tmp_path / "eval" / "leak_review.json").write_text(json.dumps({"matches": 2, "approved_generic": 2, "rejected": 0}))
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
    assert macro(tex, "model-repo") == "Fan1000-Prog/fy-m1-8B"
    assert (macro(tex, "leak-matches"), macro(tex, "leak-generic"), macro(tex, "leak-rejected")) == ("2", "2", "0")


def test_text_values_are_latex_escaped(results_dir):
    tex = numbers.render_tex(resolve(results_dir))
    assert macro(tex, "base-model") == r"org/Some\_Model-8B"


def test_missing_values_render_pending(results_dir):
    tex = numbers.render_tex(resolve(results_dir))
    assert macro(tex, "data-approved") == r"\textbf{pending}"  # key missing from a present file
    assert macro(tex, "ppl-fym1") == r"\textbf{pending}"  # nested key missing
    assert macro(tex, "train-steps") == r"\textbf{pending}"  # whole file missing


def test_only_known_result_files_are_read(results_dir):
    # The judge key would unblind the judge; a broken file proves it is never opened.
    (results_dir / "eval" / "judge_key.json").write_text("{not json")
    (results_dir / "eval" / "judge_raw.json").write_text("{not json")
    loaded = numbers.load_results(results_dir)
    assert set(loaded) == {"data_stats.json", "eval/scores.json", "eval/judge.json", "eval/leak_review.json", "model_selection.json"}


def test_every_number_gets_a_macro_even_with_no_results(tmp_path):
    tex = numbers.render_tex(resolve(tmp_path))
    assert tex.count(r"\textbf{pending}") == len(numbers.NUMBERS) + 1  # +1: empty selection table
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


def test_strict_list_has_every_unmeasured_number(results_dir):
    missing = resolve(results_dir).missing
    assert "train_log.json: steps" in missing
    assert "eval/arena.json: votes" in missing  # early arena votes are due at v0.1
    assert "eval/scores.json: fym1.drift_rate" not in missing


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
    assert "| - | pending |" in out  # arena row: no base column
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


@pytest.mark.parametrize(
    "review, problem",
    [
        ({"matches": 2, "approved_generic": 2, "rejected": 0}, None),
        ({"matches": 3, "approved_generic": 2, "rejected": 0}, "!= matches"),
        ({"matches": 2, "approved_generic": 1, "rejected": 1}, "rejected; release is blocked"),
    ],
)
def test_leak_gate_needs_every_run_reviewed_and_none_rejected(review, problem):
    problems = numbers.leak_gate_problems({numbers.LEAK_FILE: review})
    assert problems == [] if problem is None else any(problem in p for p in problems)


def test_leak_gate_without_review_is_left_to_missing_values():
    assert numbers.leak_gate_problems({}) == []


def test_cli_strict_fails_on_rejected_leak_run(results_dir, tmp_path):
    (results_dir / "eval" / "leak_review.json").write_text(json.dumps({"matches": 1, "approved_generic": 0, "rejected": 1}))
    run = run_cli("--results", str(results_dir), "--out", str(tmp_path / "n.tex"), "--strict")
    assert run.returncode != 0
    assert "release is blocked" in run.stderr


def test_lookup_picks_list_items_and_dotted_names():
    data = {"candidates": [{"model": "a/x", "v": 1}, {"model": "b/y.z", "v": 2}], "models": {"y.z": {"n": 3}}}
    assert numbers.lookup(data, ("candidates", numbers.Pick("model", "b/y.z"), "v")) == 2
    assert numbers.lookup(data, ("models", "y.z", "n")) == 3
    assert numbers.lookup(data, ("candidates", numbers.Pick("model", "nope"), "v")) is None
    assert numbers.lookup(data, ("models", numbers.Pick("model", "a/x"))) is None  # a pick on a dict misses
    assert numbers.key_text(("candidates", numbers.Pick("model", "a/x"), "v")) == "candidates[model=a/x].v"


def test_candidate_and_greeting_numbers_come_from_selection_and_gate(tmp_path):
    (tmp_path / "model_selection.json").write_text(
        json.dumps({"candidates": [{"model": numbers.QWEN, "chrf_eng_plt": 49.73, "sec_per_step": 83.3}]})
    )
    key = numbers.GREETING_KEYS[numbers.GEMMA4]
    (tmp_path / "greeting_gate.json").write_text(
        json.dumps({"models": {key: {"judge_mean": 2.8500001, "is_malagasy_rate": 0.9, "n": 20}}})
    )
    tex = numbers.render_tex(resolve(tmp_path))
    assert macro(tex, "sel-qwen-chrf-en-mg") == "49.7"
    assert macro(tex, "sel-qwen-sec-step") == "83.3"
    assert macro(tex, "greet-gemma4-mean") == "2.85"
    assert macro(tex, "greet-gemma4-malagasy") == r"90.0\%"
    assert macro(tex, "sel-gemma4-chrf-en-mg") == r"\textbf{pending}"


def test_comparison_values_do_not_block_strict_but_the_base_does(tmp_path):
    (tmp_path / "model_selection.json").write_text(
        json.dumps(
            {
                "chosen": "org/base",
                "candidates": [
                    {"model": "org/base", "license": "MIT", "tokens_per_word": 2.0},
                    {"model": "org/other", "license": "MIT", "greeting_gate": "pending"},
                ],
            }
        )
    )
    resolved = resolve(tmp_path)
    assert "model_selection.json: candidates[0].chrf_eng_plt" in resolved.missing
    assert "model_selection.json: candidates[1].chrf_eng_plt" not in resolved.missing
    assert "model_selection.json: candidates[1].greeting_gate.judge_mean" in resolved.missing_keys
    gemma4 = f"model_selection.json: candidates[model={numbers.GEMMA4}].chrf_eng_plt"
    assert gemma4 not in resolved.missing  # comparison row: may stay pending
    assert f"model_selection.json: candidates[model={numbers.QWEN}].chrf_eng_plt" in resolved.missing
    assert resolved.rows[1][-1] is None  # a "pending" string is not a judge mean


def test_data_gate_counts_are_cited(tmp_path):
    (tmp_path / "data_gate.json").write_text(json.dumps({"conversations_before_gate": 260, "kept": 192}))
    values = resolve(tmp_path).values
    assert (values["gate-before"], values["gate-kept"], values["gate-dropped"]) == ("260", "192", None)


def test_greeting_judge_is_disclosed(tmp_path):
    (tmp_path / "greeting_gate.json").write_text(json.dumps({"judge": "Judge X, blind"}))
    assert resolve(tmp_path).values["greet-judge"] == "Judge X, blind"
