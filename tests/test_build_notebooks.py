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


def eval_code_lines() -> list[str]:
    import json

    nb = json.loads((ROOT / "notebooks" / "02_eval_export.ipynb").read_text())
    return [line for c in nb["cells"] if c["cell_type"] == "code" for line in "".join(c["source"]).splitlines()]


def test_eval_notebook_scores_the_a100_run_from_its_train_log():
    code = "\n".join(eval_code_lines())
    assert 'RUN_ID = "fym1-9b-a100-r1"' in code
    assert 'CKPT_REPO, DATA_REPO = "Fan1000-Prog/fyM1-ckpt", "Fan1000-Prog/fyM1-chats"' in code
    assert 'MODEL_ID, DATA_REVISION = TRAIN_LOG["model"], TRAIN_LOG["config"]["data_revision"]' in code
    assert "use_exact_model_name=True" in code and "load_in_4bit=True" in code
    assert 'allow_patterns=[f"{RUN_ID}/final/*"]' in code
    assert "SEED, MAX_NEW_TOKENS = 42, 256" in code


def test_eval_notebook_answer_files_are_the_ones_the_laptop_scripts_read():
    """score.py maps answers_<model>.jsonl to a model name; paper/make_numbers.py reads base and fym1."""
    from fym1.eval.scoring import model_name

    code = "\n".join(eval_code_lines())
    names = ["answers_base", "answers_fym1", "answers_fym1_identity"]
    for name in names:
        assert f'"{name}": (' in code
    assert [model_name(Path(f"{n}.jsonl")) for n in names[:2]] == ["base", "fym1"]
    assert '{"id": r["id"], "prompt": r["prompt"], "response": a.strip()}' in code  # scoring/identity row keys
    assert '"base": benchmarks(' in code and '"fym1": benchmarks(' in code
    assert '"leak_rate":' not in code  # measured on the laptop by score.py leaks


def test_eval_notebook_writes_only_to_private_ckpt_results_eval():
    code = "\n".join(eval_code_lines())
    assert 'OUT_PREFIX = "results/eval"' in code
    assert "assert api.repo_info(CKPT_REPO).private" in code
    writes = [line for line in eval_code_lines() if "api." in line and "api.repo_info" not in line]
    assert writes == ['    api.upload_file(path_or_fileobj=body, path_in_repo=f"{OUT_PREFIX}/{name}", repo_id=CKPT_REPO,']
    for forbidden in ("create_commit", "upload_folder", "push_to_hub", "save_pretrained", "repo_type=\"dataset\", commit"):
        assert forbidden not in code, forbidden


def test_eval_notebook_never_prints_answer_or_chat_text():
    prints = [line for line in eval_code_lines() if "print(" in line]
    assert prints
    for needle in ("replies", "rows", "response", "texts", "held", "hyp_", "SYSTEM_PROMPT", '["messages"]', "(items", " items"):
        assert not any(needle in line for line in prints), needle


def try_code() -> str:
    import json

    nb = json.loads((ROOT / "notebooks" / "05_try.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_try_notebook_loads_the_final_adapter_with_the_training_prompt():
    code = try_code()
    assert 'f"{RUN_ID}/final/*"' in code and "PeftModel.from_pretrained(model, ADAPTER)" in code
    assert "load_in_4bit=True" in code
    assert '"system_prompt.txt"' in code and "revision=DATA_REVISION" in code
    assert "**CHAT_TEMPLATE_KWARGS" in code


def test_try_notebook_chat_ends_on_empty_line_or_quit_and_saves_nothing():
    code = try_code()
    assert 'if not user or user.lower() == "quit":' in code
    assert "upload" not in code and "open(" not in code.replace("open(hf_hub_download", "")


def test_try_notebook_multi_turn_prompt_is_the_training_render_plus_generation_prompt():
    """Past assistant turns keep the thinking-stub header training gave them (Qwen3 template, thinking off)."""
    import json

    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    from fym1.train.chat_format import CHAT_TEMPLATE_KWARGS, training_text

    source = (ROOT / "tests" / "train" / "fixtures" / "qwen3_chat_template.jinja").read_text()
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                             extensions=[loopcontrols]).from_string(source)

    class StandInTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False
            return template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)

    nb = json.loads((ROOT / "notebooks" / "05_try.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "def chat_prompt" in "".join(c["source"]))
    ns = {"tok": StandInTokenizer(), "CHAT_TEMPLATE_KWARGS": CHAT_TEMPLATE_KWARGS, "training_text": training_text}
    exec(cell, ns)

    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}]
    stub = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prompt = ns["chat_prompt"](messages)
    assert prompt == training_text(ns["render"], messages) + stub
    assert prompt == ("<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nU1<|im_end|>\n"
                      f"{stub}A1<|im_end|>\n<|im_start|>user\nU2<|im_end|>\n{stub}")
    # the plain render drops the stub from the past assistant turn, so the fix is doing work
    assert "<|im_start|>assistant\nA1" in ns["render"](messages, True)
