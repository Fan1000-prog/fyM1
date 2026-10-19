import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from fym1 import generation
from fym1.generation import GEN_KWARGS, HISTORY_EXCHANGES, LLAMA_CPP_KWARGS, trim_history

ROOT = Path(__file__).resolve().parents[1]
SAMPLING_LITERAL = re.compile(r"\b(temperature|top_p|top_k|repetition_penalty|repeat_penalty)\s*=\s*\d")


def chat(n_exchanges, end_on_user=True):
    turns = []
    for i in range(n_exchanges):
        turns += [{"role": "user", "content": f"U{i}"}, {"role": "assistant", "content": f"A{i}"}]
    if end_on_user:
        turns.append({"role": "user", "content": "NEW"})
    return turns


def contents(messages):
    return [m["content"] for m in messages]


def notebook_code(name):
    nb = json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def test_owner_approved_defaults():
    assert GEN_KWARGS == {"do_sample": True, "temperature": 0.6, "top_p": 0.9, "top_k": 50,
                          "repetition_penalty": 1.1}
    assert HISTORY_EXCHANGES == 3


def test_llama_cpp_kwargs_are_the_same_values_under_llama_cpp_names():
    assert LLAMA_CPP_KWARGS == {"temperature": GEN_KWARGS["temperature"], "top_p": GEN_KWARGS["top_p"],
                                "top_k": GEN_KWARGS["top_k"], "repeat_penalty": GEN_KWARGS["repetition_penalty"]}


def test_trim_keeps_system_prompt_plus_last_n_exchanges_and_the_new_message():
    out = trim_history("SYS", chat(6), max_exchanges=3)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert contents(out[1:]) == ["U3", "A3", "U4", "A4", "U5", "A5", "NEW"]


def test_trim_default_is_history_exchanges():
    assert trim_history("SYS", chat(10)) == trim_history("SYS", chat(10), HISTORY_EXCHANGES)
    assert len(trim_history("SYS", chat(10))) == 1 + 2 * HISTORY_EXCHANGES + 1


def test_short_history_is_kept_whole():
    assert contents(trim_history("SYS", chat(1))) == ["SYS", "U0", "A0", "NEW"]
    assert contents(trim_history("SYS", [])) == ["SYS"]


def test_history_ending_on_an_assistant_turn_keeps_the_last_n_exchanges():
    assert contents(trim_history("SYS", chat(5, end_on_user=False), 2)) == ["SYS", "U3", "A3", "U4", "A4"]


def test_zero_exchanges_keeps_only_the_new_message():
    assert contents(trim_history("SYS", chat(4), 0)) == ["SYS", "NEW"]


def test_kept_history_starts_on_a_user_turn():
    history = [{"role": "assistant", "content": "A-orphan"}] + chat(1)
    assert contents(trim_history("SYS", history, 5)) == ["SYS", "U0", "A0", "NEW"]


def test_system_prompt_is_never_doubled():
    history = [{"role": "system", "content": "SYS"}] + chat(2)
    out = trim_history("SYS", history, 5)
    assert [m["role"] for m in out].count("system") == 1
    assert contents(out) == ["SYS", "U0", "A0", "U1", "A1", "NEW"]


def test_trim_does_not_change_its_input():
    history = chat(6)
    before = json.dumps(history)
    trim_history("SYS", history, 1)
    assert json.dumps(history) == before


def test_negative_exchanges_is_an_error():
    with pytest.raises(ValueError):
        trim_history("SYS", chat(1), -1)


def test_chat_space_ships_an_exact_copy_of_the_shared_module():
    copy = ROOT / "spaces" / "chat" / "generation.py"
    assert copy.read_bytes() == Path(generation.__file__).read_bytes(), (
        "spaces/chat/generation.py is stale: cp src/fym1/generation.py spaces/chat/generation.py")


def test_chat_space_samples_with_the_shared_defaults():
    app = (ROOT / "spaces" / "chat" / "app.py").read_text()
    assert "from generation import LLAMA_CPP_KWARGS" in app
    assert 'SAMPLING = {**LLAMA_CPP_KWARGS, "max_tokens": 256}' in app
    assert "**SAMPLING" in app
    assert not SAMPLING_LITERAL.search(app)


def test_arena_space_does_not_generate():
    """The arena shows answers 02_eval_export already generated, so it has no sampling settings of its own."""
    for path in (ROOT / "spaces" / "arena").glob("*.py"):
        text = path.read_text()
        assert "generate(" not in text and "create_chat_completion" not in text and "llama_cpp" not in text


def test_eval_notebook_samples_answers_with_the_shared_defaults_and_keeps_flores_greedy():
    code = "\n".join(notebook_code("02_eval_export"))
    assert "from fym1.generation import GEN_KWARGS" in code
    assert "kw = GEN_KWARGS if sample else dict(do_sample=False)" in code
    assert not SAMPLING_LITERAL.search(code)
    assert "TEMPERATURE" not in code and "TOP_P" not in code
    answers = re.findall(r"generate\(\[r\[\"prompt\"\] for r in items\], MAX_NEW_TOKENS, (\w+), SYSTEM_PROMPT\)", code)
    assert answers == ["True"]
    flores = re.findall(r"generate\(\[bl\.translation_prompt\([^\]]*\], 256, (\w+)\)", code)
    assert flores == ["False", "False"]  # greedy, the 00_baseline protocol


def test_try_notebook_samples_with_the_shared_defaults():
    code = "\n".join(notebook_code("05_try"))
    assert "from fym1.generation import GEN_KWARGS, HISTORY_EXCHANGES, trim_history" in code
    assert not SAMPLING_LITERAL.search(code)
    assert "TEMPERATURE" not in code and "HISTORY_TURNS" not in code


def test_try_notebook_trims_history_before_rendering_and_generates_with_gen_kwargs(monkeypatch, capsys):
    """Runs the chat cell with stand-ins: the prompt holds one system prompt plus the last N exchanges."""
    cell = next(c for c in notebook_code("05_try") if "def reply(history)" in c)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.TextStreamer = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")  # the chat loop ends at once

    rendered, generate_kwargs = [], {}

    class Encoded(dict):
        def to(self, device):
            return self

    class StandInTokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            return Encoded(input_ids=np.zeros((1, 3), dtype=int))

        def decode(self, ids, **kwargs):
            return " reply "

    class StandInModel:
        def generate(self, **kwargs):
            generate_kwargs.update(kwargs)
            return np.zeros((1, 5), dtype=int)

    def chat_prompt(messages):
        rendered.append(messages)
        return "PROMPT"

    torch = types.SimpleNamespace(inference_mode=lambda: (lambda f: f))
    ns = {"torch": torch, "tok": StandInTokenizer(), "model": StandInModel(), "chat_prompt": chat_prompt,
          "system_prompt": "SYS", "MAX_NEW_TOKENS": 256, "GEN_KWARGS": GEN_KWARGS,
          "HISTORY_EXCHANGES": HISTORY_EXCHANGES, "trim_history": trim_history}
    exec(cell, ns)
    assert "veloma" in capsys.readouterr().out

    assert ns["reply"](chat(6)) == "reply"
    assert rendered == [trim_history("SYS", chat(6), HISTORY_EXCHANGES)]
    assert [m["role"] for m in rendered[0]].count("system") == 1
    for key, value in GEN_KWARGS.items():
        assert generate_kwargs[key] == value
    assert generate_kwargs["max_new_tokens"] == 256
