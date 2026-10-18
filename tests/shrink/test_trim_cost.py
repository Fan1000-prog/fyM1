"""scripts/trim_cost.py: aggregate counts only, never text. Synthetic tokenizer and synthetic chats."""

import importlib.util
import json
from pathlib import Path

from synthetic_text import HELD_OUT, OTHER_LANGUAGES

from fym1.shrink.public_keep import PublicSource

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("trim_cost", ROOT / "scripts" / "trim_cost.py")
trim_cost = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trim_cost)

SOURCES = (PublicSource("wikimedia/wikipedia", "0" * 40, 100, config="20231101.mg"),)


def _load(path, *args, **kwargs):
    assert path == "wikimedia/wikipedia"
    return iter([{"text": line} for line in OTHER_LANGUAGES])


def _write_chats(path: Path) -> Path:
    records = [{"id": "c1", "messages": [{"role": "user", "content": HELD_OUT[0]},
                                         {"role": "assistant", "content": HELD_OUT[1]}]},
               {"id": "c2", "messages": [{"role": "user", "content": HELD_OUT[2]},
                                         {"role": "assistant", "content": ""}]}]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def test_chat_lines_reads_non_empty_messages_and_caps(tmp_path):
    chats = _write_chats(tmp_path / "chats.jsonl")
    assert sorted(trim_cost.chat_lines(chats, 10)) == sorted(HELD_OUT[:3])
    assert len(trim_cost.chat_lines(chats, 2)) == 2


def test_trim_cost_reports_counts_only(tokenizer, tmp_path):
    held_out = {"chats_anonymised_private": trim_cost.chat_lines(_write_chats(tmp_path / "c.jsonl"), 10)}
    result = trim_cost.trim_cost(tokenizer, held_out, load=_load, sources=SOURCES)
    cost = result["held_out"]["chats_anonymised_private"]
    assert cost["decode_equal"] == cost["untrimmed"]["lines"] == 3  # byte fallback: nothing lost
    assert cost["trimmed"]["tokens"] > cost["untrimmed"]["tokens"]  # chat-only tokens were dropped
    assert result["vocab_after"] < result["vocab_before"]
    dumped = json.dumps(result, ensure_ascii=False)
    for line in HELD_OUT[:3]:
        for word in line.split():
            if len(word) > 4:
                assert word not in dumped


def test_main_writes_only_aggregates(tokenizer, tmp_path, monkeypatch, capsys):
    from transformers import AutoTokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *_a, **_k: tokenizer)
    monkeypatch.setattr(trim_cost, "DEFAULT_SOURCES", SOURCES)
    real = trim_cost.trim_cost
    monkeypatch.setattr(trim_cost, "trim_cost", lambda tok, held, sp: real(tok, held, sp, load=_load, sources=SOURCES))
    chats, out = _write_chats(tmp_path / "chats.jsonl"), tmp_path / "out" / "trim_cost.json"
    assert trim_cost.main(["--chats", str(chats), "--tokenizer", "synthetic", "--out", str(out)]) == 0
    result = json.loads(out.read_text("utf-8"))
    assert set(result["held_out"]) == {"chats_anonymised_private", "eval_prompts"}
    text = out.read_text("utf-8") + capsys.readouterr().out
    assert str(tmp_path) not in text  # no private path
    assert not any(line in text for line in HELD_OUT)
