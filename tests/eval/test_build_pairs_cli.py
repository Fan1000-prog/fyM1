import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_refuses_to_write_the_answer_key_inside_spaces(tmp_path):
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"id": "p1", "prompt": "q", "response": "r"}) + "\n", encoding="utf-8")
    key_out = REPO / "spaces" / "arena" / "pairs_key.test-never-written.jsonl"
    out = tmp_path / "pairs.jsonl"
    command = [sys.executable, str(REPO / "scripts" / "build_pairs.py")]
    command += ["--base", str(answers), "--fym1", str(answers), "--out", str(out), "--key-out", str(key_out)]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode != 0
        assert "inside" in result.stderr
        assert not key_out.exists() and not out.exists()
    finally:
        key_out.unlink(missing_ok=True)
