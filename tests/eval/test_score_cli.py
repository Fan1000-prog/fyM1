"""End-to-end checks of the leak gate commands in scripts/score.py (made-up data only)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCORE = Path(__file__).resolve().parents[2] / "scripts" / "score.py"
LEAKED = "a b c d e f g h"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCORE), *args], capture_output=True, text=True)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def setup_leak(tmp_path: Path, response: str) -> tuple[Path, Path, Path, Path]:
    answers = write_jsonl(
        tmp_path / "answers_fym1.jsonl", [{"id": "p1", "prompt": "x", "response": response}]
    )
    chats = write_jsonl(
        tmp_path / "chats.jsonl",
        [{"id": "c1", "messages": [{"role": "user", "content": f"x {LEAKED} y"}]}],
    )
    return answers, chats, tmp_path / "leak_matches.jsonl", tmp_path / "leak_review.json"


def leaks(answers, chats, matches, review) -> subprocess.CompletedProcess:
    return run(
        "leaks", "--answers", str(answers), "--train", str(chats),
        "--matches", str(matches), "--review", str(review),
    )


def gate(review: Path) -> subprocess.CompletedProcess:
    return run("leak-gate", "--matches", str(review.parent / "leak_matches.jsonl"),
               "--review", str(review))


def test_no_leak_passes_the_gate(tmp_path):
    answers, chats, matches, review = setup_leak(tmp_path, "z y x w v u t s")
    result = leaks(answers, chats, matches, review)
    assert result.returncode == 0
    assert "leak_rate 0.0000" in result.stdout
    assert json.loads(review.read_text()) == {"matches": 0, "approved_generic": 0, "rejected": 0}
    assert gate(review).returncode == 0


def test_leak_blocks_until_owner_approves_every_match(tmp_path):
    answers, chats, matches, review = setup_leak(tmp_path, LEAKED)
    result = leaks(answers, chats, matches, review)
    assert result.returncode == 0
    assert "leak_rate 1.0000" in result.stdout
    rows = [json.loads(line) for line in matches.read_text().splitlines()]
    assert rows == [{"ngram": LEAKED, "ids": ["p1"], "verdict": None}]
    assert json.loads(review.read_text()) == {"matches": 1, "approved_generic": 0, "rejected": 0}
    assert gate(review).returncode != 0

    write_jsonl(matches, [row | {"verdict": "generic"} for row in rows])
    assert run("leak-review", "--matches", str(matches), "--review", str(review)).returncode == 0
    assert gate(review).returncode == 0


def test_rejected_match_keeps_the_gate_blocked(tmp_path):
    answers, chats, matches, review = setup_leak(tmp_path, LEAKED)
    leaks(answers, chats, matches, review)
    write_jsonl(matches, [{"ngram": LEAKED, "ids": ["p1"], "verdict": "rejected"}])
    run("leak-review", "--matches", str(matches), "--review", str(review))
    assert json.loads(review.read_text()) == {"matches": 1, "approved_generic": 0, "rejected": 1}
    assert gate(review).returncode != 0


def test_a_run_shared_by_two_answers_is_reviewed_once(tmp_path):
    answers, chats, matches, review = setup_leak(tmp_path, LEAKED)
    write_jsonl(answers, [
        {"id": "p1", "prompt": "x", "response": LEAKED},
        {"id": "p2", "prompt": "y", "response": f"ok {LEAKED}"},
    ])
    leaks(answers, chats, matches, review)
    rows = [json.loads(line) for line in matches.read_text().splitlines()]
    assert rows == [{"ngram": LEAKED, "ids": ["p1", "p2"], "verdict": None}]
    assert json.loads(review.read_text())["matches"] == 1


def test_gate_fails_when_review_counts_are_stale(tmp_path):
    answers, chats, matches, review = setup_leak(tmp_path, LEAKED)
    leaks(answers, chats, matches, review)
    # Owner marks the match generic but forgets to rerun leak-review.
    write_jsonl(matches, [{"ngram": LEAKED, "ids": ["p1"], "verdict": "generic"}])
    result = gate(review)
    assert result.returncode != 0
    assert "out of date" in result.stderr


def test_gate_fails_when_review_is_edited_by_hand(tmp_path):
    answers, chats, matches, review = setup_leak(tmp_path, LEAKED)
    leaks(answers, chats, matches, review)
    review.write_text(json.dumps({"matches": 1, "approved_generic": 1, "rejected": 0}))
    result = gate(review)
    assert result.returncode != 0
    assert "out of date" in result.stderr


def test_leaks_never_overwrites_owner_verdicts(tmp_path):
    answers, chats, matches, review = setup_leak(tmp_path, LEAKED)
    leaks(answers, chats, matches, review)
    reviewed = [{"ngram": LEAKED, "ids": ["p1"], "verdict": "generic"}]
    write_jsonl(matches, reviewed)
    assert leaks(answers, chats, matches, review).returncode != 0
    assert [json.loads(line) for line in matches.read_text().splitlines()] == reviewed


def blind(tmp_path: Path, out: Path, key: Path) -> subprocess.CompletedProcess:
    rows = [{"id": "p1", "prompt": "x", "response": "Salama"}]
    base = write_jsonl(tmp_path / "answers_base.jsonl", rows)
    fym1 = write_jsonl(tmp_path / "answers_fym1.jsonl", rows)
    return run("blind", "--answers", str(base), str(fym1), "--out", str(out), "--key", str(key))


def test_blind_writes_the_judge_batch_apart_from_the_key(tmp_path):
    out = tmp_path / "judge_batch" / "judge_input.jsonl"
    result = blind(tmp_path, out, tmp_path / "judge_key.json")
    assert result.returncode == 0
    assert sorted(p.name for p in out.parent.iterdir()) == ["judge_input.jsonl"]


def test_blind_refuses_a_key_in_a_subdir_of_the_judge_batch_dir(tmp_path):
    out = tmp_path / "judge_batch" / "judge_input.jsonl"
    result = blind(tmp_path, out, out.parent / "keys" / "judge_key.json")
    assert result.returncode != 0
    assert not out.exists()


def test_blind_refuses_a_key_in_the_batch_dir_spelled_with_other_case(tmp_path):
    out = tmp_path / "judge_batch" / "judge_input.jsonl"
    out.parent.mkdir()
    if not (tmp_path / "JUDGE_BATCH").exists():
        pytest.skip("file system is case-sensitive; JUDGE_BATCH is a different folder here")
    result = blind(tmp_path, out, tmp_path / "JUDGE_BATCH" / "judge_key.json")
    assert result.returncode != 0
    assert not out.exists()


def test_blind_refuses_a_key_reaching_the_batch_dir_through_dotdot(tmp_path):
    out = tmp_path / "judge_batch" / "judge_input.jsonl"
    result = blind(tmp_path, out, tmp_path / "nosuch" / ".." / "judge_batch" / "judge_key.json")
    assert result.returncode != 0
    assert not out.exists()


def test_blind_allows_a_key_next_to_the_batch_dir(tmp_path):
    out = tmp_path / "judge_batch" / "judge_input.jsonl"
    result = blind(tmp_path, out, tmp_path / "judge_batch" / ".." / "judge_key.json")
    assert result.returncode == 0
    assert (tmp_path / "judge_key.json").exists()


def test_blind_refuses_a_key_inside_the_judge_batch_dir(tmp_path):
    out = tmp_path / "judge_batch" / "judge_input.jsonl"
    result = blind(tmp_path, out, out.parent / "judge_key.json")
    assert result.returncode != 0
    assert not out.exists()
