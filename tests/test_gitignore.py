import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def is_ignored(path: str) -> bool:
    # --no-index checks the rules themselves, even for paths already tracked
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=REPO,
        capture_output=True,
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/x.json",
        "data/private/x.jsonl",
        "foo.zip",
        ".env",
        "pairs_key.jsonl",
        "spaces/arena/pairs_key.jsonl",
        ".playwright-mcp/page-1.yml",
    ],
)
def test_private_paths_are_ignored(path):
    assert is_ignored(path)


@pytest.mark.parametrize(
    "path", ["src/fym1/data/new_module.py", "tests/data/fixtures/x.json", "spaces/arena/pairs.jsonl"]
)
def test_code_and_fixtures_are_tracked(path):
    assert not is_ignored(path)


# Raw answers and the blind key name which model wrote each answer. They stay private
# until the arena closes (v0.2), so raters cannot unblind their votes.
@pytest.mark.parametrize(
    "path",
    [
        "results/eval/answers_base.jsonl",
        "results/eval/answers_fym1.jsonl",
        "results/eval/judge_batch/judge_input.jsonl",
        "results/eval/judge_key.json",
        "results/eval/judge_raw.json",
        "results/eval/leak_matches.jsonl",  # 8-grams copied from real chats
    ],
)
def test_unblinding_eval_files_are_ignored(path):
    assert is_ignored(path)


@pytest.mark.parametrize(
    "path",
    [
        "results/eval/scores.json",
        "results/eval/judge.json",
        "results/eval/arena.json",
        "results/eval/leak_review.json",
    ],
)
def test_aggregate_eval_results_are_tracked(path):
    assert not is_ignored(path)
