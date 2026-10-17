import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def is_ignored(path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO, capture_output=True
    )
    return result.returncode == 0


@pytest.mark.parametrize(
    "path", ["data/raw/x.json", "data/private/x.jsonl", "foo.zip", ".env"]
)
def test_private_paths_are_ignored(path):
    assert is_ignored(path)


@pytest.mark.parametrize(
    "path", ["src/fym1/data/__init__.py", "tests/data/fixtures/x.json"]
)
def test_code_and_fixtures_are_tracked(path):
    assert not is_ignored(path)
