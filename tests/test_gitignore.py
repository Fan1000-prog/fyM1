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
    ["data/raw/x.json", "data/private/x.jsonl", "foo.zip", ".env", ".playwright-mcp/page-1.yml"],
)
def test_private_paths_are_ignored(path):
    assert is_ignored(path)


@pytest.mark.parametrize(
    "path", ["src/fym1/data/new_module.py", "tests/data/fixtures/x.json"]
)
def test_code_and_fixtures_are_tracked(path):
    assert not is_ignored(path)
