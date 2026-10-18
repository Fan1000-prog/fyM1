"""In the public snapshot only: skip test modules that need the word lists it does not ship.

The snapshot (scripts/release_snapshot.py) leaves out the lists scripts/build_known_words.py writes, and adds a
NOTICE file the private repo never tracks. There, a test module whose import needs one of those lists is skipped
with the command that builds it. In the private repo a missing list still fails the run.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IS_PUBLIC_SNAPSHOT = (ROOT / "NOTICE").is_file()


def _missing_word_list(error: FileNotFoundError) -> bool:
    path = Path(error.filename or "")
    return path.parts[-3:-1] == ("fym1", "data") and path.suffix == ".txt" and not path.exists()


class _SnapshotModule(pytest.Module):
    def _getobj(self):
        try:
            return super()._getobj()
        except FileNotFoundError as error:
            if not _missing_word_list(error):
                raise
            pytest.skip(error.strerror, allow_module_level=True)


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.Module | None:
    if IS_PUBLIC_SNAPSHOT:
        return _SnapshotModule.from_parent(parent, path=module_path)
    return None
