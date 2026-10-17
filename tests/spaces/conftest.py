import sys
from pathlib import Path

# Each Space ships as flat files (no package), so tests import its modules by path.
SPACES = Path(__file__).resolve().parents[2] / "spaces"
for space in ("arena", "chat"):
    sys.path.insert(0, str(SPACES / space))
