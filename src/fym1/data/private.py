"""Where private chat data may be written: anywhere but inside a git work tree."""

import subprocess
import sys
from pathlib import Path


def private_path(path: Path) -> Path:
    """`path` resolved, refusing anywhere inside a git work tree (chat text must never be committable)."""
    resolved = path.expanduser().resolve()
    existing = next(p for p in (resolved, *resolved.parents) if p.exists())
    if not existing.is_dir():
        existing = existing.parent
    toplevel = subprocess.run(
        ["git", "-C", str(existing), "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if toplevel.returncode == 0:
        sys.exit(
            f"{path} is inside the git work tree {toplevel.stdout.strip()}. "
            "Write chat data outside any repo, e.g. ~/malagasy-data/private/."
        )
    return resolved
