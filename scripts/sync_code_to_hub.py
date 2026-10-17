"""Copy the merged code (origin/main) into the private Fan1000-Prog/fyM1-ckpt repo under `code/`.

Colab notebooks import `fym1` from there, so they run merged code only. Run from a clean checkout of
origin/main, e.g. `git worktree add --detach ../fyM1-sync origin/main`. Refuses otherwise.
Uploads `src/**/*.py`, `eval/prompts.jsonl` and `code/SHA` (the commit) in one Hub commit; never deletes.
"""

import subprocess
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "Fan1000-Prog/fyM1-ckpt"
PATTERNS = ["src/**/*.py", "eval/prompts.jsonl"]


def commit_operations(root: Path, sha: str) -> list[CommitOperationAdd]:
    """Every synced file plus `code/SHA`, so code and SHA land in one commit or not at all."""
    files = sorted({f for pattern in PATTERNS for f in root.glob(pattern) if f.is_file()})
    ops = [CommitOperationAdd(path_in_repo=f"code/{f.relative_to(root).as_posix()}", path_or_fileobj=str(f))
           for f in files]
    return ops + [CommitOperationAdd(path_in_repo="code/SHA", path_or_fileobj=sha.encode())]


def refusal_reason(head: str, origin_main: str, porcelain: str) -> str | None:
    if head != origin_main:
        return f"HEAD {head[:9]} is not origin/main {origin_main[:9]}"
    if porcelain.strip():
        return "working tree is not clean (local changes or untracked files)"
    return None


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    git("fetch", "-q", "origin")
    reason = refusal_reason(git("rev-parse", "HEAD"), git("rev-parse", "origin/main"),
                            git("status", "--porcelain", "--untracked-files=all"))
    if reason:
        sys.exit(f"refusing to sync: {reason}")
    commit = git("rev-parse", "--short", "HEAD")
    # Notebooks record code/SHA in their results, so every number traces back to the code that made it.
    HfApi().create_commit(repo_id=REPO, operations=commit_operations(Path("."), commit),
                          commit_message=f"code: sync from origin/main {commit}")
    print(f"synced origin/main {commit} to {REPO}/code")


if __name__ == "__main__":
    main()
