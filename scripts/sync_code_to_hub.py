"""Copy the merged code (origin/main) into the private Fan1000-Prog/fyM1-ckpt repo under `code/`.

Colab notebooks import `fym1` from there, so they run merged code only. Run from a clean checkout of
origin/main, e.g. `git worktree add --detach ../fyM1-sync origin/main`. Refuses otherwise.
Uploads `src/**/*.py` and `eval/prompts.jsonl`; it never deletes files on the Hub.
"""

import subprocess
import sys

from huggingface_hub import HfApi

REPO = "Fan1000-Prog/fyM1-ckpt"
PATTERNS = ["src/**/*.py", "eval/prompts.jsonl"]


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
    HfApi().upload_folder(folder_path=".", path_in_repo="code", repo_id=REPO, allow_patterns=PATTERNS,
                          commit_message=f"code: sync from origin/main {commit}")
    print(f"synced origin/main {commit} to {REPO}/code")


if __name__ == "__main__":
    main()
