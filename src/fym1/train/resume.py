"""Find the checkpoint to resume from after a Colab disconnect."""

import re

from huggingface_hub import list_repo_files

_CHECKPOINT_DIR = re.compile(r"^checkpoint-(\d+)/")


def latest_checkpoint(repo_id: str, files: list[str] | None = None) -> str | None:
    """Return the top-level `checkpoint-<step>` folder with the highest step, or None.

    `files` are repo-relative paths; when omitted they are listed from `repo_id` on the Hub.
    Only the folder name is returned: the caller must `snapshot_download` that folder before passing
    its local path to `trainer.train(resume_from_checkpoint=...)`.
    """
    if files is None:
        files = list_repo_files(repo_id)
    steps = {int(m.group(1)) for f in files if (m := _CHECKPOINT_DIR.match(f))}
    return f"checkpoint-{max(steps)}" if steps else None
