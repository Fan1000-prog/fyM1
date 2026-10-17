"""Find the checkpoint to resume from after a Colab disconnect."""

import re

from huggingface_hub import list_repo_files


def latest_checkpoint(repo_id: str, files: list[str] | None = None, run: str | None = None) -> str | None:
    """Return the `checkpoint-<step>` folder with the highest step, or None.

    With `run`, only checkpoints inside the `<run>/` folder count and the result is `<run>/checkpoint-<step>`;
    without it, only top-level ones. `files` are repo-relative paths; when omitted they are listed from
    `repo_id` on the Hub.
    Only the folder name is returned: the caller must `snapshot_download` that folder before passing
    its local path to `trainer.train(resume_from_checkpoint=...)`.
    """
    if files is None:
        files = list_repo_files(repo_id)
    prefix = f"{run}/" if run else ""
    pattern = re.compile(rf"^{re.escape(prefix)}checkpoint-(\d+)/")
    steps = {int(m.group(1)) for f in files if (m := pattern.match(f))}
    return f"{prefix}checkpoint-{max(steps)}" if steps else None
