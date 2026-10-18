"""Guards the public model name: fy-m1 (lowercase, one hyphen).

Private HF repos keep `fyM1-chats`, `fyM1-ckpt`, `fyM1-votes`; the package stays `fym1`.
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_NAME = "fy-m" + "-1"  # split so this file does not trip its own check
PRIVATE_HF_REPOS = {"fyM1-chats", "fyM1-ckpt", "fyM1-votes"}
PUBLIC_SPACES = {"fy-m1-chat", "fy-m1-arena"}
# Files a visitor of the public repo, model page or Space reads first.
PUBLIC_FILES = [
    "README.md",
    "MODEL_CARD.md",
    "paper/fym1.tex",
    "spaces/chat/README.md",
    "spaces/chat/app.py",
    "spaces/arena/README.md",
    "spaces/arena/app.py",
]


def tracked_text_files() -> dict[str, str]:
    """Every git-tracked file that decodes as UTF-8 text, keyed by repo-relative path."""
    listing = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True
    ).stdout.decode()
    texts = {}
    for name in filter(None, listing.split("\0")):
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        try:
            texts[name] = path.read_text("utf-8")
        except UnicodeDecodeError:
            continue  # binary (PDF, images)
    return texts


@pytest.fixture(scope="module")
def texts() -> dict[str, str]:
    return tracked_text_files()


def test_old_hyphenated_name_is_gone(texts):
    offenders = sorted(name for name, text in texts.items() if OLD_NAME.lower() in text.lower())
    assert offenders == []


def test_hub_ids_use_public_name_or_a_private_repo(texts):
    """Every HF id under the Fan1000-Prog org that starts with "fy" is public fy-m1-* or a private fyM1 repo."""
    pattern = re.compile(r"Fan1000-Prog/(fy[\w.<>{}()-]*)")
    bad = {
        (name, repo)
        for name, text in texts.items()
        for repo in pattern.findall(text)
        if not repo.startswith("fy-m1-") and repo not in PRIVATE_HF_REPOS
    }
    assert sorted(bad) == []


@pytest.mark.parametrize("name", PUBLIC_FILES)
def test_public_files_link_only_the_public_github_repo(name):
    text = (REPO_ROOT / name).read_text("utf-8")
    assert "github.com/Fan1000-prog/fyM1" not in text


def test_spaces_link_the_public_ids(texts):
    linked = set(re.findall(r"huggingface\.co/spaces/Fan1000-Prog/([\w-]+)", "".join(texts.values())))
    assert linked and linked <= PUBLIC_SPACES


@pytest.mark.parametrize("space", ["chat", "arena"])
def test_space_titles_use_public_name(space, texts):
    front_matter = texts[f"spaces/{space}/README.md"].split("---")[1]
    assert f"title: fy-m1 {space}" in front_matter
    assert f'title="fy-m1' in texts[f"spaces/{space}/app.py"]
