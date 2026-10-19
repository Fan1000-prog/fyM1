"""Guards the voice (one author, first person singular) of the paper, README and model card, and the paper outline."""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper" / "fym1.tex"
README = REPO_ROOT / "README.md"

SECTION_ORDER = [
    "Introduction",
    "Related work",
    "Data",
    "Method",
    "Experiments",
    "Results",
    "Limitations",
    "Ethics",
    "Future work",
    "Acknowledgements",
]
PLURAL_OR_THIRD_PERSON = re.compile(r"\b(we|our|ours|us|ourselves|the owner|the author|data owner)\b", re.IGNORECASE)


def prose() -> str:
    """The paper's text without LaTeX comments."""
    return re.sub(r"(?<!\\)%.*", "", PAPER.read_text("utf-8"))


def test_paper_speaks_in_first_person_singular():
    assert PLURAL_OR_THIRD_PERSON.findall(prose()) == []


@pytest.mark.parametrize("name", ["README.md", "MODEL_CARD.md"])
def test_public_docs_speak_in_first_person_singular(name):
    assert PLURAL_OR_THIRD_PERSON.findall((REPO_ROOT / name).read_text("utf-8")) == []


def citation_author() -> str:
    """The author in the README's citation, read from there so this file holds no personal name."""
    found = re.findall(r"^\s*author\s*=\s*\{([^}]*)\},?$", README.read_text("utf-8"), re.M)
    assert len(found) == 1 and len(found[0].split()) == 2, "README citation needs one two-word author"
    return found[0]


def test_paper_has_one_author_with_the_byline():
    authors = re.findall(r"\\author\{([^}]*)\}", prose())
    assert authors == [f"by {citation_author()}"]


def test_paper_has_the_standard_sections_in_order():
    tex = prose()
    assert "\\begin{abstract}" in tex
    sections = re.findall(r"\\section\*?\{([^}]*)\}", tex)
    assert sections == SECTION_ORDER
    assert tex.index("\\end{abstract}") < tex.index("\\section{Introduction}")
    assert tex.index("\\section*{Acknowledgements}") < tex.index("\\bibliography{refs}")
