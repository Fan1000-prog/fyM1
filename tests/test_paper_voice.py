"""Guards the paper's voice and outline: one author, first person singular, the standard section order."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper" / "fym1.tex"

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


def test_paper_has_the_standard_sections_in_order():
    tex = prose()
    assert "\\begin{abstract}" in tex
    sections = re.findall(r"\\section\*?\{([^}]*)\}", tex)
    assert sections == SECTION_ORDER
    assert tex.index("\\end{abstract}") < tex.index("\\section{Introduction}")
    assert tex.index("\\section*{Acknowledgements}") < tex.index("\\bibliography{refs}")
