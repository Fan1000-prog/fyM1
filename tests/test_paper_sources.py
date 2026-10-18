"""The paper's figures are text sources and its citations resolve: the public snapshot ships text files only."""

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"


def tex() -> str:
    return re.sub(r"(?<!\\)%.*", "", (PAPER / "fym1.tex").read_text("utf-8"))


def test_figures_are_drawn_in_tex_not_included_as_images():
    assert "\\includegraphics" not in tex()
    assert not [p for p in PAPER.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".eps"}]


def test_plots_read_only_generated_figure_data():
    tables = re.findall(r"\\addplot[^\n]*?table\[[^\]]*\]\{([^}]*)\}", tex())
    assert tables and all(t.startswith("figdata/") and (PAPER / t).is_file() for t in tables)


def test_every_citation_has_a_bib_entry_and_every_entry_is_cited():
    cited = {k.strip() for group in re.findall(r"\\cite[pt]?\w*\{([^}]*)\}", tex()) for k in group.split(",")}
    entries = set(re.findall(r"^@\w+\{([^,]+),", (PAPER / "refs.bib").read_text("utf-8"), re.M))
    assert cited - entries == set()
    assert entries - cited == set()
