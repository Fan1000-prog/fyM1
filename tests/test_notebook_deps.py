"""Every third-party module the notebooks' fym1 code imports must be installed on the GPU runtime.

04_synth v3 died on Kaggle at `import ftfy`: fym1.data.messenger needs it, the notebook's pip line did not
install it, and the local tests passed because the dev venv has every pyproject dependency. This walks the
fym1 imports of each notebook, follows them through src/fym1, and checks each third-party top-level module
is either in that notebook's `!pip install` lines or already on the Colab/Kaggle images.
"""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NOTEBOOKS = sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb"))

# On the Colab and Kaggle GPU images without a pip install (google / kaggle_secrets: each platform's own
# secrets module, imported only on that platform by fym1.runtime.hf_token).
PREINSTALLED = {"torch", "transformers", "tokenizers", "safetensors", "huggingface_hub", "datasets", "numpy",
                "google", "kaggle_secrets"}
# pip distribution name -> top-level module, where they differ beyond case and - vs _.
PIP_MODULE = {"fasttext-numpy2-wheel": "fasttext", "llama-cpp-python": "llama_cpp"}
PIP_OPTIONS_WITH_VALUE = {"-e", "--editable", "-r", "--requirement", "--extra-index-url", "--index-url", "-i"}


def code_lines(notebook: str) -> list[str]:
    cells = json.loads((ROOT / "notebooks" / notebook).read_text())["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code").splitlines()


def python_source(lines: list[str]) -> str:
    """Drop IPython shell/magic lines (`!cmd`, `%magic`, `x = !cmd`) so the rest parses as Python."""
    return "\n".join(l for l in lines if not l.lstrip().startswith(("!", "%")) and not re.search(r"=\s*!", l))


def imported_modules(source: str) -> set[str]:
    """Absolute imports anywhere in the source, including `from pkg import sub` as `pkg.sub`."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def fym1_file(module: str) -> Path | None:
    path = SRC.joinpath(*module.split("."))
    for candidate in (path / "__init__.py", path.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None  # `from fym1.x import name` also yields "fym1.x.name", which is not a module


def third_party_imports(notebook: str) -> set[str]:
    """Top-level third-party modules reached from the notebook's fym1 imports, through src/fym1."""
    todo = [m for m in imported_modules(python_source(code_lines(notebook))) if m.split(".")[0] == "fym1"]
    seen, found = set(), set()
    while todo:
        module = todo.pop()
        path = fym1_file(module)
        if module in seen or path is None:
            continue
        seen.add(module)
        for name in imported_modules(path.read_text()):
            top = name.split(".")[0]
            if top == "fym1":
                todo.append(name)
            elif top not in sys.stdlib_module_names:
                found.add(top)
    return found


def pip_modules(lines: list[str]) -> set[str]:
    """Top-level modules installed by `!pip install` lines."""
    modules = set()
    for line in lines:
        words = line.split()
        if not re.search(r"\bpip3? install\b", line):
            continue
        skip_next = False
        for word in words[words.index("install") + 1:]:
            if skip_next or word.startswith("-"):
                skip_next = not skip_next and word in PIP_OPTIONS_WITH_VALUE
                continue
            dist = re.split(r"[<>=!~\[;]", word)[0].lower()
            modules.add(PIP_MODULE.get(dist, dist.replace("-", "_")))
    return modules


@pytest.mark.parametrize("notebook", NOTEBOOKS)
def test_fym1_third_party_imports_are_installed(notebook):
    missing = third_party_imports(notebook) - pip_modules(code_lines(notebook)) - PREINSTALLED
    assert not missing, f"{notebook}: fym1 code imports {sorted(missing)}, which its pip line does not install"


def test_walker_finds_the_synth_dependencies():
    # Guards the test itself: 04_synth reaches ftfy through synth -> segment -> messenger.
    assert {"ftfy", "fasttext"} <= third_party_imports("04_synth.ipynb")


def test_pip_parser_skips_options_and_maps_names():
    lines = ["!pip install -q -U foo-bar fasttext-numpy2-wheel baz>=1.0 -e /x/gguf-py --extra-index-url https://u qux"]
    assert pip_modules(lines) == {"foo_bar", "fasttext", "baz", "qux"}
