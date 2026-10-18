"""Every baseline candidate must be known to build_model_selection, or the rebuild fails on its result."""

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_model_selection", ROOT / "scripts" / "build_model_selection.py")
build_model_selection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_model_selection)


def baseline_candidates() -> dict:
    nb = json.loads((ROOT / "notebooks" / "00_baseline.ipynb").read_text())
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    (cell,) = [s for s in sources if s.startswith("CANDIDATES = {")]  # other cells hold `!pip`, not Python
    (assign,) = [n for n in ast.parse(cell).body
                 if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "CANDIDATES"]
    return ast.literal_eval(assign.value)


def test_every_candidate_has_a_license():
    hub_ids = {hub_id for hub_id, *_ in baseline_candidates().values()}
    assert hub_ids - set(build_model_selection.LICENSES) == set()


def test_base_models_match_the_notebook_flags():
    base = {hub_id for hub_id, _, _, is_base in baseline_candidates().values() if is_base}
    assert base == build_model_selection.BASE_MODELS
