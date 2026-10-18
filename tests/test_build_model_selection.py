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


# Synthetic fixtures: made-up numbers, no real model output.
def fake_result(hub_id: str, slug: str) -> dict:
    return {"model": {"hub_id": hub_id, "slug": slug}, "belebele": {"acc": 0.5}, "tokens_per_word": {"value": 2.0}}


FAKE_GATE = {
    "judge": "test judge",
    "rubric": "rubric.md",
    "models": {
        "gemma-4-12b-it": {
            "judge_mean": 2.123456, "is_malagasy_rate": 0.9, "drift_rate": 0.1, "n": 20,
            "shared_prompts": {"judge_mean": 2.987654, "n": 15},
        },
    },
}


def test_judged_model_carries_its_greeting_gate_numbers():
    row = build_model_selection.candidate(fake_result("google/gemma-4-12B-it", "gemma-4-12b-it"), FAKE_GATE["models"])
    assert row["greeting_gate"] == {"judge_mean": 2.123, "shared_prompts_judge_mean": 2.988, "shared_prompts_n": 15,
                                    "is_malagasy_rate": 0.9, "drift_rate": 0.1, "n": 20}
    assert row["status"] == "candidate"
    assert "dropped_reason" not in row


def test_unjudged_model_stays_pending():
    row = build_model_selection.candidate(fake_result("google/gemma-3-12b-it", "gemma-3-12b-it"), FAKE_GATE["models"])
    assert row["greeting_gate"] == "pending"


def test_afriqueqwen_8b_is_dropped_with_the_reason():
    row = build_model_selection.candidate(fake_result("McGill-NLP/AfriqueQwen-8B", "afriqueqwen-8b"), {})
    assert row["status"] == "dropped"
    assert row["dropped_reason"] == "base model, no chat; lead decision"


def test_selection_keeps_the_lead_fields_and_names_the_gate_source():
    previous = {"chosen": "x", "chosen_size": "9B", "license": "L", "hf_license": "l", "chosen_reason": ["r"]}
    selection = build_model_selection.build_selection(
        [fake_result("google/gemma-4-12B-it", "gemma-4-12b-it")], previous, FAKE_GATE)
    assert {k: selection[k] for k in previous} == previous
    assert selection["greeting_gate"] == {"source": "results/greeting_gate.json", "judge": "test judge",
                                          "rubric": "rubric.md"}
    assert len(selection["candidates"]) == 1


def test_no_gate_file_leaves_every_gate_field_pending():
    selection = build_model_selection.build_selection(
        [fake_result("google/gemma-4-12B-it", "gemma-4-12b-it")], {}, {})
    assert selection["greeting_gate"] == "pending"
    assert selection["candidates"][0]["greeting_gate"] == "pending"


def test_gemma_3_is_dropped_for_its_licence_but_keeps_its_numbers():
    row = build_model_selection.candidate(fake_result("google/gemma-3-12b-it", "gemma-3-12b-it"), {})
    assert row["status"] == "dropped"
    assert row["dropped_reason"] == "licence: Gemma Terms of Use; owner wants a simple open licence"
    assert row["license"] == "Gemma Terms of Use"
    assert row["belebele_chat_acc"] == 0.5  # measured numbers stay for comparison
