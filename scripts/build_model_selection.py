"""Assemble results/model_selection.json from the per-model baseline files.

Usage: python scripts/build_model_selection.py [--fetch]
--fetch first downloads results/baseline/*.json from the private Fan1000-Prog/fyM1-ckpt repo.
`chosen`/`chosen_size`/`license`/`hf_license` stay as already recorded in the output file (the lead fills them in).
Each candidate also carries its blind greeting-gate scores from results/greeting_gate.json ("pending" when the
model was not judged) and a `status` ("candidate", or "dropped" with the reason).
"""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from fym1.eval.baseline import selection_row

REPO = "Fan1000-Prog/fyM1-ckpt"
BASELINE_DIR = Path("results/baseline")
OUT = Path("results/model_selection.json")
GREETING_GATE = Path("results/greeting_gate.json")
LICENSES = {
    "google/gemma-4-12B-it": "Apache-2.0",
    "McGill-NLP/AfriqueQwen-8B": "CC-BY-4.0",
    "google/gemma-3-12b-it": "Gemma Terms of Use",
    "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1": "CC-BY-4.0",
}
BASE_MODELS = {"McGill-NLP/AfriqueQwen-8B"}  # not instruction-tuned: no chat-template Belebele column
# Base -> chat with little data is a risk, so record whether an instruct sibling exists (checked 2026-10-17).
INSTRUCT_VERSION = {
    "McGill-NLP/AfriqueQwen-8B": {
        "exists": False,
        "nearest": "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1",
        "nearest_license": "cc-by-4.0",
        "nearest_note": "different size and architecture (Qwen3.5, 9B); its base lists mg; not measured",
    },
}
# Candidates taken out of the pick, with the reason.
DROPPED = {
    "McGill-NLP/AfriqueQwen-8B": "base model, no chat; lead decision",
}


def greeting_scores(gate_models: dict, slug: str) -> dict | str:
    """The blind greeting-gate numbers for one model, or "pending" when the gate did not judge it."""
    scores = gate_models.get(slug)
    if scores is None:
        return "pending"
    return {
        "judge_mean": round(scores["judge_mean"], 3),
        "shared_prompts_judge_mean": round(scores["shared_prompts"]["judge_mean"], 3),
        "shared_prompts_n": scores["shared_prompts"]["n"],
        "is_malagasy_rate": scores["is_malagasy_rate"],
        "drift_rate": scores["drift_rate"],
        "n": scores["n"],
    }


def candidate(result: dict, gate_models: dict) -> dict:
    hub_id = result["model"]["hub_id"]
    row = selection_row(result, LICENSES[hub_id], base_model=hub_id in BASE_MODELS)
    row["instruct_version"] = INSTRUCT_VERSION.get(hub_id, {"exists": True, "note": "is the instruct model"})
    row["greeting_gate"] = greeting_scores(gate_models, result["model"]["slug"])
    row["status"] = "dropped" if hub_id in DROPPED else "candidate"
    if hub_id in DROPPED:
        row["dropped_reason"] = DROPPED[hub_id]
    return row


def build_selection(results: list[dict], previous: dict, gate: dict) -> dict:
    """results/model_selection.json from the baseline files, the previous output and results/greeting_gate.json."""
    return {
        "chosen": previous.get("chosen"),
        "chosen_size": previous.get("chosen_size"),
        "license": previous.get("license"),
        "hf_license": previous.get("hf_license"),
        "greeting_gate": {"source": str(GREETING_GATE), "judge": gate.get("judge"), "rubric": gate.get("rubric")}
        if gate else "pending",
        "candidates": [candidate(r, gate.get("models", {})) for r in results],
    }


def main(fetch: bool) -> None:
    if fetch:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        for f in list_repo_files(REPO):
            if f.startswith("results/baseline/") and f.endswith(".json"):
                shutil.copy(hf_hub_download(REPO, f), BASELINE_DIR / Path(f).name)
    results = [json.loads(p.read_text()) for p in sorted(BASELINE_DIR.glob("*.json"))]
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    gate = json.loads(GREETING_GATE.read_text()) if GREETING_GATE.exists() else {}
    selection = build_selection(results, previous, gate)
    OUT.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} with {len(results)} candidates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    main(parser.parse_args().fetch)
