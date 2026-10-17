"""Assemble results/model_selection.json from the per-model baseline files.

Usage: python scripts/build_model_selection.py [--fetch]
--fetch first downloads results/baseline/*.json from the private Fan1000-Prog/fyM1-ckpt repo.
`chosen`/`chosen_size`/`license`/`hf_license` stay as already recorded in the output file (the lead fills them in).
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
LICENSES = {
    "google/gemma-4-12B-it": "Apache-2.0",
    "McGill-NLP/AfriqueQwen-8B": "CC-BY-4.0",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit": "Gemma Terms of Use",
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


def candidate(result: dict) -> dict:
    hub_id = result["model"]["hub_id"]
    row = selection_row(result, LICENSES[hub_id], base_model=hub_id in BASE_MODELS)
    row["instruct_version"] = INSTRUCT_VERSION.get(hub_id, {"exists": True, "note": "is the instruct model"})
    return row


def main(fetch: bool) -> None:
    if fetch:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        for f in list_repo_files(REPO):
            if f.startswith("results/baseline/") and f.endswith(".json"):
                shutil.copy(hf_hub_download(REPO, f), BASELINE_DIR / Path(f).name)
    results = [json.loads(p.read_text()) for p in sorted(BASELINE_DIR.glob("*.json"))]
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    selection = {
        "chosen": previous.get("chosen"),
        "chosen_size": previous.get("chosen_size"),
        "license": previous.get("license"),
        "hf_license": previous.get("hf_license"),
        "candidates": [candidate(r) for r in results],
    }
    OUT.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} with {len(results)} candidates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    main(parser.parse_args().fetch)
