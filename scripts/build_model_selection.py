"""Assemble results/model_selection.json from the per-model baseline files.

Usage: python scripts/build_model_selection.py [--fetch]
--fetch first downloads results/baseline/*.json from the private Fan1000-Prog/fyM1-ckpt repo.
`chosen`/`chosen_size` stay as already recorded in the output file (the lead fills them in).
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
}


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
        "candidates": [selection_row(r, LICENSES[r["model"]["hub_id"]]) for r in results],
    }
    OUT.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} with {len(results)} candidates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    main(parser.parse_args().fetch)
