"""Push a notebook to Kaggle as a private GPU kernel (T4) and start a run.

Usage: python scripts/kaggle_push.py {00_baseline,01_train} [--user KAGGLE_USERNAME]
Builds a folder with the notebook and its kernel-metadata.json and runs the pinned Kaggle CLI
`kernels push`. The username (not a secret) comes from --user or KAGGLE_USERNAME. Kaggle credentials
are left to the CLI (its default access token); this script never uses or prints them. The HF_TOKEN
secret is attached once by the owner in the Kaggle UI (see docs/kaggle_operator.md).
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_CLI = ["uvx", "--from", "kaggle==2.2.4", "kaggle"]
NOTEBOOKS = {
    "00_baseline": ("fym1-00-baseline", "fyM1 00 baseline"),
    "01_train": ("fym1-01-train", "fyM1 01 train"),
}
# T4 (Turing): no bf16, so Unsloth's float32 forcing applies as on Colab. P100 (Pascal) is older still.
MACHINE_SHAPE = "NvidiaTeslaT4"


def kernel_metadata(username: str, notebook: str) -> dict:
    slug, title = NOTEBOOKS[notebook]
    return {
        "id": f"{username}/{slug}",
        "title": title,
        "code_file": f"{notebook}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": MACHINE_SHAPE,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }


def build_folder(folder: Path, metadata: dict, notebook_path: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(notebook_path, folder / metadata["code_file"])
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return folder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", choices=sorted(NOTEBOOKS))
    parser.add_argument("--user", default=os.environ.get("KAGGLE_USERNAME"),
                        help="Kaggle username (default: $KAGGLE_USERNAME)")
    args = parser.parse_args(argv)
    if not args.user:
        parser.error("give --user or set KAGGLE_USERNAME")
    metadata = kernel_metadata(args.user, args.notebook)
    with tempfile.TemporaryDirectory() as tmp:
        folder = build_folder(Path(tmp), metadata, REPO_ROOT / "notebooks" / f"{args.notebook}.ipynb")
        subprocess.run([*KAGGLE_CLI, "kernels", "push", "-p", str(folder)], check=True)
    print(f"pushed {metadata['id']}; poll with: {' '.join(KAGGLE_CLI)} kernels status {metadata['id']}")


if __name__ == "__main__":
    main()
