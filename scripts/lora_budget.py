"""Write results/lora_budget.json: the size of one LoRA adapter next to the frozen matrix it adapts.

    uv run python scripts/lora_budget.py [--model ORG/NAME] [--out results/lora_budget.json]

The paper's QLoRA figure shows, for one MLP up-projection W0 (d x k), how many weights full fine-tuning would
train (d*k) and how many a rank-r adapter trains (r*(d+k)). d and k come from the base model's public
config.json (downloaded at the revision recorded in the output); r and alpha are the values 01_train uses,
read from scripts/build_notebooks.py so the figure cannot drift from the training settings. Only model
dimensions and settings are written, never data. The total trainable count of the real run is not computed
here: Unsloth picks the adapted modules of the hybrid attention layers itself, and 01_train does not log it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER = REPO_ROOT / "scripts" / "build_notebooks.py"
OUT = REPO_ROOT / "results" / "lora_budget.json"
SELECTION = REPO_ROOT / "results" / "model_selection.json"
LORA_LINE = re.compile(r"^LORA_R, LORA_ALPHA = (\d+), (\d+)\s*$", re.M)


def training_lora(builder_source: str) -> tuple[int, int]:
    """(r, alpha) from the 01_train settings cell in scripts/build_notebooks.py."""
    found = LORA_LINE.findall(builder_source)
    if len(found) != 1:
        raise ValueError(f"expected one 'LORA_R, LORA_ALPHA = r, alpha' line, found {len(found)}")
    r, alpha = found[0]
    return int(r), int(alpha)


def budget(config: dict[str, Any], r: int, alpha: int) -> dict[str, Any]:
    """Weights of one MLP up-projection (d = intermediate_size rows, k = hidden_size columns) and its adapter."""
    text = config.get("text_config", config)
    d, k = text["intermediate_size"], text["hidden_size"]
    if min(d, k, r) <= 0:
        raise ValueError("sizes and rank must be positive")
    full, adapter = d * k, r * (d + k)
    return {
        "hidden_size": k,
        "intermediate_size": d,
        "num_hidden_layers": text["num_hidden_layers"],
        "lora_r": r,
        "lora_alpha": alpha,
        "scale": alpha / r,
        "example": {"matrix": "mlp.up_proj", "d": d, "k": k, "full_params": full, "lora_params": adapter,
                    "lora_share": adapter / full},
    }


def main(argv: list[str] | None = None) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=json.loads(SELECTION.read_text())["chosen"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    revision = HfApi().model_info(args.model).sha
    config = json.loads(Path(hf_hub_download(args.model, "config.json", revision=revision)).read_text())
    r, alpha = training_lora(BUILDER.read_text(encoding="utf-8"))
    report = {"model": args.model, "revision": revision, "rank_source": "scripts/build_notebooks.py (01_train)",
              **budget(config, r, alpha)}
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    ex = report["example"]
    print(f"wrote {args.out.name}: {ex['d']}x{ex['k']} -> {ex['lora_params']:,} of {ex['full_params']:,} weights")
    return 0


if __name__ == "__main__":
    sys.exit(main())
