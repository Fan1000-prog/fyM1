"""Fetch a training run's log from the private fyM1-ckpt repo into results/train_log.json (counts and losses only).

Usage: uv run python scripts/fetch_train_log.py [--run-id fym1-9b-a100-r1] [--out results/train_log.json]

notebooks/01_train.ipynb writes `<RUN_ID>/train_log.json`; each `<RUN_ID>/checkpoint-<step>/` holds the Trainer's
`trainer_state.json`. This copies the keys paper/make_numbers.py and paper/make_figures.py read (gpu, steps, epochs,
wall_time_s, final/best losses, config numbers, data.train, loss_curve, eval_curve), adds `sec_per_step` (wall_time_s /
steps: wall clock, so it includes model loading and evals) and the newest checkpoint's step counts. Keys are copied by
an allow-list, so file names, data revisions and anything else in the log never reach git.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from fym1.train.resume import latest_checkpoint

REPO = "Fan1000-Prog/fyM1-ckpt"
RUN_ID = "fym1-9b-a100-r1"
OUT = Path(__file__).resolve().parents[1] / "results" / "train_log.json"

TOP_KEYS = ("run_id", "model", "gpu", "steps", "epochs", "wall_time_s", "sessions", "final_train_loss",
            "best_eval_loss", "best_step", "stopped_early")
CONFIG_KEYS = ("lora_r", "lora_alpha", "lr", "lr_scheduler", "warmup_steps", "max_seq_len", "batch", "grad_accum",
               "effective_batch")
DATA_KEYS = ("examples", "train", "eval", "tokens", "over_max_len")
STATE_KEYS = ("global_step", "max_steps", "epoch", "num_train_epochs")


def fetch_json(path: str) -> dict:
    return json.loads(Path(hf_hub_download(REPO, path, force_download=True)).read_text())


def list_files() -> list[str]:
    return list_repo_files(REPO)


def _pick(source: dict, keys: tuple[str, ...]) -> dict:
    return {k: source[k] for k in keys if k in source}


def _number(value, what: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{what} must be a number, got {value!r}")
    return value


def summary(log: dict, state: dict, checkpoint: str) -> dict:
    """results/train_log.json content from the Hub's train_log.json and the newest checkpoint's trainer_state.json."""
    out = _pick(log, TOP_KEYS)
    for key in ("steps", "wall_time_s"):
        _number(out.get(key), key)
    if out["steps"] <= 0:
        raise ValueError(f"steps must be positive, got {out['steps']}")
    out["sec_per_step"] = round(out["wall_time_s"] / out["steps"], 1)
    config = log.get("config") or {}
    out["config"] = _pick(config, CONFIG_KEYS)
    dtype = (config.get("hardware") or {}).get("compute_dtype")
    if dtype is not None:
        out["config"]["compute_dtype"] = dtype
    out["data"] = _pick(log.get("data") or {}, DATA_KEYS)
    out["loss_curve"] = [{"step": _number(p["step"], "loss_curve step"), "loss": _number(p["loss"], "loss")}
                         for p in log.get("loss_curve") or []]
    out["eval_curve"] = [{"step": _number(p["step"], "eval_curve step"),
                          "eval_loss": _number(p["eval_loss"], "eval_loss")}
                         for p in log.get("eval_curve") or []]
    out["checkpoint"] = {"name": checkpoint.rsplit("/", 1)[-1]} | _pick(state, STATE_KEYS)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    checkpoint = latest_checkpoint(REPO, list_files(), run=args.run_id)
    if checkpoint is None:
        print(f"no {args.run_id}/checkpoint-* in {REPO}", file=sys.stderr)
        return 1
    try:
        log, state = fetch_json(f"{args.run_id}/train_log.json"), fetch_json(f"{checkpoint}/trainer_state.json")
        result = summary(log, state, checkpoint)
    except (KeyError, ValueError) as error:
        print(f"refused {args.run_id}/train_log.json: {error}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out.name}: {result['steps']} steps, {result['sec_per_step']} s/step, "
          f"best eval loss {result.get('best_eval_loss')} at step {result.get('best_step')}, {checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
