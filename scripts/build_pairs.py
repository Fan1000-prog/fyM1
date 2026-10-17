"""Build the arena's blind pairs from the eval answer files.

    uv run python scripts/build_pairs.py --base results/eval/answers_base.jsonl \
        --fym1 results/eval/answers_fym1.jsonl --out spaces/arena/pairs.jsonl --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from fym1.eval.arena import build_pairs


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="answers_base.jsonl")
    parser.add_argument("--fym1", type=Path, required=True, help="answers_fym1.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs, skipped = build_pairs(read_jsonl(args.base), read_jsonl(args.fym1), random.Random(args.seed))
    if not pairs:
        raise SystemExit("no id appears in both answer files; nothing to write")
    args.out.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")
    print(f"wrote {len(pairs)} pairs to {args.out}; skipped {len(skipped)} ids missing on one side: {skipped}")


if __name__ == "__main__":
    main()
