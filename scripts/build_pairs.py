"""Build the arena's blind pairs and their answer key from the eval answer files.

    uv run python scripts/build_pairs.py --base results/eval/answers_base.jsonl \
        --fym1 results/eval/answers_fym1.jsonl --out spaces/arena/pairs.jsonl --seed 42 --upload-key

`--out` is public (it ships in the Space). The answer key (which model is A) is written to `--key-out`
(git-ignored) and, with `--upload-key`, to the private votes dataset, the only place it may live online.
"""

import argparse
import json
import random
from pathlib import Path

from huggingface_hub import HfApi

from fym1.eval.arena import ANSWER_KEY_FILE, VOTES_REPO, build_pairs

REPO = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def upload_key(key_path: Path) -> None:
    api = HfApi()
    api.create_repo(VOTES_REPO, repo_type="dataset", private=True, exist_ok=True)
    if not api.repo_info(VOTES_REPO, repo_type="dataset").private:
        raise SystemExit(f"{VOTES_REPO} is public; refusing to upload the answer key there")
    api.upload_file(
        path_or_fileobj=key_path,
        path_in_repo=ANSWER_KEY_FILE,
        repo_id=VOTES_REPO,
        repo_type="dataset",
        commit_message="Update arena answer key",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="answers_base.jsonl")
    parser.add_argument("--fym1", type=Path, required=True, help="answers_fym1.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="public pairs file, e.g. spaces/arena/pairs.jsonl")
    parser.add_argument("--key-out", type=Path, default=REPO / ANSWER_KEY_FILE, help="private answer key")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--upload-key", action="store_true", help=f"also upload the key to {VOTES_REPO}")
    args = parser.parse_args()

    pairs, key, skipped = build_pairs(read_jsonl(args.base), read_jsonl(args.fym1), random.Random(args.seed))
    if not pairs:
        raise SystemExit("no id appears in both answer files; nothing to write")
    write_jsonl(args.out, pairs)
    write_jsonl(args.key_out, key)
    print(f"wrote {len(pairs)} pairs to {args.out} and the key to {args.key_out}")
    print(f"skipped {len(skipped)} ids missing on one side: {skipped}")
    if args.upload_key:
        upload_key(args.key_out)
        print(f"uploaded the key to {VOTES_REPO}/{ANSWER_KEY_FILE}")


if __name__ == "__main__":
    main()
