"""Tally arena votes into results/eval/arena.json.

    uv run python scripts/tally_votes.py                     # votes + answer key from the private dataset
    uv run python scripts/tally_votes.py --votes-dir DIR --key pairs_key.jsonl   # or local files

Downloading needs a Hugging Face login (`hf auth login`) or HF_TOKEN with read access.
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from fym1.eval.arena import ANSWER_KEY_FILE, VOTES_REPO, a_is_by_pair, tally

REPO = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_votes(votes_dir: Path) -> list[dict]:
    files = sorted(votes_dir.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl vote files under {votes_dir}")
    return [vote for path in files for vote in read_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--votes-dir", type=Path, help=f"local vote files (default: download {VOTES_REPO})")
    parser.add_argument("--key", type=Path, help=f"local answer key (default: download from {VOTES_REPO})")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "eval" / "arena.json")
    args = parser.parse_args()

    if args.votes_dir:
        votes_dir = args.votes_dir
    else:
        votes_dir = Path(snapshot_download(VOTES_REPO, repo_type="dataset", allow_patterns="votes/*")) / "votes"
    key_path = args.key or Path(hf_hub_download(VOTES_REPO, ANSWER_KEY_FILE, repo_type="dataset"))
    result = {
        "votes_source": "local files" if args.votes_dir else VOTES_REPO,  # no local paths in public results
        **tally(read_votes(votes_dir), a_is_by_pair(read_jsonl(key_path))),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {result['votes']} votes, fym1 win rate {result['fym1_win_rate']}")


if __name__ == "__main__":
    main()
