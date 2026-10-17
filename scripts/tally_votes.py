"""Tally arena votes into results/eval/arena.json.

    uv run python scripts/tally_votes.py                       # downloads the private votes dataset
    uv run python scripts/tally_votes.py --votes-dir some/dir  # or reads local *.jsonl vote files

Downloading needs a Hugging Face login (`hf auth login`) or HF_TOKEN with read access.
"""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from fym1.eval.arena import tally

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "spaces" / "arena"))
from votes import load_pairs  # noqa: E402  (Space module, shipped as a flat file)

VOTES_REPO = "Fan1000-Prog/fyM1-votes"


def read_votes(votes_dir: Path) -> list[dict]:
    files = sorted(votes_dir.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"no *.jsonl vote files under {votes_dir}")
    return [json.loads(line) for path in files for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--votes-dir", type=Path, help=f"local vote files (default: download {VOTES_REPO})")
    parser.add_argument("--pairs", type=Path, default=REPO / "spaces" / "arena" / "pairs.jsonl")
    parser.add_argument("--out", type=Path, default=REPO / "results" / "eval" / "arena.json")
    args = parser.parse_args()

    votes_dir = args.votes_dir or Path(snapshot_download(VOTES_REPO, repo_type="dataset"))
    pairs = load_pairs(args.pairs)
    result = {
        "votes_source": "local files" if args.votes_dir else VOTES_REPO,  # no local paths in public results
        "pairs_file": args.pairs.name,
        **tally(read_votes(votes_dir), {pair["pair_id"]: pair["a_is"] for pair in pairs}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {result['votes']} votes, fym1 win rate {result['fym1_win_rate']}")


if __name__ == "__main__":
    main()
