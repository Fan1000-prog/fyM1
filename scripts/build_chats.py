"""Messenger export -> anonymised Malagasy conversations (JSONL) + an HTML review sheet.

Runs on the owner's laptop only. Prints counts, never text.

    uv run python scripts/build_chats.py --export ~/malagasy-data/raw --owner "<name>" \
        --out ~/malagasy-data/private/chats.jsonl --sheet ~/malagasy-data/private/review.html
"""

import argparse
import json
import sys
from pathlib import Path

from fym1.data.messenger import load_threads
from fym1.data.pipeline import build_records
from fym1.data.review_sheet import render_review_sheet, sample_for_review

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True, help="unzipped Messenger export folder")
    parser.add_argument("--owner", required=True, help="owner's name exactly as Messenger shows it")
    parser.add_argument("--out", type=Path, required=True, help="JSONL of anonymised conversations")
    parser.add_argument("--sheet", type=Path, required=True, help="HTML review sheet")
    parser.add_argument("--sample", type=int, default=100, help="conversations on the review sheet")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def private_path(path: Path) -> Path:
    """`path` resolved, refusing anywhere inside this git repo (chat text must never be committable)."""
    resolved = path.expanduser().resolve()
    if resolved.is_relative_to(REPO):
        sys.exit(f"{path} is inside the repo {REPO}. Write chat data outside it, e.g. ~/malagasy-data/private/.")
    return resolved


def main() -> None:
    args = parse_args()
    out, sheet = private_path(args.out), private_path(args.sheet)
    export = args.export.expanduser()
    threads = load_threads(export)
    if not threads:
        sys.exit(f"No Messenger threads found under {export}. Point --export at the unzipped export folder.")
    if not any(args.owner in t.participants for t in threads):
        sys.exit(f"--owner {args.owner!r} is not a participant of any thread. Use the name exactly as Messenger shows it.")

    try:
        from fym1.eval.lang import is_malagasy  # loads the language-ID model; only needed here
    except ModuleNotFoundError as error:
        if error.name != "fym1.eval.lang":
            raise
        sys.exit("fym1.eval.lang is missing: merge Task 3 (eval metrics) before running this.")

    records, common_word_replacements = build_records(threads, owner=args.owner, is_malagasy=is_malagasy)

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    sample = sample_for_review(records, n=args.sample, seed=args.seed)
    sheet.write_text(render_review_sheet(sample), encoding="utf-8")

    print(json.dumps({
        "threads": len(threads),
        "messages": sum(len(t.messages) for t in threads),
        "conversations": len(records),
        "review_sample": len(sample),
        "common_word_replacements": common_word_replacements,
    }))  # fmt: skip


if __name__ == "__main__":
    main()
