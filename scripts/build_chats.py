"""Messenger export -> anonymised Malagasy conversations (JSONL) + an HTML review sheet.

Runs on the owner's laptop only. Prints counts, never text. Also writes leftover_names.html
next to the review sheet: capitalised words that survived anonymising, with counts. Tick the
real names, save blocklist.txt, and run again with --blocklist to replace them everywhere.

    uv run python scripts/build_chats.py --export ~/malagasy-data/raw --owner "<name>" \
        --out ~/malagasy-data/private/chats.jsonl --sheet ~/malagasy-data/private/review.html
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fym1.data.leftovers import read_blocklist, render_leftover_sheet
from fym1.data.messenger import load_threads
from fym1.data.pipeline import build_records, merge_data_stats
from fym1.data.review_sheet import render_review_sheet, sample_for_review

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True, help="unzipped Messenger export folder")
    parser.add_argument("--owner", required=True, help="owner's name exactly as Messenger shows it")
    parser.add_argument("--out", type=Path, required=True, help="JSONL of anonymised conversations")
    parser.add_argument("--sheet", type=Path, required=True, help="HTML review sheet")
    parser.add_argument("--blocklist", type=Path, help="blocklist.txt saved from leftover_names.html")
    parser.add_argument("--sample", type=int, default=100, help="conversations on the review sheet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats", type=Path, default=REPO / "results/data_stats.json", help="counts only")
    return parser.parse_args()


def private_path(path: Path) -> Path:
    """`path` resolved, refusing anywhere inside a git work tree (chat text must never be committable)."""
    resolved = path.expanduser().resolve()
    existing = next(p for p in (resolved, *resolved.parents) if p.exists())
    if not existing.is_dir():
        existing = existing.parent
    toplevel = subprocess.run(
        ["git", "-C", str(existing), "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if toplevel.returncode == 0:
        sys.exit(
            f"{path} is inside the git work tree {toplevel.stdout.strip()}. "
            "Write chat data outside any repo, e.g. ~/malagasy-data/private/."
        )
    return resolved


def main() -> None:
    args = parse_args()
    out, sheet = private_path(args.out), private_path(args.sheet)
    leftover_sheet = private_path(sheet.with_name("leftover_names.html"))
    blocklist = read_blocklist(args.blocklist.expanduser()) if args.blocklist else []
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

    records, common_word_replacements, leftovers = build_records(
        threads, owner=args.owner, is_malagasy=is_malagasy, blocklist=blocklist
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    sample = sample_for_review(records, n=args.sample, seed=args.seed)
    sheet.write_text(render_review_sheet(sample), encoding="utf-8")
    leftover_sheet.write_text(render_leftover_sheet(leftovers), encoding="utf-8")

    existing = json.loads(args.stats.read_text()) if args.stats.exists() else {}
    stats = merge_data_stats(existing, threads=len(threads), conversations=len(records))
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")

    print(json.dumps({
        "threads": len(threads),
        "messages": sum(len(t.messages) for t in threads),
        "conversations": len(records),
        "review_sample": len(sample),
        "common_word_replacements": common_word_replacements,
        "unharvested_nickname_events": sum(t.unharvested_nickname_events for t in threads),
        "blocklist": len(blocklist),
        "leftover_names": len(leftovers),
    }))  # fmt: skip


if __name__ == "__main__":
    main()
