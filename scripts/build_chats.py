"""Messenger export -> anonymised Malagasy conversations (JSONL), through a strict automatic gate.

Runs on the owner's laptop only. Prints counts, never text. After anonymising, the strict gate
(src/fym1/data/gate.py) replaces every unknown capitalised word with a fake name and drops every
conversation with a 7+ digit number that is not a price, an email/link/@handle, or more than 30%
of its words replaced. Nobody reads the text. The gate counts
(counts only) go to data_gate.json next to --out; --gate-results also copies them into the repo.

--sheet (a 100-sample review page) and --leftover-sheet (leftover capitalised words with counts)
are optional extras, off by default. --blocklist still replaces listed words everywhere.

--export takes a "Download your information" folder, a folder of one-file JSON exports, or one
such file. --dry-run runs everything, prints segmenting and gate counts, and writes nothing.

    uv run python scripts/build_chats.py --export ~/malagasy-data/raw --all-senders \
        --out ~/malagasy-data/private/chats.jsonl --gate-results results/data_gate.json
    uv run python scripts/build_chats.py --export ~/malagasy-data/raw --all-senders --dry-run
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from fym1.data.leftovers import read_blocklist, render_leftover_sheet
from fym1.data.messenger import Thread, load_threads
from fym1.data.pipeline import build_records, merge_data_stats
from fym1.data.review_sheet import render_review_sheet, sample_for_review
from fym1.data.segment import count_sessions, to_conversations

REPO = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True, help="unzipped export folder, or one export file")
    speaker = parser.add_mutually_exclusive_group(required=True)
    speaker.add_argument("--owner", help="owner's name exactly as Messenger shows it")
    speaker.add_argument(
        "--all-senders", action="store_true", help="every sender takes the assistant side in turn (all consented)"
    )
    parser.add_argument("--out", type=Path, help="JSONL of anonymised conversations that passed the gate")
    parser.add_argument("--dry-run", action="store_true", help="print counts only; write nothing")
    parser.add_argument("--gate-results", type=Path, help="also copy the counts-only gate JSON here (results/)")
    parser.add_argument("--sheet", type=Path, help="optional HTML review sheet of gated conversations")
    parser.add_argument("--leftover-sheet", type=Path, help="optional HTML list of leftover words with counts")
    parser.add_argument("--blocklist", type=Path, help="words to replace as names everywhere, one per line")
    parser.add_argument("--sample", type=int, default=100, help="conversations on the review sheet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stats", type=Path, default=REPO / "results/data_stats.json", help="counts only")
    args = parser.parse_args()
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run")
    return args


def segment_counts(threads: list[Thread], owner: str | None) -> dict[str, int]:
    """Counts after time-gap segmenting, before anonymising and the Malagasy filter. No text.

    Exact-duplicate conversations count once, as in `build_records`.
    """
    conversations = {json.dumps(c, sort_keys=True) for t in threads for c in to_conversations(t, owner)}
    return {
        "threads": len(threads),
        "messages": sum(len(t.messages) for t in threads),
        "sessions": sum(count_sessions(t) for t in threads),
        "conversations": len(conversations),
    }


def load_is_malagasy() -> Callable[[str], bool]:
    """The language filter; loads the language-ID model, so only called when building."""
    try:
        from fym1.eval.lang import is_malagasy
    except ModuleNotFoundError as error:
        if error.name != "fym1.eval.lang":
            raise
        sys.exit("fym1.eval.lang is missing: merge Task 3 (eval metrics) before running this.")
    return is_malagasy


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
    export = args.export.expanduser()
    threads = load_threads(export)
    if not threads:
        sys.exit(f"No Messenger threads found under {export}. Point --export at the unzipped export folder.")
    if args.owner is not None and not any(args.owner in t.participants for t in threads):
        sys.exit(f"--owner {args.owner!r} is not a participant of any thread. Use the name exactly as Messenger shows it.")
    blocklist = read_blocklist(args.blocklist.expanduser()) if args.blocklist else []
    out = sheet = leftover_sheet = None
    if not args.dry_run:  # refuse a repo path before the slow build, not after
        out = private_path(args.out)
        sheet = private_path(args.sheet) if args.sheet else None
        leftover_sheet = private_path(args.leftover_sheet) if args.leftover_sheet else None

    result = build_records(threads, owner=args.owner, is_malagasy=load_is_malagasy(), blocklist=blocklist)
    counts = {
        **segment_counts(threads, args.owner),
        **result.gate,
        "common_word_replacements": result.common_word_replacements,
        "unharvested_nickname_events": sum(t.unharvested_nickname_events for t in threads),
        "blocklist": len(blocklist),
        "leftover_names": len(result.leftover_names),
    }
    if args.dry_run:
        print(json.dumps(counts))
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result.records), encoding="utf-8")
    gate_json = json.dumps(result.gate, indent=2) + "\n"
    out.with_name("data_gate.json").write_text(gate_json, encoding="utf-8")
    if args.gate_results:
        args.gate_results.write_text(gate_json, encoding="utf-8")
    if sheet:
        sheet.parent.mkdir(parents=True, exist_ok=True)
        sample = sample_for_review(result.records, n=args.sample, seed=args.seed)
        sheet.write_text(render_review_sheet(sample), encoding="utf-8")
    if leftover_sheet:
        leftover_sheet.parent.mkdir(parents=True, exist_ok=True)
        leftover_sheet.write_text(render_leftover_sheet(result.leftover_names), encoding="utf-8")

    existing = json.loads(args.stats.read_text()) if args.stats.exists() else {}
    stats = merge_data_stats(existing, threads=len(threads), conversations=len(result.records))
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
