"""Make the public release snapshot of a clean commit and check it (counts only).

    uv run python scripts/release_snapshot.py --export ~/malagasy-data/raw --out ../fy-m1-public \
        --results results/release_check.json --private-ids /path/outside/the/repo/private_ids.txt [--ref origin/main]

Copies the commit's tracked files (no git history) into --out, outside every worktree, minus the private paths
in fym1.release.EXCLUDES (including the word lists scripts/build_known_words.py writes), and adds a NOTICE for
the third-party word lists. Sender names are read from the export at run time, removed from the hand-written
name lists and searched for in every file but the lists a script builds from public data alone, as the strict
gate reads words (case changes, spaced-out letters, digits typed for letters, hidden characters, escapes); then
a secret scan runs. Only known text types and PDFs (text and metadata) are read; any other file fails, and any
archive (zip, gz, tar, whl, ...) is refused before anything is written. Writes the counts report to --results
only (never into --out: the report is a private build record).
--private-ids names a file outside the repo (one private account id per line); any whole-word,
case-insensitive match in a snapshot file or path fails the check, counted per file, never shown.
Prints and writes counts and repo paths only, never a name, message text or secret value.
Exit 0: clean, snapshot in --out. Exit 1: name hits, secret hits or an unreadable file; nothing written to --out.
Exit 2: refused before anything was written.
"""

import argparse
import sys
from pathlib import Path

from fym1.release import RefusedError, build_snapshot, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="raw export folder or file (local only)")
    parser.add_argument("--out", type=Path, required=True, help="empty or new folder outside the repo")
    parser.add_argument("--ref", default="HEAD", help="commit to snapshot (default HEAD)")
    parser.add_argument("--repo", type=Path, default=Path("."), help="repo to snapshot (default: here)")
    parser.add_argument("--results", type=Path, required=True,
                        help="where the counts report goes, outside --out, e.g. results/release_check.json")
    parser.add_argument("--private-ids", type=Path, required=True,
                        help="file outside the repo, one private id per line (never printed)")
    args = parser.parse_args(argv)
    out, results = args.out.expanduser().resolve(), args.results.expanduser().resolve()
    try:
        if results == out or out in results.parents:
            raise RefusedError("--results must be outside --out (the report never ships)")
        report = build_snapshot(args.repo, args.ref, out, args.export.expanduser(), args.private_ids)
    except RefusedError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    write_report(report, results)
    summary = report.as_dict()
    print(f"ref {report.ref[:9]}: {report.files_copied} files copied, {len(report.excluded)} excluded")
    print(f"name tokens {summary['names']['name_tokens']}, list entries removed "
          f"{summary['names']['list_entries_removed_total']}, name hits {summary['names']['hits_total']} "
          f"in {len(report.name_hits)} files, unscannable {len(report.unscannable)}")
    print(f"private ids {report.private_ids}, hits {summary['private_ids']['hits_total']} "
          f"in {len(report.private_id_hits)} files")
    print(f"secret hits {summary['secrets']['hits_total']} in {len(report.secret_hits)} files")
    print(f"report: {results}")
    print("OK" if report.ok else "FAILED: see the report's per-file counts; nothing was written to --out")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
