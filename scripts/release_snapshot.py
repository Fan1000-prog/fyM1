"""Make the public release snapshot of a clean commit and check it (counts only).

    uv run python scripts/release_snapshot.py --export ~/malagasy-data/raw --out ../fy-m1-public \
        [--ref origin/main] [--results results/release_check.json]

Copies the commit's tracked files (no git history) into --out, outside every worktree, minus the private paths
in fym1.release.EXCLUDES, and adds a NOTICE for the third-party word lists. Sender names are read from the
export at run time, removed from every word/name list and searched for in every file, whole word,
case/accent-insensitive; then a secret scan runs. Writes RELEASE_CHECK.json into --out (and to --results).
Prints and writes counts and repo paths only, never a name, message text or secret value.
Exit 0: clean. Exit 1: name hits, secret hits or an unreadable file. Exit 2: refused, nothing copied.
"""

import argparse
import sys
from pathlib import Path

from fym1.release import REPORT_NAME, RefusedError, build_snapshot, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="raw export folder or file (local only)")
    parser.add_argument("--out", type=Path, required=True, help="empty or new folder outside the repo")
    parser.add_argument("--ref", default="HEAD", help="commit to snapshot (default HEAD)")
    parser.add_argument("--repo", type=Path, default=Path("."), help="repo to snapshot (default: here)")
    parser.add_argument("--results", type=Path, help="also write the report here, e.g. results/release_check.json")
    args = parser.parse_args(argv)
    try:
        report = build_snapshot(args.repo, args.ref, args.out.expanduser(), args.export.expanduser())
    except RefusedError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    paths = [args.out.expanduser() / REPORT_NAME, *([args.results] if args.results else [])]
    write_report(report, *paths)
    summary = report.as_dict()
    print(f"ref {report.ref[:9]}: {report.files_copied} files copied, {len(report.excluded)} excluded")
    print(f"name tokens {summary['names']['name_tokens']}, list entries removed "
          f"{summary['names']['list_entries_removed_total']}, name hits {summary['names']['hits_total']} "
          f"in {len(report.name_hits)} files, unscannable {len(report.unscannable)}")
    print(f"secret hits {summary['secrets']['hits_total']} in {len(report.secret_hits)} files")
    print(f"report: {', '.join(map(str, paths))}")
    print("OK" if report.ok else "FAILED: see the report's per-file counts")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
