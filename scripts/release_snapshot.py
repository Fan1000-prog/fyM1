"""Make the public release snapshot of a clean commit and check it (counts only).

    uv run python scripts/release_snapshot.py --export ~/malagasy-data/raw --out ../fy-m1-public \
        --results results/release_check.json --private-ids /path/outside/the/repo/private_ids.txt \
        --chats ~/malagasy-data/private/<run>/chats.jsonl [--ref origin/main]

Copies the commit's tracked files (no git history) into --out, outside every worktree, minus the private paths
in fym1.release.EXCLUDES (including the word lists scripts/build_known_words.py writes), and adds a NOTICE for
the third-party word lists. Sender names are read from the export at run time, removed from the hand-written
name lists and searched for in every file but the lists a script builds from public data alone, as the strict
gate reads words (case changes, spaced-out letters, digits typed for letters, hidden characters, escapes). Any
file or path sharing an 8-word run with the export's messages or the turns of the anonymised chats file (--chats:
fake names, joined messages), runs across message boundaries included, or holding a whole shorter message with 4+
distinctive words, fails (copied chat text, counted per file, never shown; words compared as the name scan
compares them); then a secret scan runs. The release ships text files only (known text types, decoded in full,
no control characters, base64 data: URIs or Unicode tag characters): any other file (image, PDF, binary data hidden
after a text file) and any archive (zip, gz, tar, whl, ...) is refused before anything is written. The paper ships
as paper/fym1.tex; its PDF is built from it. Every file under --export must be a thread page (message_*.json) or a
one-file JSON export, at any depth; any other file (hidden ones such as .DS_Store too) is refused (counted, never
named). Every key of every export file must be in its format's schema (fym1.release.DYI_SCHEMA / FLAT_SCHEMA:
each key a name, free text, or data such as ids and timestamps); any other key refuses the export (counted, never
named). Names, nicknames and the distinctive words of titles and notes join the name list; every free-text field
is checked as chat text too. Writes the counts report to
--results only (never into --out: the report is a private build record).
--private-ids names a file outside the repo (one private account id per line); any whole-word,
case-insensitive match in a snapshot file or path fails the check, counted per file, never shown.
Prints and writes counts and repo paths only, never a name, message text or secret value.
Exit 0: clean, snapshot in --out.
Exit 1: name hits, private-id hits, copied chat text or secret hits; nothing written to --out.
Exit 2: refused before anything was written (a non-text file, an archive, or a missing, unreadable or empty
--chats file, among others).
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
    parser.add_argument("--chats", type=Path, required=True,
                        help="the private anonymised chats.jsonl, outside the repo (never printed)")
    args = parser.parse_args(argv)
    out, results = args.out.expanduser().resolve(), args.results.expanduser().resolve()
    try:
        if results == out or out in results.parents:
            raise RefusedError("--results must be outside --out (the report never ships)")
        report = build_snapshot(args.repo, args.ref, out, args.export.expanduser(), args.private_ids,
                                args.chats)
    except RefusedError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    write_report(report, results)
    summary = report.as_dict()
    print(f"ref {report.ref[:9]}: {report.files_copied} files copied, {len(report.excluded)} excluded")
    print(f"name tokens {summary['names']['name_tokens']}, list entries removed "
          f"{summary['names']['list_entries_removed_total']}, name hits {summary['names']['hits_total']} "
          f"in {len(report.name_hits)} files")
    print(f"export schema keys {report.export_schema_keys}, title words left out as common "
          f"{report.label_words_common}")
    print(f"private ids {report.private_ids}, hits {summary['private_ids']['hits_total']} "
          f"in {len(report.private_id_hits)} files")
    print(f"chat messages {report.chat_messages}, anonymised turns {report.chat_turns}, short messages "
          f"{report.chat_short_messages}, free-text fields {report.chat_free_texts}, files with copied chat text {summary['chat_text']['files_total']}")
    print(f"secret hits {summary['secrets']['hits_total']} in {len(report.secret_hits)} files")
    print(f"report: {results}")
    print("OK" if report.ok else "FAILED: see the report's per-file counts; nothing was written to --out")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
