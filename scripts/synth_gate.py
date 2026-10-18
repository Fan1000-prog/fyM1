"""Strict automatic sample gate on the filtered synthetic chats in fyM1-chats. Counts and chat ids only, never text.

    uv run python scripts/synth_gate.py
    uv run python scripts/synth_gate.py --revision <fyM1-chats commit> --export ~/malagasy-data/raw

Downloads `synth/filtered.ndjson` at --revision (the Hugging Face token comes from the local cache, never an
argument), reads every sender, participant and message of the raw export in --export (local only), and runs
`fym1.data.synth_gate` on every chat. Writes the counts report (with dropped chat ids) to --results. When any
chat is dropped, writes the kept ids, one per line, to <--private-out>/kept_ids.txt (outside every git work
tree) so the trainer promotes only those. Never uploads and never promotes.
Exit 0: pass. Exit 1: the gate failed (an 8-word match with the raw export, or nothing kept). Exit 2: refused.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download

from fym1 import release
from fym1.data.private import private_path
from fym1.data.synth_gate import SAMPLE_SEED, SAMPLE_SIZE, raw_reference, run_gate

DATA_REPO = "Fan1000-Prog/fyM1-chats"
FILTERED = "synth/filtered.ndjson"
REVISION = "d0d076f60ec691b2f82ff3c4c5f494db81a4e545"  # the re-filtered set this gate was first run on
KEPT_IDS = "kept_ids.txt"
REPO = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default=REVISION, help="fyM1-chats commit to check")
    parser.add_argument("--export", type=Path, default=Path("~/malagasy-data/raw"), help="raw export (local only)")
    parser.add_argument("--results", type=Path, default=REPO / "results" / "synth_gate.json")
    parser.add_argument("--private-out", type=Path, default=Path("~/malagasy-data/private/synth-gate"),
                        help="folder outside every git work tree for kept_ids.txt")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    return parser.parse_args(argv)


def download_filtered(revision: str) -> Path:
    return Path(hf_hub_download(DATA_REPO, FILTERED, repo_type="dataset", revision=revision))


def read_raw(export: Path) -> tuple[list[str], list[str]]:
    """(every sender and participant, every message text) of the raw export."""
    threads = release.read_export(export.expanduser())
    return release.export_senders(threads), release.export_texts(threads)


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def main(argv: list[str] | None = None, download: Callable[[str], Path] = download_filtered,
         load_raw: Callable[[Path], tuple[list[str], list[str]]] = read_raw,
         is_malagasy: Callable[[str], bool] | None = None) -> int:  # fmt: skip
    args = parse_args(argv)
    if is_malagasy is None:
        from fym1.eval.lang import is_malagasy  # loads GlotLID; tests pass a stand-in
    private_out = private_path(args.private_out)
    try:
        raw = raw_reference(*load_raw(args.export))
    except release.RefusedError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    path = download(args.revision)
    records = read_records(path)
    result = run_gate(records, raw, is_malagasy, args.sample_size, args.seed)
    report = {"dataset": DATA_REPO, "file": FILTERED, "revision": args.revision,
              "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **result.report}  # fmt: skip
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result.report["dropped"]:
        private_out.mkdir(parents=True, exist_ok=True)
        (private_out / KEPT_IDS).write_text("".join(f"{i}\n" for i in result.kept_ids), encoding="utf-8")
    print(f"checked {report['checked']}, kept {report['kept']}, dropped {report['dropped']}")
    print("dropped by reason: " + ", ".join(f"{k} {v}" for k, v in report["dropped_by_reason"].items()))
    print(f"report: {args.results}" + (f"; kept ids: {private_out / KEPT_IDS}" if report["dropped"] else ""))
    print(report["status"].upper())
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
