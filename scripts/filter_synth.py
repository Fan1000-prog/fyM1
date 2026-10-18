"""Re-filter the raw synthetic chats in fyM1-chats with the current code, on CPU. Counts only, never text.

    uv run python scripts/filter_synth.py --real ~/malagasy-data/private/chats-2026-10-18d/chats.jsonl
    uv run python scripts/filter_synth.py --real ~/malagasy-data/private/chats-2026-10-18d/chats.jsonl --upload

notebooks/04_synth.ipynb filters with the code and the real chats.jsonl it started with. This script redoes that
step after either changes: it downloads every `synth/raw/part-*.ndjson`, runs
`fym1.data.synth.filter_conversations` against the LOCAL real chats in --real (few-shot leak reference), and writes
`filtered.ndjson` plus counts-only `synth_stats.json`. Default is a dry run into --out-dir (outside every git work
tree). --upload writes both to `synth/filtered.ndjson` and `synth/synth_stats.json` in fyM1-chats, in one commit.
Run settings the raw rows do not carry (seed, few_shot_k, target_raw, gpus) are kept from the Hub's current
synth_stats.json when there is one; `data_revision` is a hash of the --real file. The Hugging Face token comes
from the local cache, never an argument.
"""

import argparse
import fnmatch
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import EntryNotFoundError

from fym1.data import synth
from fym1.data.private import private_path

DATA_REPO = "Fan1000-Prog/fyM1-chats"
RAW_GLOB = "synth/raw/part-*.ndjson"
FILTERED, STATS = "synth/filtered.ndjson", "synth/synth_stats.json"
CARRIED_META = ("seed", "few_shot_k", "target_raw", "gpus")  # notebook settings the raw rows do not record
REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("~/malagasy-data/private/synth-refilter")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", type=Path, required=True, help="local real chats.jsonl (leak reference)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="dry-run output folder, outside any repo")
    parser.add_argument("--upload", action="store_true", help=f"write {FILTERED} and {STATS} to {DATA_REPO}")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def download_raw_rows(api: HfApi, revision: str) -> tuple[list[dict], int]:
    """Every raw row in the raw parts at `revision`, sorted by seed index, and the number of parts read."""
    files = sorted(f for f in api.list_repo_files(DATA_REPO, repo_type="dataset", revision=revision)
                   if fnmatch.fnmatch(f, RAW_GLOB))  # fmt: skip
    rows = [row for f in files
            for row in read_jsonl(Path(api.hf_hub_download(DATA_REPO, f, repo_type="dataset", revision=revision)))]  # fmt: skip
    return sorted(rows, key=lambda r: r["index"]), len(files)


def hub_stats(api: HfApi, revision: str) -> dict:
    """The synth_stats.json at `revision`, or {} when there is none yet."""
    try:
        return json.loads(Path(api.hf_hub_download(DATA_REPO, STATS, repo_type="dataset", revision=revision)).read_text())
    except EntryNotFoundError:
        return {}


def code_sha() -> str:
    """The commit this filter code runs from, marked "-dirty" when the work tree has changes."""
    git = ["git", "-C", str(REPO)]
    sha = subprocess.run([*git, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run([*git, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return f"{sha}-dirty" if dirty else sha


def local_revision(real: Path) -> str:
    """The real chats' identity for the stats: a hash of the local file (it may differ from any Hub revision)."""
    return "local-sha256-" + hashlib.sha256(real.expanduser().read_bytes()).hexdigest()[:12]


def bodies(kept: list[dict], stats: dict) -> dict[str, bytes]:
    return {FILTERED: "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept).encode(),
            STATS: (json.dumps(stats, indent=2) + "\n").encode()}  # fmt: skip


def main(argv: list[str] | None = None, api: HfApi | None = None,
         is_malagasy: Callable[[str], bool] | None = None, sha: Callable[[], str] = code_sha) -> int:  # fmt: skip
    args = parse_args(argv)
    api = api or HfApi()
    if is_malagasy is None:
        from fym1.eval.lang import is_malagasy  # loads the language-ID model: only when actually filtering
    out_dir = None if args.upload else private_path(args.out_dir)  # refuse a repo path before downloading

    real = read_jsonl(args.real.expanduser())  # private chat text: in memory only, never printed
    revision = api.dataset_info(DATA_REPO).sha  # one Hub revision for every file read
    rows, parts = download_raw_rows(api, revision)
    if not rows:
        print(f"no raw parts matching {RAW_GLOB} in {DATA_REPO}", file=sys.stderr)
        return 1
    previous = hub_stats(api, revision)
    meta = {key: previous.get(key) for key in CARRIED_META}
    meta |= {"code_sha": sha(), "data_revision": local_revision(args.real)}
    kept, stats = synth.filter_raw_rows(rows, real, is_malagasy, meta)
    files = bodies(kept, stats)

    if args.upload:
        api.create_commit(DATA_REPO, repo_type="dataset", parent_commit=revision,  # fails if the Hub moved on
                          operations=[CommitOperationAdd(path, body) for path, body in files.items()],
                          commit_message=f"synth refilter: {len(kept)}/{len(rows)} kept (code {stats['code_sha']})")  # fmt: skip
        where = f"{DATA_REPO} ({FILTERED}, {STATS})"
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        for path, body in files.items():
            (out_dir / Path(path).name).write_bytes(body)
        where = str(out_dir)
    print(f"hub {revision[:7]} | real chats {len(real)} | raw parts {parts} | raw {stats['raw']} | kept {stats['kept']} | "
          f"dropped {json.dumps(stats['dropped'])} | wrote {where}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
