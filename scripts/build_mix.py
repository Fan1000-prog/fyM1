"""Build the r2 training mix (fym1.data.mix) and, with --upload, publish it to fyM1-chats. Counts only, never text.

    uv run python scripts/build_mix.py --dry-run       # counts only: write the results file, no mix files
    uv run python scripts/build_mix.py                 # build into ~/malagasy-data/private/mix-r2, write results
    uv run python scripts/build_mix.py --upload        # same build, then one commit of mix-r2/* to fyM1-chats

Reads, all pinned: `chats.jsonl` (real chats), `identity.jsonl` and `synth/synth_chats.jsonl` (the 1,284 gate-kept
synthetic chats) from Fan1000-Prog/fyM1-chats at --revision; the Aya Dataset parquet at mix.AYA_REVISION; the raw
export (--export, laptop only) for the sender-name check; the public eval prompts (EVAL_FILES, this repo) for the
`eval_prompt` check. Writes `<source>.jsonl` for each of mix.SOURCES plus
`mix_stats.json` into --out-dir (outside every git work tree) and the same counts to --results. --upload commits
`mix-r2/<source>.jsonl` and `mix-r2/mix_stats.json` in one commit on top of the current Hub head and prints the new
dataset revision; 01_train with DATA_DIR = "mix-r2/" trains on it. The token comes from the local Hugging Face cache.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from fym1.data import mix, name_scan
from fym1.data.private import private_path

DATA_REPO = "Fan1000-Prog/fyM1-chats"
REVISION = "0c7ca5d5e8d8631457d437dea7675dd91e29a909"  # the fyM1-chats revision fym1-9b-a100-r1 trained on
INPUTS = {"real": "chats.jsonl", "identity": "identity.jsonl", "synth": "synth/synth_chats.jsonl"}
AYA_LICENCE = "apache-2.0"  # CohereLabs/aya_dataset card, checked at AYA_REVISION
REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("~/malagasy-data/private/mix-r2")
DEFAULT_EXPORT = Path("~/malagasy-data/raw")
DEFAULT_RESULTS = REPO / "results" / "mix_r2.json"
EVAL_FILES = ("eval/prompts.jsonl", "eval/identity_check.jsonl")  # prompts that must never be trained on
STATS = "mix_stats.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default=REVISION, help=f"{DATA_REPO} revision the inputs are read at")
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT, help="raw Messenger export (sender names)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="private output folder, outside any repo")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS, help="counts-only report")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="write only the counts to --results, no mix files")
    mode.add_argument("--upload", action="store_true", help=f"commit {mix.MIX_DIR}/* to {DATA_REPO}")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def raw_names(export: Path) -> frozenset[str]:
    names, _, _ = name_scan.name_tokens(name_scan.export_senders(name_scan.read_export(export.expanduser())))
    if not names:  # an empty set would clear every row of real_name
        raise name_scan.RefusedError("export gave no sender name tokens")
    return names


def read_aya(api: HfApi) -> list[dict]:
    import pyarrow.parquet as pq  # comes with `datasets`

    path = api.hf_hub_download(mix.AYA_REPO, mix.AYA_FILE, repo_type="dataset", revision=mix.AYA_REVISION)
    return pq.read_table(path, columns=["inputs", "targets", "language_code", "annotation_type"]).to_pylist()


def read_eval_prompts() -> tuple[mix.EvalPrompts, int]:
    """The public eval prompts and how many there are."""
    prompts = [row["prompt"] for file in EVAL_FILES for row in read_jsonl(REPO / file)]
    return mix.EvalPrompts(prompts), len(prompts)


def code_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                          check=True).stdout.strip()


def ndjson_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")


def build(inputs: dict[str, list[dict]], aya: list[dict], names: frozenset[str], is_malagasy: Callable[[str], bool],
          eval_set: mix.EvalPrompts) -> tuple[dict[str, list[dict]], dict]:  # fmt: skip
    """The mix rows per source and the counts-only stats. `eval_set`: the eval prompts no row may carry."""
    synth, synth_counts = mix.synth_rows(inputs["synth"], eval_set)
    identity = mix.with_repeat(inputs["identity"], mix.IDENTITY_REPEAT)
    real, real_drops = mix.real_pairs(inputs["real"], is_malagasy, names, eval_set)
    aya_malagasy, aya_counts = mix.aya_malagasy(aya)
    aya_kept, aya_drops = mix.aya_rows(aya_malagasy, is_malagasy, names, eval_set)

    base = mix.mix_rows({"synth": synth, "identity": identity})
    shares = {"real": mix.REAL_MAX_SHARE, "aya": mix.AYA_MAX_SHARE}
    caps = mix.capped_counts(base, {"real": len(real), "aya": len(aya_kept)}, shares)
    rows = {"synth": synth, "identity": identity, "real": mix.sample(real, caps["real"]),
            "aya": mix.sample(aya_kept, caps["aya"])}  # fmt: skip
    total = mix.mix_rows(rows)
    per_source = {source: mix.mix_rows({source: rows[source]}) for source in mix.SOURCES}
    stats = {
        "mix_dir": mix.MIX_DIR,
        "seed": mix.SEED,
        "synth": {"input_chats": len(inputs["synth"]), "kept_chats": len(synth), **synth_counts},
        "identity": {"chats": len(inputs["identity"]), "repeat": mix.IDENTITY_REPEAT, "file_rows": len(identity)},
        "real": {"input_chats": len(inputs["real"]), "pairs_passed": len(real), "dropped_pairs_by_reason": real_drops,
                 "max_share": mix.REAL_MAX_SHARE, "cap": caps["real"]},
        "aya": {"repo": mix.AYA_REPO, "revision": mix.AYA_REVISION, "licence": AYA_LICENCE,
                "language_code": mix.AYA_LANGUAGE_CODE, "annotation": mix.AYA_ANNOTATION, **aya_counts,
                "rows_passed": len(aya_kept), "dropped_rows_by_reason": aya_drops, "max_share": mix.AYA_MAX_SHARE,
                "cap": caps["aya"]},
        "mix": {"rows": per_source, "training_rows": total, "files_rows": {s: len(rows[s]) for s in mix.SOURCES},
                "shares": {s: round(per_source[s] / total, 4) for s in mix.SOURCES}},
    }  # fmt: skip
    return rows, stats


def bodies(rows: dict[str, list[dict]], stats: dict) -> dict[str, bytes]:
    files = {f"{mix.MIX_DIR}/{source}.jsonl": ndjson_bytes(rows[source]) for source in mix.SOURCES}
    return files | {f"{mix.MIX_DIR}/{STATS}": (json.dumps(stats, indent=2, sort_keys=True) + "\n").encode()}


def main(argv: list[str] | None = None, api: HfApi | None = None, is_malagasy: Callable[[str], bool] | None = None,
         load_names: Callable[[Path], frozenset[str]] = raw_names, load_aya: Callable[[HfApi], list[dict]] = read_aya,
         sha: Callable[[], str] = code_sha) -> int:  # fmt: skip
    args = parse_args(argv)
    api = api or HfApi()
    out_dir = private_path(args.out_dir.expanduser())  # refuse a repo path before reading anything private
    if is_malagasy is None:
        from fym1.eval.lang import is_malagasy  # loads GlotLID; tests pass a stand-in
    try:
        names = load_names(args.export)
    except name_scan.RefusedError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    paths = {source: Path(api.hf_hub_download(DATA_REPO, file, repo_type="dataset", revision=args.revision))
             for source, file in INPUTS.items()}  # fmt: skip
    inputs = {source: read_jsonl(path) for source, path in paths.items()}
    eval_set, n_eval_prompts = read_eval_prompts()
    rows, stats = build(inputs, load_aya(api), names, is_malagasy, eval_set)
    stats["eval_prompts"] = {"files": list(EVAL_FILES), "prompts": n_eval_prompts, "run_words": mix.EVAL_RUN_WORDS}
    stats["inputs"] = {"dataset": DATA_REPO, "revision": args.revision,
                       "sha256": {INPUTS[s]: hashlib.sha256(p.read_bytes()).hexdigest() for s, p in paths.items()}}
    stats["code_sha"] = sha()
    files = bodies(rows, stats)

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        for path, body in files.items():
            (out_dir / Path(path).name).write_bytes(body)
    print("mix rows " + ", ".join(f"{s} {n}" for s, n in stats["mix"]["rows"].items())
          + f" | training rows {stats['mix']['training_rows']} | wrote "
          + (f"{args.results} (dry run)" if args.dry_run else f"{out_dir} and {args.results}"))
    if args.upload:
        head = api.dataset_info(DATA_REPO).sha
        commit = api.create_commit(DATA_REPO, repo_type="dataset", parent_commit=head,  # fails if the Hub moved on
                                   operations=[CommitOperationAdd(p, b) for p, b in files.items()],
                                   commit_message=f"{mix.MIX_DIR}: {stats['mix']['training_rows']} training rows "
                                                  f"(code {stats['code_sha'][:7]})")  # fmt: skip
        print(f"uploaded {len(files)} files to {DATA_REPO} | DATA_REVISION {commit.oid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
