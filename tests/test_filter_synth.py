"""filter_synth re-filters raw synthetic chats with a mocked Hub. Every chat here is made up for the test."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.utils import EntryNotFoundError

from fym1.data.synth import DROP_REASONS, validate_stats

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("filter_synth", ROOT / "scripts" / "filter_synth.py")
filter_synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filter_synth)

GEN = "McGill-NLP/AfriqueQwen3.5-9B-50Langs-Instruct-v1"
REAL_LINE = "Aza adino ny mitondra ilay boky manga rahampitso maraina fa ilaina any am-pianarana"
KEPT_OUTPUT = "A: Manao ahoana ianao androany?\nB: Tsara fa vizaka kely fotsiny, mbola miasa"
LEAK_OUTPUT = f"A: Inona ny vaovao?\nB: {REAL_LINE}"
FRENCH_OUTPUT = "A: Salama e\nB: fr: je suis au travail maintenant"


def says_malagasy(text):
    """Stand-in for is_malagasy: French lines in these fixtures start with "fr:"."""
    return not text.startswith("fr:")


def raw_row(index, output):
    return {"index": index, "seed": {}, "generator": GEN, "dtype": "float16", "licence": "cc-by-4.0",
            "code_sha": "old", "data_revision": "old", "output": output, "new_tokens": 10, "gen_seconds": 1.5}  # fmt: skip


class FakeHub:
    """The few HfApi calls filter_synth makes, over files in a temp folder."""

    def __init__(self, folder: Path, files: dict[str, str], sha: str = "abc1234def"):
        self.folder, self.sha, self.commits, self.downloads = folder, sha, [], []
        for path, text in files.items():
            (folder / path).parent.mkdir(parents=True, exist_ok=True)
            (folder / path).write_text(text, encoding="utf-8")
        self.files = sorted(files)

    def dataset_info(self, repo_id):
        return SimpleNamespace(sha=self.sha)

    def list_repo_files(self, repo_id, repo_type, revision):
        assert (repo_type, revision) == ("dataset", self.sha)
        return [*self.files, "chats.jsonl", "synth/raw/notes.txt"]

    def hf_hub_download(self, repo_id, filename, repo_type, revision):
        assert (repo_type, revision) == ("dataset", self.sha)
        self.downloads.append(filename)
        if filename not in self.files:
            raise EntryNotFoundError(filename)
        return str(self.folder / filename)

    def create_commit(self, repo_id, repo_type, parent_commit, operations, commit_message):
        self.commits.append({"repo": repo_id, "parent": parent_commit, "message": commit_message,
                             "files": {op.path_in_repo: op.path_or_fileobj for op in operations}})  # fmt: skip


def ndjson(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


@pytest.fixture
def real(tmp_path):
    path = tmp_path / "private" / "chats.jsonl"
    path.parent.mkdir()
    messages = [{"role": "user", "content": "Inona no tokony ho entiko?"}, {"role": "assistant", "content": REAL_LINE}]
    path.write_text(json.dumps({"id": "r1", "messages": messages}) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def hub(tmp_path):
    parts = {"synth/raw/part-00002.ndjson": ndjson([raw_row(2, FRENCH_OUTPUT)]),
             "synth/raw/part-00000.ndjson": ndjson([raw_row(0, KEPT_OUTPUT), raw_row(1, LEAK_OUTPUT)]),
             "synth/synth_stats.json": json.dumps({"seed": 0, "few_shot_k": 3, "target_raw": 2500, "gpus": "2x T4"})}  # fmt: skip
    (tmp_path / "hub").mkdir()
    return FakeHub(tmp_path / "hub", parts)


def run(args, hub, capsys):
    code = filter_synth.main(args, api=hub, is_malagasy=says_malagasy, sha=lambda: "newcode")
    return code, capsys.readouterr()


def test_dry_run_writes_filtered_and_stats_locally_and_commits_nothing(tmp_path, real, hub, capsys):
    out = tmp_path / "out"
    code, printed = run(["--real", str(real), "--out-dir", str(out)], hub, capsys)
    assert code == 0 and hub.commits == []
    kept = [json.loads(line) for line in (out / "filtered.ndjson").read_text().splitlines()]
    stats = json.loads((out / "synth_stats.json").read_text())
    validate_stats(stats)
    assert [r["messages"][1]["content"] for r in kept] == ["Tsara fa vizaka kely fotsiny, mbola miasa"]
    assert (stats["raw"], stats["kept"]) == (3, 1)
    assert {r: n for r, n in stats["dropped"].items() if n} == {"leak_8gram": 1, "not_malagasy": 1}
    assert "raw parts 2 | raw 3 | kept 1" in printed.out


def test_leak_reference_is_the_local_real_file(tmp_path, hub, capsys):
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"id": "r9", "messages": [{"role": "user", "content": "ny alahady izao dia handeha hitsidika an'i dadabe any ambanivohitra izahay"}]}) + "\n")
    out = tmp_path / "out"
    run(["--real", str(other), "--out-dir", str(out)], hub, capsys)
    assert json.loads((out / "synth_stats.json").read_text())["kept"] == 2  # no leak without REAL_LINE


def test_stats_carry_notebook_settings_and_record_this_code_and_the_real_file(tmp_path, real, hub, capsys):
    out = tmp_path / "out"
    run(["--real", str(real), "--out-dir", str(out)], hub, capsys)
    stats = json.loads((out / "synth_stats.json").read_text())
    assert {k: stats[k] for k in ("seed", "few_shot_k", "target_raw", "gpus")} == {
        "seed": 0, "few_shot_k": 3, "target_raw": 2500, "gpus": "2x T4"}  # fmt: skip
    assert stats["code_sha"] == "newcode"
    assert stats["data_revision"] == filter_synth.local_revision(real)
    assert stats["by_generator"] == {GEN: 3} and stats["generator_licence"] == "cc-by-4.0"


def test_missing_hub_stats_leave_notebook_settings_empty(tmp_path, real, hub, capsys):
    hub.files.remove("synth/synth_stats.json")
    out = tmp_path / "out"
    assert run(["--real", str(real), "--out-dir", str(out)], hub, capsys)[0] == 0
    assert json.loads((out / "synth_stats.json").read_text())["seed"] is None


def test_upload_makes_one_commit_on_the_read_revision_and_writes_nothing_locally(tmp_path, real, hub, capsys):
    out = tmp_path / "out"
    code, _ = run(["--real", str(real), "--out-dir", str(out), "--upload"], hub, capsys)
    assert code == 0 and not out.exists()
    [commit] = hub.commits
    assert commit["repo"] == "Fan1000-Prog/fyM1-chats" and commit["parent"] == hub.sha
    assert set(commit["files"]) == {"synth/filtered.ndjson", "synth/synth_stats.json"}
    stats = json.loads(commit["files"]["synth/synth_stats.json"])
    validate_stats(stats)
    assert stats["kept"] == len(commit["files"]["synth/filtered.ndjson"].decode().splitlines()) == 1


def test_never_prints_chat_text(tmp_path, real, hub, capsys):
    for args in (["--out-dir", str(tmp_path / "out")], ["--upload"]):
        _, printed = run(["--real", str(real), *args], hub, capsys)
        for text in ("vizaka", "boky", "travail", "Inona"):
            assert text not in printed.out + printed.err


def test_refuses_an_out_dir_inside_a_git_work_tree_before_downloading(real, hub, capsys):
    with pytest.raises(SystemExit, match="git work tree"):
        run(["--real", str(real), "--out-dir", str(ROOT / "synth-out")], hub, capsys)
    assert hub.downloads == []


def test_no_raw_parts_is_an_error(tmp_path, real, capsys):
    (tmp_path / "hub").mkdir()
    empty = FakeHub(tmp_path / "hub", {})
    code, printed = run(["--real", str(real), "--out-dir", str(tmp_path / "out")], empty, capsys)
    assert code == 1 and "no raw parts" in printed.err and empty.commits == []


def test_every_drop_reason_is_counted(tmp_path, real, hub, capsys):
    out = tmp_path / "out"
    run(["--real", str(real), "--out-dir", str(out)], hub, capsys)
    assert set(json.loads((out / "synth_stats.json").read_text())["dropped"]) == set(DROP_REASONS)
