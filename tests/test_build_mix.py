"""scripts/build_mix.py with a fake Hub. Every chat and name here is made up for the test."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fym1.data import mix, name_scan

SPEC = importlib.util.spec_from_file_location("build_mix", Path(__file__).parents[1] / "scripts" / "build_mix.py")
build_mix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_mix)


def chat(*turns):
    messages = [{"role": ("user", "assistant")[i % 2], "content": t} for i, t in enumerate(turns)]
    return {"id": mix.record_id(messages), "messages": messages}


SYNTH = [chat(f"Salama {i}", f"Salama tsara {i}") for i in range(40)] + [chat("Aiza?", "Ho tonga aho rahariva")]
IDENTITY = [{"messages": [{"role": "user", "content": "Iza ianao?"},
                          {"role": "assistant", "content": "fy-m1 aho, mpanampy AI"}]}]  # fmt: skip
REAL = [chat(f"Inona no vaovao {i}?", f"Tsy misy {i}") for i in range(20)] + [chat("Aiza?", "Haja no tonga")]
AYA = [{"inputs": f"Fanontaniana {i}?", "targets": f"Valiny {i}", "language_code": "plt",
        "annotation_type": "original-annotations"} for i in range(100)]  # fmt: skip
AYA += [{"inputs": "a", "targets": "b", "language_code": "plt", "annotation_type": "re-annotations"}]


class FakeHub:
    def __init__(self, folder: Path):
        self.folder, self.commits = folder, []
        for name, rows in {"chats.jsonl": REAL, "identity.jsonl": IDENTITY, "synth_chats.jsonl": SYNTH}.items():
            (folder / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def hf_hub_download(self, repo, file, repo_type, revision):
        assert (repo, repo_type, revision) == (build_mix.DATA_REPO, "dataset", build_mix.REVISION)
        return str(self.folder / Path(file).name)

    def dataset_info(self, repo):
        return SimpleNamespace(sha="head123")

    def create_commit(self, repo, repo_type, parent_commit, operations, commit_message):
        self.commits.append((parent_commit, {op.path_in_repo for op in operations}))
        return SimpleNamespace(oid="newrev456")


def run(tmp_path, *extra):
    hub = FakeHub(tmp_path)
    out, results = tmp_path / "out", tmp_path / "results.json"
    code = build_mix.main(["--out-dir", str(out), "--results", str(results), *extra], api=hub,
                          is_malagasy=lambda text: True, load_names=lambda _: name_scan.name_tokens(["Zebulon Q"])[0],
                          load_aya=lambda _: AYA, sha=lambda: "abc1234")  # fmt: skip
    return code, hub, out, json.loads(results.read_text())


def test_build_writes_the_mix_and_counts_only(tmp_path):
    code, hub, out, stats = run(tmp_path)
    assert code == 0 and hub.commits == []
    assert stats["synth"]["kept_chats"] == 40 and stats["synth"]["dropped"] == 1
    assert stats["identity"] == {"chats": 1, "repeat": mix.IDENTITY_REPEAT, "rows": 1}
    assert stats["real"]["pairs_passed"] == 20 and stats["real"]["dropped_pairs_by_reason"]["fake_name"] == 1
    assert stats["aya"]["malagasy_rows"] == 101 and stats["aya"]["original_annotations"] == 100
    rows = stats["mix"]["rows"]
    total = stats["mix"]["training_rows"]
    assert rows["synth"] == 40 and rows["identity"] == mix.IDENTITY_REPEAT
    assert rows["real"] <= mix.REAL_MAX_SHARE * total and rows["aya"] <= mix.AYA_MAX_SHARE * total
    assert {p.name for p in out.iterdir()} == {f"{s}.jsonl" for s in mix.SOURCES} | {"mix_stats.json"}
    identity = [json.loads(line) for line in (out / "identity.jsonl").read_text().splitlines()]
    assert identity[0]["repeat"] == mix.IDENTITY_REPEAT
    report = json.dumps(stats)
    assert "Salama" not in report and "Valiny" not in report and "Zebulon" not in report  # counts only, never text


def test_upload_is_one_commit_of_the_mix_folder(tmp_path):
    code, hub, _, _ = run(tmp_path, "--upload")
    assert code == 0
    assert hub.commits == [("head123", {f"{mix.MIX_DIR}/{s}.jsonl" for s in mix.SOURCES}
                            | {f"{mix.MIX_DIR}/mix_stats.json"})]  # fmt: skip


def test_dry_run_writes_the_counts_and_no_mix_files(tmp_path):
    code, hub, out, stats = run(tmp_path, "--dry-run")
    assert code == 0 and hub.commits == [] and not out.exists()
    assert stats["mix"]["training_rows"] > 0


def test_dry_run_and_upload_are_refused_together(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, "--dry-run", "--upload")


def test_out_dir_inside_a_repo_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        build_mix.main(["--out-dir", str(Path(__file__).parent / "private-out")], api=FakeHub(tmp_path),
                       is_malagasy=lambda t: True, load_names=lambda _: frozenset({"x"}), load_aya=lambda _: AYA)
