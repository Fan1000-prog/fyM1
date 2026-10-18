"""scripts/synth_gate.py with a stand-in Hub and raw export. Every chat, sender and message here is made up."""

import importlib.util
import json
import subprocess
from pathlib import Path

from fym1.data import name_scan
from fym1.data.synth import record_id

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("synth_gate_script", ROOT / "scripts" / "synth_gate.py")
synth_gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(synth_gate)

RAW_LINE = "Aza adino ny mitondra ilay boky manga rahampitso maraina fa ilaina any am-pianarana"


def says_malagasy(text):
    return not text.startswith("fr:")


def chat(user, reply):
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": reply}]
    return {"id": record_id(messages), "messages": messages}


def run(tmp_path, records, raw=(["Zebulon Quixbert"], [[RAW_LINE]]), private=None):
    filtered = tmp_path / "hub" / "filtered.ndjson"
    filtered.parent.mkdir(parents=True)
    filtered.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    revisions = []

    def download(revision):
        revisions.append(revision)
        return filtered

    def load_raw(export):
        if raw is None:
            raise name_scan.RefusedError("export holds no conversations")
        return raw

    results, private = tmp_path / "results" / "synth_gate.json", private or tmp_path / "private"
    code = synth_gate.main(["--results", str(results), "--private-out", str(private), "--revision", "abc"],
                           download=download, load_raw=load_raw, is_malagasy=says_malagasy)  # fmt: skip
    return code, results, private, revisions


def test_clean_set_passes_and_writes_every_id(tmp_path, capsys):
    good = chat("Salama", "Salama tompoko, tsara aho")
    code, results, private, revisions = run(tmp_path, [good])
    report = json.loads(results.read_text())
    assert code == 0 and revisions == ["abc"]
    assert report["status"] == "pass" and report["checked"] == 1 and report["dropped"] == 0
    assert report["revision"] == "abc" and len(report["file_sha256"]) == 64
    assert (private / "kept_ids.txt").read_text() == f"{good['id']}\n"
    assert "PASS" in capsys.readouterr().out


def test_clean_run_after_a_dropping_run_replaces_the_kept_list(tmp_path):
    good, other = chat("Salama", "Salama tompoko"), chat("Veloma", "Veloma tompoko")
    named = chat("Iza?", "Tonga i Zebulon")
    run(tmp_path / "first", [good, named], private=tmp_path / "private")
    code, _, private, _ = run(tmp_path / "second", [good, other], private=tmp_path / "private")
    assert code == 0
    assert (private / "kept_ids.txt").read_text() == f"{good['id']}\n{other['id']}\n"


def test_failed_gate_removes_the_old_kept_list(tmp_path):
    good = chat("Salama", "Salama tompoko")
    run(tmp_path / "first", [good], private=tmp_path / "private")
    assert (tmp_path / "private" / "kept_ids.txt").exists()
    code, _, private, _ = run(tmp_path / "second", [good, chat("Inona?", RAW_LINE)], private=tmp_path / "private")
    assert code == 1 and not (private / "kept_ids.txt").exists()


def test_name_drop_writes_kept_ids_outside_the_repo(tmp_path, capsys):
    good, named = chat("Salama", "Salama tompoko, tsara aho"), chat("Iza?", "Tonga i Zebulon")
    code, results, private, _ = run(tmp_path, [good, named])
    assert code == 0
    assert (private / "kept_ids.txt").read_text() == f"{good['id']}\n"
    assert json.loads(results.read_text())["dropped_ids"] == {named["id"]: ["real_name", "unknown_name"]}
    out = capsys.readouterr().out
    assert "Zebulon" not in out and "Tonga" not in out


def test_leak_fails_with_exit_1_and_writes_no_kept_list(tmp_path):
    code, results, private, _ = run(tmp_path, [chat("Salama", "Salama tompoko"), chat("Inona?", RAW_LINE)])
    assert code == 1 and not (private / "kept_ids.txt").exists()
    assert json.loads(results.read_text())["status"] == "fail"
    assert "mitondra" not in results.read_text()


def test_unreadable_export_is_refused(tmp_path):
    code, results, _, revisions = run(tmp_path, [chat("Salama", "Salama tompoko")], raw=None)
    assert code == 2 and not results.exists() and revisions == []


def test_private_out_inside_the_repo_is_refused(tmp_path):
    try:
        synth_gate.main(["--results", str(tmp_path / "r.json"), "--private-out", str(ROOT / "private")],
                        download=lambda revision: None, load_raw=lambda export: ([], []),
                        is_malagasy=says_malagasy)  # fmt: skip
    except SystemExit as error:
        assert "outside any repo" in str(error)
    else:
        raise AssertionError("expected a refusal")
    assert not (ROOT / "private").exists()


def test_help_runs():
    out = subprocess.run(["uv", "run", "python", str(ROOT / "scripts" / "synth_gate.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT, check=True).stdout  # fmt: skip
    assert "kept_ids.txt" in out


def test_export_with_no_sender_names_is_refused(tmp_path):
    code, results, private, revisions = run(tmp_path, [chat("Salama", "Salama tompoko")],
                                            raw=(["Facebook User"], [RAW_LINE]))  # fmt: skip
    assert code == 2 and not results.exists() and revisions == [] and not private.exists()
