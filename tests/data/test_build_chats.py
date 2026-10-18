import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from fym1.data.gate import FAKE_STATS_KEYS

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_chats.py"
spec = importlib.util.spec_from_file_location("build_chats", SCRIPT)
build_chats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_chats)


def test_private_path_refuses_any_git_work_tree(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)

    with pytest.raises(SystemExit, match="git work tree"):
        build_chats.private_path(tmp_path / "not" / "yet" / "chats.jsonl")


def test_private_path_refuses_this_repo():
    with pytest.raises(SystemExit, match="git work tree"):
        build_chats.private_path(SCRIPT.parent / "chats.jsonl")


def test_private_path_allows_folders_outside_git(tmp_path):
    target = tmp_path / "private" / "chats.jsonl"

    assert build_chats.private_path(target) == target.resolve()


def _flat_export(path: Path) -> Path:
    def msg(i: int, sender: str, text: str, ts: str) -> dict:
        return {"id": i, "sender": sender, "text": text, "timestamp": ts}

    messages = [
        msg(1, "Alpha", "inona no vaovao", "2026-07-01T08:00:00"),
        msg(2, "Owner", "tsy misy", "2026-07-01T08:05:00"),
        msg(3, "Alpha", "aiza ianao", "2026-07-01T14:00:00"),
        msg(4, "Owner", "ao an-trano", "2026-07-01T14:10:00"),
    ]
    conversation = {"title": "Synthetic", "participants": ["Alpha", "Bravo", "Owner"], "timezone": "UTC"}
    path.write_text(json.dumps({"conversation": conversation, "messages": messages}), encoding="utf-8")
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def _main(monkeypatch, capsys, *args: str) -> dict:
    """Run the script in-process with a stand-in language filter (no model download)."""
    monkeypatch.setattr(build_chats, "load_is_malagasy", lambda: lambda text: True)
    monkeypatch.setattr(sys, "argv", ["build_chats.py", *args])
    build_chats.main()
    return json.loads(capsys.readouterr().out)


GATE_KEYS = {
    "conversations_before_gate",
    "dropped",
    "dropped_long_number",
    "dropped_contact",
    "dropped_too_many_unknown",
    "dropped_too_many_fakes",
    "dropped_hidden_char",
    "kept",
    "cleared_mg_common",
    "cleared_mg_wiki",
    "cleared_known_names",
    "cleared_fr",
    "cleared_en",
    "tokens_replaced",
    "words_kept",
    "fakes_per_kept_median",
    "fakes_per_kept_max",
}


def test_dry_run_prints_segment_and_gate_counts_and_writes_nothing(tmp_path, monkeypatch, capsys):
    export = _flat_export(tmp_path / "group.json")
    stats = tmp_path / "stats.json"

    counts = _main(monkeypatch, capsys, "--export", str(export), "--owner", "Owner", "--dry-run", "--stats", str(stats))

    assert {k: counts[k] for k in ("threads", "messages", "sessions", "conversations")} == {
        "threads": 1,
        "messages": 4,
        "sessions": 2,
        "conversations": 2,
    }
    assert GATE_KEYS <= counts.keys()
    assert counts["kept"] == 2 and counts["dropped"] == 0
    assert all(isinstance(v, int) for v in counts.values())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["group.json"]


def test_dry_run_gate_counts_drop_numbers_and_replace_names(tmp_path, monkeypatch, capsys):
    export = _flat_export(tmp_path / "group.json")
    data = json.loads(export.read_text())
    data["messages"][1]["text"] = "antsoy amin'ny 12 34 567 aho"  # odd grouping: phone-ish digits
    data["messages"][3]["text"] = "miaraka amin'i Bemazava"
    export.write_text(json.dumps(data))

    counts = _main(monkeypatch, capsys, "--export", str(export), "--owner", "Owner", "--dry-run")

    assert counts["dropped"] == counts["dropped_long_number"] == 1
    assert counts["kept"] == 1 and counts["tokens_replaced"] == 1
    assert "Bemazava" not in json.dumps(counts)


def test_dry_run_still_checks_the_owner(tmp_path):
    export = _flat_export(tmp_path / "group.json")

    result = _run("--export", str(export), "--owner", "Nobody", "--dry-run")

    assert result.returncode != 0
    assert "not a participant" in result.stderr


def test_full_run_still_requires_out(tmp_path):
    export = _flat_export(tmp_path / "group.json")

    result = _run("--export", str(export), "--owner", "Owner")

    assert result.returncode != 0
    assert "--out" in result.stderr


def test_full_run_writes_gated_chats_and_counts_only_gate_json(tmp_path, monkeypatch, capsys):
    export = _flat_export(tmp_path / "group.json")
    data = json.loads(export.read_text())
    data["messages"][3]["text"] = "ao an-trano, kaody 12 34 567"
    export.write_text(json.dumps(data))
    private = tmp_path / "private"
    results = tmp_path / "data_gate.json"
    args = ["--export", str(export), "--owner", "Owner", "--out", str(private / "chats.jsonl")]

    counts = _main(monkeypatch, capsys, *args, "--gate-results", str(results), "--stats", str(tmp_path / "s.json"))

    chats = (private / "chats.jsonl").read_text().splitlines()
    gate = json.loads((private / "data_gate.json").read_text())
    assert len(chats) == counts["kept"] == gate["kept"] == 1
    assert set(gate) == GATE_KEYS and gate == json.loads(results.read_text())
    assert gate["dropped_long_number"] == 1
    assert sorted(p.name for p in private.iterdir()) == ["chats.jsonl", "data_gate.json"]  # sheets are opt-in


def test_full_run_writes_role_and_leak_counts_to_stats(tmp_path, monkeypatch, capsys):
    export = _flat_export(tmp_path / "group.json")
    stats = tmp_path / "s.json"
    stats.write_text(json.dumps({"approved": 7}))
    args = ["--export", str(export), "--owner", "Owner", "--out", str(tmp_path / "private" / "chats.jsonl")]

    counts = _main(monkeypatch, capsys, *args, "--stats", str(stats))

    written = json.loads(stats.read_text())
    assert written["kept_messages"] == written["kept_messages_user"] + written["kept_messages_assistant"] == 4
    assert written["kept_messages_user"] == written["kept_messages_assistant"] == 2
    leaks = {k: v for k, v in written.items() if k.startswith("leak_")}
    assert len(leaks) == 5 and set(leaks.values()) == {0}
    assert written["approved"] == 7 and written["conversations"] == counts["kept"]
    assert {k: written[k] for k in FAKE_STATS_KEYS} == {k: counts[k] for k in FAKE_STATS_KEYS}
    assert all(v is None or isinstance(v, int) for v in written.values())


def test_review_and_leftover_sheets_are_opt_in(tmp_path, monkeypatch, capsys):
    export = _flat_export(tmp_path / "group.json")
    private = tmp_path / "private"
    args = ["--export", str(export), "--owner", "Owner", "--out", str(private / "chats.jsonl")]
    sheets = ["--sheet", str(private / "review.html"), "--leftover-sheet", str(private / "leftover_names.html")]

    _main(monkeypatch, capsys, *args, *sheets, "--stats", str(tmp_path / "s.json"))

    assert {"review.html", "leftover_names.html"} <= {p.name for p in private.iterdir()}


def test_all_senders_dry_run_counts_every_side_without_duplicates(tmp_path, monkeypatch, capsys):
    export = _flat_export(tmp_path / "group.json")

    counts = _main(monkeypatch, capsys, "--export", str(export), "--all-senders", "--dry-run")

    # Owner answers twice; Alpha answers nothing (always speaks first, so trimmed); Bravo is silent.
    assert counts["conversations"] == 2


def test_dry_run_counts_exact_duplicates_once(tmp_path, monkeypatch, capsys):
    export = _flat_export(tmp_path / "group.json")
    data = json.loads(export.read_text())
    for msg in data["messages"]:
        msg["text"] = "mitovy" if msg["sender"] == "Alpha" else "eny"
    export.write_text(json.dumps(data))

    counts = _main(monkeypatch, capsys, "--export", str(export), "--owner", "Owner", "--dry-run")

    assert counts["conversations"] == 1


def test_owner_and_all_senders_are_exclusive(tmp_path):
    export = _flat_export(tmp_path / "group.json")

    assert _run("--export", str(export), "--owner", "Owner", "--all-senders", "--dry-run").returncode != 0
    assert _run("--export", str(export), "--dry-run").returncode != 0
