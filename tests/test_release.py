"""fym1.release and scripts/release_snapshot.py, on synthetic repos and exports only."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from fym1 import release
from fym1.release import RefusedError, build_snapshot, count_name_hits, name_tokens, secret_hits, strip_list

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "release_snapshot.py"
spec = importlib.util.spec_from_file_location("release_snapshot", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

FAKE_SENDERS = ["Zafy Test", "Bodo Fakenaina"]
NAMES = name_tokens(FAKE_SENDERS)[0]
FAKE_HF = "hf_" + "Q" * 24  # built at run time so no token-shaped string sits in the source


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t.invalid",
                           "-c", "commit.gpgsign=false", *args], check=True, capture_output=True, text=True).stdout


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    git(repo.parent, "init", "-q", str(repo))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


def make_export(tmp_path: Path, senders: list[str] = FAKE_SENDERS) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    messages = [{"id": str(i), "sender": s, "text": "salama", "timestamp": f"2026-01-01T10:0{i}:00"}
                for i, s in enumerate(senders)]
    (raw / "export.json").write_text(json.dumps({"conversation": {"title": "t"}, "messages": messages}))
    return raw


CLEAN_FILES = {
    "README.md": "# demo\nNothing private here.\n",
    "src/fym1/data/mg_first_names.txt": "# hand list\nmialy\nZÀFY\nfanja\n",
    "src/fym1/data/fr_words.txt": "# wordfreq\nbonjour\nbodo\n",
    "AGENTS.md": "agent rules\n",
    "docs/superpowers/plan.md": "plan\n",
}


# ---- names ----------------------------------------------------------------------------------------------------


def test_name_tokens_fold_case_and_accents_and_skip_short_and_placeholders():
    tokens, short, placeholders = name_tokens(["Zàfy TEST", "Ny Bodo", "Facebook User", "zafy test"])
    assert tokens == {"zafy", "test", "bodo"}
    assert (short, placeholders) == (1, 1)


def test_strip_list_removes_matching_entries_any_case_or_accent_and_keeps_comments():
    text, removed = strip_list("# zafy in a comment stays\nmialy\nZÁFY\nbodo be\nfanja\n", NAMES)
    assert removed == 2
    assert text == "# zafy in a comment stays\nmialy\nfanja\n"


@pytest.mark.parametrize(("text", "hits"), [
    ("Salama Zafy!", 1),
    ("ZÂFY and zafy", 2),
    ("zafy_test", 2),
    ('"line\\nZafy"', 1),
    ('"\\u005aafy"', 1),
    ("Zafyo Bodoina", 0),
    ("nothing here", 0),
])
def test_count_name_hits_is_whole_word(text, hits):
    assert count_name_hits(text, NAMES) == hits


# ---- secrets --------------------------------------------------------------------------------------------------


def test_secret_hits_count_patterns_and_files_without_values():
    pem = "-----BEGIN RSA PRIVATE " + "KEY-----"
    text = f"token = {FAKE_HF}\n{pem}\n" + '{"username": "u", "key": "' + "a" * 32 + '"}\nmail me: zz@mail.example\n'
    hits = secret_hits("kaggle.json", text)
    assert hits == {"secret_file": 1, "hf_token": 1, "private_key": 1, "kaggle_key_json": 1, "email": 1}
    assert FAKE_HF not in json.dumps(hits)


def test_secret_hits_env_files_and_email_allow_list():
    assert secret_hits("config/.env", "") == {"secret_file": 1}
    assert secret_hits("scripts/x.py", "see x@y.mg") == {}
    assert secret_hits("tests/data/test_x.py", "probe zz@mail.example") == {}
    assert release.allowed_email_count("tests/data/test_x.py", "probe zz@mail.example") == 1


# ---- end to end -----------------------------------------------------------------------------------------------


def test_snapshot_drops_excluded_paths_strips_lists_and_passes(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    out = tmp_path / "public"
    report = build_snapshot(repo, "HEAD", out, make_export(tmp_path))
    assert report.ok
    assert not (out / "AGENTS.md").exists() and not (out / "docs/superpowers").exists()
    assert not (out / ".git").exists()
    assert (out / "src/fym1/data/mg_first_names.txt").read_text() == "# hand list\nmialy\nfanja\n"
    assert report.list_entries_removed == {"src/fym1/data/fr_words.txt": 1, "src/fym1/data/mg_first_names.txt": 1}
    assert "CC BY-SA 4.0" in (out / "NOTICE").read_text()
    assert {e["path"] for e in report.excluded} == {"AGENTS.md", "docs/superpowers/plan.md"}
    assert "AGENTS.md" not in report.excludes_unmatched


def test_name_and_secret_hits_fail_and_never_leak(tmp_path, capsys):
    files = {**CLEAN_FILES, "docs/notes.md": f"Thanks to Zafy.\nkey {FAKE_HF}\n", "zafy/readme.md": "x\n"}
    repo = make_repo(tmp_path, files)
    out = tmp_path / "public"
    code = cli.main(["--repo", str(repo), "--export", str(make_export(tmp_path)), "--out", str(out),
                     "--results", str(tmp_path / "results/release_check.json")])
    assert code == 1
    report = json.loads((out / release.REPORT_NAME).read_text())
    assert report["ok"] is False
    assert report["names"]["hits_per_file"] == {"docs/notes.md": 1, "<redacted path 1>": 1}
    assert report["secrets"]["hits_per_file"] == {"docs/notes.md": {"hf_token": 1}}
    written = (out / release.REPORT_NAME).read_text() + (tmp_path / "results/release_check.json").read_text()
    printed = capsys.readouterr()
    for blob in (written, printed.out + printed.err):
        assert "zafy" not in blob.lower() and "bodo" not in blob.lower() and FAKE_HF not in blob


def test_refuses_dirty_tree(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    (repo / "README.md").write_text("changed\n")
    with pytest.raises(RefusedError, match="not clean"):
        build_snapshot(repo, "HEAD", tmp_path / "public", make_export(tmp_path))


@pytest.mark.parametrize("inside", ["repo/out", "repo"])
def test_refuses_output_inside_repo(tmp_path, inside):
    repo = make_repo(tmp_path, CLEAN_FILES)
    with pytest.raises(RefusedError, match="inside the repo"):
        build_snapshot(repo, "HEAD", tmp_path / inside, make_export(tmp_path))


def test_refuses_non_empty_output_and_unreadable_export(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    used = tmp_path / "used"
    used.mkdir()
    (used / "x").write_text("x")
    with pytest.raises(RefusedError, match="not empty"):
        build_snapshot(repo, "HEAD", used, make_export(tmp_path))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "zafy.json").write_text('{"other": 1}')
    with pytest.raises(RefusedError, match="could not be read") as error:
        build_snapshot(repo, "HEAD", tmp_path / "public", bad)
    assert "zafy" not in str(error.value).lower()


def test_cli_refusal_exits_2(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    assert cli.main(["--repo", str(repo), "--export", str(make_export(tmp_path)), "--out", str(repo / "x")]) == 2


# ---- policy on this repo --------------------------------------------------------------------------------------


def tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split()


def test_every_exclude_has_a_reason():
    assert all(entry.reason.strip() for entry in release.EXCLUDES)


def test_every_shipped_list_file_has_a_licence_line():
    lists = {p for p in tracked() if release.is_list_file(p) and not release.exclusion_reason(p)}
    assert lists == set(release.DATA_LICENCES)


def test_no_shipped_file_imports_an_excluded_script():
    excluded_modules = {Path(e.path).stem for e in release.EXCLUDES if e.path.endswith(".py")}
    for path in tracked():
        if path.endswith(".py") and not release.exclusion_reason(path):
            text = (REPO_ROOT / path).read_text("utf-8")
            assert not any(f'"{m}"' in text or f"/ \"{m}.py\"" in text for m in excluded_modules), path
