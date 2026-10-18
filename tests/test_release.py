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


FAKE_ID = "fake.user-42"  # synthetic private account id


def make_ids(tmp_path: Path, text: str = f"# private ids\n{FAKE_ID}\n") -> Path:
    ids = tmp_path / "private" / "ids.txt"  # outside the synthetic repo
    ids.parent.mkdir(exist_ok=True)
    ids.write_text(text)
    return ids


def snap(repo: Path, out: Path, export: Path):
    return build_snapshot(repo, "HEAD", out, export, make_ids(repo.parent))


def cli_args(tmp_path: Path, repo: Path, out: Path) -> list[str]:
    return ["--repo", str(repo), "--export", str(make_export(tmp_path)), "--private-ids", str(make_ids(tmp_path)),
            "--out", str(out)]


CLEAN_FILES = {
    "README.md": "# demo\nNothing private here.\n",
    "src/fym1/data/mg_first_names.txt": "# hand list\nmialy\nZÀFY\nfanja\n",
    "src/fym1/data/fr_words.txt": "# wordfreq\nbonjour\nbodo\n",
    "AGENTS.md": "agent rules\n",
    "docs/superpowers/plan.md": "plan\n",
}


# ---- names ----------------------------------------------------------------------------------------------------


def test_name_tokens_fold_case_and_accents_and_skip_short_and_placeholders():
    tokens, short, placeholders = name_tokens(["Zàfy TEST", "Ny Bodo", "Facebook User", "zafy test", "Zafyyy"])
    assert tokens == {"zafy", "test", "bodo"}
    assert (short, placeholders) == (1, 1)


def test_prefix_rule_spares_everyday_words_like_the_anonymiser():
    names = frozenset({"fara"})
    assert release.is_name_word("faraso", names)
    assert not release.is_name_word("farany", names)  # "farany" is not "Fara" + "-ny"


def test_strip_list_catches_stretched_and_glued_entries():
    text, removed = strip_list("mialy\nZafyyy\niBodo\nfanja\n", NAMES)
    assert (text, removed) == ("mialy\nfanja\n", 2)


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
    ("Zafyo Bodoina", 2),  # the anonymiser's 4+ letter prefix rule
    ("zafybodo_42", 1),
    ("Zaf Bod Tes", 0),
    ("nothing here", 0),
    ("Salama iZafy", 1),
    ("tamin'n'iZafy sy an'iBodo", 2),
    ("n’iZafy", 1),
    ("ryZafy raBodo", 2),
    ("Zafyyy!! ZAAAFY", 2),
    ("iZafyyyko", 1),
    ("Za\u200bfy", 1),
    ("ｚａｆｙ", 1),
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
    assert secret_hits("tests/data/test_anonymize_probes.py", "PROBE" + "@GMAIL.COM") == {}
    assert release.allowed_email_count("tests/data/test_x.py", "probe zz@mail.example") == 1
    # Only the made-up domains and the listed probes are allowed under tests/, and nothing outside tests/.
    assert secret_hits("tests/data/test_x.py", "real.person" + "@gmail.com") == {"email": 1}
    assert secret_hits("docs/x.md", "zz@mail.example") == {"email": 1}


def test_secret_hits_google_oauth_jwt_and_telegram_tokens():
    oauth = "GOCSPX-" + "Ab1_" * 7
    jwt = ".".join(["eyJ" + "hbGciOiJI", "eyJ" + "zdWIiOiIx", "S" * 20])
    bot = "1234567" + "89:" + "A" * 35
    hits = secret_hits("scripts/x.py", f"{oauth}\n{jwt}\nbot {bot}\n")
    assert hits == {"google_oauth_secret": 1, "jwt": 1, "telegram_bot_token": 1}


# ---- end to end -----------------------------------------------------------------------------------------------


def test_snapshot_drops_excluded_paths_strips_hand_lists_only_and_passes(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    out = tmp_path / "public"
    report = snap(repo, out, make_export(tmp_path))
    assert report.ok
    assert not (out / "AGENTS.md").exists() and not (out / "docs/superpowers").exists()
    assert not (out / ".git").exists()
    assert (out / "src/fym1/data/mg_first_names.txt").read_text() == "# hand list\nmialy\nfanja\n"
    # A generated list ships exactly as built (a rebuild + diff would expose a removal) and is not name-scanned.
    assert (out / "src/fym1/data/fr_words.txt").read_text() == CLEAN_FILES["src/fym1/data/fr_words.txt"]
    assert report.list_entries_removed == {"src/fym1/data/mg_first_names.txt": 1}
    assert report.name_hits == {}
    notice = (out / "NOTICE").read_text()
    assert "CC BY-SA 4.0" in notice
    assert not any(word in notice.lower() for word in ("privacy", "removed", "release_check"))
    assert sorted(p.name for p in out.iterdir()) == ["NOTICE", "README.md", "src"]
    assert {e["path"] for e in report.excluded} == {"AGENTS.md", "docs/superpowers/plan.md"}
    assert "AGENTS.md" not in report.excludes_unmatched


def test_name_and_secret_hits_fail_and_never_leak(tmp_path, capsys):
    files = {**CLEAN_FILES, "docs/notes.md": f"Thanks to Zafy.\nkey {FAKE_HF}\n", "zafy/readme.md": "x\n",
             "docs/inbox/xbodox_1.md": f"{FAKE_HF}\n"}  # a name inside a path word: redacted too
    repo = make_repo(tmp_path, files)
    out = tmp_path / "public"
    code = cli.main([*cli_args(tmp_path, repo, out), "--results", str(tmp_path / "results/release_check.json")])
    assert code == 1
    assert not list(out.rglob("*.json"))  # the report never goes into the snapshot
    report = json.loads((tmp_path / "results/release_check.json").read_text())
    assert report["ok"] is False
    assert report["names"]["hits_per_file"] == {"docs/notes.md": 1, "<redacted path 1>": 1}
    assert report["secrets"]["hits_per_file"] == {"<redacted path 2>": {"hf_token": 1},
                                                  "docs/notes.md": {"hf_token": 1}}
    written = (tmp_path / "results/release_check.json").read_text()
    printed = capsys.readouterr()
    for blob in (written, printed.out + printed.err):
        assert "zafy" not in blob.lower() and "bodo" not in blob.lower() and FAKE_HF not in blob


def test_refuses_dirty_tree(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    (repo / "README.md").write_text("changed\n")
    with pytest.raises(RefusedError, match="not clean"):
        snap(repo, tmp_path / "public", make_export(tmp_path))


@pytest.mark.parametrize("inside", ["repo/out", "repo", "linked/out", "linked"])
def test_refuses_output_inside_repo_or_a_linked_worktree(tmp_path, inside):
    repo = make_repo(tmp_path, CLEAN_FILES)
    git(repo, "worktree", "add", "-q", str(tmp_path / "linked"))
    with pytest.raises(RefusedError, match="inside the repo"):
        snap(repo, tmp_path / inside, make_export(tmp_path))


def test_refuses_non_empty_output_and_unreadable_export(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    used = tmp_path / "used"
    used.mkdir()
    (used / "x").write_text("x")
    with pytest.raises(RefusedError, match="not empty"):
        snap(repo, used, make_export(tmp_path))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "zafy.json").write_text('{"other": 1}')
    with pytest.raises(RefusedError, match="could not be read") as error:
        snap(repo, tmp_path / "public", bad)
    assert "zafy" not in str(error.value).lower()


def test_symlink_is_refused_unless_excluded(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    (repo / "CLAUDE.md").symlink_to("AGENTS.md")
    (repo / "link.md").symlink_to("README.md")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "links")
    export = make_export(tmp_path)
    with pytest.raises(RefusedError, match="1 unsafe or non-regular") as error:
        snap(repo, tmp_path / "public", export)
    assert "link.md" not in str(error.value)  # counted, never named: a path may hold a name
    assert not (tmp_path / "public").exists()
    git(repo, "rm", "-q", "link.md")
    git(repo, "commit", "-q", "-m", "drop link")
    assert snap(repo, tmp_path / "public", export).ok


def test_cli_refusal_exits_2(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    results = str(tmp_path / "r.json")
    assert cli.main([*cli_args(tmp_path, repo, repo / "x"), "--results", results]) == 2


def test_cli_refuses_results_inside_the_snapshot(tmp_path, capsys):
    repo = make_repo(tmp_path, CLEAN_FILES)
    out = tmp_path / "public"
    assert cli.main([*cli_args(tmp_path, repo, out), "--results", str(out / "check.json")]) == 2
    assert "outside --out" in capsys.readouterr().err
    assert not out.exists()


# ---- policy on this repo --------------------------------------------------------------------------------------


def tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.split()


def test_every_exclude_has_a_reason():
    assert all(entry.reason.strip() for entry in release.EXCLUDES)


def test_every_shipped_list_file_has_a_licence_line():
    lists = {p for p in tracked() if release.is_list_file(p) and not release.exclusion_reason(p)}
    assert lists == set(release.DATA_LICENCES)


def test_every_list_is_either_hand_written_or_generated_with_a_reason():
    lists = {p for p in tracked() if release.is_list_file(p)}
    assert not set(release.HAND_LISTS) & set(release.GENERATED_LISTS)
    assert lists == set(release.HAND_LISTS) | set(release.GENERATED_LISTS)
    assert all(reason.strip() for reason in [*release.HAND_LISTS.values(), *release.GENERATED_LISTS.values()])


@pytest.mark.parametrize("path", sorted(release.GENERATED_LISTS))
def test_generated_list_is_written_by_its_build_script(path):
    script = "build_common_words.py" if path.endswith("mg_common_words.txt") else "build_known_words.py"
    assert script in release.GENERATED_LISTS[path]
    assert Path(path).name in (REPO_ROOT / "scripts" / script).read_text("utf-8")


@pytest.mark.parametrize("path", sorted(release.HAND_LISTS))
def test_hand_list_says_it_is_hand_written(path):
    assert (REPO_ROOT / path).read_text("utf-8").startswith("# Hand list")


def test_no_shipped_file_imports_an_excluded_script():
    excluded_modules = {Path(e.path).stem for e in release.EXCLUDES if e.path.endswith(".py")}
    for path in tracked():
        if path.endswith(".py") and not release.exclusion_reason(path):
            text = (REPO_ROOT / path).read_text("utf-8")
            assert not any(f'"{m}"' in text or f"/ \"{m}.py\"" in text for m in excluded_modules), path


# ---- private ids ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "hits"), [
    (f"kaggle user {FAKE_ID}", 1),
    (f"{FAKE_ID.upper()}/fym1-00-baseline", 1),
    ('"id": "fake.user\\u002d42"', 1),
    (f"x{FAKE_ID} {FAKE_ID}2 {FAKE_ID}_b", 0),
    ("nothing here", 0),
])
def test_private_ids_match_whole_words_any_case(tmp_path, text, hits):
    ids, count = release.load_private_ids(make_ids(tmp_path), [tmp_path / "repo"])
    assert count == 1
    assert release.count_private_ids(text, ids) == hits


def test_private_id_hits_fail_with_counts_and_paths_only(tmp_path, capsys):
    files = {**CLEAN_FILES, "scripts/push.py": f'USER = "{FAKE_ID}"\n', f"docs/{FAKE_ID}.md": "x\n"}
    repo = make_repo(tmp_path, files)
    results = tmp_path / "results.json"
    assert cli.main([*cli_args(tmp_path, repo, tmp_path / "public"), "--results", str(results)]) == 1
    report = json.loads(results.read_text())
    assert report["private_ids"] == {"ids_checked": 1, "hits_per_file": {"scripts/push.py": 1,
                                                                          "<redacted path 1>": 1}, "hits_total": 2}
    printed = capsys.readouterr()
    assert FAKE_ID not in results.read_text() + printed.out + printed.err


@pytest.mark.parametrize("where", ["repo/ids.txt", "worktree/ids.txt"])
def test_private_ids_file_inside_the_repo_or_a_worktree_is_refused(tmp_path, where):
    repo = make_repo(tmp_path, CLEAN_FILES)
    git(repo, "worktree", "add", "-q", str(tmp_path / "worktree"))
    (repo / ".git/info/exclude").write_text("ids.txt\n")  # ignored, so the tree stays clean
    (tmp_path / where).write_text(f"{FAKE_ID}\n")
    with pytest.raises(RefusedError, match="inside the repo") as error:
        build_snapshot(repo, "HEAD", tmp_path / "public", make_export(tmp_path), tmp_path / where)
    assert FAKE_ID not in str(error.value)
    assert not (tmp_path / "public").exists()


def test_private_ids_file_empty_or_missing_is_refused(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    export = make_export(tmp_path)
    with pytest.raises(RefusedError, match="holds no ids"):
        build_snapshot(repo, "HEAD", tmp_path / "public", export, make_ids(tmp_path, "# only a comment\n\n"))
    with pytest.raises(RefusedError, match="could not be read"):
        build_snapshot(repo, "HEAD", tmp_path / "public", export, tmp_path / "missing.txt")


# ---- unreadable files and NOTICE ------------------------------------------------------------------------------


def test_unreadable_pdf_fails_the_check(tmp_path):
    repo = make_repo(tmp_path, {**CLEAN_FILES, "paper/main.pdf": "not a pdf at all\n"})
    report = snap(repo, tmp_path / "public", make_export(tmp_path))
    assert report.unscannable == ["paper/main.pdf"]
    assert not report.ok


def test_tracked_notice_is_refused_before_anything_is_written(tmp_path):
    repo = make_repo(tmp_path, {**CLEAN_FILES, "NOTICE": "existing notes\n"})
    with pytest.raises(RefusedError, match="NOTICE already tracked"):
        snap(repo, tmp_path / "public", make_export(tmp_path))
    assert not (tmp_path / "public").exists()
