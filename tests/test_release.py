"""fym1.release and scripts/release_snapshot.py, on synthetic repos and exports only."""

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path, PurePath

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


def make_repo(tmp_path: Path, files: dict[str, str | bytes]) -> Path:
    repo = tmp_path / "repo"
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            (repo / name).write_bytes(text)
        else:
            (repo / name).write_text(text, encoding="utf-8")
    git(repo.parent, "init", "-q", str(repo))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo


# Synthetic chat line in the synthetic export: shipped text sharing 8 words in a row with it is copied chat text.
FAKE_CHAT = "rahampitso maraina dia handeha hiantsena izahay miaraka amin'ny namanay any an-tanàna"
FAKE_CHAT_WORDS = FAKE_CHAT.replace("'", " ").replace("-", " ").split()  # as fym1.eval.metrics splits words


def make_export(tmp_path: Path, senders: list[str] = FAKE_SENDERS, chat: str = FAKE_CHAT) -> Path:
    """Each sender says "salama"; the first one also says `chat`."""
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = [*((s, "salama") for s in senders), (senders[0], chat)]
    messages = [{"id": str(i), "sender": s, "text": text, "timestamp": f"2026-01-01T10:0{i}:00"}
                for i, (s, text) in enumerate(lines)]
    (raw / "export.json").write_text(json.dumps({"conversation": {"title": "t"}, "messages": messages}))
    return raw


FAKE_ID = "fake.user-42"  # synthetic private account id


def make_ids(tmp_path: Path, text: str = f"# private ids\n{FAKE_ID}\n") -> Path:
    ids = tmp_path / "private" / "ids.txt"  # outside the synthetic repo
    ids.parent.mkdir(exist_ok=True)
    ids.write_text(text)
    return ids


# Synthetic anonymised chat turn (a fake name, two messages joined) that is in no export message: the anonymised
# chats.jsonl holds text the raw export does not, so the check indexes it too.
FAKE_TURN = "Salama Tsikivola! efa tonga ve ianao?\nmbola miandry anao eto amin'ny fiantsonana izahay rehetra"
FAKE_TURN_WORDS = FAKE_TURN.replace("'", " ").replace("?", " ").replace("!", " ").split()


def make_chats(tmp_path: Path, turns: list[str] | None = None, text: str | None = None) -> Path:
    """The anonymised chats file, outside the synthetic repo: one record of `turns` (user, assistant, ...), plus a
    system turn the check leaves out; or exactly `text` when given."""
    chats = tmp_path / "private" / "chats.jsonl"
    chats.parent.mkdir(exist_ok=True)
    if text is None:
        turns = [FAKE_TURN, "eny"] if turns is None else turns
        messages = [{"role": "system", "content": "system prompt"},
                    *({"role": ("user", "assistant")[i % 2], "content": t} for i, t in enumerate(turns))]
        text = json.dumps({"id": "c1", "messages": messages}) + "\n"
    chats.write_text(text, encoding="utf-8")
    return chats


def snap(repo: Path, out: Path, export: Path):
    return build_snapshot(repo, "HEAD", out, export, make_ids(repo.parent), make_chats(repo.parent))


def cli_args(tmp_path: Path, repo: Path, out: Path) -> list[str]:
    return ["--repo", str(repo), "--export", str(make_export(tmp_path)), "--private-ids", str(make_ids(tmp_path)),
            "--chats", str(make_chats(tmp_path)), "--out", str(out)]


CLEAN_FILES = {
    "README.md": "# demo\nNothing private here.\n",
    "src/fym1/data/mg_first_names.txt": "# hand list\nmialy\nZÀFY\nfanja\n",
    "src/fym1/data/mg_common_words.txt": "# wikipedia\nbonjour\nbodo\n",
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


def test_strip_list_catches_split_and_digit_spelled_entries():
    text, removed = strip_list("mialy\nZa\u3164fy\nB0d0\nfanja\n", NAMES)
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
    # split by an invisible or combining character the gate flags
    ("Za\u0903fy", 1),  # spacing mark (Mc)
    ("Za\u3164fy Bo\uffa0do", 2),  # Hangul fillers: default-ignorable letters
    ("Za\u115ffy", 1),
    ("Za\u0336fy", 1),  # loose strike-through (Mn)
    ("Z\u200da\u200df\u200dy", 1),  # ZWJ (Cf)
    ('"Za\\u0903fy"', 1),  # as a JSON/Python escape
    # digits typed for letters
    ("Z4fy", 1),
    ("z4fy_t3st", 2),
    ("B0d0 tonga", 1),
    ("iZ4fyko", 1),
    ("Z4fy2", 1),
    ("h4ck3r sha256 42 2026", 0),
])
def test_count_name_hits_is_whole_word(text, hits):
    assert count_name_hits(text, NAMES) == hits


@pytest.mark.parametrize(("text", "hits"), [
    # the strict gate's tokeniser: a lower-to-upper case change splits a word
    ("salutZafy", 1),
    ("helloBodoAndZafy", 2),
    # spaced-out letters read as one word
    ("Z a f y", 1),
    ("Z.a.f.y", 1),
    ("B_o_d_o", 1),
    # \\x, \\U, HTML and percent escapes, and escapes inside escapes
    ("\\x5aafy", 1),
    ("\\U0000005aafy", 1),
    ("&#90;afy &#x42;odo", 2),
    ("&amp;#90;afy", 1),
    ("%5Aafy", 1),
    ("%255Aafy", 1),
    # a 4+ letter name inside a longer word no known-word list has; known words stay
    ("xxfakenainaxx", 1),
    ("bodofakenaina", 1),
    ("xxzafyxx", 1),  # a 4-letter name inside a word no list has
    ("contest attestation", 0),
    # digits for letters around a gate word
    ("B0d0", 1),
])
def test_count_name_hits_reads_text_as_the_gate_does(text, hits):
    assert count_name_hits(text, NAMES) == hits


def test_a_name_inside_a_known_word_is_not_a_hit():
    names = name_tokens(["Ertain"])[0]
    assert count_name_hits("certainly", names) == 0  # a known English word
    assert count_name_hits("xertainx", names) == 1


def test_a_4_letter_name_inside_an_unknown_word_is_caught():
    assert count_name_hits("see xxzafyxx and qbodoq", NAMES) == 2


def _allow(monkeypatch, pairs: dict[tuple[str, str], str]) -> None:
    monkeypatch.setattr(release, "INNER_NAME_ALLOWED", pairs)


def test_inner_name_allow_list_is_exact_per_file_and_word(monkeypatch):
    _allow(monkeypatch, {("docs/a.md", "xxzafyxx"): "synthetic"})
    allowed = release.inner_allowed("docs/a.md")
    assert count_name_hits("xxzafyxx XXZAFYXX", NAMES, allowed) == 0
    assert count_name_hits("yyzafyyy", NAMES, allowed) == 1  # another word
    assert count_name_hits("xxzafyxxko", NAMES, allowed) == 1  # a longer word
    assert count_name_hits("Zafy zafyxx", NAMES, allowed) == 2  # the name alone, the name at a word's start
    assert count_name_hits("xxzafyxx", NAMES, release.inner_allowed("docs/b.md")) == 1  # another file


def test_inner_name_allow_list_never_covers_paths_end_to_end(tmp_path, monkeypatch):
    _allow(monkeypatch, {("docs/a.md", "xxzafyxx"): "synthetic", ("docs/xxzafyxx.md", "xxzafyxx"): "synthetic"})
    repo = make_repo(tmp_path, {**CLEAN_FILES, "docs/a.md": "xxzafyxx twice: xxzafyxx\n",
                                "docs/b.md": "xxzafyxx\n", "docs/xxzafyxx.md": "ok\n"})
    report = snap(repo, tmp_path / "public", make_export(tmp_path))
    assert report.name_hits == {"<redacted path 1>": 1, "docs/b.md": 1}
    assert report.inner_name_words_allowed == 2
    assert "xxzafyxx" not in json.dumps(report.as_dict())


def test_every_inner_name_allow_list_entry_has_a_reason_and_its_word_is_in_its_file():
    assert release.INNER_NAME_ALLOWED
    for (path, word), reason in release.INNER_NAME_ALLOWED.items():
        assert reason.strip() and "\n" not in reason, (path, word)
        assert word == release.compare_key(word) and word.isalpha(), (path, word)
        if path.endswith(".pdf") and not all(map(shutil.which, ("pdftotext", "pdfinfo", "pdfdetach"))):
            continue  # poppler missing (CI): the PDF cannot be read here
        text = release.read_text(REPO_ROOT / path)
        assert text is not None and word in release.words(text), (path, word)


def test_short_names_are_checked_before_stretched_letters_collapse():
    names, short, _ = name_tokens(["Ann Lee", "Bo"])
    assert names == {"an", "le"} and short == 1  # "Bo" is short; "Ann" and "Lee" are not
    assert count_name_hits("Ann, ANNN and Lée", names) == 3
    assert count_name_hits("an apple, le chat", names) == 0  # the everyday words their keys collapse to
    assert strip_list("# hand list\nann\nan\nlee\nle\n", names) == ("# hand list\nan\nle\n", 2)


def test_hidden_characters_are_read_both_as_a_break_and_as_nothing():
    lee = name_tokens(["Lee"])[0]
    assert count_name_hits("xq\u200blee", lee) == 1  # as a break: "xq lee" (3 letters: no infix rule)
    assert count_name_hits("Le\u0903e", lee) == 1  # as nothing: "Lee"
    names = name_tokens(["Zafy\u0903bodo"])[0]
    assert {"zafy", "bodo", "zafybodo"} <= names  # the sender read both ways
    assert count_name_hits("salama bodo", names) == 1


# ---- secrets --------------------------------------------------------------------------------------------------


def test_secret_hits_count_patterns_and_files_without_values():
    pem = "-----BEGIN RSA PRIVATE " + "KEY-----"
    text = f"token = {FAKE_HF}\n{pem}\n" + '{"username": "u", "key": "' + "a" * 32 + '"}\nmail me: zz' + "@corp.mg\n"
    hits = secret_hits("kaggle.json", text)
    assert hits == {"secret_file": 1, "hf_token": 1, "private_key": 1, "kaggle_key_json": 1, "email": 1}
    assert FAKE_HF not in json.dumps(hits)


def test_secret_hits_env_files_and_email_allow_list():
    placeholder = "x" + "@y.mg"  # built at run time so this file holds no address the scan would see
    assert secret_hits("config/.env", "") == {"secret_file": 1}
    # Reserved test domains pass anywhere.
    assert secret_hits("docs/x.md", "zz@mail.example, a@b.invalid, c@d.TEST") == {}
    assert release.allowed_email_count("tests/data/test_x.py", "probe zz@mail.example") == 1
    # Any other address passes only where it is listed, exactly.
    assert secret_hits("scripts/build_known_words.py", f"see {placeholder}") == {}
    assert secret_hits("scripts/x.py", f"see {placeholder}") == {"email": 1}
    assert secret_hits("tests/data/test_x.py", f"see {placeholder}") == {"email": 1}  # no folder-wide pass


def test_email_probe_files_allow_only_their_listed_addresses():
    probes = "tests/data/test_known_words.py"
    assert secret_hits(probes, "PROBE" + "@MAIL.MG") == {"email": 1}
    assert secret_hits(probes, "FANJA" + "@MAIL.MG") == {}  # listed, any case
    assert secret_hits(probes, "olona" + "@example.mg") == {"email": 1}  # listed for another file only
    assert secret_hits("tests/data/test_gate.py", "olona" + "@example.mg") == {}


def test_every_listed_probe_address_is_in_its_file():
    for path, addresses in release.EMAIL_PROBE_FILES.items():
        text = (REPO_ROOT / path).read_text()
        found = {tuple(email.casefold().rsplit("@", 1)) for email in release.EMAIL.findall(text)}
        assert addresses <= found, path


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
    assert (out / "src/fym1/data/mg_common_words.txt").read_text() == CLEAN_FILES["src/fym1/data/mg_common_words.txt"]
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
             "docs/inbox/xbodox_1.md": f"{FAKE_HF}\n"}  # a 4-letter name inside a path word: a hit, redacted
    repo = make_repo(tmp_path, files)
    out = tmp_path / "public"
    code = cli.main([*cli_args(tmp_path, repo, out), "--results", str(tmp_path / "results/release_check.json")])
    assert code == 1
    assert not list(out.rglob("*.json"))  # the report never goes into the snapshot
    report = json.loads((tmp_path / "results/release_check.json").read_text())
    assert report["ok"] is False
    assert report["names"]["hits_per_file"] == {"<redacted path 1>": 1, "docs/notes.md": 1, "<redacted path 2>": 1}
    assert report["secrets"]["hits_per_file"] == {"<redacted path 1>": {"hf_token": 1},
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


def test_every_list_is_hand_written_generated_or_rebuilt_with_a_reason():
    lists = {p for p in tracked() if release.is_list_file(p)}
    groups = [set(release.HAND_LISTS), set(release.GENERATED_LISTS), set(release.REBUILT_LISTS)]
    assert sum(map(len, groups)) == len(set().union(*groups))  # no list in two groups
    assert lists == set().union(*groups)
    assert all(reason.strip() for reason in [*release.HAND_LISTS.values(), *release.GENERATED_LISTS.values()])


def _known_words_outputs() -> set[str]:
    """The lists build_known_words.py writes: the keys of its SOURCES dict (each written file's header)."""
    tree = ast.parse((REPO_ROOT / "scripts" / "build_known_words.py").read_text("utf-8"))
    sources = next(node.value for node in tree.body
                   if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "SOURCES" for t in node.targets))
    return {f"src/fym1/data/{key.value}" for key in sources.keys}


def test_lists_built_from_the_hand_lists_never_ship():
    # build_known_words.py drops the hand lists' names from everything it writes, so a rebuild from the shipped
    # (stripped) hand lists, diffed against the repo's lists, would show the removed names.
    assert _known_words_outputs() == set(release.REBUILT_LISTS)
    assert all(release.exclusion_reason(path) for path in release.REBUILT_LISTS)


@pytest.mark.parametrize("path", sorted(release.GENERATED_LISTS))
def test_shipped_generated_list_is_built_from_public_data_alone(path):
    script = "build_common_words.py"
    source = (REPO_ROOT / "scripts" / script).read_text("utf-8")
    assert script in release.GENERATED_LISTS[path] and Path(path).name in source
    assert "known_words" not in source
    assert not any(Path(hand).name in source for hand in release.HAND_LISTS)


def test_snapshot_ships_without_the_rebuilt_lists_and_says_how_to_build_them(tmp_path):
    files = {**CLEAN_FILES, **{path: "# built\nbonjour\n" for path in release.REBUILT_LISTS}}
    repo = make_repo(tmp_path, files)
    out = tmp_path / "public"
    report = snap(repo, out, make_export(tmp_path))
    assert report.ok
    assert not any((out / path).exists() for path in release.REBUILT_LISTS)
    notice = (out / "NOTICE").read_text()
    assert release.REBUILD_COMMAND in notice and all(path in notice for path in release.REBUILT_LISTS)


def test_build_known_words_reads_hand_lists_from_disk_not_through_known_words():
    # known_words needs the lists build_known_words writes, so the public snapshot could not rebuild them.
    source = (REPO_ROOT / "scripts" / "build_known_words.py").read_text("utf-8")
    assert "fym1.data.known_words" not in source


@pytest.mark.parametrize("path", sorted(release.HAND_LISTS))
def test_hand_list_says_it_is_hand_written(path):
    assert (REPO_ROOT / path).read_text("utf-8").startswith("# Hand list")


def test_no_shipped_file_imports_an_excluded_script():
    excluded_modules = {Path(e.path).stem for e in release.EXCLUDES if e.path.endswith(".py")}
    for path in tracked():
        if path.endswith(".py") and not release.exclusion_reason(path):
            text = (REPO_ROOT / path).read_text("utf-8")
            assert not any(f'"{m}"' in text or f"/ \"{m}.py\"" in text for m in excluded_modules), path



def test_release_tooling_never_ships(tmp_path):
    # Lead decision 32: the tooling and its exact (file, word) allow-lists are a private build record.
    assert set(release.RELEASE_TOOLING) <= set(tracked())
    assert all(release.exclusion_reason(path) for path in release.RELEASE_TOOLING)
    repo = make_repo(tmp_path, {**CLEAN_FILES, **{path: "# tool\n" for path in release.RELEASE_TOOLING}})
    out = tmp_path / "public"
    report = snap(repo, out, make_export(tmp_path, ["Zafy Bodo"]))  # FAKE_SENDERS' "Test" would redact a path
    assert report.ok
    assert not any((out / path).exists() for path in release.RELEASE_TOOLING)
    assert set(release.RELEASE_TOOLING) <= {e["path"] for e in report.excluded}


def _imports_release(source: str) -> bool:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(a.name.startswith("fym1.release") for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (
                (node.module or "").startswith("fym1.release")
                or (node.module == "fym1" and any(a.name == "release" for a in node.names))):
            return True
    return False


def test_nothing_shipped_imports_or_describes_the_release_tooling():
    shipped = {p: release.read_text(REPO_ROOT / p) for p in tracked() if not release.exclusion_reason(p)}
    assert shipped
    for path, text in shipped.items():
        assert text is not None, path
        assert "fym1.release" not in text and "INNER_NAME_ALLOWED" not in text, path
        if path.endswith(".py"):
            assert not _imports_release(text), path


def test_no_inner_name_allow_list_entry_names_an_unshipped_file():
    assert not any(release.exclusion_reason(path) for path, _ in release.INNER_NAME_ALLOWED)

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
        build_snapshot(repo, "HEAD", tmp_path / "public", make_export(tmp_path), tmp_path / where,
                       make_chats(tmp_path))
    assert FAKE_ID not in str(error.value)
    assert not (tmp_path / "public").exists()


def test_private_ids_file_empty_or_missing_is_refused(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    export = make_export(tmp_path)
    with pytest.raises(RefusedError, match="holds no ids"):
        build_snapshot(repo, "HEAD", tmp_path / "public", export, make_ids(tmp_path, "# only a comment\n\n"),
                       make_chats(tmp_path))
    with pytest.raises(RefusedError, match="could not be read"):
        build_snapshot(repo, "HEAD", tmp_path / "public", export, tmp_path / "missing.txt", make_chats(tmp_path))


# ---- copied chat text (lead decision 34) ------------------------------------------------------------------------


def _run(words: list[str]) -> str:
    return " ".join(words)


@pytest.mark.parametrize(("text", "hit"), [
    (f"Example reply: {_run(FAKE_CHAT_WORDS[2:10])}.", True),  # 8 words in a row
    (f"Example reply: {_run(FAKE_CHAT_WORDS[2:10]).upper()}!", True),  # any case
    (f'"text": "{_run(FAKE_CHAT_WORDS[:4])}\\n{_run(FAKE_CHAT_WORDS[4:8])}"', True),  # JSON escape between words
    (f"{_run(FAKE_CHAT_WORDS[:4])}\u200b {_run(FAKE_CHAT_WORDS[4:8])}", True),  # hidden character
    (f"Example reply: {_run(FAKE_CHAT_WORDS[2:9])}.", False),  # only 7 words in a row
    (f"{_run(FAKE_CHAT_WORDS[:4])} sy {_run(FAKE_CHAT_WORDS[4:8])}", False),  # a word between the halves
])
def test_has_chat_text_needs_an_8_word_run(tmp_path, text, hit):
    messages = release.export_texts(release.read_export(make_export(tmp_path)))
    assert len(messages) == len(FAKE_SENDERS) + 1
    assert release.has_chat_text(text, [release.chat_text_index(messages, "export")]) == hit


@pytest.mark.parametrize("letters", [
    lambda c: chr(ord(c) - ord("a") + 0xFF41),  # full-width
    lambda c: chr(ord(c) - ord("a") + 0x24D0),  # circled
])
def test_export_in_styled_letters_matches_plain_shipped_text_through_normalise(tmp_path, letters):
    styled = "".join(letters(c) if "a" <= c <= "z" else c for c in FAKE_CHAT)
    plain = f"Example reply: {_run(FAKE_CHAT_WORDS[1:9])}."
    as_written = release.NgramIndex([styled, FAKE_TURN], n=8)  # FAKE_TURN: circled letters alone give no 8-gram
    assert not as_written.shares_with(plain)  # read as written, the export misses it
    messages = release.export_texts(release.read_export(make_export(tmp_path, chat=styled)))
    assert release.has_chat_text(plain, [release.chat_text_index(messages, "export")])
    repo = make_repo(tmp_path, {**CLEAN_FILES, "docs/copied.md": plain})
    report = snap(repo, tmp_path / "public", tmp_path / "raw")
    assert report.chat_text_hits == ["docs/copied.md"] and not report.ok


def test_export_without_an_8_word_message_is_refused_before_anything_is_written(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    with pytest.raises(RefusedError, match="8-word message"):
        snap(repo, tmp_path / "public", make_export(tmp_path, chat=_run(FAKE_CHAT_WORDS[:7])))
    assert not (tmp_path / "public").exists() and not _leftovers(tmp_path)


def test_copied_chat_text_fails_with_counts_and_paths_only(tmp_path, capsys):
    copied = _run(FAKE_CHAT_WORDS[1:9])
    files = {**CLEAN_FILES, "paper/examples.tex": f"Reply: {copied}\n", "docs/near.md": _run(FAKE_CHAT_WORDS[:7]),
             f"docs/{FAKE_ID}.md": f"{copied}\n"}
    repo = make_repo(tmp_path, files)
    out, results = tmp_path / "public", tmp_path / "results.json"
    assert cli.main([*cli_args(tmp_path, repo, out), "--results", str(results)]) == 1
    assert not out.exists() and not _leftovers(tmp_path)  # refused: nothing written
    report = json.loads(results.read_text())
    assert report["ok"] is False
    assert report["chat_text"] == {"messages_indexed": 3, "anonymised_turns_indexed": 2, "ngram_words": 8,
                                   "files_total": 2,
                                   "files_with_hits": ["<redacted path 1>", "paper/examples.tex"]}
    printed = capsys.readouterr()
    for blob in (results.read_text(), printed.out + printed.err):
        assert copied not in blob and not any(word in blob for word in FAKE_CHAT_WORDS if len(word) > 5)
    assert "files sharing an 8-word run 2" in printed.out


@pytest.mark.parametrize("copied", [
    FAKE_TURN,  # the whole anonymised turn, fake name and joined messages included
    _run(FAKE_TURN_WORDS[3:11]),  # 8 words across the join
    _run(FAKE_TURN_WORDS[3:11]).replace("a", "\uff41"),  # full-width letters: read as the anonymiser normalises
])
def test_an_anonymised_chat_turn_copied_into_a_shipped_file_fails(tmp_path, capsys, copied):
    export = make_export(tmp_path)
    assert not release.has_chat_text(copied, [release.chat_text_index(
        release.export_texts(release.read_export(export)), "export")])  # the raw export alone misses it
    repo = make_repo(tmp_path, {**CLEAN_FILES, "spaces/chat/examples.md": f"Example:\n{copied}\n"})
    out, results = tmp_path / "public", tmp_path / "results.json"
    args = ["--repo", str(repo), "--export", str(export), "--private-ids", str(make_ids(tmp_path)),
            "--chats", str(make_chats(tmp_path)), "--out", str(out), "--results", str(results)]
    assert cli.main(args) == 1
    assert not out.exists() and not _leftovers(tmp_path)
    report = json.loads(results.read_text())
    assert report["chat_text"]["files_with_hits"] == ["spaces/chat/examples.md"]
    assert report["chat_text"]["anonymised_turns_indexed"] == 2
    printed = capsys.readouterr()
    for blob in (results.read_text(), printed.out + printed.err):
        assert not any(word in blob for word in FAKE_TURN_WORDS if len(word) > 5)
    assert "anonymised turns 2" in printed.out


def test_the_system_turn_is_not_indexed(tmp_path):
    repo = make_repo(tmp_path, {**CLEAN_FILES, "docs/prompt.md": "system prompt\n"})
    assert release.read_chats(make_chats(tmp_path), []) == [FAKE_TURN, "eny"]
    assert snap(repo, tmp_path / "public", make_export(tmp_path)).ok


@pytest.mark.parametrize(("text", "match"), [
    (None, "could not be read"),  # missing
    ("", "holds no turns"),
    ('{"id": "c1", "messages": [{"role": "system", "content": "only the prompt"}]}\n', "holds no turns"),
    ("not json\n", "could not be read"),
    ('{"id": "c1"}\n', "could not be read"),
    ('{"id": "c1", "messages": [{"role": "user", "content": 7}]}\n', "could not be read"),
    (json.dumps({"id": "c1", "messages": [{"role": "user", "content": _run(FAKE_TURN_WORDS[:7])}]}) + "\n",
     "chats file has no 8-word message"),
])
def test_chats_file_missing_empty_or_malformed_is_refused_with_exit_2(tmp_path, capsys, text, match):
    repo = make_repo(tmp_path, CLEAN_FILES)
    out, results = tmp_path / "public", tmp_path / "results.json"
    args = cli_args(tmp_path, repo, out)
    chats = Path(args[args.index("--chats") + 1])
    if text is None:
        chats.unlink()
    else:
        chats.write_text(text, encoding="utf-8")
    assert cli.main([*args, "--results", str(results)]) == 2
    assert match in capsys.readouterr().err
    assert not out.exists() and not results.exists() and not _leftovers(tmp_path)


@pytest.mark.parametrize("where", ["repo/chats.jsonl", "worktree/chats.jsonl"])
def test_chats_file_inside_the_repo_or_a_worktree_is_refused(tmp_path, where):
    repo = make_repo(tmp_path, CLEAN_FILES)
    git(repo, "worktree", "add", "-q", str(tmp_path / "worktree"))
    (repo / ".git/info/exclude").write_text("chats.jsonl\n")  # ignored, so the tree stays clean
    shutil.copy(make_chats(tmp_path), tmp_path / where)
    with pytest.raises(RefusedError, match="chats file is inside the repo"):
        build_snapshot(repo, "HEAD", tmp_path / "public", make_export(tmp_path), make_ids(tmp_path), tmp_path / where)
    assert not (tmp_path / "public").exists()


def test_a_7_word_overlap_passes_end_to_end(tmp_path):
    repo = make_repo(tmp_path, {**CLEAN_FILES, "docs/near.md": f"{_run(FAKE_CHAT_WORDS[:7])}\n"})
    report = snap(repo, tmp_path / "public", make_export(tmp_path))
    assert report.ok and report.chat_text_hits == []
    assert (tmp_path / "public/docs/near.md").exists()


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


# ---- reading files: only known text types and PDFs; archives are refused ------------------------------------------


def _pdf(info: str, page_text: str = "") -> bytes:
    """A minimal one-page PDF with `info` as its document info dictionary and `page_text` (if any) on its page."""
    page = "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >>"
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>", page, info]
    if page_text:
        stream = f"BT /F1 8 Tf 10 400 Td ({page_text}) Tj ET"
        objects[2] = ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 2000 800] /Contents 5 0 R "
                      "/Resources << /Font << /F1 6 0 R >> >> >>")
        objects += [f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
                    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    body, offsets = "%PDF-1.4\n", []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{obj}\nendobj\n"
    xref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 4 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return body.encode("latin-1")


@pytest.mark.parametrize(("name", "data", "readable", "hits"), [
    ("docs/a.md", "Salama Zafy".encode(), True, 1),
    ("docs/a.md", b"\xef\xbb\xbfSalama Zafy", True, 1),  # UTF-8 BOM
    ("docs/a.md", "Salama Zafy".encode("utf-16"), True, 1),  # UTF-16 with its BOM
    ("docs/a.md", "Salama Zafy".encode("utf-16-le"), False, 0),  # no BOM: NUL bytes
    ("docs/a.md", "Salama Zafy caf\u00e9".encode("latin-1"), False, 0),  # not UTF-8: no guessing
    ("assets/logo.png", b"\x89PNG\r\n\x1a\nZafy", False, 0),  # not a text type
    ("LICENSE", b"Apache", True, 0),
    ("data/a.txt.gz", b"Zafy", False, 0),  # an archive is never read (and is refused before the scan)
])
def test_only_known_text_types_are_read(tmp_path, name, data, readable, hits):
    path = tmp_path / PurePath(name).name
    path.write_bytes(data)
    text = release.read_text(path)
    assert (text is not None) == readable
    assert count_name_hits(text or "", NAMES) == hits


needs_poppler = pytest.mark.skipif(not all(map(shutil.which, ("pdftotext", "pdfinfo", "pdfdetach"))),
                                   reason="poppler not installed (a PDF is then unscannable and fails the check)")


@needs_poppler
def test_pdf_metadata_is_scanned(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(_pdf("<< /Author (Zafy Bodo) /Title (notes) >>"))
    assert count_name_hits(release.read_text(path), NAMES) == 2


@needs_poppler
@pytest.mark.parametrize("where", ["metadata", "page"])
def test_chat_text_in_a_pdf_fails(tmp_path, where):
    copied = _run(FAKE_CHAT_WORDS[:8])
    pdf = _pdf(f"<< /Title ({copied}) >>") if where == "metadata" else _pdf("<< /Title (notes) >>", copied)
    repo = make_repo(tmp_path, {**CLEAN_FILES, "paper/main.pdf": pdf})
    report = snap(repo, tmp_path / "public", make_export(tmp_path))
    assert report.chat_text_hits == ["paper/main.pdf"] and not report.unscannable and not report.ok


@pytest.mark.parametrize("path", ["data/a.zip", "data/a.txt.gz", "dist/x.tar.gz", "dist/x.TGZ", "vendor/x.whl",
                                  "lib/x.jar", "data/x.7z", "data/x.rar", "data/x.tar", "data/x.bz2", "data/x.xz"])
def test_an_archive_fails_the_release_before_anything_is_written(tmp_path, capsys, path):
    repo = make_repo(tmp_path, {**CLEAN_FILES, path: b"not read"})
    out = tmp_path / "public"
    with pytest.raises(RefusedError, match="1 archive files") as error:
        snap(repo, out, make_export(tmp_path))
    assert path not in str(error.value)  # counted, never named: a path may hold a name
    assert not out.exists() and not _leftovers(tmp_path)
    (tmp_path / "cli").mkdir()
    assert cli.main([*cli_args(tmp_path / "cli", repo, out), "--results", str(tmp_path / "r.json")]) == 2
    assert not out.exists() and not (tmp_path / "r.json").exists()
    assert "refused" in capsys.readouterr().err


@pytest.mark.parametrize(("path", "archive"), [("a.tar.gz", True), ("a.ZIP", True), ("v0.1.jar.md", True),
                                               ("notes.md", False), ("zip.md", False), ("data/gz", False)])
def test_is_archive_reads_every_suffix_of_the_file_name(path, archive):
    assert release.is_archive(path) == archive


def _tar_header() -> bytes:
    return b"\0" * 257 + b"ustar\x0000" + b"\0" * 242


ARCHIVE_BYTES = {  # synthetic: just the magic, then filler
    "zip": b"PK\x03\x04 filler", "empty-zip": b"PK\x05\x06 filler", "gzip": b"\x1f\x8b filler",
    "bzip2": b"BZh91 filler", "xz": b"\xfd7zXZ\x00 filler", "7z": b"7z\xbc\xaf\x27\x1c filler",
    "rar": b"Rar!\x1a\x07\x00 filler", "zstd": b"\x28\xb5\x2f\xfd filler", "tar": _tar_header(),
    "zip-after-text": b"# notes\nplain text\nPK\x03\x04 filler", "zip-end-after-text": b"plain\nPK\x05\x06\0\0",
}


@pytest.mark.parametrize("kind", sorted(ARCHIVE_BYTES))
@pytest.mark.parametrize("name", ["docs/notes.md", "data/words.txt"])
def test_an_archive_under_a_text_name_fails_before_anything_is_written(tmp_path, capsys, kind, name):
    repo = make_repo(tmp_path, {**CLEAN_FILES, name: ARCHIVE_BYTES[kind]})
    out = tmp_path / "public"
    with pytest.raises(RefusedError, match="1 archive files") as error:
        snap(repo, out, make_export(tmp_path))
    assert name not in str(error.value) and PurePath(name).name not in str(error.value)
    assert not out.exists() and not _leftovers(tmp_path)
    assert name not in capsys.readouterr().out


def test_a_pdf_with_an_appended_zip_fails_before_anything_is_written(tmp_path):
    pdf = _pdf("<< /Title (notes) >>") + b"PK\x03\x04 filler PK\x05\x06\0\0"
    repo = make_repo(tmp_path, {**CLEAN_FILES, "paper/x.pdf": pdf})
    out = tmp_path / "public"
    with pytest.raises(RefusedError, match="1 archive files") as error:
        snap(repo, out, make_export(tmp_path))
    assert "x.pdf" not in str(error.value)
    assert not out.exists() and not _leftovers(tmp_path)


@pytest.mark.parametrize(("data", "archive"), [
    (b"# notes\nplain text, PK and zip as words\n", False),
    (b"\x89PNG\r\n\x1a\n", False),
    (b"x" * 300, False),
    (b"x" * 257 + b"ustar", True),
    (b"x" * 256 + b"ustar", False),  # ustar only counts at offset 257
    (b"text BZh later", False),  # bzip2 magic only counts at the start
    (b"text PK\x05\x06 later", True),  # a zip record counts anywhere
])
def test_has_archive_magic(data, archive):
    assert release.has_archive_magic(data) == archive


def test_unscannable_files_fail_end_to_end(tmp_path):
    repo = make_repo(tmp_path, {**CLEAN_FILES, "docs/old.md": "caf\u00e9 Zafy".encode("latin-1"),
                                "assets/logo.png": b"\x89PNG\r\n"})
    report = snap(repo, tmp_path / "public", make_export(tmp_path))
    assert sorted(report.unscannable) == ["assets/logo.png", "docs/old.md"]
    assert not report.ok


# ---- no partial output ----------------------------------------------------------------------------------------


def _leftovers(tmp_path: Path) -> list[Path]:
    return [p for p in tmp_path.iterdir() if p.name.startswith(".public.partial-")]


def test_failed_check_writes_nothing(tmp_path):
    repo = make_repo(tmp_path, {**CLEAN_FILES, "docs/notes.md": "Thanks to Zafy.\n"})
    out = tmp_path / "public"
    report = snap(repo, out, make_export(tmp_path))
    assert not report.ok
    assert not out.exists() and not _leftovers(tmp_path)


def test_failed_check_keeps_an_empty_output_dir_empty(tmp_path):
    repo = make_repo(tmp_path, {**CLEAN_FILES, "docs/notes.md": "Thanks to Zafy.\n"})
    out = tmp_path / "public"
    out.mkdir()
    assert not snap(repo, out, make_export(tmp_path)).ok
    assert list(out.iterdir()) == [] and not _leftovers(tmp_path)


def test_error_while_writing_deletes_the_partial_snapshot(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, CLEAN_FILES)
    out = tmp_path / "public"
    write_files = release.write_files

    def write_then_fail(files, folder):
        write_files(files, folder)
        raise OSError("disk full")

    monkeypatch.setattr(release, "write_files", write_then_fail)
    with pytest.raises(OSError):
        snap(repo, out, make_export(tmp_path))
    assert not out.exists() and not _leftovers(tmp_path)


def test_clean_snapshot_is_moved_into_place(tmp_path):
    repo = make_repo(tmp_path, CLEAN_FILES)
    out = tmp_path / "public"
    out.mkdir()
    assert snap(repo, out, make_export(tmp_path)).ok
    assert (out / "README.md").exists() and not _leftovers(tmp_path)


# ---- report paths -----------------------------------------------------------------------------------------------


def test_every_flagged_path_is_redacted(tmp_path):
    named = ["docs/%5Aafy.md", "docs/salutBodo.md", "docs/Z_a_f_y.md", "docs/&#90;afy.md", "docs/B0d0.md",
             f"docs/{FAKE_ID}.md"]
    repo = make_repo(tmp_path, {**CLEAN_FILES, **{path: f"{FAKE_HF}\n" for path in named}})
    report = snap(repo, tmp_path / "public", make_export(tmp_path))
    assert not report.ok
    assert len(report.secret_hits) == len(named)
    assert all(path.startswith("<redacted path ") for path in report.secret_hits)
    written = json.dumps(report.as_dict())
    assert not any(path in written for path in named)


# ---- the owner-chosen author string (lead decision 31) ---------------------------------------------------------


def _author(monkeypatch, author: str = "Zafy Bodo") -> None:
    monkeypatch.setattr(release, "AUTHOR_SHA256", hashlib.sha256(author.encode()).hexdigest())


@pytest.mark.parametrize(("path", "text", "hits", "allowed"), [
    ("README.md", "by Zafy Bodo, 2026", 0, 1),
    ("paper/fym1.tex", "\\author{Zafy Bodo}\nZafy Bodo again", 0, 2),
    ("README.md", "thanks Zafy", 1, 0),  # the name alone
    ("README.md", "Zafy  Bodo", 2, 0),  # another form: two spaces
    ("README.md", "zafy bodo", 2, 0),  # another form: lower case
    ("README.md", "Zafy Bodoko", 2, 0),  # glued to a longer word
    ("docs/notes.md", "by Zafy Bodo", 2, 0),  # another file
])
def test_only_the_exact_author_string_in_author_files_passes(monkeypatch, path, text, hits, allowed):
    _author(monkeypatch)
    stripped, found = release.without_author(path, text)
    assert (count_name_hits(stripped, NAMES), found) == (hits, allowed)


def test_author_string_passes_end_to_end_and_is_counted(tmp_path, monkeypatch):
    _author(monkeypatch)
    repo = make_repo(tmp_path, {**CLEAN_FILES, "README.md": "# demo\nby Zafy Bodo\n"})
    report = snap(repo, tmp_path / "public", make_export(tmp_path))
    assert report.ok and report.author_strings_allowed == 1


def test_the_real_author_string_is_in_every_author_file():
    for path in release.AUTHOR_FILES - {"paper/fym1.pdf"}:
        assert release.without_author(path, (REPO_ROOT / path).read_text("utf-8"))[1] >= 1, path
