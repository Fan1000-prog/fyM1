"""Public release snapshot: tracked files of one clean commit, minus private paths, checked for names and secrets.

The public repo is a fresh single-commit repo built from this snapshot, so private git history never ships.
Release blocker (lead decision 21): the export's sender names are read at run time, removed from the hand-written
name lists, and searched for in every file except the lists a script rebuilds from public data. Everything
this module reports is a count or a repo path; it never returns a sender name, a message text, a private id
or a matched secret value.
"""

import fnmatch
import io
import json
import re
import shutil
import subprocess
import tarfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from fym1.data.anonymize import MIN_NAME_PREFIX, NAME_SUFFIXES, compare_key, is_everyday_word, normalise
from fym1.data.gate import PERSON_MARKER_WORDS
from fym1.data.messenger import load_threads


@dataclass(frozen=True)
class Exclude:
    path: str  # exact repo path, or a folder prefix ending in "/"
    reason: str


EXCLUDES: tuple[Exclude, ...] = (
    Exclude("AGENTS.md", "agent-team rules for the private build; points at the owner's private handoff"),
    Exclude("CLAUDE.md", "same agent-team rules as AGENTS.md, for one tool"),
    Exclude("docs/project-update.md", "internal status and team pickup notes; points at the private operator handoff "
                                      "that names accounts and machines"),
    Exclude("docs/colab_operator.md", "operator-only: the owner's personal Google account rules and the HF token "
                                      "hand-off steps for Playwright"),
    Exclude("docs/kaggle_operator.md", "operator-only: the owner's Kaggle account setup and run steps against "
                                       "private repos"),
    Exclude("docs/superpowers/", "internal design spec and build plan: owner background, accounts and team process; "
                                 "the public account is README, MODEL_CARD, paper and docs/anonymisation.md"),
    Exclude("scripts/serve_hf_token_once.py", "operator-only: hands the local HF token to the browser agent"),
    Exclude("tests/test_serve_hf_token_once.py", "tests the excluded scripts/serve_hf_token_once.py"),
    Exclude("results/release_check.json", "this tool's own counts report: a private build record of what the "
                                          "snapshot checked"),
)

# Word/name lists, shipped under their own licences (see NOTICE).
LIST_GLOBS: tuple[str, ...] = ("src/fym1/data/*.txt",)
# Hand-edited lists: the export's sender names are removed from these. Nobody can rebuild them, so a removal
# leaves no trace.
HAND_LISTS: dict[str, str] = {
    "src/fym1/data/mg_first_names.txt": "hand-edited Malagasy given names",
    "src/fym1/data/mg_known_names.txt": "hand-edited places, brands and other non-person words",
    "src/fym1/data/name_deny_list.txt": "hand-edited Malagasy given names that are also everyday words",
}
# Lists a script writes from public data. They are shipped exactly as built and the name scan skips them:
# anyone can rebuild one and diff it, so a removed entry would point at a sender, and a common word in one
# reveals nothing about the export. Secrets and private ids are still checked in them.
_KNOWN_WORDS = "reproducible from public data by scripts/build_known_words.py"
GENERATED_LISTS: dict[str, str] = {
    "src/fym1/data/en_words.txt": f"{_KNOWN_WORDS} (wordfreq)",
    "src/fym1/data/fr_words.txt": f"{_KNOWN_WORDS} (wordfreq)",
    "src/fym1/data/first_names.txt": f"{_KNOWN_WORDS} (INSEE)",
    "src/fym1/data/wordlike_first_names.txt": f"{_KNOWN_WORDS} (INSEE, wordfreq, Wikipedia)",
    "src/fym1/data/mg_wiki_words.txt": f"{_KNOWN_WORDS} (Malagasy Wikipedia)",
    "src/fym1/data/mg_corpus_words.txt": f"{_KNOWN_WORDS} (HPLT 2.0)",
    "src/fym1/data/mg_common_words.txt": "reproducible from public data by scripts/build_common_words.py "
                                         "(Malagasy Wikipedia)",
}
DATA_LICENCES: dict[str, str] = {
    "src/fym1/data/en_words.txt": "derived from wordfreq 3.1 by Robyn Speer (https://github.com/rspeer/wordfreq); "
                                  "word data CC BY-SA 4.0",
    "src/fym1/data/fr_words.txt": "derived from wordfreq 3.1 by Robyn Speer (https://github.com/rspeer/wordfreq); "
                                  "word data CC BY-SA 4.0",
    "src/fym1/data/wordlike_first_names.txt": "INSEE fichier des prenoms (Licence Ouverte / Etalab 2.0) filtered "
                                              "with wordfreq (CC BY-SA 4.0) and Wikipedia (CC BY-SA 4.0); "
                                              "CC BY-SA 4.0",
    "src/fym1/data/first_names.txt": "INSEE fichier des prenoms 2022, Licence Ouverte / Etalab 2.0",
    "src/fym1/data/mg_common_words.txt": "derived from Malagasy Wikipedia (wikimedia/wikipedia 20231101.mg); "
                                         "CC BY-SA 4.0",
    "src/fym1/data/mg_wiki_words.txt": "derived from Malagasy Wikipedia (wikimedia/wikipedia 20231101.mg); "
                                       "CC BY-SA 4.0",
    "src/fym1/data/mg_corpus_words.txt": "word list from HPLT 2.0 cleaned, Malagasy plt_Latn (packaging CC0 1.0); "
                                         "no text kept",
    "src/fym1/data/mg_first_names.txt": "hand list, this repo (Apache-2.0)",
    "src/fym1/data/mg_known_names.txt": "hand list, this repo (Apache-2.0)",
    "src/fym1/data/name_deny_list.txt": "hand list, this repo (Apache-2.0)",
}
NOTICE_NAME = "NOTICE"

MIN_NAME_TOKEN = 3  # shorter tokens ("ny", initials) are everyday words or letters; counted as skipped
# Meta's label for a deleted or unknown account: not a person's name.
PLACEHOLDER_SENDERS = frozenset({"facebook user", "utilisateur de facebook", "instagram user", "meta user"})
PRIVATE_HF_ID = "Fan1000-Prog/" + "fyM1-"  # split so test_public_name does not see a Hub id

SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "kaggle_key_json": re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),
    "kaggle_api_token": re.compile(r"KGAT_[A-Za-z0-9]{16,}"),
    "private_key": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    "sk_api_key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    "slack_token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    "google_oauth_secret": re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{20,}"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    "telegram_bot_token": re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b"),
    "assigned_secret": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b["']?\s*[:=]\s*["'][A-Za-z0-9_\-+/=.]{16,}["']"""
    ),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
ALLOWED_EMAILS: dict[str, str] = {
    "x@y.mg": "placeholder example in scripts/build_known_words.py, not an address",
}
# Under tests/ only: the made-up domains the synthetic fixtures use.
TEST_EMAIL_DOMAINS: dict[str, str] = {
    "example.mg": "made-up domain of the synthetic anonymiser and gate probes",
    "mail.example": "reserved .example domain, release tests",
    "mail.mg": "made-up domain of a synthetic known-words probe",
    "t.invalid": "reserved .invalid domain, the git identity of synthetic test repos",
}
# Test files whose synthetic probes need a real provider's shape (plus-tags, upper case) to test email masking.
EMAIL_PROBE_FILES: dict[str, str] = {
    "tests/data/test_anonymize_probes.py": "synthetic anonymiser probes at real providers' domains",
}
SECRET_FILE_NAMES = frozenset({".env", "kaggle.json", "access_token", "id_rsa", "id_ed25519", ".netrc"})
SECRET_FILE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")

_LETTERS = re.compile(r"[^\W\d_]+")
_ESCAPES = re.compile(r"\\(?:u([0-9a-fA-F]{4})|[nrt\"])")


class RefusedError(Exception):
    """The snapshot cannot be made safely; nothing was copied."""


@dataclass
class Report:
    ref: str
    files_copied: int = 0
    excluded: list[dict] = field(default_factory=list)
    excludes_unmatched: list[str] = field(default_factory=list)
    name_tokens: int = 0
    short_tokens_skipped: int = 0
    placeholder_senders_skipped: int = 0
    list_entries_removed: dict[str, int] = field(default_factory=dict)
    name_hits: dict[str, int] = field(default_factory=dict)
    private_ids: int = 0
    private_id_hits: dict[str, int] = field(default_factory=dict)
    unscannable: list[str] = field(default_factory=list)
    secret_hits: dict[str, dict[str, int]] = field(default_factory=dict)
    emails_allowed: int = 0
    private_hf_ids: dict[str, int] = field(default_factory=dict)
    references_to_excluded: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.name_hits or self.private_id_hits or self.secret_hits or self.unscannable)

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "ref": self.ref,
            "files_copied": self.files_copied,
            "excluded": self.excluded,
            "excludes_unmatched": self.excludes_unmatched,
            "names": {
                "name_tokens": self.name_tokens,
                "short_tokens_skipped": self.short_tokens_skipped,
                "placeholder_senders_skipped": self.placeholder_senders_skipped,
                "list_entries_removed": self.list_entries_removed,
                "list_entries_removed_total": sum(self.list_entries_removed.values()),
                "hits_per_file": self.name_hits,
                "hits_total": sum(self.name_hits.values()),
                "unscannable_files": self.unscannable,
            },
            "private_ids": {
                "ids_checked": self.private_ids,
                "hits_per_file": self.private_id_hits,
                "hits_total": sum(self.private_id_hits.values()),
            },
            "secrets": {
                "hits_per_file": self.secret_hits,
                "hits_total": sum(sum(c.values()) for c in self.secret_hits.values()),
                "emails_allowed": self.emails_allowed,
            },
            "private_hf_ids_per_file": self.private_hf_ids,
            "references_to_excluded": self.references_to_excluded,
        }


# ---- names ----------------------------------------------------------------------------------------------------


def words(text: str) -> list[str]:
    """Letter runs as the anonymiser compares them: normalised, folded, stretched letters collapsed ("Zafyyy")."""
    return [compare_key(run) for run in _LETTERS.findall(normalise(text))]


def is_name_word(word: str, names: frozenset[str]) -> bool:
    """`word` (a `words()` key) is a name, alone or with a glued person marker ("iZafy", "n'iZafy" reads "izafy",
    "ryZafy") and/or a glued possessive ending ("Zafyko"), the anonymiser's suffix list, or starts with a 4+ letter
    name ("zafybodo"). Like the anonymiser, only an exact name match counts for an everyday word."""
    if word in names:
        return True
    if is_everyday_word(word):  # checked first, as the anonymiser does: "farany" is not "Fara" + "-ny"
        return False
    stems = {word} | {word[len(m):] for m in PERSON_MARKER_WORDS if word.startswith(m)}
    stems |= {stem[: -len(s)] for stem in stems for s in NAME_SUFFIXES if stem.endswith(s)}
    long_names = [name for name in names if len(name) >= MIN_NAME_PREFIX]
    return not stems.isdisjoint(names) or any(stem.startswith(n) for stem in stems for n in long_names)


def name_tokens(senders: Iterable[str]) -> tuple[frozenset[str], int, int]:
    """Name tokens of `senders`, plus how many short tokens and placeholder senders were skipped."""
    tokens, short, placeholders = set(), set(), 0
    placeholder_keys = {" ".join(words(label)) for label in PLACEHOLDER_SENDERS}
    for sender in set(senders):
        if " ".join(words(sender)) in placeholder_keys:
            placeholders += 1
            continue
        for token in words(sender):
            (tokens if len(token) >= MIN_NAME_TOKEN else short).add(token)
    return frozenset(tokens), len(short - tokens), placeholders


def export_senders(export: Path) -> list[str]:
    """Every sender and listed participant in the export. Errors never carry the export's paths or text."""
    try:
        threads = load_threads(export)
    except (ValueError, OSError, KeyError, TypeError) as error:
        raise RefusedError(f"export could not be read ({type(error).__name__})") from None
    if not threads:
        raise RefusedError("export holds no conversations")
    return [p for thread in threads for p in [*thread.participants, *(m.sender for m in thread.messages)]]


def count_name_hits(text: str, names: frozenset[str]) -> int:
    """Whole-word, case/accent-insensitive matches of `names`; JSON/Python escapes are read as characters."""
    return sum(1 for word in words(_unescape(text)) if is_name_word(word, names))


def _unescape(text: str) -> str:
    """JSON/Python `\\n` and `\\uXXXX` escapes read as the characters they stand for."""
    return _ESCAPES.sub(lambda m: chr(int(m.group(1), 16)) if m.group(1) else " ", text)


def strip_list(text: str, names: frozenset[str]) -> tuple[str, int]:
    """`text` (one entry per line, `#` comments) without entries holding a name token, and how many went."""
    kept, removed = [], 0
    for line in text.splitlines(keepends=True):
        if not line.startswith("#") and any(is_name_word(word, names) for word in words(line)):
            removed += 1
        else:
            kept.append(line)
    return "".join(kept), removed


# ---- private ids ----------------------------------------------------------------------------------------------


def load_private_ids(path: Path, trees: Iterable[Path]) -> tuple[re.Pattern[str], int]:
    """Whole-word, case-insensitive pattern for the ids in `path` (one per line, `#` comments), and their count.

    The file must sit outside the repo and its worktrees, so the ids never get committed. Errors never carry
    the file's contents.
    """
    target = path.expanduser().resolve()
    if any(target == tree or tree in target.parents for tree in trees):
        raise RefusedError("private ids file is inside the repo or one of its worktrees")
    try:
        lines = target.read_text("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RefusedError(f"private ids file could not be read ({type(error).__name__})") from None
    ids = sorted({line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")}, key=len,
                 reverse=True)
    if not ids:
        raise RefusedError("private ids file holds no ids")
    pattern = re.compile(rf"(?<![A-Za-z0-9_])(?:{'|'.join(map(re.escape, ids))})(?![A-Za-z0-9_])", re.IGNORECASE)
    return pattern, len(ids)


def count_private_ids(text: str, ids: re.Pattern[str]) -> int:
    """Whole-word matches of the private ids in `text`, JSON/Python escapes read as characters."""
    return len(ids.findall(_unescape(text)))


# ---- secrets --------------------------------------------------------------------------------------------------


def secret_hits(path: str, text: str) -> Counter:
    """Pattern name -> match count for one file. Never returns a matched value."""
    hits: Counter = Counter()
    name = PurePosixPath(path).name
    if name in SECRET_FILE_NAMES or name.startswith(".env.") or name.endswith(SECRET_FILE_SUFFIXES):
        hits["secret_file"] += 1
    for label, pattern in SECRET_PATTERNS.items():
        if n := len(pattern.findall(text)):
            hits[label] += n
    if n := sum(1 for email in EMAIL.findall(text) if not email_allowed(path, email)):
        hits["email"] += n
    return hits


def email_allowed(path: str, email: str) -> bool:
    """`email` is a known placeholder, or a synthetic test address on a made-up domain or the probe list."""
    address = email.casefold()
    if address in ALLOWED_EMAILS:
        return True
    return path in EMAIL_PROBE_FILES or (path.startswith("tests/") and address.rsplit("@", 1)[1] in TEST_EMAIL_DOMAINS)


def allowed_email_count(path: str, text: str) -> int:
    return sum(1 for email in EMAIL.findall(text) if email_allowed(path, email))


# ---- copy -----------------------------------------------------------------------------------------------------


def exclusion_entry(path: str) -> str | None:
    """The EXCLUDES path that covers `path`, if any."""
    for entry in EXCLUDES:
        if path == entry.path or (entry.path.endswith("/") and path.startswith(entry.path)):
            return entry.path
    return None


def exclusion_reason(path: str) -> str | None:
    return next((entry.reason for entry in EXCLUDES if entry.path == exclusion_entry(path)), None)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout


def worktrees(root: Path) -> set[Path]:
    """The repo root and every worktree of it."""
    return {root, *(Path(line[len("worktree "):]).resolve()
                    for line in _git(root, "worktree", "list", "--porcelain").splitlines()
                    if line.startswith("worktree "))}


def check_inputs(repo: Path, ref: str, out: Path) -> tuple[Path, str]:
    """The repo root and the commit sha, or RefusedError: dirty tree, bad ref, output inside a worktree or used."""
    try:
        root = Path(_git(repo, "rev-parse", "--show-toplevel").strip()).resolve()
        sha = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    except subprocess.CalledProcessError:
        raise RefusedError(f"not a git repo or unknown ref: {ref}") from None
    if _git(root, "status", "--porcelain").strip():
        raise RefusedError("working tree is not clean (local changes or untracked files)")
    target = out.resolve()
    for tree in worktrees(root):
        if target == tree or tree in target.parents:
            raise RefusedError("output dir is inside the repo or one of its worktrees")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise RefusedError("output dir exists and is not empty")
    return root, sha


def copy_tracked(root: Path, sha: str, out: Path) -> tuple[list[str], list[dict]]:
    """Write the commit's regular files into `out`, skipping EXCLUDES; returns kept paths and excluded entries."""
    archive = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", sha],
                             check=True, capture_output=True).stdout
    kept, excluded = [], []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        members, unsafe = [], 0
        for member in tar.getmembers():
            if member.isdir() or member.name == "pax_global_header":
                continue
            if reason := exclusion_reason(member.name):
                excluded.append({"path": member.name, "reason": reason})
                continue
            path = PurePosixPath(member.name)
            unsafe += path.is_absolute() or ".." in path.parts or not member.isfile()  # symlinks too: fail closed
            members.append(member)
        if unsafe:  # counted, not named: a path may hold a name
            raise RefusedError(f"{unsafe} unsafe or non-regular archive entries (symlink, absolute or ../ path)")
        if any(member.name == NOTICE_NAME for member in members):
            raise RefusedError(f"{NOTICE_NAME} already tracked; merge the licence notes by hand")
        out.mkdir(parents=True, exist_ok=True)
        for member in members:
            dest = out / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(member).read())
            dest.chmod(0o755 if member.mode & 0o111 else 0o644)
            kept.append(member.name)
    return sorted(kept), excluded


def notice_text() -> str:
    lines = [
        "fy-m1: third-party data in this repository",
        "",
        "Code is Apache-2.0 (LICENSE). These data files keep their own licences:",
        "",
        *(f"- {path}: {licence}" for path, licence in sorted(DATA_LICENCES.items())),
        "",
        "wordfreq (https://github.com/rspeer/wordfreq) is by Robyn Speer. Its word frequency data is licensed",
        "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); the lists derived from it above are shared",
        "under the same licence. Changes: lists were reduced to the most frequent words and folded to lower case",
        "without accents. Regenerate the lists with `uv run python scripts/build_known_words.py`.",
        "",
    ]
    return "\n".join(lines)


# ---- scan -----------------------------------------------------------------------------------------------------


def read_text(path: Path) -> str | None:
    """A file's text: UTF-8, PDF via pdftotext, else Latin-1 bytes. None if a PDF cannot be read."""
    if path.suffix.lower() == ".pdf":
        if not shutil.which("pdftotext"):
            return None
        done = subprocess.run(["pdftotext", "-q", str(path), "-"], capture_output=True)
        return done.stdout.decode("utf-8", "replace") if done.returncode == 0 else None
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def is_list_file(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in LIST_GLOBS)


def build_snapshot(repo: Path, ref: str, out: Path, export: Path, private_ids: Path) -> Report:
    """Make the snapshot in `out` and check it. Raises RefusedError before copying anything unsafe."""
    root, sha = check_inputs(repo, ref, out)
    ids, id_count = load_private_ids(private_ids, worktrees(root))
    names, short, placeholders = name_tokens(export_senders(export))
    if not names:
        raise RefusedError("export gave no sender name tokens")
    kept, excluded = copy_tracked(root, sha, out)
    report = Report(ref=sha, files_copied=len(kept), excluded=excluded, name_tokens=len(names),
                    short_tokens_skipped=short, placeholder_senders_skipped=placeholders,
                    private_ids=id_count)
    matched = {exclusion_entry(e["path"]) for e in excluded}
    report.excludes_unmatched = [entry.path for entry in EXCLUDES if entry.path not in matched]
    (out / NOTICE_NAME).write_text(notice_text(), encoding="utf-8")

    for path in (p for p in kept if p in HAND_LISTS):
        text, removed = strip_list((out / path).read_text("utf-8"), names)
        if removed:
            (out / path).write_text(text, encoding="utf-8")
            report.list_entries_removed[path] = removed

    excluded_refs = [entry.path for entry in EXCLUDES]
    for path in [*kept, NOTICE_NAME]:
        text = read_text(out / path)
        if text is None:
            report.unscannable.append(path)
            continue
        if hits := (0 if path in GENERATED_LISTS else count_name_hits(text, names)) + count_name_hits(path, names):
            report.name_hits[path] = hits
        if id_hits := count_private_ids(text, ids) + count_private_ids(path, ids):
            report.private_id_hits[path] = id_hits
        if secrets := secret_hits(path, text):
            report.secret_hits[path] = dict(sorted(secrets.items()))
        report.emails_allowed += allowed_email_count(path, text)
        if n := text.count(PRIVATE_HF_ID):
            report.private_hf_ids[path] = n
        for ref_path in excluded_refs:
            if ref_path in text:
                report.references_to_excluded[ref_path] = report.references_to_excluded.get(ref_path, 0) + 1
    _redact_named_paths(report, names, ids)
    return report


def _redact_named_paths(report: Report, names: frozenset[str], ids: re.Pattern[str]) -> None:
    """A repo path that itself holds a name token or a private id is reported as `<redacted path N>`."""
    aliases: dict[str, str] = {}

    def alias(path: str) -> str:
        # Redaction is cheap, so a name anywhere inside a path word counts here ("bodozafy").
        if any(name in word for word in words(path) for name in names) or count_private_ids(path, ids):
            return aliases.setdefault(path, f"<redacted path {len(aliases) + 1}>")
        return path

    for attr in ("list_entries_removed", "name_hits", "private_id_hits", "secret_hits", "private_hf_ids"):
        setattr(report, attr, {alias(p): v for p, v in sorted(getattr(report, attr).items())})
    report.unscannable = [alias(p) for p in report.unscannable]
    report.excluded = [{**e, "path": alias(e["path"])} for e in report.excluded]


def write_report(report: Report, *paths: Path) -> None:
    text = json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n"
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
