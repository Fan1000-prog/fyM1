"""Public release snapshot: tracked files of one clean commit, minus private paths, checked for names and secrets.

The public repo is a fresh single-commit repo built from this snapshot, so private git history never ships.
Release blocker (lead decision 21): the export's sender names are read at run time, removed from every word/name
list, and searched for in every file. Everything this module reports is a count or a repo path; it never returns
a sender name, a message text or a matched secret value.
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

from fym1.data.anonymize import fold
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
)

# Word/name lists the export's sender names are removed from. Shipped under their own licences (see NOTICE).
LIST_GLOBS: tuple[str, ...] = ("src/fym1/data/*.txt",)
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
REPORT_NAME = "RELEASE_CHECK.json"

MIN_NAME_TOKEN = 3  # shorter tokens ("ny", initials) are everyday words or letters; counted as skipped
# Meta's label for a deleted or unknown account: not a person's name.
PLACEHOLDER_SENDERS = frozenset({"facebook user", "utilisateur de facebook", "instagram user", "meta user"})
PRIVATE_HF_ID = "Fan1000-Prog/fyM1-"

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
    "assigned_secret": re.compile(
        r"""(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b["']?\s*[:=]\s*["'][A-Za-z0-9_\-+/=.]{16,}["']"""
    ),
}
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
ALLOWED_EMAILS: dict[str, str] = {
    "x@y.mg": "placeholder example in scripts/build_known_words.py, not an address",
}
ALLOWED_EMAIL_PREFIXES: dict[str, str] = {
    "tests/": "synthetic anonymiser probe addresses (fixtures are synthetic by repo rule)",
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
    unscannable: list[str] = field(default_factory=list)
    secret_hits: dict[str, dict[str, int]] = field(default_factory=dict)
    emails_allowed: int = 0
    private_hf_ids: dict[str, int] = field(default_factory=dict)
    references_to_excluded: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.name_hits or self.secret_hits or self.unscannable)

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
    """Folded (lower case, no accents) letter runs: the whole words names are compared as."""
    return _LETTERS.findall(fold(text))


def name_tokens(senders: Iterable[str]) -> tuple[frozenset[str], int, int]:
    """Name tokens of `senders`, plus how many short tokens and placeholder senders were skipped."""
    tokens, short, placeholders = set(), set(), 0
    for sender in set(senders):
        if " ".join(words(sender)) in PLACEHOLDER_SENDERS:
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
    unescaped = _ESCAPES.sub(lambda m: chr(int(m.group(1), 16)) if m.group(1) else " ", text)
    return sum(1 for word in words(unescaped) if word in names)


def strip_list(text: str, names: frozenset[str]) -> tuple[str, int]:
    """`text` (one entry per line, `#` comments) without entries holding a name token, and how many went."""
    kept, removed = [], 0
    for line in text.splitlines(keepends=True):
        if not line.startswith("#") and any(word in names for word in words(line)):
            removed += 1
        else:
            kept.append(line)
    return "".join(kept), removed


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
    if not path.startswith(tuple(ALLOWED_EMAIL_PREFIXES)):
        if n := sum(1 for email in EMAIL.findall(text) if email not in ALLOWED_EMAILS):
            hits["email"] += n
    return hits


def allowed_email_count(path: str, text: str) -> int:
    emails = EMAIL.findall(text)
    if path.startswith(tuple(ALLOWED_EMAIL_PREFIXES)):
        return len(emails)
    return sum(1 for email in emails if email in ALLOWED_EMAILS)


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


def check_inputs(repo: Path, ref: str, out: Path) -> tuple[Path, str]:
    """The repo root and the commit sha, or RefusedError: dirty tree, bad ref, output inside a worktree or used."""
    try:
        root = Path(_git(repo, "rev-parse", "--show-toplevel").strip()).resolve()
        sha = _git(root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    except subprocess.CalledProcessError:
        raise RefusedError(f"not a git repo or unknown ref: {ref}") from None
    if _git(root, "status", "--porcelain").strip():
        raise RefusedError("working tree is not clean (local changes or untracked files)")
    worktrees = [Path(line[len("worktree "):]).resolve()
                 for line in _git(root, "worktree", "list", "--porcelain").splitlines() if line.startswith("worktree ")]
    target = out.resolve()
    for tree in {root, *worktrees}:
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
        for member in tar.getmembers():
            if member.isdir() or member.name == "pax_global_header":
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise RefusedError(f"unsafe or non-regular archive entry: {member.name}")
            if reason := exclusion_reason(member.name):
                excluded.append({"path": member.name, "reason": reason})
                continue
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
        "under the same licence. Changes: lists were reduced to the most frequent words, folded to lower case",
        "without accents, and entries were removed for privacy when this snapshot was made (counts in",
        f"{REPORT_NAME}). Regenerate the lists with `uv run python scripts/build_known_words.py`.",
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


def build_snapshot(repo: Path, ref: str, out: Path, export: Path) -> Report:
    """Make the snapshot in `out` and check it. Raises RefusedError before copying anything unsafe."""
    root, sha = check_inputs(repo, ref, out)
    names, short, placeholders = name_tokens(export_senders(export))
    if not names:
        raise RefusedError("export gave no sender name tokens")
    out.mkdir(parents=True, exist_ok=True)
    kept, excluded = copy_tracked(root, sha, out)
    report = Report(ref=sha, files_copied=len(kept), excluded=excluded, name_tokens=len(names),
                    short_tokens_skipped=short, placeholder_senders_skipped=placeholders)
    matched = {exclusion_entry(e["path"]) for e in excluded}
    report.excludes_unmatched = [entry.path for entry in EXCLUDES if entry.path not in matched]
    if (out / NOTICE_NAME).exists():
        raise RefusedError(f"{NOTICE_NAME} already tracked; merge the licence notes by hand")
    (out / NOTICE_NAME).write_text(notice_text(), encoding="utf-8")

    for path in filter(is_list_file, kept):
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
        if hits := count_name_hits(text, names) + count_name_hits(path, names):
            report.name_hits[path] = hits
        if secrets := secret_hits(path, text):
            report.secret_hits[path] = dict(sorted(secrets.items()))
        report.emails_allowed += allowed_email_count(path, text)
        if n := text.count(PRIVATE_HF_ID):
            report.private_hf_ids[path] = n
        for ref_path in excluded_refs:
            if ref_path in text:
                report.references_to_excluded[ref_path] = report.references_to_excluded.get(ref_path, 0) + 1
    _redact_named_paths(report, names)
    return report


def _redact_named_paths(report: Report, names: frozenset[str]) -> None:
    """A repo path that itself holds a name token is reported as `<redacted path N>`."""
    aliases: dict[str, str] = {}

    def alias(path: str) -> str:
        if count_name_hits(path, names):
            return aliases.setdefault(path, f"<redacted path {len(aliases) + 1}>")
        return path

    for attr in ("list_entries_removed", "name_hits", "secret_hits", "private_hf_ids"):
        setattr(report, attr, {alias(p): v for p, v in sorted(getattr(report, attr).items())})
    report.unscannable = [alias(p) for p in report.unscannable]
    report.excluded = [{**e, "path": alias(e["path"])} for e in report.excluded]


def write_report(report: Report, *paths: Path) -> None:
    text = json.dumps(report.as_dict(), indent=2, ensure_ascii=False) + "\n"
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
