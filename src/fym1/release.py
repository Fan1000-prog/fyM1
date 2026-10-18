"""Public release snapshot: tracked files of one clean commit, minus private paths, checked for names and secrets.

The public repo is a fresh single-commit repo built from this snapshot, so private git history never ships.
Release blocker (lead decision 21): the export's sender names are read at run time, removed from the hand-written
name lists, and searched for in every file except the lists a script rebuilds from public data alone. The lists
scripts/build_known_words.py builds from public data minus the hand lists do not ship (REBUILT_LISTS). Everything
this module reports is a count or a repo path; it never returns a sender name, a message text, a private id
or a matched secret value.
"""

import fnmatch
import gzip
import hashlib
import html
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import unicodedata
import urllib.parse
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from fym1.data.anonymize import MIN_NAME_PREFIX, NAME_SUFFIXES, compare_key, fold, is_everyday_word, normalise
from fym1.data.gate import PERSON_MARKER_WORDS, is_hidden_char
from fym1.data.gate import tokens as gate_tokens
from fym1.data.known_words import KNOWN_WORDS, known_by
from fym1.data.messenger import load_threads


@dataclass(frozen=True)
class Exclude:
    path: str  # exact repo path, or a folder prefix ending in "/"
    reason: str


# Lists scripts/build_known_words.py writes. It builds them from public data MINUS the hand lists' names, and the
# repo's copies were built from the unstripped hand lists: a rebuild from the shipped (stripped) hand lists and a diff
# would show exactly the names the snapshot removed. So they never ship; anyone rebuilds them from the shipped hand
# lists with REBUILD_COMMAND (large downloads: Wikipedia, INSEE, HPLT).
REBUILD_COMMAND = "uv sync --group data && uv run python scripts/build_known_words.py"
_NOT_SHIPPED = f"built from the unstripped hand lists (a rebuild + diff would show the removed names); rebuild with " \
               f"`{REBUILD_COMMAND}`"
REBUILT_LISTS: tuple[str, ...] = tuple(f"src/fym1/data/{name}" for name in (
    "en_words.txt", "fr_words.txt", "first_names.txt", "wordlike_first_names.txt", "mg_wiki_words.txt",
    "mg_corpus_words.txt",
))

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
    *(Exclude(path, _NOT_SHIPPED) for path in REBUILT_LISTS),
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
# Shipped lists a script writes from public data alone (no hand list goes in). They are shipped exactly as built
# and the name scan skips them: anyone can rebuild one and diff it, so a removed entry would point at a sender, and
# a common word in one reveals nothing about the export. Secrets and private ids are still checked in them.
GENERATED_LISTS: dict[str, str] = {
    "src/fym1/data/mg_common_words.txt": "reproducible from public data by scripts/build_common_words.py "
                                         "(Malagasy Wikipedia)",
}
DATA_LICENCES: dict[str, str] = {
    "src/fym1/data/mg_common_words.txt": "derived from Malagasy Wikipedia (wikimedia/wikipedia 20231101.mg); "
                                         "CC BY-SA 4.0",
    "src/fym1/data/mg_first_names.txt": "hand list, this repo (Apache-2.0)",
    "src/fym1/data/mg_known_names.txt": "hand list, this repo (Apache-2.0)",
    "src/fym1/data/name_deny_list.txt": "hand list, this repo (Apache-2.0)",
}
NOTICE_NAME = "NOTICE"

# The only files read as text; any other type is unscannable and fails the check (zip, gz and PDF have readers).
TEXT_SUFFIXES = frozenset({".py", ".md", ".txt", ".json", ".jsonl", ".ipynb", ".toml", ".yml", ".yaml", ".lock",
                           ".jinja", ".tex", ".bib", ".cfg", ".ini", ".csv", ".tsv", ".sh"})
TEXT_NAMES = frozenset({"LICENSE", "NOTICE", ".gitignore", ".gitkeep", ".gitattributes", ".python-version"})
BOMS = ((b"\xef\xbb\xbf", "utf-8"), (b"\xff\xfe\x00\x00", "utf-32-le"), (b"\x00\x00\xfe\xff", "utf-32-be"),
        (b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"))
MAX_ARCHIVE_DEPTH = 3
MAX_UNPACKED_BYTES = 200 << 20

# The owner chose to publish the paper and the citations under their own name (lead decision 31), and the owner is a
# sender in the export. So exactly that author string, two words, one space, as written, passes the name scan, and
# only in these files; any other form, any other file, or either word alone is still a hit. Kept as a hash so this
# file does not hold the name.
AUTHOR_SHA256 = "44c9ff23a0c635faa8d70a28269daac1dcffa04e70ee5156c06ef7cbf3e453de"
AUTHOR_FILES = frozenset({"README.md", "MODEL_CARD.md", "paper/fym1.tex", "paper/fym1.pdf",
                          "tests/test_public_name.py"})
_WORD_START = re.compile(r"(?<![^\W\d_])[^\W\d_]")
_TWO_WORDS = re.compile(r"(?<![^\W\d_])[^\W\d_]+ [^\W\d_]+(?![^\W\d_])")

MIN_NAME_TOKEN = 3  # shorter tokens ("ny", initials) are everyday words or letters; counted as skipped
# A name this long is a hit anywhere inside a word no known-word list has ("xbodozafyx"). Shorter names are common
# letter runs: a real 4-letter name sits inside everyday English words and code names the lists lack.
MIN_INNER_NAME = 5
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
# An address on a reserved test domain (RFC 2606/6761) is never a person's, anywhere in the snapshot.
RESERVED_EMAIL_TLDS = (".example", ".invalid", ".test")
# Every other address must be listed here, exactly, for the one file that holds it: a placeholder in a script or a
# synthetic probe that needs a real provider's shape (plus-tags, upper case) to test email masking. Any other
# address in any file is a secret hit. Stored as (local part, domain), case-folded, so this file holds no address.
EMAIL_PROBE_FILES: dict[str, frozenset[tuple[str, str]]] = {
    "scripts/build_known_words.py": frozenset({("x", "y.mg")}),  # placeholder in a comment
    "tests/data/test_synth.py": frozenset({("x", "y.mg")}),
    "tests/data/test_anonymize.py": frozenset({("hanta.r", "example.mg")}),
    "tests/data/test_kept_checks.py": frozenset({("rabe.sample", "example.mg")}),
    "tests/data/test_gate.py": frozenset({("olona", "example.mg")}),
    "tests/data/test_known_words.py": frozenset({("fanja", "mail.mg")}),
    "tests/data/test_anonymize_probes.py": frozenset({
        ("hanta+tag", "gmail.com"), ("hanta.rk+fy", "yahoo.fr"), ("hanta", "gmail.com"),
        ("marie-claire.rasoanaivo", "gmail.com"),
    }),
}
SECRET_FILE_NAMES = frozenset({".env", "kaggle.json", "access_token", "id_rsa", "id_ed25519", ".netrc"})
SECRET_FILE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")

_LETTERS = re.compile(r"[^\W\d_]+")
_LETTERS_AND_DIGITS = re.compile(r"[^\W_]+")
# Digits typed for the letters they look like ("H4nt4", "Rak0to"): the name scan also reads them as letters.
DIGIT_LETTERS = str.maketrans("140357", "iaoest")
_ESCAPES = re.compile(r"\\(?:x([0-9a-fA-F]{2})|u([0-9a-fA-F]{4})|U([0-9a-fA-F]{8})|[nrt\"'\\])")
MAX_UNESCAPE_ROUNDS = 5


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
    author_strings_allowed: int = 0
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
                "author_strings_allowed": self.author_strings_allowed,
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


def readings(text: str) -> tuple[str, str]:
    """`text` with escapes decoded (`_unescape`) and normalised as the anonymiser does, read two ways: every
    invisible or combining character the gate flags taken as a word break ("Bodo\u200bzafy" -> "Bodo zafy"), and
    taken out ("Za\u0903fy" -> "Zafy"). A name found in either reading counts."""
    text = _unescape(text)
    split = normalise("".join(" " if is_hidden_char(c) else c for c in unicodedata.normalize("NFC", text)))
    joined = "".join(c for c in normalise(text) if not is_hidden_char(c))
    return split, joined


def _forms(word: str) -> list[str]:
    """Letter runs of one gate token; a token with digits inside ("Rak0to") is also read with DIGIT_LETTERS."""
    forms = _LETTERS.findall(word)
    if any(c.isdigit() for c in word):
        forms += _LETTERS.findall(word.translate(DIGIT_LETTERS))
    return forms


def _reading_words(reading: str) -> list[str]:
    """Words of one reading, split by the strict gate's own tokeniser (`fym1.data.gate.tokens`)."""
    return [form for token in gate_tokens(reading) for form in _forms(token.word)]


def words(text: str) -> list[str]:
    """Compare keys (folded, stretched letters collapsed) of the gate's words in both `readings` of `text`."""
    return [compare_key(word) for reading in dict.fromkeys(readings(text)) for word in _reading_words(reading)]


def is_name_word(word: str, names: frozenset[str]) -> bool:
    """`word` (as typed) is a name, alone or with a glued person marker ("izafy", "n'izafy" reads "izafy", "ryzafy")
    and/or a glued possessive ending ("Zafyko"), the anonymiser's suffix list, or holds a 4+ letter name: at its start
    ("zafybodo") or, for a MIN_INNER_NAME+ letter name when no known-word list has the word, anywhere inside
    ("xfakenainax"). Like the anonymiser, only
    an exact name match counts for an everyday word. A word under MIN_NAME_TOKEN letters is never a name, so a short
    name's collapsed key ("Ann" -> "an") does not match the everyday word "an"."""
    if len(fold(word)) < MIN_NAME_TOKEN:
        return False
    key = compare_key(word)
    if key in names:
        return True
    if is_everyday_word(key):  # checked first, as the anonymiser does: "farany" is not "Fara" + "-ny"
        return False
    stems = {key} | {key[len(m):] for m in PERSON_MARKER_WORDS if key.startswith(m)}
    stems |= {stem[: -len(s)] for stem in stems for s in NAME_SUFFIXES if stem.endswith(s)}
    if not stems.isdisjoint(names):
        return True
    if any(stem.startswith(n) for stem in stems for n in names if len(n) >= MIN_NAME_PREFIX):
        return True
    inner = [n for n in names if len(n) >= MIN_INNER_NAME]
    return bool(inner) and known_by(word, KNOWN_WORDS) is None and any(n in key for n in inner)


def name_tokens(senders: Iterable[str]) -> tuple[frozenset[str], int, int]:
    """Name tokens of `senders` (compare keys of the gate's words in both readings), plus how many short tokens
    (under MIN_NAME_TOKEN letters as typed, before stretched letters collapse) and placeholder senders were skipped.
    """
    tokens, short, placeholders = set(), set(), 0
    placeholder_keys = {" ".join(words(label)) for label in PLACEHOLDER_SENDERS}
    for sender in set(senders):
        if " ".join(words(sender)) in placeholder_keys:
            placeholders += 1
            continue
        for reading in readings(sender):
            for word in _reading_words(reading):
                (tokens if len(fold(word)) >= MIN_NAME_TOKEN else short).add(compare_key(word))
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
    """Case/accent-insensitive name matches (`is_name_word`) in the gate's words of `text`, in whichever of its
    two `readings` finds more. Escapes are decoded first; a run of letters and digits with no name in its gate
    words is read again with DIGIT_LETTERS ("B0d0", "Z4fy2")."""
    return max(_reading_hits(reading, names) for reading in readings(text))


def _token_hits(text: str, names: frozenset[str]) -> list[tuple[int, int]]:
    """Spans of the gate tokens in `text` that hold a name."""
    return [(t.start, t.end) for t in gate_tokens(text) if any(is_name_word(f, names) for f in _forms(t.word))]


def _reading_hits(reading: str, names: frozenset[str]) -> int:
    spans = _token_hits(reading, names)
    hits = len(spans)
    for run in _LETTERS_AND_DIGITS.finditer(reading):
        word = run.group()
        if word.isalpha() or not any(c.isalpha() for c in word):
            continue
        if not any(start < run.end() and end > run.start() for start, end in spans):
            hits += len(_token_hits(word.translate(DIGIT_LETTERS), names))
    return hits


def _unescape(text: str) -> str:
    """`text` with escapes read as the characters they stand for, repeated until nothing changes (escapes inside
    escapes): JSON/Python `\\n`, `\\xXX`, `\\uXXXX`, `\\UXXXXXXXX`, HTML entities (`&#90;`, `&amp;`) and URL
    percent escapes (`%5A`)."""
    for _ in range(MAX_UNESCAPE_ROUNDS):
        decoded = _ESCAPES.sub(_escape_char, text)
        decoded = html.unescape(decoded)
        decoded = urllib.parse.unquote(decoded, errors="replace")
        if decoded == text:
            break
        text = decoded
    return text


def _escape_char(match: re.Match) -> str:
    code = next((g for g in match.groups() if g), None)
    if code is None:  # \n \r \t \" \' \\: a break between words
        return " "
    try:
        return chr(int(code, 16))
    except (ValueError, OverflowError):  # past U+10FFFF: not a character
        return " "


def strip_list(text: str, names: frozenset[str]) -> tuple[str, int]:
    """`text` (one entry per line, `#` comments) without entries holding a name token, and how many went."""
    kept, removed = [], 0
    for line in text.splitlines(keepends=True):
        if not line.startswith("#") and count_name_hits(line, names):
            removed += 1
        else:
            kept.append(line)
    return "".join(kept), removed


def without_author(path: str, text: str) -> tuple[str, int]:
    """`text` with each exact author string (AUTHOR_SHA256) blanked when `path` is in AUTHOR_FILES, and how many."""
    if path not in AUTHOR_FILES:
        return text, 0
    pieces, pos, found = [], 0, 0
    for start in (m.start() for m in _WORD_START.finditer(text)):  # every word start: "Dr A B" also tries "A B"
        if start < pos:
            continue
        match = _TWO_WORDS.match(text, start)
        if match and hashlib.sha256(match.group().encode("utf-8")).hexdigest() == AUTHOR_SHA256:
            pieces += [text[pos:start], " " * len(match.group())]
            pos, found = match.end(), found + 1
    return "".join(pieces) + text[pos:], found


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
    """`email` is on a reserved test domain, or listed exactly for `path` in EMAIL_PROBE_FILES."""
    local, domain = email.casefold().rsplit("@", 1)
    return domain.endswith(RESERVED_EMAIL_TLDS) or (local, domain) in EMAIL_PROBE_FILES.get(path, frozenset())


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


def tracked_files(root: Path, sha: str) -> tuple[dict[str, tuple[bytes, int]], list[dict]]:
    """The commit's regular files minus EXCLUDES, as path -> (bytes, mode), and the excluded entries.

    Every refusal is raised here, before anything is written.
    """
    archive = subprocess.run(["git", "-C", str(root), "archive", "--format=tar", sha],
                             check=True, capture_output=True).stdout
    files, excluded, unsafe = {}, [], 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar.getmembers():
            if member.isdir() or member.name == "pax_global_header":
                continue
            if reason := exclusion_reason(member.name):
                excluded.append({"path": member.name, "reason": reason})
                continue
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():  # symlinks too: fail closed
                unsafe += 1
                continue
            files[member.name] = (tar.extractfile(member).read(), member.mode)
    if unsafe:  # counted, not named: a path may hold a name
        raise RefusedError(f"{unsafe} unsafe or non-regular archive entries (symlink, absolute or ../ path)")
    if NOTICE_NAME in files:
        raise RefusedError(f"{NOTICE_NAME} already tracked; merge the licence notes by hand")
    return files, excluded


def write_files(files: dict[str, tuple[bytes, int]], out: Path) -> list[str]:
    """Write `files` under `out`; returns their paths, sorted."""
    for name, (data, mode) in files.items():
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        dest.chmod(0o755 if mode & 0o111 else 0o644)
    return sorted(files)


def notice_text() -> str:
    lines = [
        "fy-m1: third-party data in this repository",
        "",
        "Code is Apache-2.0 (LICENSE). These data files keep their own licences:",
        "",
        *(f"- {path}: {licence}" for path, licence in sorted(DATA_LICENCES.items())),
        "",
        "The strict gate's other word lists are not included. Build them before using fym1.data.gate or",
        f"fym1.data.known_words: `{REBUILD_COMMAND}`.",
        "It writes:",
        "",
        *(f"- {path}" for path in REBUILT_LISTS),
        "",
        "from Malagasy, French and English Wikipedia (CC BY-SA 4.0), wordfreq by Robyn Speer",
        "(https://github.com/rspeer/wordfreq; word data CC BY-SA 4.0), the INSEE fichier des prenoms 2022",
        "(Licence Ouverte / Etalab 2.0) and HPLT 2.0 cleaned (CC0 1.0). Lists derived from CC BY-SA data are",
        "shared under the same licence (https://creativecommons.org/licenses/by-sa/4.0/).",
        "",
    ]
    return "\n".join(lines)


# ---- scan -----------------------------------------------------------------------------------------------------


def read_text(path: Path) -> str | None:
    """The text the checks read from one snapshot file, or None when it cannot be read in full (fails the check)."""
    return _read(path.name, path.read_bytes(), depth=0)


def _read(name: str, data: bytes, depth: int) -> str | None:
    """Text of a file called `name`: a TEXT_SUFFIXES/TEXT_NAMES file in UTF-8 (UTF-8/16/32 with a BOM), a PDF's
    text and metadata, or a zip/gz archive's member names and texts. Anything else, a NUL byte or a member that
    cannot be read is None: an unknown file type is never guessed at."""
    suffix = PurePosixPath(name).suffix.lower()
    if depth > MAX_ARCHIVE_DEPTH:
        return None
    if suffix == ".pdf":
        return _pdf_text(data)
    if suffix == ".zip":
        return _zip_text(data, depth)
    if suffix == ".gz":
        return _gz_text(name, data, depth)
    if suffix not in TEXT_SUFFIXES and name not in TEXT_NAMES:
        return None
    return _decode(data)


def _decode(data: bytes) -> str | None:
    for bom, codec in BOMS:
        if data.startswith(bom):
            data, encoding = data[len(bom):], codec
            break
    else:
        encoding = "utf-8"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        return None
    return None if "\x00" in text else text  # NULs: UTF-16 without a BOM, or binary posing as text


def _pdf_text(data: bytes) -> str | None:
    """A PDF's page text, document info and XMP metadata (poppler), or None: tools missing, a tool failed, or the
    PDF carries embedded files."""
    if not all(shutil.which(tool) for tool in ("pdftotext", "pdfinfo", "pdfdetach")):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "file.pdf"
        pdf.write_bytes(data)
        runs = [subprocess.run(cmd, capture_output=True) for cmd in (
            ["pdftotext", "-q", str(pdf), "-"], ["pdfinfo", str(pdf)], ["pdfinfo", "-meta", str(pdf)],
            ["pdfdetach", "-list", str(pdf)])]
    if any(run.returncode for run in runs):
        return None
    texts = [_decode(run.stdout) for run in runs]
    if any(text is None for text in texts) or not re.match(r"0 embedded files\b", texts[3]):
        return None
    return "\n".join(texts[:3])


def _zip_text(data: bytes, depth: int) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if sum(m.file_size for m in members) > MAX_UNPACKED_BYTES:
                return None
            texts = [(m.filename, _read(m.filename, archive.read(m), depth + 1)) for m in members]
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, ValueError):  # broken, encrypted, odd method
        return None
    if any(text is None for _, text in texts):
        return None
    return "\n".join(f"{name}\n{text}" for name, text in texts)


def _gz_text(name: str, data: bytes, depth: int) -> str | None:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as file:
            unpacked = file.read(MAX_UNPACKED_BYTES + 1)
    except (OSError, EOFError):
        return None
    if len(unpacked) > MAX_UNPACKED_BYTES:
        return None
    return _read(name[: -len(".gz")], unpacked, depth + 1)


def is_list_file(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in LIST_GLOBS)


def build_snapshot(repo: Path, ref: str, out: Path, export: Path, private_ids: Path) -> Report:
    """Make the snapshot and check it. `out` is written only when every check passes.

    Every RefusedError is raised before anything is written. The snapshot is built in a staging folder next to
    `out` and moved into place only when the report is ok; on a failed check or any error the staging folder is
    deleted, so `out` never holds a partial or failing snapshot.
    """
    root, sha = check_inputs(repo, ref, out)
    ids, id_count = load_private_ids(private_ids, worktrees(root))
    names, short, placeholders = name_tokens(export_senders(export))
    if not names:
        raise RefusedError("export gave no sender name tokens")
    files, excluded = tracked_files(root, sha)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.partial-", dir=out.parent))
    try:
        report = _write_and_check(files, excluded, staging, names, ids)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    report.ref, report.name_tokens, report.private_ids = sha, len(names), id_count
    report.short_tokens_skipped, report.placeholder_senders_skipped = short, placeholders
    if not report.ok:
        shutil.rmtree(staging)
        return report
    if out.exists():
        out.rmdir()  # check_inputs made sure it is empty
    staging.rename(out)
    return report


def _write_and_check(files: dict[str, tuple[bytes, int]], excluded: list[dict], out: Path, names: frozenset[str],
                     ids: re.Pattern[str]) -> Report:
    kept = write_files(files, out)
    report = Report(ref="", files_copied=len(kept), excluded=excluded)
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
            text = ""  # the path itself is still checked
        name_text, authors = without_author(path, text)
        report.author_strings_allowed += authors
        if hits := (0 if path in GENERATED_LISTS else count_name_hits(name_text, names)) + count_name_hits(path, names):
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
        # Read through the same escapes and readings as the scan. Redaction is cheap, so a name anywhere inside
        # a path word counts here, known word or not ("bodozafy").
        if (any(name in word for word in words(path) for name in names) or count_name_hits(path, names)
                or count_private_ids(path, ids)):
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
